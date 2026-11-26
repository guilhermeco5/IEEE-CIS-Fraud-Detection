"""Engenharia de features (Semana 2).

Este módulo concentra as features derivadas, o que diferencia um
projeto de "rodei um classificador" de um projeto de "fiz engenharia
de dados para o problema de fraude". Cada função tem uma justificativa
de negócio documentada no docstring.

Regra que vale para todas as features de entidade: cada linha só enxerga
transações **estritamente anteriores** no tempo (`TransactionDT < t`).
Transações com o mesmo timestamp da linha atual (empates) também ficam de
fora, porque a ordem entre empates não é determinística e incluí-las deixaria
a feature dependente de um detalhe arbitrário do sort.
"""

import numpy as np
import pandas as pd

from src.config import (
    AMOUNT_COLUMN,
    ENTITY_COLUMN,
    FREQ_ENCODE_COLUMNS,
    ID_COLUMN,
    MISSING_FLAG_COLUMNS,
    TEMPORAL_SPLIT_COLUMN,
    VELOCITY_WINDOWS_SECONDS,
)


class _EntityTimeline:
    """Índice ordenado (entidade, tempo) reutilizado pelas features de entidade.

    Codifica cada linha como uma chave única `codigo_entidade * BIG + tempo`,
    o que permite responder "quantas transações desta entidade ocorreram
    antes de t" com `np.searchsorted` em O(n log n), sem loop por grupo.
    """

    def __init__(self, df: pd.DataFrame, entity_col: str, time_col: str):
        times = df[time_col].to_numpy(dtype=np.int64)
        if len(times) and times.min() < 0:
            raise ValueError(f"{time_col} deve ser não-negativo.")
        codes, _ = pd.factorize(df[entity_col], use_na_sentinel=True)

        self.valid = codes >= 0  # entidade ausente -> features ficam NaN
        self.big = np.int64(times.max() + max(VELOCITY_WINDOWS_SECONDS.values()) + 1) if len(times) else 1
        self.order = np.lexsort((times, codes))  # posições originais na ordem (entidade, tempo)
        self.codes = codes[self.order].astype(np.int64)
        self.times = times[self.order]
        self.keys = self.codes * self.big + self.times

        # Primeira posição do grupo da entidade e primeira posição com o mesmo
        # timestamp: tudo em [group_start, first_tie) é passado estrito.
        self.group_start = np.searchsorted(self.keys, self.codes * self.big, side="left")
        self.first_tie = np.searchsorted(self.keys, self.keys, side="left")
        self.n_prev = self.first_tie - self.group_start

    def count_since(self, window_seconds: int) -> np.ndarray:
        """Número de transações anteriores da entidade em [t - window, t)."""
        lower = np.maximum(self.times - window_seconds, 0)
        lo = np.searchsorted(self.keys, self.codes * self.big + lower, side="left")
        return self.first_tie - lo

    def to_original_order(self, values: np.ndarray) -> np.ndarray:
        out = np.empty(len(values), dtype=np.float64)
        out[self.order] = values
        out[~self.valid] = np.nan
        return out


def add_velocity_features(
    df: pd.DataFrame,
    entity_col: str = ENTITY_COLUMN,
    time_col: str = TEMPORAL_SPLIT_COLUMN,
    windows: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Conta quantas transações a mesma entidade fez nas últimas N horas,
    e há quanto tempo foi a transação anterior.

    Hipótese de negócio: fraude costuma vir em rajadas; um cartão
    comprometido gera muitas transações em pouco tempo, diferente do
    padrão de uso legítimo.

    Colunas criadas: `{entity}_cnt_{janela}` (janela semiaberta [t-W, t),
    sem contar a própria transação nem empates de timestamp) e
    `{entity}_secs_since_prev` (NaN na primeira transação da entidade).
    """
    windows = windows or VELOCITY_WINDOWS_SECONDS
    tl = _EntityTimeline(df, entity_col, time_col)
    out = df.copy()

    for label, seconds in windows.items():
        out[f"{entity_col}_cnt_{label}"] = tl.to_original_order(tl.count_since(seconds))

    prev_pos = tl.first_tie - 1
    has_prev = tl.n_prev > 0
    secs = np.full(len(tl.times), np.nan)
    secs[has_prev] = tl.times[has_prev] - tl.times[prev_pos[has_prev]]
    out[f"{entity_col}_secs_since_prev"] = tl.to_original_order(secs)
    return out


def add_aggregation_features(
    df: pd.DataFrame,
    entity_col: str = ENTITY_COLUMN,
    value_col: str = AMOUNT_COLUMN,
    time_col: str = TEMPORAL_SPLIT_COLUMN,
) -> pd.DataFrame:
    """Agregações históricas por entidade (contagem, média, desvio e máximo
    do valor) e o quanto a transação atual foge desse padrão.

    Hipótese de negócio: uma transação muito acima do padrão histórico da
    entidade é um sinal de risco clássico (ex: fraudador "testando" o
    limite depois de compras pequenas).

    Só usa o passado estrito: a própria linha nunca entra na sua média.
    Colunas: `{entity}_n_prev`, `{entity}_amt_mean_prev`,
    `{entity}_amt_std_prev` (NaN com <2 transações anteriores),
    `{entity}_amt_max_prev`, `{entity}_amt_ratio_mean` (valor / média) e
    `{entity}_amt_zscore` (NaN se o desvio for 0 ou indefinido).
    """
    tl = _EntityTimeline(df, entity_col, time_col)
    values = df[value_col].to_numpy(dtype=np.float64)[tl.order]

    # Somas acumuladas com zero à esquerda: soma de [a, b) = S[b] - S[a].
    s1 = np.concatenate(([0.0], np.cumsum(values)))
    s2 = np.concatenate(([0.0], np.cumsum(values**2)))
    n = tl.n_prev.astype(np.float64)
    start, end = tl.group_start, tl.first_tie

    with np.errstate(invalid="ignore", divide="ignore"):
        total = s1[end] - s1[start]
        mean = np.where(n > 0, total / n, np.nan)
        var = np.where(n > 1, (s2[end] - s2[start] - total**2 / n) / (n - 1), np.nan)
        std = np.sqrt(np.clip(var, 0, None))

        # Máximo acumulado por entidade, lido na última posição do passado estrito.
        cummax = pd.Series(values).groupby(tl.codes).cummax().to_numpy()
        max_prev = np.where(n > 0, cummax[np.maximum(end - 1, 0)], np.nan)

        ratio = values / mean
        zscore = np.where(std > 0, (values - mean) / std, np.nan)

    out = df.copy()
    prefix = f"{entity_col}_"
    out[prefix + "n_prev"] = tl.to_original_order(n)
    out[prefix + "amt_mean_prev"] = tl.to_original_order(mean)
    out[prefix + "amt_std_prev"] = tl.to_original_order(std)
    out[prefix + "amt_max_prev"] = tl.to_original_order(max_prev)
    out[prefix + "amt_ratio_mean"] = tl.to_original_order(ratio)
    out[prefix + "amt_zscore"] = tl.to_original_order(zscore)
    return out


def fit_frequency_encoder(train_df: pd.DataFrame, columns: list[str] | None = None) -> dict[str, pd.Series]:
    """Aprende a frequência relativa de cada categoria, **só no treino**.

    Por que frequency encoding e não target encoding: não usa o rótulo,
    então não há como vazar `isFraud` para a feature. Categorias raras
    (domínio de e-mail obscuro, dispositivo pouco comum) ficam com
    frequência baixa, que já é um sinal em si. O modelo de árvore ainda
    recebe a coluna original como `category` para capturar o resto.
    """
    columns = columns if columns is not None else FREQ_ENCODE_COLUMNS
    return {
        col: train_df[col].value_counts(normalize=True, dropna=True)
        for col in columns
        if col in train_df.columns
    }


def apply_frequency_encoder(df: pd.DataFrame, encoder: dict[str, pd.Series]) -> pd.DataFrame:
    """Adiciona `{col}_freq`. Categoria nunca vista no treino -> 0; ausente -> NaN."""
    out = df.copy()
    for col, freqs in encoder.items():
        if col not in out.columns:
            continue
        raw = out[col].astype(object)
        mapped = raw.map(freqs.to_dict()).astype(np.float64)
        out[f"{col}_freq"] = mapped.where(raw.isna() | mapped.notna(), 0.0)
    return out


def add_missing_flags(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Flags binárias de ausência + `has_identity`.

    Hipótese (revisada no EDA, docs/metodologia.md seção 2): a ausência
    de dados de identidade carrega um sinal residual, mas confundido por
    `ProductCD`. As flags entram junto com `ProductCD` como categórica,
    para que a árvore aprenda a interação em vez de usarmos a ausência
    como discriminador agnóstico de produto.
    """
    columns = columns if columns is not None else MISSING_FLAG_COLUMNS
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[f"{col}_is_missing"] = out[col].isna().astype(np.int8)

    identity_cols = [c for c in out.columns if c.startswith("id_") and not c.endswith(("_is_missing", "_freq"))]
    identity_cols += [c for c in ("DeviceType", "DeviceInfo") if c in out.columns]
    if identity_cols:
        out["has_identity"] = out[identity_cols].notna().any(axis=1).astype(np.int8)
    return out


def build_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pipeline completo de features para um par treino/teste temporal.

    Velocidade e agregações são calculadas sobre a linha do tempo
    concatenada (treino + teste). Isso não vaza: cada linha só olha o
    passado estrito, então uma linha de teste usar o histórico do treino é
    exatamente o que aconteceria em produção. O frequency encoder, ao
    contrário, é ajustado só no treino e aplicado aos dois.

    Devolve os dois blocos na ordem original de linhas de cada um.
    """
    marker = "__is_test__"
    combined = pd.concat(
        [train_df.assign(**{marker: False}), test_df.assign(**{marker: True})],
        ignore_index=True,
    )
    combined = add_velocity_features(combined)
    combined = add_aggregation_features(combined)
    combined = add_missing_flags(combined)

    is_test = combined.pop(marker).to_numpy()
    train_feat = combined[~is_test].reset_index(drop=True)
    test_feat = combined[is_test].reset_index(drop=True)

    encoder = fit_frequency_encoder(train_feat)
    return apply_frequency_encoder(train_feat, encoder), apply_frequency_encoder(test_feat, encoder)


if __name__ == "__main__":
    import time

    from src.data_loader import load_raw_data
    from src.preprocessing import temporal_train_test_split

    t0 = time.perf_counter()
    train, test = temporal_train_test_split(load_raw_data())
    train_f, test_f = build_features(train, test)
    new_cols = [c for c in train_f.columns if c not in train.columns]
    print(f"Treino: {train_f.shape} | Teste: {test_f.shape} | {time.perf_counter() - t0:.1f}s")
    print(f"{len(new_cols)} features novas:")
    print(train_f[new_cols].isna().mean().rename("pct_nan_treino").to_string())
    assert ID_COLUMN in train_f.columns
