"""Serving do modelo (Semana 5): pontua UMA transação com as mesmas features do treino.

O risco central de servir um modelo é o *skew* treino/serving: calcular as
features de um jeito no treino e de outro na API. Aqui não há segunda
implementação: para uma transação nova, monta-se o histórico do cartão
(`card1`) mais a nova linha e chama-se exatamente as funções de
`src/features.py`. Como elas só olham o passado estrito, o resultado é o
mesmo do batch, e `python -m src.serving --check` confere isso nos dados reais.

Artefatos (gerados por `python -m src.serving`, sem retreinar):
  models/xgboost.joblib         modelo (gerado por `python -m src.train`)
  models/serving_bundle.joblib  colunas, categorias, encoder, histórico, limiares

Limitações assumidas (ver docs/metodologia.md, seção 7): o histórico vive na
memória do processo (uma réplica só) e reiniciar perde o que foi acrescentado
depois do bundle.
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import (
    AMOUNT_COLUMN,
    BUNDLE_ENV_VAR,
    DEMO_CARDS,
    ENTITY_COLUMN,
    ID_COLUMN,
    RANDOM_STATE,
    REFERENCE_MAX_MISSING_RATE,
    SERVING_BUNDLE_FILE,
    SERVING_DEMO_BUNDLE_FILE,
    SERVING_MEDIUM_RISK_FPR,
    TARGET_COLUMN,
    TARGET_FPR_FOR_RECALL,
    TEMPORAL_SPLIT_COLUMN,
    XGBOOST_MODEL_FILE,
)
from src.explain import top_contributions
from src.features import (
    add_aggregation_features,
    add_missing_flags,
    add_velocity_features,
    apply_frequency_encoder,
    fit_frequency_encoder,
)

PRODUCT_COLUMN = "ProductCD"
# Colunas que vêm dos campos obrigatórios do request, não do dict `features`.
RESERVED_COLUMNS = {ID_COLUMN, TEMPORAL_SPLIT_COLUMN, AMOUNT_COLUMN, ENTITY_COLUMN, PRODUCT_COLUMN, TARGET_COLUMN}

# Colunas categóricas que o formulário da demo oferece como lista de opções.
META_CATEGORY_COLUMNS = (PRODUCT_COLUMN, "card4", "card6", "P_emaildomain", "DeviceType")

_EMPTY_TIMES = np.empty(0, dtype=np.int64)
_EMPTY_AMOUNTS = np.empty(0, dtype=np.float32)


class HistoryStore:
    """Tempo e valor de cada transação passada, por cartão, ordenados no tempo.

    Os valores ficam em float32 porque o treino leu `TransactionAmt` como
    float32 (downcast em `data_loader`); manter o mesmo tipo evita diferenças
    de arredondamento nas médias e desvios.
    """

    def __init__(self, data: dict[int, tuple[np.ndarray, np.ndarray]] | None = None):
        self._data = data if data is not None else {}

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> HistoryStore:
        df = df[[ENTITY_COLUMN, TEMPORAL_SPLIT_COLUMN, AMOUNT_COLUMN]].sort_values(
            [ENTITY_COLUMN, TEMPORAL_SPLIT_COLUMN], kind="stable"
        )
        data = {
            int(card): (g[TEMPORAL_SPLIT_COLUMN].to_numpy(np.int64), g[AMOUNT_COLUMN].to_numpy(np.float32))
            for card, g in df.groupby(ENTITY_COLUMN, sort=False)
        }
        return cls(data)

    def get(self, card: int) -> tuple[np.ndarray, np.ndarray]:
        return self._data.get(int(card), (_EMPTY_TIMES, _EMPTY_AMOUNTS))

    def add(self, card: int, time_: int, amount: float) -> None:
        """Insere mantendo a ordem por tempo (cria arrays novos; leitores não veem estado parcial)."""
        times, amounts = self.get(card)
        i = int(np.searchsorted(times, time_, side="right"))
        self._data[int(card)] = (np.insert(times, i, time_), np.insert(amounts, i, np.float32(amount)))

    @property
    def n_cards(self) -> int:
        return len(self._data)

    @property
    def n_transactions(self) -> int:
        return sum(len(t) for t, _ in self._data.values())

    def to_dict(self) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        return self._data


@dataclass
class ScoreResult:
    fraud_score: float
    is_fraud_flag: bool
    risk_tier: str
    threshold: float
    top_factors: list[dict] = field(default_factory=list)
    ignored_features: list[str] = field(default_factory=list)


def _native(value):
    """Valor serializável em JSON: NaN vira None, escalares numpy viram Python."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, np.generic):
        value = value.item()
        return None if isinstance(value, float) and np.isnan(value) else value
    return value


class FraudScorer:
    """Modelo + bundle carregados uma vez; `score` pontua uma transação por chamada."""

    def __init__(self, model, bundle: dict, history: HistoryStore | None = None):
        self.model = model
        self.feature_columns: list[str] = bundle["feature_columns"]
        self.raw_columns: list[str] = bundle["raw_columns"]
        self.categories: dict[str, list] = bundle["categories"]
        self.freq_encoder = bundle["freq_encoder"]
        self.thresholds: dict[str, float] = bundle["thresholds"]
        self.demo_cards: list[int] = list(bundle.get("demo_cards", []))
        self.reference_features: dict[str, float | str] = dict(bundle.get("typical_values", {}))
        self.history = history if history is not None else HistoryStore(bundle["history"])
        self._raw_set = set(self.raw_columns)
        self._category_sets = {c: set(v) for c, v in self.categories.items()}
        self._lock = threading.Lock()

        booster_columns = list(model.get_booster().feature_names)
        if booster_columns != self.feature_columns:
            raise ValueError("O bundle não corresponde ao modelo (colunas diferentes). Rode `python -m src.serving`.")

    @classmethod
    def load(cls, model_path=XGBOOST_MODEL_FILE, bundle_path=None) -> FraudScorer:
        bundle_path = Path(bundle_path) if bundle_path else runtime_bundle_file()
        missing = [p for p in (model_path, bundle_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Artefatos ausentes: {', '.join(str(p) for p in missing)}. "
                "Rode `python -m src.train` e depois `python -m src.serving`."
            )
        return cls(joblib.load(model_path), joblib.load(bundle_path))

    def demo_card_stats(self) -> list[dict]:
        """Resumo do histórico de cada cartão de demo (a interface usa para posicionar os exemplos no tempo)."""
        stats = []
        for card in self.demo_cards:
            times, amounts = self.history.get(card)
            if len(times):
                stats.append({
                    "card1": int(card), "n_transactions": int(len(times)),
                    "last_dt": int(times[-1]), "mean_amt": round(float(amounts.mean()), 2),
                })
        return stats

    # --- construção da linha de features ---

    def _convert(self, name: str, value) -> tuple[object, str | None]:
        """Valor no formato do treino, ou (NaN, motivo) se não der para usar."""
        if value is None:
            return np.nan, None
        if name in self.categories:
            text = str(value)
            if text in self._category_sets[name]:
                return text, None
            return np.nan, "categoria não vista no treino"
        try:
            return float(value), None
        except (TypeError, ValueError):
            return np.nan, "valor não numérico"

    def _entity_features(self, card1: int, transaction_dt: int, amount: np.float32) -> dict[str, float]:
        """Features de velocidade e agregação: histórico do cartão + a nova linha, via `features.py`."""
        times, amounts = self.history.get(card1)
        frame = pd.DataFrame({
            TEMPORAL_SPLIT_COLUMN: np.append(times, np.int64(transaction_dt)),
            AMOUNT_COLUMN: np.append(amounts, amount).astype(np.float32),
            ENTITY_COLUMN: np.int64(card1),
        })
        frame = add_aggregation_features(add_velocity_features(frame))
        base = {TEMPORAL_SPLIT_COLUMN, AMOUNT_COLUMN, ENTITY_COLUMN}
        new_row = frame.iloc[-1]
        return {c: float(new_row[c]) for c in frame.columns if c not in base}

    def build_row(
        self,
        *,
        transaction_dt: int,
        transaction_amt: float,
        card1: int,
        product_cd: str | None,
        features: dict | None = None,
        transaction_id: int | None = None,
    ) -> tuple[pd.DataFrame, list[str]]:
        """Uma linha com as 451 colunas do modelo, na ordem e nos dtypes do treino."""
        ignored: list[str] = []
        values: dict[str, object] = {}
        for name, value in (features or {}).items():
            if name in RESERVED_COLUMNS:
                ignored.append(f"{name}: já é informado nos campos do request")
            elif name not in self._raw_set:
                ignored.append(f"{name}: coluna desconhecida")
            else:
                values[name], problem = self._convert(name, value)
                if problem:
                    ignored.append(f"{name}: {problem}")
        values[PRODUCT_COLUMN], problem = self._convert(PRODUCT_COLUMN, product_cd)
        if problem:
            ignored.append(f"{PRODUCT_COLUMN}: {problem}")
        if transaction_id is not None:
            values[ID_COLUMN] = float(transaction_id)

        amount = np.float32(transaction_amt)  # mesmo dtype do treino
        data: dict[str, object] = {}
        for col in self.raw_columns:
            if col in self.categories:
                data[col] = pd.Categorical([values.get(col)], categories=self.categories[col])
            elif col == TEMPORAL_SPLIT_COLUMN:
                data[col] = np.array([transaction_dt], dtype=np.int64)
            elif col == AMOUNT_COLUMN:
                data[col] = np.array([amount], dtype=np.float32)
            elif col == ENTITY_COLUMN:
                data[col] = np.array([card1], dtype=np.int64)
            else:
                data[col] = np.array([values.get(col, np.nan)], dtype=np.float64)

        # Mesma ordem do `build_features`: flags de ausência, entidade, frequency encoding.
        row = add_missing_flags(pd.DataFrame(data))
        for name, value in self._entity_features(card1, transaction_dt, amount).items():
            row[name] = value
        row = apply_frequency_encoder(row, self.freq_encoder)
        return row[self.feature_columns], ignored

    # --- pontuação ---

    def _tier(self, score: float) -> str:
        if score >= self.thresholds["alto"]:
            return "alto"
        return "médio" if score >= self.thresholds["medio"] else "baixo"

    def score(
        self,
        *,
        transaction_dt: int,
        transaction_amt: float,
        card1: int,
        product_cd: str | None,
        features: dict | None = None,
        transaction_id: int | None = None,
        record: bool = True,
        top_k: int = 5,
    ) -> ScoreResult:
        """Pontua uma transação. Com `record=True`, ela entra no histórico do cartão depois."""
        with self._lock:
            X, ignored = self.build_row(
                transaction_dt=transaction_dt, transaction_amt=transaction_amt, card1=card1,
                product_cd=product_cd, features=features, transaction_id=transaction_id,
            )
            # Um DMatrix só: converter as 451 colunas é o custo dominante, e score
            # e contribuições SHAP saem do mesmo booster.
            booster = self.model.get_booster()
            dmatrix = xgb.DMatrix(X, enable_categorical=True)
            score = float(booster.predict(dmatrix)[0])
            contribs = pd.Series(booster.predict(dmatrix, pred_contribs=True)[0, :-1], index=X.columns)
            factors = top_contributions(contribs, X, top_k)
            if record:
                self.history.add(card1, transaction_dt, transaction_amt)
        return ScoreResult(
            fraud_score=score,
            is_fraud_flag=score >= self.thresholds["alto"],
            risk_tier=self._tier(score),
            threshold=self.thresholds["alto"],
            top_factors=[
                {"feature": r.feature, "value": _native(r.valor), "shap": float(r.shap)}
                for r in factors.itertuples()
            ],
            ignored_features=ignored,
        )


def runtime_bundle_file() -> Path:
    """Bundle a carregar: `FRAUD_BUNDLE_FILE` se definida (a imagem Docker aponta para o de demo), senão o completo.

    Lida na hora da chamada, não na importação, para os testes poderem trocá-la.
    """
    return Path(os.environ.get(BUNDLE_ENV_VAR) or SERVING_BUNDLE_FILE)


# --- geração do bundle e checagem de paridade ---


def build_bundle(model_path=XGBOOST_MODEL_FILE) -> dict:
    """Monta o bundle a partir dos parquets em cache e do modelo salvo (sem retreinar)."""
    from src.evaluate import recall_at_fpr
    from src.train import engineered_feature_columns, load_feature_data, make_xy

    model = joblib.load(model_path)
    train_df, test_df = load_feature_data()
    engineered = set(engineered_feature_columns(train_df))
    raw_columns = [c for c in train_df.columns if c not in engineered]
    # Categorias do TREINO: são as que o modelo viu.
    categories = {
        c: list(train_df[c].cat.categories) for c in raw_columns if isinstance(train_df[c].dtype, pd.CategoricalDtype)
    }

    X_test, y_test = make_xy(test_df)
    proba = model.predict_proba(X_test)[:, 1]
    thresholds = {
        "alto": recall_at_fpr(y_test, proba, TARGET_FPR_FOR_RECALL)[1],
        "medio": recall_at_fpr(y_test, proba, SERVING_MEDIUM_RISK_FPR)[1],
    }
    history = HistoryStore.from_frame(pd.concat([train_df, test_df], ignore_index=True))
    return {
        "feature_columns": list(model.get_booster().feature_names),
        "raw_columns": raw_columns,
        "categories": categories,
        "freq_encoder": fit_frequency_encoder(train_df),
        "typical_values": typical_values(train_df, raw_columns),
        "history": history.to_dict(),
        "thresholds": thresholds,
    }


def typical_values(
    df: pd.DataFrame, columns: list[str], max_missing: float = REFERENCE_MAX_MISSING_RATE
) -> dict[str, float | str]:
    """Mediana (numéricas) ou moda (categóricas) das colunas que raramente faltam.

    Serve de transação de referência para a demo. Importa porque colunas que quase
    nunca faltam nos dados reais (C1 a C14, D1, ...) chegam ausentes num payload
    esparso, e o modelo trata ausência dessas colunas como forte sinal de transação
    legítima (SHAP de -2.0 em `C3`, por exemplo). Um payload só com alguns campos fica
    fora da distribuição de treino e pontua perto de zero, seja qual for o risco.
    São agregados (um valor por coluna), não linhas do dataset.
    """
    out: dict[str, float | str] = {}
    for col in columns:
        if col in RESERVED_COLUMNS:
            continue
        series = df[col]
        if series.isna().mean() > max_missing:
            continue
        values = series.dropna()
        out[col] = str(values.mode().iloc[0]) if isinstance(series.dtype, pd.CategoricalDtype) else float(values.median())
    return out


def build_demo_bundle(bundle: dict, cards=DEMO_CARDS) -> dict:
    """Bundle de demonstração: o histórico e o encoder de `card1` só dos `cards` pedidos.

    Cartões fora da lista passam a se comportar como cartão novo: sem histórico
    (n_prev = 0, médias NaN) e `card1_freq` = 0 (nunca visto). O resto do bundle
    (colunas, categorias, limiares e as frequências de e-mail, dispositivo e
    navegador) são agregados e ficam como estão.
    """
    cards = [int(c) for c in cards]
    missing = [c for c in cards if c not in bundle["history"]]
    if missing:
        raise ValueError(f"Cartões sem histórico no bundle completo: {missing}")
    encoder = dict(bundle["freq_encoder"])
    if ENTITY_COLUMN in encoder:
        encoder[ENTITY_COLUMN] = encoder[ENTITY_COLUMN][encoder[ENTITY_COLUMN].index.isin(cards)]
    return {
        **bundle,
        "history": {c: bundle["history"][c] for c in cards},
        "freq_encoder": encoder,
        "demo_cards": cards,
    }


def _same(a, b) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    if isinstance(a, str) or isinstance(b, str):
        return str(a) == str(b)
    return bool(np.isclose(float(a), float(b), rtol=1e-6, atol=1e-6))


def check_parity(scorer: FraudScorer, test_df: pd.DataFrame, n: int = 300, seed: int = RANDOM_STATE) -> dict:
    """Pontua `n` linhas reais do teste pelo caminho de serving e compara com o batch.

    Compara (1) as 451 features de cada linha com as do parquet e (2) o score
    com o `predict_proba` em lote. Usa `record=False`: o histórico não muda.
    """
    from src.train import make_xy

    sample = test_df.sample(n=min(n, len(test_df)), random_state=seed)
    X_batch, _ = make_xy(sample)
    batch_scores = scorer.model.predict_proba(X_batch)[:, 1]

    bad_columns: dict[str, int] = {}
    score_diffs, ignored_rows, elapsed = [], 0, 0.0
    for i, (_, row) in enumerate(sample.iterrows()):
        payload = dict(
            transaction_dt=int(row[TEMPORAL_SPLIT_COLUMN]),
            transaction_amt=float(row[AMOUNT_COLUMN]),
            card1=int(row[ENTITY_COLUMN]),
            product_cd=None if pd.isna(row[PRODUCT_COLUMN]) else str(row[PRODUCT_COLUMN]),
            features={
                c: v for c, v in row.items()
                if c in scorer._raw_set and c not in RESERVED_COLUMNS and pd.notna(v)
            },
        )
        X_online, ignored = scorer.build_row(**payload)
        ignored_rows += bool(ignored)
        for col in scorer.feature_columns:
            if not _same(X_online.iloc[0][col], X_batch.iloc[i][col]):
                bad_columns[col] = bad_columns.get(col, 0) + 1
        t = time.perf_counter()
        result = scorer.score(**payload, record=False)
        elapsed += time.perf_counter() - t
        score_diffs.append(abs(result.fraud_score - float(batch_scores[i])))

    return {
        "linhas": len(sample),
        "colunas_com_divergencia": bad_columns,
        "linhas_com_campos_ignorados": ignored_rows,
        "max_diff_score": float(np.max(score_diffs)),
        "linhas_diff_score_gt_1e-5": int(np.sum(np.array(score_diffs) > 1e-5)),
        "ms_por_chamada": round(1000 * elapsed / len(sample), 1),
    }


def main(check: bool = False, n: int = 300, demo: bool = False) -> None:
    if demo:
        if not SERVING_BUNDLE_FILE.exists():
            raise SystemExit(f"{SERVING_BUNDLE_FILE} não existe. Rode `python -m src.serving` antes.")
        bundle = build_demo_bundle(joblib.load(SERVING_BUNDLE_FILE))
        joblib.dump(bundle, SERVING_DEMO_BUNDLE_FILE, compress=3)
        history = HistoryStore(bundle["history"])
        print(f"Bundle de demo salvo em {SERVING_DEMO_BUNDLE_FILE} "
              f"({SERVING_DEMO_BUNDLE_FILE.stat().st_size / 1e6:.2f} MB)")
        print(f"  cartões: {bundle['demo_cards']} | {history.n_transactions} transações no histórico")
        print(f"  encoder de card1: {len(bundle['freq_encoder'][ENTITY_COLUMN])} cartões")
        return
    if check:
        from src.train import load_feature_data

        scorer = FraudScorer.load()
        _, test_df = load_feature_data()
        report = check_parity(scorer, test_df, n=n)
        print("Paridade serving x batch:")
        for key, value in report.items():
            print(f"  {key}: {value}")
        ok = not report["colunas_com_divergencia"] and report["max_diff_score"] < 1e-5
        print("OK" if ok else "DIVERGÊNCIA")
        raise SystemExit(0 if ok else 1)

    t0 = time.perf_counter()
    bundle = build_bundle()
    SERVING_BUNDLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, SERVING_BUNDLE_FILE, compress=3)
    history = HistoryStore(bundle["history"])
    print(f"Bundle salvo em {SERVING_BUNDLE_FILE} ({SERVING_BUNDLE_FILE.stat().st_size / 1e6:.1f} MB, "
          f"{time.perf_counter() - t0:.0f}s)")
    print(f"  {len(bundle['feature_columns'])} features, {len(bundle['raw_columns'])} colunas brutas, "
          f"{len(bundle['categories'])} categóricas")
    print(f"  histórico: {history.n_cards} cartões, {history.n_transactions} transações")
    print(f"  limiares: {bundle['thresholds']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="compara o serving com o batch em linhas reais do teste")
    parser.add_argument("-n", type=int, default=300, help="linhas da checagem de paridade")
    parser.add_argument("--demo", action="store_true", help="gera o bundle de demonstração a partir do completo")
    args = parser.parse_args()
    main(check=args.check, n=args.n, demo=args.demo)
