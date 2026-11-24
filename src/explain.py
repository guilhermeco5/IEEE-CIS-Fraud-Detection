"""Explicabilidade do XGBoost com SHAP (Semana 4).

Os valores SHAP vêm do TreeSHAP nativo do XGBoost
(`Booster.predict(..., pred_contribs=True)`): exato, rápido e compatível
com as categóricas nativas (`enable_categorical`), que o
`shap.TreeExplainer` não trata de forma confiável. A biblioteca `shap`
entra só nos gráficos.

Escala: log-odds (margem do modelo). Para cada linha,
    base_value + soma(SHAP) == margem,  e  sigmoid(margem) == predict_proba.
A margem já inclui o deslocamento do `scale_pos_weight`, então o
base_value não é a taxa de fraude real; a leitura útil é a relativa
(quem empurra o score para cima ou para baixo).

Duas leituras:
  - global: |SHAP| médio numa amostra do teste, comparado com o gain;
  - local: por que o modelo deu aquele score para UMA transação
    (`explain_transaction`), ilustrado com um acerto, um alarme falso e
    uma fraude perdida no limiar de FPR 1%.

Uso: `python -m src.explain` (usa o modelo salvo por `python -m src.train`).
"""

import time

import joblib
import matplotlib

matplotlib.use("Agg")  # sem janela: só salva PNG
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from src.config import (
    FIGURES_DIR,
    ID_COLUMN,
    METRICS_DIR,
    RANDOM_STATE,
    SHAP_DEPENDENCE_FEATURES,
    SHAP_SAMPLE_SIZE,
    SHAP_TOP_N,
    TARGET_FPR_FOR_RECALL,
    XGBOOST_MODEL_FILE,
)
from src.evaluate import recall_at_fpr
from src.train import engineered_feature_columns, load_feature_data, make_xy

BIAS_COLUMN = "__bias__"


def load_model(path=XGBOOST_MODEL_FILE) -> xgb.XGBClassifier:
    if not path.exists():
        raise FileNotFoundError(f"Modelo não encontrado em {path}. Rode `python -m src.train` antes.")
    return joblib.load(path)


def shap_values(model: xgb.XGBClassifier, X: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Contribuições SHAP por linha e feature (log-odds) e o valor base.

    `X` precisa ter as mesmas colunas, na mesma ordem e com os mesmos dtypes
    (incluindo `category`) usados no treino.
    """
    dmatrix = xgb.DMatrix(X, enable_categorical=True)
    contribs = model.get_booster().predict(dmatrix, pred_contribs=True)
    shap_df = pd.DataFrame(contribs[:, :-1], index=X.index, columns=X.columns)
    # A última coluna é o bias, constante em todas as linhas.
    return shap_df, float(contribs[0, -1])


def global_importance(shap_df: pd.DataFrame) -> pd.Series:
    """|SHAP| médio por feature, do maior para o menor."""
    return shap_df.abs().mean().sort_values(ascending=False).rename("mean_abs_shap")


def compare_with_gain(importance: pd.Series, gain: pd.Series) -> pd.DataFrame:
    """Tabela SHAP x gain com o rank de cada critério (1 = mais importante).

    Features sem nenhum split não aparecem no gain; ficam com gain 0 e rank NaN.
    """
    table = importance.to_frame()
    table["rank_shap"] = np.arange(1, len(table) + 1)
    table["gain"] = gain.reindex(table.index).fillna(0.0)
    table["rank_gain"] = gain.rank(ascending=False, method="min").reindex(table.index)
    table.index.name = "feature"
    return table


def sample_rows(df: pd.DataFrame, n: int = SHAP_SAMPLE_SIZE, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Amostra aleatória simples (mantém a taxa de fraude natural do teste)."""
    return df if len(df) <= n else df.sample(n=n, random_state=seed).sort_index()


def _to_explanation(shap_df: pd.DataFrame, base_value: float, X: pd.DataFrame) -> shap.Explanation:
    # Categóricas viram código numérico só para a cor do beeswarm.
    data = X.apply(lambda s: s.cat.codes.where(s.notna()) if isinstance(s.dtype, pd.CategoricalDtype) else s)
    return shap.Explanation(
        values=shap_df.to_numpy(),
        base_values=np.full(len(shap_df), base_value),
        data=data.to_numpy(dtype=float),
        feature_names=list(shap_df.columns),
    )


def plot_global(explanation: shap.Explanation, top_n: int = SHAP_TOP_N) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for kind, plot in [("bar", shap.plots.bar), ("beeswarm", shap.plots.beeswarm)]:
        plot(explanation, max_display=top_n + 1, show=False)  # +1: a linha "soma das outras"
        plt.title(f"SHAP: top {top_n} features (teste, log-odds)")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"shap_{kind}_top{top_n}.png", dpi=150, bbox_inches="tight")
        plt.close("all")


def group_share(importance: pd.Series, columns: list[str]) -> float:
    """Fatia do |SHAP| médio total que vem de um grupo de colunas (0 a 1)."""
    return float(importance[importance.index.isin(columns)].sum() / importance.sum())


def binned_trend(x: pd.Series, shap_col: pd.Series, bins: int = 5) -> pd.DataFrame:
    """SHAP médio por faixa (quantis) do valor da feature.

    Resume numericamente o que o gráfico de dependência mostra: se o SHAP
    sobe com a feature, se é monotônico ou não. Ausentes viram uma linha à
    parte, porque "faltou" é informação para o XGBoost.
    """
    valid = x.notna()
    bands = pd.qcut(x[valid], q=bins, duplicates="drop")
    out = shap_col[valid].groupby(bands, observed=True).agg(n="size", shap_medio="mean")
    out.insert(0, "x_mediano", x[valid].groupby(bands, observed=True).median())
    out.index = out.index.astype(str)
    if (~valid).any():
        out.loc["ausente"] = [np.nan, int((~valid).sum()), shap_col[~valid].mean()]
    out.index.name = "faixa"
    return out


def plot_dependence(shap_df: pd.DataFrame, X: pd.DataFrame, features: list[str]) -> None:
    """SHAP x valor da feature, com a tendência por faixas de quantil por cima.

    O eixo x é cortado nos percentis 1 e 99 e usa escala symlog, porque
    contagens e tempos têm cauda longa que esmagaria o gráfico.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ncols = 3
    nrows = -(-len(features) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows), squeeze=False)
    for ax, feature in zip(axes.ravel(), features):
        x, s = X[feature].astype(float), shap_df[feature]
        lo, hi = x.quantile([0.01, 0.99])
        ax.scatter(x, s, s=3, alpha=0.15, color="#9aa5b1", rasterized=True)
        trend = binned_trend(x, s, bins=20).dropna(subset=["x_mediano"])
        ax.plot(trend["x_mediano"], trend["shap_medio"], color="#d62728", lw=2)
        ax.axhline(0, color="black", lw=0.6)
        # Região linear só até o 5º percentil dos positivos; senão uma feature
        # em [0, 0.03] (card1_freq) ficaria toda dentro da faixa linear.
        positive = x[x > 0]
        ax.set_xscale("symlog", linthresh=float(positive.quantile(0.05)) if len(positive) else 1.0)
        ax.set_xlim(lo, hi)
        ax.set_title(feature)
        ax.set_xlabel("valor da feature")
        ax.set_ylabel("SHAP (log-odds)")
    for ax in axes.ravel()[len(features):]:
        ax.axis("off")
    fig.suptitle("Features da Semana 2: SHAP por valor (linha vermelha = média por quantil)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "shap_dependence_semana2.png", dpi=150, bbox_inches="tight")
    plt.close("all")


def explain_transaction(model: xgb.XGBClassifier, row: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    """As `top_k` contribuições SHAP de maior |SHAP| para UMA transação.

    `row` é um DataFrame de uma linha, no mesmo formato do treino. Sinal
    positivo empurra o score para fraude; negativo, para legítima. Pensado
    para ser reaproveitado no `/predict` da API (Semana 5).
    """
    if len(row) != 1:
        raise ValueError(f"esperava 1 linha, recebi {len(row)}")
    shap_df, _ = shap_values(model, row)
    contrib = shap_df.iloc[0]
    order = contrib.abs().sort_values(ascending=False).index[:top_k]
    return pd.DataFrame({
        "feature": order,
        "valor": [row.iloc[0][f] for f in order],
        "shap": contrib[order].to_numpy(),
    })


def select_cases(y_true: pd.Series, proba: pd.Series, threshold: float) -> dict[str, object]:
    """Índice de um caso de cada tipo, no limiar de decisão dado.

    tp: fraude com o maior score (o modelo mais seguro e certo).
    fp: legítima com o maior score (o alarme falso mais confiante).
    fn: fraude com o menor score (a fraude mais "invisível" ao modelo).
    """
    flagged = proba >= threshold
    fraud = y_true == 1
    return {
        "tp": proba[fraud & flagged].idxmax(),
        "fp": proba[~fraud & flagged].idxmax(),
        "fn": proba[fraud & ~flagged].idxmin(),
    }


def plot_waterfall(shap_row: pd.Series, base_value: float, x_row: pd.Series, title: str, path) -> None:
    explanation = shap.Explanation(
        values=shap_row.to_numpy(),
        base_values=base_value,
        data=x_row.to_numpy(dtype=object),  # mantém o texto das categóricas
        feature_names=list(shap_row.index),
    )
    shap.plots.waterfall(explanation, max_display=12, show=False)
    plt.title(title)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")


CASE_LABELS = {
    "tp": "Fraude detectada (verdadeiro positivo)",
    "fp": "Alarme falso (falso positivo)",
    "fn": "Fraude não detectada (falso negativo)",
}


def explain_cases(model: xgb.XGBClassifier, test_df: pd.DataFrame, top_k: int = 10) -> dict[str, pd.DataFrame]:
    """Escolhe TP/FP/FN no limiar de FPR 1% do teste inteiro e explica cada um."""
    X, y = make_xy(test_df)
    proba = pd.Series(model.predict_proba(X)[:, 1], index=X.index)
    _, threshold = recall_at_fpr(y, proba)
    cases = select_cases(y, proba, threshold)
    print(f"\nLimiar a FPR {TARGET_FPR_FOR_RECALL:.0%}: {threshold:.4f}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    tables = {}
    for kind, idx in cases.items():
        row = X.loc[[idx]]
        shap_df, base_value = shap_values(model, row)
        title = f"{CASE_LABELS[kind]}: score {proba[idx]:.4f}"
        plot_waterfall(shap_df.iloc[0], base_value, row.iloc[0], title, FIGURES_DIR / f"shap_waterfall_{kind}.png")
        tables[kind] = explain_transaction(model, row, top_k)
        print(f"\n{title} (TransactionID {test_df.loc[idx, ID_COLUMN]})\n",
              tables[kind].round(4).to_string(index=False), sep="")
    return tables


def main() -> pd.DataFrame:
    t0 = time.perf_counter()
    model = load_model()
    _, test_df = load_feature_data()
    sample = sample_rows(test_df)
    X, y = make_xy(sample)

    shap_df, base_value = shap_values(model, X)
    print(f"SHAP em {len(X)} linhas x {X.shape[1]} features "
          f"(fraude {y.mean():.2%}), base {base_value:.3f} ({time.perf_counter() - t0:.0f}s)")

    # Sanidade: base + soma(SHAP) passa pela sigmoide e tem que bater com predict_proba.
    proba = model.predict_proba(X)[:, 1]
    rebuilt = 1 / (1 + np.exp(-(base_value + shap_df.sum(axis=1).to_numpy())))
    print(f"Erro máximo da reconstrução de predict_proba: {np.abs(rebuilt - proba).max():.2e}")

    gain = pd.Series(model.get_booster().get_score(importance_type="gain"))
    table = compare_with_gain(global_importance(shap_df), gain)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(METRICS_DIR / "shap_importancia_global.csv")

    plot_global(_to_explanation(shap_df, base_value, X))

    print(f"\nTop {SHAP_TOP_N} (|SHAP| médio) x rank no gain:\n",
          table.head(SHAP_TOP_N).round(4).to_string(), sep="")

    eng_cols = engineered_feature_columns(test_df)
    importance = table["mean_abs_shap"]
    share = group_share(importance, eng_cols)
    print(f"\nFeatures da Semana 2: {len(eng_cols)} de {len(importance)} colunas, "
          f"{share:.1%} do |SHAP| médio total")
    plot_dependence(shap_df, X, SHAP_DEPENDENCE_FEATURES)
    for feature in SHAP_DEPENDENCE_FEATURES:
        print(f"\n{feature}: SHAP médio por quintil\n",
              binned_trend(X[feature].astype(float), shap_df[feature]).round(4).to_string(), sep="")

    explain_cases(model, test_df)
    print(f"\nTotal: {time.perf_counter() - t0:.0f}s")
    return table


if __name__ == "__main__":
    main()
