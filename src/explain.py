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
    METRICS_DIR,
    RANDOM_STATE,
    SHAP_SAMPLE_SIZE,
    SHAP_TOP_N,
    XGBOOST_MODEL_FILE,
)
from src.train import load_feature_data, make_xy

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


def main() -> pd.DataFrame:
    t0 = time.perf_counter()
    model = load_model()
    _, test_df = load_feature_data()
    sample = sample_rows(test_df)
    X, y = make_xy(sample)
    del test_df

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
    print(f"\nTotal: {time.perf_counter() - t0:.0f}s")
    return table


if __name__ == "__main__":
    main()
