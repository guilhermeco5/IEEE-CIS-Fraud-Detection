"""Avaliação de modelos de fraude.

Acurácia é uma métrica enganosa aqui: com ~3.5% de fraude, um modelo
que sempre prevê "não fraude" acerta >96% e é completamente inútil.
As métricas corretas avaliam a capacidade de separar as classes sob
desbalanceamento e o trade-off de negócio entre detectar fraude
(recall) e bloquear clientes legítimos por engano (falso positivo).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from src.config import TARGET_FPR_FOR_RECALL

SECONDS_PER_WEEK = 7 * 86_400
RECALL_KEY = f"recall_at_fpr_{TARGET_FPR_FOR_RECALL:.0%}"


def recall_at_fpr(y_true, y_proba, target_fpr: float = TARGET_FPR_FOR_RECALL) -> tuple[float, float]:
    """Recall no maior FPR ainda dentro do orçamento, e o limiar correspondente.

    `argmin(abs(fpr - alvo))` pode escolher um ponto com fpr ACIMA do alvo
    (se for numericamente mais próximo), violando a restrição de negócio
    "bloquear no máximo X% dos clientes legítimos". Por isso pegamos o
    último ponto com fpr <= alvo.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    within_budget = np.where(fpr <= target_fpr)[0]
    idx = within_budget[-1] if len(within_budget) else 0
    return float(tpr[idx]), float(thresholds[idx])


def evaluate_scores(y_true, y_proba) -> dict:
    """PR-AUC (métrica principal), ROC-AUC (referência) e recall no FPR-alvo."""
    recall, threshold = recall_at_fpr(y_true, y_proba)
    return {
        "pr_auc": round(float(average_precision_score(y_true, y_proba)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        RECALL_KEY: round(recall, 4),
        "decision_threshold": round(threshold, 4),
    }


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Atalho: predict_proba + evaluate_scores."""
    return evaluate_scores(y_test, model.predict_proba(X_test)[:, 1])


def evaluate_by_week(y_true, y_proba, transaction_dt) -> pd.DataFrame:
    """Métricas por semana de `TransactionDT`.

    O EDA mostrou que a taxa de fraude varia entre semanas (docs/metodologia.md,
    seção 1). Um agregado bom pode esconder uma semana ruim, então a
    performance é monitorada semana a semana. Semanas com uma classe só
    ficam com NaN (as métricas não são definidas).
    """
    frame = pd.DataFrame({
        "y": np.asarray(y_true),
        "p": np.asarray(y_proba),
        "semana": np.asarray(transaction_dt) // SECONDS_PER_WEEK,
    })
    rows = []
    for week, g in frame.groupby("semana"):
        row = {"semana": int(week), "n": len(g), "taxa_fraude": round(g["y"].mean(), 4)}
        if g["y"].nunique() == 2:
            row["pr_auc"] = round(float(average_precision_score(g["y"], g["p"])), 4)
            row[RECALL_KEY] = round(recall_at_fpr(g["y"], g["p"])[0], 4)
        else:
            row["pr_auc"] = row[RECALL_KEY] = np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("semana")


def compare_models(results: dict[str, dict]) -> pd.DataFrame:
    """Monta tabela comparativa a partir de {nome_modelo: evaluate_scores(...)}."""
    return pd.DataFrame(results).T
