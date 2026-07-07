"""Avaliação de modelos de fraude.

Acurácia é uma métrica enganosa aqui: com ~3.5% de fraude, um modelo
que sempre prevê "não fraude" acerta >96% e é completamente inútil.
As métricas corretas avaliam a capacidade de separar as classes sob
desbalanceamento e o trade-off de negócio entre detectar fraude
(recall) e bloquear clientes legítimos por engano (falso positivo).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_curve

from src.config import TARGET_FPR_FOR_RECALL


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Calcula PR-AUC e recall no FPR-alvo definido em config.py."""
    y_proba = model.predict_proba(X_test)[:, 1]

    pr_auc = average_precision_score(y_test, y_proba)

    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    # encontra o recall (tpr) no ponto onde fpr está mais próximo do alvo de negócio
    idx = np.argmin(np.abs(fpr - TARGET_FPR_FOR_RECALL))
    recall_at_target_fpr = tpr[idx]
    threshold_at_target_fpr = thresholds[idx]

    return {
        "pr_auc": round(float(pr_auc), 4),
        f"recall_at_fpr_{TARGET_FPR_FOR_RECALL:.0%}": round(float(recall_at_target_fpr), 4),
        "decision_threshold": round(float(threshold_at_target_fpr), 4),
    }


def compare_models(results: dict[str, dict]) -> pd.DataFrame:
    """Monta tabela comparativa a partir de {nome_modelo: evaluate_model(...)}."""
    return pd.DataFrame(results).T
