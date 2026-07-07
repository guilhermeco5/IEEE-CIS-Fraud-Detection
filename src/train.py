"""Treino dos modelos (Semana 3).

Baseline simples primeiro (regressão logística), depois XGBoost.
O desbalanceamento é tratado via scale_pos_weight, não SMOTE — gerar
exemplos sintéticos de fraude por interpolação tende a criar pontos
que não existem na distribuição real e pode degradar o modelo em
produção. scale_pos_weight ajusta o peso da perda sem inventar dados.
"""

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.config import MODELS_DIR, RANDOM_STATE, TARGET_COLUMN, XGBOOST_PARAMS


def train_baseline(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """Baseline de regressão logística com class_weight balanceado."""
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    """XGBoost com scale_pos_weight calculado a partir do desbalanceamento real."""
    n_negative = (y_train == 0).sum()
    n_positive = (y_train == 1).sum()
    scale_pos_weight = n_negative / n_positive

    params = {**XGBOOST_PARAMS, "scale_pos_weight": scale_pos_weight}
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model


def save_model(model, filename: str) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / filename)
    print(f"Modelo salvo em {MODELS_DIR / filename}")


if __name__ == "__main__":
    # TODO (Semana 3): carregar X_train/y_train processados,
    # treinar os dois modelos e chamar evaluate.py para comparar.
    raise NotImplementedError("Pipeline de treino completo: implementar na Semana 3.")
