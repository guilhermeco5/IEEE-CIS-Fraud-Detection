"""API de scoring de fraude.

Roda com: uvicorn api.app:app --reload
Docs interativas automáticas em /docs (Swagger UI gerado pelo FastAPI).
"""

from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException

from api.schemas import FraudPrediction, TransactionRequest
from src.config import MODELS_DIR

app = FastAPI(
    title="API de Detecção de Fraude",
    description="Scoring de risco de fraude em transações financeiras.",
    version="0.1.0",
)

MODEL_PATH = MODELS_DIR / "xgboost_fraud_model.joblib"
_model = None  # carregado em memória sob demanda, ver get_model()


def get_model():
    """Carrega o modelo serializado uma única vez (lazy loading)."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Modelo não encontrado em {MODEL_PATH}. Treine o modelo primeiro (src/train.py).",
            )
        _model = joblib.load(MODEL_PATH)
    return _model


@app.get("/health")
def health_check():
    """Endpoint de saúde da API — útil para o Render/Railway verificarem se o serviço está de pé."""
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict", response_model=FraudPrediction)
def predict(transaction: TransactionRequest):
    """Recebe os dados de uma transação e retorna a probabilidade de fraude.

    TODO (Semana 5): aplicar o mesmo pipeline de features usado no
    treino (src/features.py) antes de chamar model.predict_proba.
    """
    model = get_model()  # noqa: F841 — usado quando a lógica de features estiver pronta

    raise NotImplementedError(
        "Implementar transformação de features + predict_proba na Semana 5, "
        "reaproveitando exatamente a mesma lógica de src/features.py usada no treino."
    )
