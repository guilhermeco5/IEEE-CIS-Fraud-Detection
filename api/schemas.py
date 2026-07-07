"""Schemas de request/response da API, validados via Pydantic."""

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    """Dados mínimos de uma transação para scoring de fraude.

    Campos a expandir conforme as features definidas na Semana 2 —
    isto é um esqueleto inicial com os campos mais óbvios do IEEE-CIS.
    """

    transaction_amt: float = Field(..., description="Valor da transação", gt=0)
    card_type: str = Field(..., description="Tipo de cartão (ex: 'credit', 'debit')")
    product_cd: str = Field(..., description="Código do produto da transação")
    # TODO (Semana 5): adicionar os demais campos conforme as features finais do modelo


class FraudPrediction(BaseModel):
    fraud_probability: float = Field(..., ge=0, le=1)
    is_fraud_flag: bool
    risk_tier: str  # ex: "baixo", "médio", "alto"
