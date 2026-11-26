"""Schemas de request/response da API, validados via Pydantic."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TransactionRequest(BaseModel):
    """Uma transação para pontuar.

    O modelo usa 434 colunas brutas do IEEE-CIS. Só cinco são obrigatórias;
    as demais vão em `features`, com o nome da coluna do dataset (ex:
    `card4`, `P_emaildomain`, `id_31`, `C1`, `V258`). O que faltar é tratado
    como ausente, que o modelo já sabe lidar.
    """

    model_config = ConfigDict(json_schema_extra={"example": {
        "transaction_id": 3540441,
        "transaction_dt": 14_000_000,
        "transaction_amt": 59.0,
        "card1": 13926,
        "product_cd": "W",
        "features": {"card4": "visa", "card6": "debit", "P_emaildomain": "gmail.com", "C1": 1, "C13": 5},
    }})

    transaction_id: int | None = Field(None, description="Identificador da transação (opcional, não afeta o score)")
    transaction_dt: int = Field(
        ..., ge=0,
        description="TransactionDT: segundos desde a referência do dataset. Define a ordem no histórico do cartão.",
    )
    transaction_amt: float = Field(..., gt=0, description="Valor da transação (TransactionAmt)")
    card1: int = Field(..., description="card1, usado como identificador do cartão no histórico")
    product_cd: str = Field(..., description="ProductCD (ex: 'W', 'C', 'H', 'R', 'S')")
    features: dict[str, float | str | None] = Field(
        default_factory=dict,
        description="Demais colunas brutas, por nome. Desconhecidas ou inválidas voltam em `ignored_features`.",
    )


class Factor(BaseModel):
    feature: str
    value: float | str | None = Field(None, description="Valor da feature nesta transação (None se ausente)")
    shap: float = Field(..., description="Contribuição em log-odds: positivo empurra para fraude, negativo para legítima")


class FraudPrediction(BaseModel):
    fraud_score: float = Field(
        ..., ge=0, le=1,
        description="Score do modelo. NÃO é uma probabilidade calibrada (o treino usa scale_pos_weight): "
        "serve para ordenar transações e comparar com `threshold`.",
    )
    is_fraud_flag: bool = Field(..., description="`fraud_score >= threshold`")
    risk_tier: Literal["baixo", "médio", "alto"] = Field(
        ..., description="alto: acima do limiar de FPR 1% no teste; médio: acima do de FPR 5%; baixo: abaixo"
    )
    threshold: float = Field(..., description="Limiar de decisão (FPR 1% no teste)")
    top_factors: list[Factor] = Field(..., description="As features que mais pesaram neste score (SHAP)")
    ignored_features: list[str] = Field(
        default_factory=list, description="Campos de `features` não usados, com o motivo"
    )


class Meta(BaseModel):
    """O que a interface precisa saber do servidor, para não repetir listas no front-end."""

    demo_mode: bool = Field(..., description="Se verdadeiro, o servidor nunca grava transações no histórico")
    categories: dict[str, list[str]] = Field(..., description="Valores aceitos das colunas categóricas principais")
    demo_cards: list[int] = Field(..., description="Cartões com histórico no bundle (os demais contam como cartão novo)")
    thresholds: dict[str, float] = Field(..., description="Limiares do score: `alto` (FPR 1%) e `medio` (FPR 5%)")
