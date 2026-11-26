"""API de scoring de fraude.

Roda com: uvicorn api.app:app --reload
Docs interativas automáticas em /docs (Swagger UI gerado pelo FastAPI).

Pré-requisito: `python -m src.train` e `python -m src.serving` (geram o modelo
e o bundle em models/). Toda a lógica de features vive em `src/serving.py`.
"""

import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

from api.schemas import FraudPrediction, Meta, TransactionRequest
from src.config import DEMO_MODE_ENV_VAR
from src.serving import META_CATEGORY_COLUMNS, FraudScorer


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega modelo e bundle uma vez. Se faltarem, a API sobe e responde 503 com o motivo.

    `FRAUD_DEMO_MODE=1` (demo pública): o servidor nunca grava transações no histórico,
    mesmo que o cliente peça `record=true`. Sem autenticação, qualquer um poderia poluir
    o histórico dos cartões de demonstração.
    """
    app.state.demo_mode = os.environ.get(DEMO_MODE_ENV_VAR, "").strip().lower() in {"1", "true", "yes"}
    app.state.scorer, app.state.load_error = None, None
    try:
        app.state.scorer = FraudScorer.load()
    except FileNotFoundError as exc:
        app.state.load_error = str(exc)
    yield


app = FastAPI(
    title="API de Detecção de Fraude",
    description="Scoring de risco de fraude em transações financeiras (IEEE-CIS), com explicação SHAP.",
    version="0.2.0",
    lifespan=lifespan,
)


INDEX_PAGE = Path(__file__).parent / "static" / "index.html"


@app.get("/", include_in_schema=False)
def index():
    """Página de demonstração (HTML único, sem dependências externas). Usa /meta e /predict."""
    return FileResponse(INDEX_PAGE, media_type="text/html")


def get_scorer(request: Request) -> FraudScorer:
    scorer = request.app.state.scorer
    if scorer is None:
        raise HTTPException(status_code=503, detail=request.app.state.load_error)
    return scorer


@app.get("/health")
def health_check(request: Request):
    """Estado do serviço (para o Render/Railway/Docker). 503 se o modelo não carregou."""
    scorer = request.app.state.scorer
    if scorer is None:
        raise HTTPException(status_code=503, detail=request.app.state.load_error)
    return {
        "status": "ok",
        "model_loaded": True,
        "demo_mode": request.app.state.demo_mode,
        "cards_in_history": scorer.history.n_cards,
        "transactions_in_history": scorer.history.n_transactions,
    }


@app.get("/meta", response_model=Meta)
def meta(request: Request, scorer: FraudScorer = Depends(get_scorer)):
    """Opções para o formulário da demo: categorias aceitas, cartões de demo, transação de referência e limiares."""
    return Meta(
        demo_mode=request.app.state.demo_mode,
        categories={c: [str(v) for v in scorer.categories[c]] for c in META_CATEGORY_COLUMNS if c in scorer.categories},
        demo_cards=scorer.demo_card_stats(),
        reference_features=scorer.reference_features,
        thresholds=scorer.thresholds,
    )


@app.post("/predict", response_model=FraudPrediction)
def predict(
    transaction: TransactionRequest,
    request: Request,
    record: bool = True,
    scorer: FraudScorer = Depends(get_scorer),
):
    """Pontua uma transação e explica o score com as 5 features que mais pesaram.

    Com `record=true` (padrão) a transação entra no histórico do cartão e passa a
    contar nas features de velocidade e agregação das próximas. Use `record=false`
    para só consultar, sem alterar o estado. Em modo demo (`FRAUD_DEMO_MODE=1`) o servidor
    ignora `record=true` e nunca grava. O histórico vive na memória do
    processo (ver docs/metodologia.md, seção 7).

    É uma função síncrona de propósito: o cálculo é CPU-bound e o FastAPI o executa
    numa thread, sem travar o event loop.
    """
    result = scorer.score(
        transaction_dt=transaction.transaction_dt,
        transaction_amt=transaction.transaction_amt,
        card1=transaction.card1,
        product_cd=transaction.product_cd,
        features=transaction.features,
        transaction_id=transaction.transaction_id,
        record=record and not request.app.state.demo_mode,
    )
    return FraudPrediction(**asdict(result))
