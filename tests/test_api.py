"""Testes da API com TestClient, usando o modelo e o bundle sintéticos do conftest."""

import pytest
from fastapi.testclient import TestClient

from api.app import app
from src.config import DEMO_MODE_ENV_VAR
from src.serving import FraudScorer

VALID = {
    "transaction_dt": 10_000,
    "transaction_amt": 50.0,
    "card1": 1000,
    "product_cd": "W",
    "features": {"P_emaildomain": "gmail.com", "id_01": 0.3},
}


@pytest.fixture
def client(scorer, monkeypatch):
    """Cliente com o scorer sintético no lugar dos artefatos reais de models/."""
    monkeypatch.setattr(FraudScorer, "load", lambda *a, **k: scorer)
    monkeypatch.delenv(DEMO_MODE_ENV_VAR, raising=False)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def demo_client(scorer, monkeypatch):
    """Mesmo cliente, com o servidor em modo demonstração (FRAUD_DEMO_MODE=1)."""
    monkeypatch.setattr(FraudScorer, "load", lambda *a, **k: scorer)
    monkeypatch.setenv(DEMO_MODE_ENV_VAR, "1")
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client_without_artifacts(monkeypatch):
    def missing(*a, **k):
        raise FileNotFoundError("Artefatos ausentes. Rode `python -m src.train` e depois `python -m src.serving`.")

    monkeypatch.setattr(FraudScorer, "load", missing)
    with TestClient(app) as c:
        yield c


def test_health_reports_history_size(client, scorer):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert body["cards_in_history"] == scorer.history.n_cards
    assert body["transactions_in_history"] == scorer.history.n_transactions


def test_predict_returns_documented_fields(client):
    response = client.post("/predict", json=VALID)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"fraud_score", "is_fraud_flag", "risk_tier", "threshold", "top_factors", "ignored_features"}
    assert 0 <= body["fraud_score"] <= 1
    assert body["risk_tier"] in {"baixo", "médio", "alto"}
    assert body["is_fraud_flag"] == (body["fraud_score"] >= body["threshold"])
    assert len(body["top_factors"]) == 5
    assert all(set(f) == {"feature", "value", "shap"} for f in body["top_factors"])
    assert body["ignored_features"] == []


def test_predict_matches_scorer_called_directly(client, scorer):
    direct = scorer.score(
        transaction_dt=10_000, transaction_amt=50.0, card1=1000, product_cd="W",
        features=VALID["features"], record=False,
    )
    body = client.post("/predict?record=false", json=VALID).json()
    assert body["fraud_score"] == pytest.approx(direct.fraud_score)
    assert [f["feature"] for f in body["top_factors"]] == [f["feature"] for f in direct.top_factors]


def test_record_flag_controls_history(client):
    def n_transactions():
        return client.get("/health").json()["transactions_in_history"]

    before = n_transactions()
    client.post("/predict?record=false", json=VALID)
    assert n_transactions() == before
    client.post("/predict", json=VALID)  # padrão: record=true
    assert n_transactions() == before + 1


def test_unknown_and_invalid_features_come_back_ignored(client):
    payload = {**VALID, "features": {"coluna_inexistente": 1, "DeviceType": "toaster", "id_01": "abc"}}
    body = client.post("/predict?record=false", json=payload).json()
    assert {text.split(":")[0] for text in body["ignored_features"]} == {"coluna_inexistente", "DeviceType", "id_01"}


@pytest.mark.parametrize(
    "patch",
    [
        {"transaction_amt": 0},
        {"transaction_amt": -5},
        {"transaction_dt": -1},
        {"card1": "abc"},
        {"product_cd": None},
    ],
)
def test_invalid_body_is_rejected_with_422(client, patch):
    assert client.post("/predict", json={**VALID, **patch}).status_code == 422


@pytest.mark.parametrize("missing", ["transaction_dt", "transaction_amt", "card1", "product_cd"])
def test_required_fields_are_enforced(client, missing):
    payload = {k: v for k, v in VALID.items() if k != missing}
    assert client.post("/predict", json=payload).status_code == 422


def test_503_with_actionable_message_when_artifacts_are_missing(client_without_artifacts):
    for response in (client_without_artifacts.get("/health"), client_without_artifacts.post("/predict", json=VALID)):
        assert response.status_code == 503
        assert "python -m src.train" in response.json()["detail"]


def test_openapi_documents_the_request_example(client):
    schema = client.get("/openapi.json").json()["components"]["schemas"]["TransactionRequest"]
    assert schema["example"]["card1"] == 13926


def test_demo_mode_ignores_record_true(demo_client):
    def n_transactions():
        return demo_client.get("/health").json()["transactions_in_history"]

    assert demo_client.get("/health").json()["demo_mode"] is True
    before = n_transactions()
    assert demo_client.post("/predict?record=true", json=VALID).status_code == 200
    assert demo_client.post("/predict", json=VALID).status_code == 200  # padrão também é record=true
    assert n_transactions() == before


def test_normal_mode_reports_demo_mode_off(client):
    assert client.get("/health").json()["demo_mode"] is False


def test_meta_exposes_what_the_form_needs(client, scorer):
    body = client.get("/meta").json()
    assert set(body) == {"demo_mode", "categories", "demo_cards", "thresholds"}
    assert body["demo_mode"] is False
    assert set(body["categories"]["ProductCD"]) == {"W", "C", "H"}
    assert set(body["categories"]) <= {"ProductCD", "card4", "card6", "P_emaildomain", "DeviceType"}
    assert body["thresholds"] == scorer.thresholds
    assert body["demo_cards"] == scorer.demo_cards == []  # o bundle sintético não é de demo


def test_meta_is_503_without_artifacts(client_without_artifacts):
    assert client_without_artifacts.get("/meta").status_code == 503
