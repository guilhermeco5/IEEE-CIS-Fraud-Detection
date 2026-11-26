"""Testes do serving: o caminho online tem que reproduzir o batch do treino.

Os fixtures `world` e `scorer` (modelo e bundle sintéticos) estão em conftest.py.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.config import BUNDLE_ENV_VAR, SERVING_BUNDLE_FILE
from src.serving import FraudScorer, HistoryStore, build_demo_bundle, runtime_bundle_file


def _payload(row: pd.Series, raw_columns) -> dict:
    reserved = {"TransactionID", "TransactionDT", "TransactionAmt", "card1", "ProductCD", "isFraud"}
    return dict(
        transaction_dt=int(row["TransactionDT"]),
        transaction_amt=float(row["TransactionAmt"]),
        card1=int(row["card1"]),
        product_cd=str(row["ProductCD"]),
        features={c: v for c, v in row.items() if c in raw_columns and c not in reserved and pd.notna(v)},
    )


def _same_value(a, b) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    if isinstance(a, str) or isinstance(b, str):
        return str(a) == str(b)
    return bool(np.isclose(float(a), float(b), rtol=1e-6, atol=1e-6))


def test_online_features_match_batch_sequentially(world, scorer):
    """Pontuando o teste em ordem de tempo, com `record=True`, cada linha online
    tem que ser igual à do batch (que vê treino + teste anterior)."""
    _, bundle, _, test_f = world
    raw_columns = set(bundle["raw_columns"])
    for _, row in test_f.iterrows():
        payload = _payload(row, raw_columns)
        X_online, ignored = scorer.build_row(**payload)
        assert ignored == []
        for col in bundle["feature_columns"]:
            assert _same_value(X_online.iloc[0][col], row[col]), f"{col}: {X_online.iloc[0][col]} != {row[col]}"
        scorer.score(**payload)  # record=True: entra no histórico para as próximas linhas


def test_score_matches_batch_predict_proba(world, scorer):
    model, bundle, _, test_f = world
    first = test_f.iloc[0]  # a única linha cujo passado é só o treino
    result = scorer.score(**_payload(first, set(bundle["raw_columns"])), record=False)
    batch = model.predict_proba(test_f.iloc[[0]][bundle["feature_columns"]])[0, 1]
    assert result.fraud_score == pytest.approx(batch, abs=1e-6)


def test_record_false_leaves_history_untouched_and_true_adds(world, scorer):
    _, bundle, _, test_f = world
    payload = _payload(test_f.iloc[0], set(bundle["raw_columns"]))
    before = scorer.history.n_transactions
    scorer.score(**payload, record=False)
    assert scorer.history.n_transactions == before
    scorer.score(**payload, record=True)
    assert scorer.history.n_transactions == before + 1


def test_recorded_transaction_shows_up_in_next_card_features(world, scorer):
    _, bundle, _, test_f = world
    payload = _payload(test_f.iloc[0], set(bundle["raw_columns"]))
    n_before = scorer.build_row(**payload)[0].iloc[0]["card1_n_prev"]
    scorer.score(**payload, record=True)
    later = {**payload, "transaction_dt": payload["transaction_dt"] + 60}
    assert scorer.build_row(**later)[0].iloc[0]["card1_n_prev"] == n_before + 1


def test_unknown_card_has_no_history_features(scorer):
    X, _ = scorer.build_row(transaction_dt=10_000, transaction_amt=50.0, card1=999_999, product_cd="W")
    assert X.iloc[0]["card1_n_prev"] == 0
    assert np.isnan(X.iloc[0]["card1_amt_mean_prev"])
    assert np.isnan(X.iloc[0]["card1_secs_since_prev"])


def test_ignored_features_are_reported(scorer):
    result = scorer.score(
        transaction_dt=10_000, transaction_amt=50.0, card1=1000, product_cd="W", record=False,
        features={"coluna_inexistente": 1, "TransactionAmt": 3, "id_01": "abc", "DeviceType": "toaster", "id_02": 0.5},
    )
    names = {text.split(":")[0] for text in result.ignored_features}
    assert names == {"coluna_inexistente", "TransactionAmt", "id_01", "DeviceType"}


def test_unseen_category_and_bad_number_become_nan(scorer):
    X, _ = scorer.build_row(
        transaction_dt=10_000, transaction_amt=50.0, card1=1000, product_cd="Z",
        features={"DeviceType": "toaster", "id_01": "abc"},
    )
    assert pd.isna(X.iloc[0]["DeviceType"]) and pd.isna(X.iloc[0]["ProductCD"]) and pd.isna(X.iloc[0]["id_01"])


def test_missing_flags_and_frequency_encoding_follow_training_rules(scorer):
    X, _ = scorer.build_row(
        transaction_dt=10_000, transaction_amt=50.0, card1=1000, product_cd="W",
        features={"P_emaildomain": "gmail.com", "id_01": 0.3},
    )
    row = X.iloc[0]
    assert row["id_02_is_missing"] == 1 and row["id_01_is_missing"] == 0
    assert row["has_identity"] == 1
    assert row["P_emaildomain_freq"] > 0
    assert np.isnan(row["R_emaildomain_freq"])  # ausente continua NaN, não 0


@pytest.mark.parametrize(
    "thresholds, tier, flag",
    [({"alto": 0.0, "medio": 0.0}, "alto", True),
     ({"alto": 2.0, "medio": 0.0}, "médio", False),
     ({"alto": 2.0, "medio": 2.0}, "baixo", False)],
)
def test_risk_tier_and_flag_follow_thresholds(scorer, thresholds, tier, flag):
    scorer.thresholds = thresholds
    result = scorer.score(transaction_dt=10_000, transaction_amt=50.0, card1=1000, product_cd="W", record=False)
    assert result.risk_tier == tier and result.is_fraud_flag is flag
    assert result.threshold == thresholds["alto"]


def test_top_factors_are_sorted_and_json_serializable(scorer):
    result = scorer.score(
        transaction_dt=10_000, transaction_amt=50.0, card1=1000, product_cd="W", record=False,
        features={"P_emaildomain": "gmail.com"}, top_k=4,
    )
    assert len(result.top_factors) == 4
    assert all(set(f) == {"feature", "value", "shap"} for f in result.top_factors)
    shaps = [abs(f["shap"]) for f in result.top_factors]
    assert shaps == sorted(shaps, reverse=True)
    json.dumps(result.top_factors)  # NaN/numpy não podem vazar para a resposta


def test_bundle_that_does_not_match_model_is_rejected(world):
    model, bundle, *_ = world
    with pytest.raises(ValueError, match="bundle"):
        FraudScorer(model, {**bundle, "feature_columns": bundle["feature_columns"][::-1]})


def test_load_without_artifacts_explains_what_to_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m src.train"):
        FraudScorer.load(model_path=tmp_path / "x.joblib", bundle_path=tmp_path / "y.joblib")


def test_history_store_keeps_time_order():
    store = HistoryStore()
    for t, a in [(30, 3.0), (10, 1.0), (20, 2.0), (20, 2.5)]:
        store.add(7, t, a)
    times, amounts = store.get(7)
    assert list(times) == [10, 20, 20, 30] and amounts.dtype == np.float32
    assert store.n_cards == 1 and store.n_transactions == 4
    assert len(store.get(8)[0]) == 0


def test_demo_bundle_keeps_only_requested_cards(world):
    _, bundle, *_ = world
    demo = build_demo_bundle(bundle, cards=[1000, 1001])
    assert set(demo["history"]) == {1000, 1001} and demo["demo_cards"] == [1000, 1001]
    assert demo["feature_columns"] == bundle["feature_columns"] and demo["thresholds"] == bundle["thresholds"]
    assert set(demo["freq_encoder"]["card1"].index) == {1000, 1001}
    # o bundle completo não pode ser alterado por quem gera o de demo
    assert len(bundle["history"]) > 2 and len(bundle["freq_encoder"]["card1"]) > 2
    assert "demo_cards" not in bundle


def test_demo_bundle_rejects_card_without_history(world):
    _, bundle, *_ = world
    with pytest.raises(ValueError, match="999999"):
        build_demo_bundle(bundle, cards=[1000, 999_999])


def test_card_outside_demo_bundle_behaves_as_new_card(world):
    model, bundle, *_ = world
    demo = build_demo_bundle(bundle, cards=[1000])
    scorer = FraudScorer(model, demo, history=HistoryStore(dict(demo["history"])))
    assert scorer.demo_cards == [1000]
    kwargs = dict(transaction_dt=5_000_000, transaction_amt=50.0, product_cd="W")
    known = scorer.build_row(card1=1000, **kwargs)[0].iloc[0]
    outside = scorer.build_row(card1=1002, **kwargs)[0].iloc[0]
    assert known["card1_n_prev"] > 0 and known["card1_freq"] > 0
    assert outside["card1_n_prev"] == 0 and np.isnan(outside["card1_amt_mean_prev"])
    assert outside["card1_freq"] == 0  # card1 nunca visto, como no treino


def test_runtime_bundle_file_follows_env_at_call_time(monkeypatch, tmp_path):
    monkeypatch.delenv(BUNDLE_ENV_VAR, raising=False)
    assert runtime_bundle_file() == SERVING_BUNDLE_FILE
    monkeypatch.setenv(BUNDLE_ENV_VAR, str(tmp_path / "demo.joblib"))
    assert runtime_bundle_file() == tmp_path / "demo.joblib"
