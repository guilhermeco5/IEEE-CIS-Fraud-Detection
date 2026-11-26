"""Fixtures compartilhados: um mundo sintético pequeno (dados, features em lote, modelo e bundle)."""

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from src.features import build_features, fit_frequency_encoder
from src.serving import FraudScorer, HistoryStore, typical_values

NON_FEATURES = ["TransactionID", "TransactionDT", "isFraud"]
CATEGORICAL = ["ProductCD", "P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo", "id_31"]
N_TRAIN, N_TEST = 400, 100


def _synthetic_raw() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = N_TRAIN + N_TEST

    def cat(values, p_missing=0.2):
        drawn = rng.choice(np.array(values + [None], dtype=object), size=n,
                           p=[(1 - p_missing) / len(values)] * len(values) + [p_missing])
        return pd.Categorical(drawn, categories=values)

    amount = rng.gamma(2, 50, n).astype("float32")
    df = pd.DataFrame({
        "TransactionID": np.arange(n),
        "isFraud": 0,
        # Poucos instantes distintos de propósito: força empates de timestamp.
        "TransactionDT": np.sort(rng.integers(0, 600, n) * 5_000),
        "TransactionAmt": amount,
        "card1": rng.integers(1000, 1006, n),
        "ProductCD": cat(["W", "C", "H"], 0.0),
        "P_emaildomain": cat(["gmail.com", "yahoo.com", "hotmail.com"]),
        "R_emaildomain": cat(["gmail.com", "yahoo.com"], 0.5),
        "DeviceType": cat(["mobile", "desktop"], 0.4),
        "DeviceInfo": cat(["SM-G920P", "Windows", "iOS Device"], 0.4),
        "id_31": cat(["chrome 66.0", "firefox 52.0"], 0.4),
        "id_01": np.where(rng.random(n) < 0.4, np.nan, rng.normal(size=n)),
        "id_02": np.where(rng.random(n) < 0.4, np.nan, rng.normal(size=n)),
        "V1": rng.normal(size=n),
    })
    df["isFraud"] = (amount + 40 * rng.normal(size=n) > 130).astype(int)
    return df


@pytest.fixture(scope="module")
def world():
    """Treino/teste sintéticos, features em lote (`build_features`) e modelo treinado nelas."""
    raw = _synthetic_raw()
    train_f, test_f = build_features(raw.iloc[:N_TRAIN].copy(), raw.iloc[N_TRAIN:].copy())
    X = train_f.drop(columns=NON_FEATURES)
    model = XGBClassifier(n_estimators=15, max_depth=3, tree_method="hist", enable_categorical=True, random_state=0)
    model.fit(X, train_f["isFraud"])

    bundle = {
        "feature_columns": list(model.get_booster().feature_names),
        "raw_columns": list(raw.columns),
        "categories": {c: list(train_f[c].cat.categories) for c in CATEGORICAL},
        "freq_encoder": fit_frequency_encoder(train_f),
        "typical_values": typical_values(train_f, list(raw.columns)),
        "history": HistoryStore.from_frame(train_f).to_dict(),
        "thresholds": {"alto": 0.7, "medio": 0.3},
    }
    return model, bundle, train_f, test_f


@pytest.fixture
def scorer(world):
    model, bundle, *_ = world
    # Cópia do dict: `record=True` troca arrays do histórico e não pode vazar entre testes.
    return FraudScorer(model, bundle, history=HistoryStore(dict(bundle["history"])))
