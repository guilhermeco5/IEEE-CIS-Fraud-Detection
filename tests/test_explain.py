"""Testes da explicabilidade (SHAP nativo do XGBoost)."""

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from src.explain import (
    binned_trend,
    compare_with_gain,
    explain_transaction,
    global_importance,
    group_share,
    sample_rows,
    select_cases,
    shap_values,
)


@pytest.fixture(scope="module")
def model_and_data():
    rng = np.random.default_rng(0)
    n = 600
    X = pd.DataFrame({
        "forte": rng.normal(size=n),
        "fraca": rng.normal(size=n),
        "ruido": rng.normal(size=n),
        "cat": pd.Categorical(rng.choice(["a", "b", "c"], size=n)),
    })
    X.loc[rng.random(n) < 0.1, "fraca"] = np.nan  # NaN também precisa funcionar
    logit = 2.5 * X["forte"] + 0.5 * X["fraca"].fillna(0) + (X["cat"] == "c") * 1.0
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    model = XGBClassifier(n_estimators=40, max_depth=3, tree_method="hist",
                          enable_categorical=True, random_state=0)
    model.fit(X, y)
    return model, X


def test_shap_is_additive_to_margin(model_and_data):
    model, X = model_and_data
    shap_df, base = shap_values(model, X)
    margin = model.predict(X, output_margin=True)
    np.testing.assert_allclose(base + shap_df.sum(axis=1), margin, atol=1e-4)


def test_shap_shape_and_columns(model_and_data):
    model, X = model_and_data
    shap_df, _ = shap_values(model, X)
    assert shap_df.shape == X.shape
    assert list(shap_df.columns) == list(X.columns)
    assert shap_df.index.equals(X.index)


def test_global_importance_ranks_strong_feature_first(model_and_data):
    model, X = model_and_data
    imp = global_importance(shap_values(model, X)[0])
    assert imp.index[0] == "forte"
    assert imp.is_monotonic_decreasing


def test_compare_with_gain_handles_unused_feature():
    imp = pd.Series({"a": 0.5, "b": 0.3, "c": 0.0}, name="mean_abs_shap")
    gain = pd.Series({"b": 10.0, "a": 5.0})  # "c" nunca foi usada num split
    table = compare_with_gain(imp, gain)
    assert table.loc["a", "rank_shap"] == 1 and table.loc["a", "rank_gain"] == 2
    assert table.loc["c", "gain"] == 0 and np.isnan(table.loc["c", "rank_gain"])


def test_sample_rows_is_reproducible():
    df = pd.DataFrame({"x": range(100)})
    a, b = sample_rows(df, n=10, seed=1), sample_rows(df, n=10, seed=1)
    assert a.equals(b) and len(a) == 10
    assert sample_rows(df, n=1000) is df


def test_explain_transaction_top_k_sorted_by_abs_shap(model_and_data):
    model, X = model_and_data
    row = X.iloc[[5]]
    out = explain_transaction(model, row, top_k=3)
    assert list(out.columns) == ["feature", "valor", "shap"]
    assert len(out) == 3
    assert out["shap"].abs().is_monotonic_decreasing
    assert out.loc[0, "valor"] == row.iloc[0][out.loc[0, "feature"]]


def test_explain_transaction_all_features_add_up_to_margin(model_and_data):
    model, X = model_and_data
    row = X.iloc[[7]]
    out = explain_transaction(model, row, top_k=X.shape[1])
    _, base = shap_values(model, row)
    assert base + out["shap"].sum() == pytest.approx(model.predict(row, output_margin=True)[0], abs=1e-4)


def test_explain_transaction_rejects_multiple_rows(model_and_data):
    model, X = model_and_data
    with pytest.raises(ValueError):
        explain_transaction(model, X.iloc[:2])


def test_select_cases():
    y = pd.Series([1, 1, 1, 0, 0, 0], index=list("abcdef"))
    p = pd.Series([0.9, 0.6, 0.1, 0.8, 0.7, 0.2], index=list("abcdef"))
    assert select_cases(y, p, threshold=0.5) == {"tp": "a", "fp": "d", "fn": "c"}


def test_group_share_sums_to_one_over_all_columns():
    imp = pd.Series({"a": 0.6, "b": 0.3, "c": 0.1})
    assert group_share(imp, ["a", "b", "c"]) == pytest.approx(1.0)
    assert group_share(imp, ["b", "c"]) == pytest.approx(0.4)
    assert group_share(imp, ["inexistente"]) == 0.0


def test_binned_trend_monotonic_and_missing_row():
    x = pd.Series(np.r_[np.arange(100, dtype=float), [np.nan] * 5])
    s = pd.Series(np.r_[np.arange(100) / 100, [9.0] * 5])  # SHAP cresce com x; ausentes à parte
    out = binned_trend(x, s, bins=4)
    assert out.index[-1] == "ausente" and out.loc["ausente", "n"] == 5
    assert out.loc["ausente", "shap_medio"] == 9.0
    bands = out.drop(index="ausente")
    assert len(bands) == 4 and bands["shap_medio"].is_monotonic_increasing
    assert bands["n"].sum() == 100


def test_binned_trend_collapses_duplicate_edges():
    x = pd.Series([0.0] * 90 + list(range(1, 11)), dtype=float)  # maioria zero, como cnt_1h
    out = binned_trend(x, pd.Series(np.zeros(100)), bins=5)
    assert out["n"].sum() == 100 and len(out) < 5
