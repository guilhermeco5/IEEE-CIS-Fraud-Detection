"""Testes da avaliação e da montagem de X/y."""

import numpy as np
import pandas as pd
import pytest

from src.evaluate import RECALL_KEY, evaluate_by_week, evaluate_scores, recall_at_fpr
from src.train import baseline_columns, make_xy


def test_recall_at_fpr_never_exceeds_budget():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(1000), np.ones(100)]
    p = np.r_[rng.normal(0, 1, 1000), rng.normal(2, 1, 100)]

    recall, threshold = recall_at_fpr(y, p, target_fpr=0.01)

    fpr_real = (p[y == 0] >= threshold).mean()
    assert fpr_real <= 0.01
    assert recall == pytest.approx((p[y == 1] >= threshold).mean())


def test_perfect_model_scores_one():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    scores = evaluate_scores(y, p)
    assert scores["pr_auc"] == 1.0
    assert scores[RECALL_KEY] == 1.0


def test_evaluate_by_week_groups_and_handles_single_class_weeks():
    week = 7 * 86_400
    dt = np.array([0, 1, 2, 3, week, week + 1])
    y = np.array([0, 1, 0, 1, 0, 0])            # semana 1 sem fraude
    p = np.array([0.1, 0.9, 0.2, 0.8, 0.5, 0.4])

    out = evaluate_by_week(y, p, dt)

    assert out.index.tolist() == [0, 1]
    assert out.loc[0, "n"] == 4
    assert out.loc[0, "pr_auc"] == 1.0
    assert np.isnan(out.loc[1, "pr_auc"])


def test_make_xy_drops_id_time_and_target():
    df = pd.DataFrame({"TransactionID": [1], "TransactionDT": [0], "isFraud": [1], "a": [3.0]})
    X, y = make_xy(df)
    assert X.columns.tolist() == ["a"]
    assert y.tolist() == [1]


def test_baseline_columns_skips_categorical_and_mostly_missing():
    X = pd.DataFrame({
        "ok": [1.0, 2.0, np.nan, 4.0],
        "quase_vazia": [np.nan, np.nan, np.nan, 1.0],
        "cat": pd.Series(["a", "b", "a", "b"], dtype="category"),
    })
    assert baseline_columns(X) == ["ok"]
