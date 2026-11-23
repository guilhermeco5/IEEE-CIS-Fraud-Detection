"""Testes das features da Semana 2.

O foco principal é vazamento: nenhuma feature de entidade pode depender
da própria linha, de empates de timestamp ou de qualquer transação futura.
"""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    add_aggregation_features,
    add_missing_flags,
    add_velocity_features,
    apply_frequency_encoder,
    build_features,
    fit_frequency_encoder,
)

H = 3_600


def _toy() -> pd.DataFrame:
    # Linhas propositalmente fora de ordem temporal; entidade B intercalada.
    return pd.DataFrame({
        "TransactionID": [1, 2, 3, 4, 5, 6, 7],
        "card1": [10, 10, 20, 10, 10, 20, 10],
        "TransactionDT": [5 * H, 0, 1 * H, 2 * H, 2 * H, 30 * H, 30 * H],
        "TransactionAmt": [50.0, 10.0, 999.0, 20.0, 30.0, 1.0, 100.0],
    })


def test_velocity_counts_past_only_and_excludes_ties():
    out = add_velocity_features(_toy()).set_index("TransactionID")

    # Entidade 10, tempos: 0, 2h, 2h (empate), 5h, 30h.
    assert out.loc[2, "card1_cnt_24h"] == 0               # primeira transação
    assert out.loc[4, "card1_cnt_24h"] == 1               # só t=0; o empate em 2h não conta
    assert out.loc[5, "card1_cnt_24h"] == 1
    assert out.loc[1, "card1_cnt_1h"] == 0                # nada em [4h, 5h)
    assert out.loc[1, "card1_cnt_6h"] == 3                # 0, 2h, 2h
    assert out.loc[7, "card1_cnt_24h"] == 0               # [6h, 30h) está vazio (5h fica fora)
    assert out.loc[7, "card1_secs_since_prev"] == 25 * H  # anterior foi em 5h
    assert out.loc[4, "card1_secs_since_prev"] == 2 * H
    assert np.isnan(out.loc[2, "card1_secs_since_prev"])


def test_velocity_isolates_entities():
    out = add_velocity_features(_toy()).set_index("TransactionID")
    # Entidade 20 só tem 1h e 30h; a 10 não pode contaminar.
    assert out.loc[3, "card1_cnt_24h"] == 0
    assert out.loc[6, "card1_cnt_24h"] == 0
    assert out.loc[6, "card1_secs_since_prev"] == 29 * H


def test_aggregations_match_manual_calculation():
    out = add_aggregation_features(_toy()).set_index("TransactionID")

    assert out.loc[2, "card1_n_prev"] == 0
    assert np.isnan(out.loc[2, "card1_amt_mean_prev"])
    # Empates (ids 4 e 5, ambos em 2h) só enxergam id 2.
    assert out.loc[4, "card1_amt_mean_prev"] == pytest.approx(10.0)
    assert np.isnan(out.loc[4, "card1_amt_std_prev"])     # <2 anteriores
    # id 1 (5h) enxerga 10, 20, 30.
    assert out.loc[1, "card1_amt_mean_prev"] == pytest.approx(20.0)
    assert out.loc[1, "card1_amt_std_prev"] == pytest.approx(10.0)
    assert out.loc[1, "card1_amt_max_prev"] == pytest.approx(30.0)
    assert out.loc[1, "card1_amt_ratio_mean"] == pytest.approx(2.5)
    assert out.loc[1, "card1_amt_zscore"] == pytest.approx(3.0)
    # id 7 (30h) enxerga 10, 20, 30, 50.
    assert out.loc[7, "card1_amt_max_prev"] == pytest.approx(50.0)


@pytest.mark.parametrize("builder", [add_velocity_features, add_aggregation_features])
def test_no_lookahead_future_changes_do_not_affect_past(builder):
    df = _toy()
    base = builder(df).set_index("TransactionID")

    changed = df.copy()
    future = changed["TransactionID"] == 7  # a última transação no tempo
    changed.loc[future, "TransactionAmt"] = 1e6
    changed.loc[future, "TransactionDT"] = 31 * H
    after = builder(changed).set_index("TransactionID")

    past_ids = [1, 2, 3, 4, 5, 6]
    new_cols = [c for c in base.columns if c not in df.columns]
    pd.testing.assert_frame_equal(base.loc[past_ids, new_cols], after.loc[past_ids, new_cols])


def test_frequency_encoder_fit_on_train_only():
    train = pd.DataFrame({"P_emaildomain": pd.Series(["a", "a", "b", None], dtype="category")})
    test = pd.DataFrame({"P_emaildomain": pd.Series(["a", "zzz", None], dtype="category")})

    enc = fit_frequency_encoder(train, ["P_emaildomain"])
    out = apply_frequency_encoder(test, enc)["P_emaildomain_freq"]

    assert out.iloc[0] == pytest.approx(2 / 3)   # frequência relativa entre não-ausentes do treino
    assert out.iloc[1] == 0.0                    # nunca vista no treino
    assert np.isnan(out.iloc[2])                 # ausente continua ausente


def test_missing_flags_and_has_identity():
    df = pd.DataFrame({"id_01": [1.0, np.nan], "id_02": [np.nan, np.nan], "DeviceType": [None, None]})
    out = add_missing_flags(df, ["id_01", "id_02"])
    assert out["id_01_is_missing"].tolist() == [0, 1]
    assert out["has_identity"].tolist() == [1, 0]


def test_build_features_preserves_rows_and_order():
    df = _toy().assign(P_emaildomain=["a", "b", "a", None, "c", "a", "d"])
    train, test = df.iloc[:5], df.iloc[5:]

    train_f, test_f = build_features(train, test)

    assert train_f["TransactionID"].tolist() == train["TransactionID"].tolist()
    assert test_f["TransactionID"].tolist() == test["TransactionID"].tolist()
    assert "__is_test__" not in train_f.columns
    # Teste usa histórico do treino: id 7 (30h) vê as 4 transações anteriores do card 10.
    assert test_f.set_index("TransactionID").loc[7, "card1_n_prev"] == 4
    # Encoder ajustado só no treino: "d" só aparece no teste.
    assert test_f.set_index("TransactionID").loc[7, "P_emaildomain_freq"] == 0.0
