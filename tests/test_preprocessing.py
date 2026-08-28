"""Teste do split temporal — a decisão de design mais crítica do projeto.

Garante que train sempre vem antes de test no tempo, e que não há
overlap entre os dois conjuntos.
"""

import pandas as pd

from src.preprocessing import temporal_train_test_split


def test_temporal_split_train_before_test():
    df = pd.DataFrame({
        "TransactionDT": range(100),
        "isFraud": [0] * 90 + [1] * 10,
    })

    train_df, test_df = temporal_train_test_split(df)

    assert train_df["TransactionDT"].max() < test_df["TransactionDT"].min(), (
        "Vazamento temporal: há registros de teste com tempo anterior ao treino."
    )


def test_temporal_split_respects_fraction():
    df = pd.DataFrame({
        "TransactionDT": range(100),
        "isFraud": [0] * 90 + [1] * 10,
    })

    train_df, test_df = temporal_train_test_split(df)

    assert len(train_df) == 80
    assert len(test_df) == 20


def test_temporal_split_keeps_tied_timestamps_together():
    """Reproduz o cenário que um split posicional (iloc pós-sort) quebrava:
    várias transações com o mesmo TransactionDT bem no ponto de corte.
    """
    df = pd.DataFrame({
        "TransactionDT": list(range(79)) + [79] * 21,
        "isFraud": [0] * 100,
    })

    train_df, test_df = temporal_train_test_split(df)

    boundary_value = 79
    in_train = (train_df["TransactionDT"] == boundary_value).any()
    in_test = (test_df["TransactionDT"] == boundary_value).any()
    assert not (in_train and in_test), (
        "Transações com o mesmo TransactionDT foram divididas entre treino e teste."
    )
