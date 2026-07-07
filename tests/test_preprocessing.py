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
