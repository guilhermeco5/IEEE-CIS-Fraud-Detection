"""Carregamento dos dados brutos do IEEE-CIS Fraud Detection."""

import pandas as pd

from src.config import RAW_IDENTITY_FILE, RAW_TRANSACTION_FILE


def load_raw_data() -> pd.DataFrame:
    """Carrega e une transaction + identity pela chave TransactionID.

    A tabela identity cobre só uma fração das transações (left join
    proposital: a ausência de dados de identidade é, ela mesma, um
    sinal potencialmente relevante para fraude).
    """
    if not RAW_TRANSACTION_FILE.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {RAW_TRANSACTION_FILE}\n"
            "Ver data/raw/README.md para instruções de download via Kaggle."
        )

    transactions = pd.read_csv(RAW_TRANSACTION_FILE)
    identity = pd.read_csv(RAW_IDENTITY_FILE)

    df = transactions.merge(identity, on="TransactionID", how="left")
    return df


if __name__ == "__main__":
    df = load_raw_data()
    print(f"Shape: {df.shape}")
    print(f"Taxa de fraude: {df['isFraud'].mean():.4%}")
