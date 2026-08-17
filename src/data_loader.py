"""Carregamento dos dados brutos do IEEE-CIS Fraud Detection."""

import pandas as pd

from src.config import DATA_PROCESSED_DIR, RAW_IDENTITY_FILE, RAW_TRANSACTION_FILE

CACHE_FILE = DATA_PROCESSED_DIR / "train_merged.parquet"


def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Reduz o uso de memória: object de baixa cardinalidade -> category, downcast numérico."""
    for col in df.select_dtypes(include="object").columns:
        if df[col].nunique(dropna=True) / len(df) < 0.5:
            df[col] = df[col].astype("category")
    for col in df.select_dtypes(include="float64").columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include="int64").columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def load_raw_data(use_cache: bool = True) -> pd.DataFrame:
    """Carrega e une transaction + identity pela chave TransactionID.

    A tabela identity cobre só uma fração das transações (left join
    proposital: a ausência de dados de identidade é, ela mesma, um
    sinal potencialmente relevante para fraude).

    Usa um cache em Parquet (dtypes otimizados) em data/processed/ para
    evitar reprocessar os CSVs brutos (434 colunas) a cada load.
    """
    if use_cache and CACHE_FILE.exists():
        return pd.read_parquet(CACHE_FILE)

    if not RAW_TRANSACTION_FILE.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {RAW_TRANSACTION_FILE}\n"
            "Ver data/raw/README.md para instruções de download via Kaggle."
        )

    transactions = pd.read_csv(RAW_TRANSACTION_FILE)
    identity = pd.read_csv(RAW_IDENTITY_FILE)

    df = transactions.merge(identity, on="TransactionID", how="left")
    df = _optimize_dtypes(df)

    if use_cache:
        DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(CACHE_FILE)

    return df


if __name__ == "__main__":
    df = load_raw_data()
    print(f"Shape: {df.shape}")
    print(f"Taxa de fraude: {df['isFraud'].mean():.4%}")
