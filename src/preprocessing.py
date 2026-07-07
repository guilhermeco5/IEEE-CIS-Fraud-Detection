"""Pré-processamento: split temporal e limpeza básica.

Decisão de design mais importante deste módulo: o split treino/teste é
feito por TEMPO, não aleatoriamente. Fraude tem padrões que mudam ao
longo do tempo (novos vetores de ataque, sazonalidade); um split
aleatório otimista vaza informação do "futuro" para o treino e infla
métricas de forma que não se sustenta em produção.
"""

import pandas as pd

from src.config import TARGET_COLUMN, TEMPORAL_SPLIT_COLUMN, TEST_SIZE_FRACTION


def temporal_train_test_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide o dataframe em treino/teste ordenando por tempo.

    Os últimos `TEST_SIZE_FRACTION` registros (no tempo) vão para teste,
    simulando como o modelo veria dados em produção: sempre o futuro
    em relação ao que foi usado no treino.
    """
    df_sorted = df.sort_values(TEMPORAL_SPLIT_COLUMN).reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - TEST_SIZE_FRACTION))

    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]

    return train_df, test_df


def report_missing_by_class(df: pd.DataFrame) -> pd.DataFrame:
    """Compara a taxa de valores ausentes entre fraude e não-fraude.

    Em datasets de fraude, a ausência de um dado (ex: identidade não
    verificada) costuma ser, ela mesma, informativa — por isso vale
    checar isso antes de decidir como imputar, em vez de aplicar
    dropna/fillna sem investigar.
    """
    missing_fraud = df[df[TARGET_COLUMN] == 1].isnull().mean()
    missing_legit = df[df[TARGET_COLUMN] == 0].isnull().mean()

    comparison = pd.DataFrame({
        "missing_rate_fraud": missing_fraud,
        "missing_rate_legit": missing_legit,
    })
    comparison["diff"] = (comparison["missing_rate_fraud"] - comparison["missing_rate_legit"]).abs()
    return comparison.sort_values("diff", ascending=False)
