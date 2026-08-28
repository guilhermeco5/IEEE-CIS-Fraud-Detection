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
    """Divide o dataframe em treino/teste por um corte de VALOR de tempo.

    Os últimos `TEST_SIZE_FRACTION` (no tempo) vão para teste, simulando
    como o modelo veria dados em produção: sempre o futuro em relação ao
    que foi usado no treino.

    O corte é por valor de `TEMPORAL_SPLIT_COLUMN` (quantil), não por
    posição (`iloc`) após sort: um corte posicional pode dividir
    transações com o mesmo timestamp entre treino e teste, já que o
    sort não é estável e a ordem entre empates não é determinística.
    Cortar por valor garante que todo grupo de mesmo timestamp fica
    inteiro de um único lado.
    """
    cutoff = df[TEMPORAL_SPLIT_COLUMN].quantile(1 - TEST_SIZE_FRACTION)

    train_df = df[df[TEMPORAL_SPLIT_COLUMN] < cutoff].sort_values(
        TEMPORAL_SPLIT_COLUMN, kind="stable"
    ).reset_index(drop=True)
    test_df = df[df[TEMPORAL_SPLIT_COLUMN] >= cutoff].sort_values(
        TEMPORAL_SPLIT_COLUMN, kind="stable"
    ).reset_index(drop=True)

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
