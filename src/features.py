"""Engenharia de features (Semana 2).

Este módulo concentra as features derivadas — o que diferencia um
projeto de "rodei um classificador" de um projeto de "fiz engenharia
de dados para o problema de fraude". Cada função deve ter uma
justificativa de negócio clara, documentada no docstring.
"""

import pandas as pd


def add_velocity_features(df: pd.DataFrame, entity_col: str, time_col: str) -> pd.DataFrame:
    """Conta quantas transações a mesma entidade (cartão/usuário) fez
    nas últimas N horas. Fraude costuma vir em rajadas — um cartão
    comprometido gera muitas transações em pouco tempo, diferente do
    padrão de uso legítimo.

    TODO (Semana 2): implementar janelas de 1h, 6h, 24h usando rolling
    window sobre o tempo ordenado, agrupado por `entity_col`.
    """
    raise NotImplementedError("Implementar na Semana 2.")


def add_aggregation_features(df: pd.DataFrame, entity_col: str, value_col: str) -> pd.DataFrame:
    """Agregações históricas por entidade (média, desvio, máximo do
    valor da transação). Uma transação muito acima do padrão histórico
    da entidade é um sinal de risco clássico.

    TODO (Semana 2): implementar com groupby + expanding/rolling,
    cuidando de não vazar informação do próprio registro atual.
    """
    raise NotImplementedError("Implementar na Semana 2.")


def encode_high_cardinality_categoricals(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Encoding de categóricas de alta cardinalidade (ex: domínio de
    e-mail, dispositivo). One-hot não escala aqui — considerar target
    encoding com regularização (suavização bayesiana) para evitar
    overfitting em categorias raras.

    TODO (Semana 2): decidir e implementar a estratégia de encoding.
    """
    raise NotImplementedError("Implementar na Semana 2.")
