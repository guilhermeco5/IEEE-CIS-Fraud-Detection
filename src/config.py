"""Configuração central do projeto: caminhos e parâmetros.

Mantém todos os caminhos e hiperparâmetros em um único lugar, para que
notebooks e scripts em src/ nunca tenham caminhos hardcoded espalhados.
"""

from pathlib import Path

# --- Caminhos ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"

RAW_TRANSACTION_FILE = DATA_RAW_DIR / "train_transaction.csv"
RAW_IDENTITY_FILE = DATA_RAW_DIR / "train_identity.csv"

# --- Split temporal ---
# IEEE-CIS fornece TransactionDT (segundos desde um ponto de referência).
# Split aleatório vaza informação do futuro para o treino — usar sempre
# split temporal, treinando no passado e validando no "futuro".
TEMPORAL_SPLIT_COLUMN = "TransactionDT"
TEST_SIZE_FRACTION = 0.2  # últimos 20% no tempo vão para validação

# --- Alvo ---
TARGET_COLUMN = "isFraud"

# --- Modelagem ---
RANDOM_STATE = 42

XGBOOST_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "scale_pos_weight": None,  # calculado em train.py a partir do desbalanceamento real
    "eval_metric": "aucpr",
    "random_state": RANDOM_STATE,
}

# --- Avaliação ---
# Recall reportado a este FPR fixo, métrica de negócio relevante para fraude
TARGET_FPR_FOR_RECALL = 0.01
