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

# --- Engenharia de features (Semana 2) ---
# Entidade usada nas features de velocidade e agregação. `card1` nunca é
# ausente; o custo é misturar clientes distintos que compartilham o mesmo
# card1 (ver docs/metodologia.md, seção 3).
ENTITY_COLUMN = "card1"
AMOUNT_COLUMN = "TransactionAmt"
ID_COLUMN = "TransactionID"

# Janelas das features de velocidade, em segundos (1h, 6h, 24h).
VELOCITY_WINDOWS_SECONDS = {"1h": 3_600, "6h": 21_600, "24h": 86_400}

# Categóricas de alta cardinalidade com frequency encoding (fit só no treino).
FREQ_ENCODE_COLUMNS = ["P_emaildomain", "R_emaildomain", "DeviceInfo", "id_31", "card1"]

# Colunas que ganham flag binária `{col}_is_missing`.
MISSING_FLAG_COLUMNS = ["id_01", "id_02", "DeviceType", "R_emaildomain"]

# --- Modelagem ---
RANDOM_STATE = 42

XGBOOST_PARAMS = {
    "n_estimators": 1500,  # teto; o número real sai do early stopping
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.5,
    "tree_method": "hist",
    "enable_categorical": True,  # usa as colunas `category` nativamente
    "scale_pos_weight": None,  # calculado em train.py a partir do desbalanceamento real
    "eval_metric": "aucpr",
    "random_state": RANDOM_STATE,
}
EARLY_STOPPING_ROUNDS = 100
# Fração final (no tempo) do TREINO usada como validação do early stopping.
VALIDATION_SIZE_FRACTION = 0.2

# Baseline: só colunas numéricas com até este % de ausência no treino.
BASELINE_MAX_MISSING_RATE = 0.5

FEATURES_TRAIN_FILE = DATA_PROCESSED_DIR / "features_train.parquet"
FEATURES_TEST_FILE = DATA_PROCESSED_DIR / "features_test.parquet"

# --- Avaliação ---
# Recall reportado a este FPR fixo, métrica de negócio relevante para fraude
TARGET_FPR_FOR_RECALL = 0.01
