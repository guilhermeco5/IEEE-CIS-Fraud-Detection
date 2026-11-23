"""Treino e comparação dos modelos (Semana 3).

Baseline simples primeiro (regressão logística), depois XGBoost.
O desbalanceamento é tratado via pesos (class_weight / scale_pos_weight),
não SMOTE: gerar exemplos sintéticos de fraude por interpolação tende a
criar pontos que não existem na distribuição real. O peso ajusta a perda
sem inventar dados.

Protocolo de validação, todo temporal:
    treino (80%) ──► treino interno (80%) + validação (20% final do treino)
                     └─ early stopping escolhe o nº de árvores na validação
                 ──► refit no treino inteiro com esse nº de árvores
    teste (20% final) ── usado UMA vez, só para reportar.

Uso: `python -m src.train` (ou `--no-ablation` para pular o XGBoost sem as
features da Semana 2).
"""

import argparse
import gc
import json
import time

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import (
    BASELINE_MAX_MISSING_RATE,
    EARLY_STOPPING_ROUNDS,
    FEATURES_TEST_FILE,
    FEATURES_TRAIN_FILE,
    ID_COLUMN,
    METRICS_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEMPORAL_SPLIT_COLUMN,
    VALIDATION_SIZE_FRACTION,
    XGBOOST_PARAMS,
)
from src.evaluate import compare_models, evaluate_by_week, evaluate_scores
from src.preprocessing import temporal_train_test_split

# Identificador, tempo absoluto e alvo nunca entram como feature. O tempo
# absoluto não generaliza: no futuro, todo TransactionDT é maior que no treino.
NON_FEATURE_COLUMNS = [ID_COLUMN, TEMPORAL_SPLIT_COLUMN, TARGET_COLUMN]


def load_feature_data(use_cache: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Treino/teste com as features da Semana 2, com cache em parquet."""
    if use_cache and FEATURES_TRAIN_FILE.exists() and FEATURES_TEST_FILE.exists():
        return pd.read_parquet(FEATURES_TRAIN_FILE), pd.read_parquet(FEATURES_TEST_FILE)

    from src.data_loader import load_raw_data
    from src.features import build_features

    train, test = temporal_train_test_split(load_raw_data())
    train, test = build_features(train, test)
    if use_cache:
        train.to_parquet(FEATURES_TRAIN_FILE)
        test.to_parquet(FEATURES_TEST_FILE)
    return train, test


def engineered_feature_columns(df: pd.DataFrame) -> list[str]:
    """Colunas criadas na Semana 2 (as que não existem no dataset bruto)."""
    from src.data_loader import CACHE_FILE

    raw_columns = set(pq.read_schema(CACHE_FILE).names)
    return [c for c in df.columns if c not in raw_columns]


def make_xy(df: pd.DataFrame, drop: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    to_drop = [c for c in NON_FEATURE_COLUMNS + (drop or []) if c in df.columns]
    return df.drop(columns=to_drop), df[TARGET_COLUMN]


def baseline_columns(X_train: pd.DataFrame) -> list[str]:
    """Colunas numéricas com ausência <= BASELINE_MAX_MISSING_RATE no treino.

    Regressão logística não aceita NaN nem `category`; imputar centenas de
    colunas quase vazias pela mediana só adiciona ruído (e memória). As
    flags de ausência da Semana 2 já carregam a informação de "faltou".
    """
    numeric = X_train.select_dtypes(include="number")
    return numeric.columns[numeric.isna().mean() <= BASELINE_MAX_MISSING_RATE].tolist()


def train_baseline(X_train: pd.DataFrame, y_train: pd.Series):
    """Imputação pela mediana + padronização + regressão logística balanceada."""
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE),
    )
    model.fit(X_train.astype(np.float32), y_train)
    return model


def _xgb(n_estimators: int | None = None, early_stopping: bool = False, **overrides) -> XGBClassifier:
    params = {**XGBOOST_PARAMS, **overrides}
    if n_estimators is not None:
        params["n_estimators"] = n_estimators
    if early_stopping:
        params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
    return XGBClassifier(**params)


def train_xgboost(train_df: pd.DataFrame, drop: list[str] | None = None) -> tuple[XGBClassifier, int]:
    """Early stopping numa validação temporal interna, depois refit no treino todo.

    `scale_pos_weight` = negativos / positivos do treino. Ele desloca as
    probabilidades (o modelo não fica calibrado), mas PR-AUC e recall a
    FPR fixo só dependem da ordenação, então não é um problema aqui.
    """
    inner, valid = temporal_train_test_split(train_df, test_size=VALIDATION_SIZE_FRACTION)
    X_in, y_in = make_xy(inner, drop)
    X_va, y_va = make_xy(valid, drop)
    spw = float((y_in == 0).sum() / (y_in == 1).sum())

    probe = _xgb(early_stopping=True, scale_pos_weight=spw)
    probe.fit(X_in, y_in, eval_set=[(X_va, y_va)], verbose=False)
    best_n = int(probe.best_iteration) + 1
    del probe, X_in, X_va, inner, valid
    gc.collect()

    X, y = make_xy(train_df, drop)
    spw = float((y == 0).sum() / (y == 1).sum())
    model = _xgb(n_estimators=best_n, scale_pos_weight=spw)
    model.fit(X, y, verbose=False)
    return model, best_n


def save_model(model, filename: str) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / filename)
    print(f"Modelo salvo em {MODELS_DIR / filename}")


def main(run_ablation: bool = True) -> pd.DataFrame:
    t0 = time.perf_counter()
    train_df, test_df = load_feature_data()
    eng_cols = engineered_feature_columns(train_df)
    print(f"Dados: treino {train_df.shape}, teste {test_df.shape}, {len(eng_cols)} features da Semana 2 "
          f"({time.perf_counter() - t0:.0f}s)")

    results, extras = {}, {}
    X_test, y_test = make_xy(test_df)

    # 1) Baseline
    t = time.perf_counter()
    X_train, y_train = make_xy(train_df)
    cols = baseline_columns(X_train)
    baseline = train_baseline(X_train[cols], y_train)
    del X_train
    gc.collect()
    results["regressao_logistica"] = evaluate_scores(y_test, baseline.predict_proba(X_test[cols].astype(np.float32))[:, 1])
    extras["regressao_logistica"] = {"n_features": len(cols), "segundos": round(time.perf_counter() - t)}
    save_model(baseline, "baseline_logreg.joblib")
    print("baseline:", results["regressao_logistica"], extras["regressao_logistica"])
    del baseline
    gc.collect()

    # 2) XGBoost com todas as features
    t = time.perf_counter()
    xgb, best_n = train_xgboost(train_df)
    proba = xgb.predict_proba(X_test)[:, 1]
    results["xgboost"] = evaluate_scores(y_test, proba)
    extras["xgboost"] = {"n_features": X_test.shape[1], "n_arvores": best_n, "segundos": round(time.perf_counter() - t)}
    save_model(xgb, "xgboost.joblib")
    print("xgboost:", results["xgboost"], extras["xgboost"])

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    by_week = evaluate_by_week(y_test, proba, test_df[TEMPORAL_SPLIT_COLUMN])
    by_week.to_csv(METRICS_DIR / "xgboost_por_semana.csv")
    importance = pd.Series(xgb.get_booster().get_score(importance_type="gain")).sort_values(ascending=False)
    importance.rename("gain").to_csv(METRICS_DIR / "xgboost_importancia_gain.csv")
    del xgb
    gc.collect()

    # 3) Ablação: mesmo XGBoost sem as features da Semana 2. Mede o quanto elas ajudam.
    if run_ablation:
        t = time.perf_counter()
        xgb_raw, best_n_raw = train_xgboost(train_df, drop=eng_cols)
        results["xgboost_sem_features_semana2"] = evaluate_scores(
            y_test, xgb_raw.predict_proba(X_test.drop(columns=eng_cols))[:, 1]
        )
        extras["xgboost_sem_features_semana2"] = {
            "n_features": X_test.shape[1] - len(eng_cols), "n_arvores": best_n_raw,
            "segundos": round(time.perf_counter() - t),
        }
        print("ablação:", results["xgboost_sem_features_semana2"], extras["xgboost_sem_features_semana2"])

    table = compare_models(results).join(pd.DataFrame(extras).T)
    table.to_csv(METRICS_DIR / "comparacao_modelos.csv")
    (METRICS_DIR / "comparacao_modelos.json").write_text(
        json.dumps({k: {**results[k], **extras[k]} for k in results}, indent=2), encoding="utf-8"
    )
    print("\n", table.to_string(), sep="")
    print("\nPor semana (XGBoost, teste):\n", by_week.to_string(), sep="")
    print("\nTop 15 features (gain):\n", importance.head(15).round(1).to_string(), sep="")
    print(f"\nTotal: {time.perf_counter() - t0:.0f}s")
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-ablation", action="store_true", help="pula o XGBoost sem features da Semana 2")
    main(run_ablation=not parser.parse_args().no_ablation)
