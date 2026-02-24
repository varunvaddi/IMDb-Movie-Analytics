"""
IMDb Rating Predictor — SageMaker Training Script
--------------------------------------------------
SageMaker runs this script inside a managed ML container.
It reads features from S3, trains XGBoost, evaluates,
and saves the model artifact back to S3.

SageMaker injects these environment variables automatically:
- SM_MODEL_DIR     → where to save the model
- SM_CHANNEL_TRAIN → where training data lives
- SM_CHANNEL_TEST  → where test data lives
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()

    # SageMaker injects these automatically
    parser.add_argument("--model-dir",  default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train",      default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test",       default=os.environ.get("SM_CHANNEL_TEST"))

    # Hyperparameters — tunable
    parser.add_argument("--n-estimators",   type=int,   default=200)
    parser.add_argument("--max-depth",      type=int,   default=6)
    parser.add_argument("--learning-rate",  type=float, default=0.1)
    parser.add_argument("--subsample",      type=float, default=0.8)
    parser.add_argument("--min-votes",      type=int,   default=100)

    return parser.parse_args()


def load_data(data_dir):
    """Load all parquet files from a directory."""
    files = [f for f in os.listdir(data_dir) if f.endswith(".parquet")]
    dfs = [pd.read_parquet(os.path.join(data_dir, f)) for f in files]
    return pd.concat(dfs, ignore_index=True)


def get_feature_columns():
    """Return the exact feature columns the model trains on."""
    TOP_GENRES = [
        "Drama", "Comedy", "Thriller", "Romance", "Action",
        "Horror", "Crime", "Documentary", "Adventure", "Sci-Fi",
        "Family", "Fantasy", "Mystery", "Biography", "Animation"
    ]
    genre_cols = [f"genre_{g.lower().replace('-', '_')}" for g in TOP_GENRES]

    return genre_cols + [
        "year_normalized", "is_modern", "is_classic",
        "decade_2000s", "decade_2010s", "decade_2020s",
        "runtime_normalized", "is_short_film", "is_epic",
        "votes_log", "votes_normalized", "is_well_known", "is_blockbuster"
    ]


if __name__ == "__main__":
    args = parse_args()
    logger.info("Starting IMDb rating predictor training...")
    logger.info(f"Hyperparameters: {vars(args)}")

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("Loading training data...")
    df_train = load_data(args.train)
    df_test  = load_data(args.test)

    logger.info(f"Train set: {len(df_train):,} rows")
    logger.info(f"Test set:  {len(df_test):,} rows")

    FEATURE_COLS = get_feature_columns()
    TARGET = "target_rating"

    X_train = df_train[FEATURE_COLS].fillna(0)
    y_train = df_train[TARGET]
    X_test  = df_test[FEATURE_COLS].fillna(0)
    y_test  = df_test[TARGET]

    logger.info(f"Training on {len(FEATURE_COLS)} features")

    # ── Train XGBoost ─────────────────────────────────────────────────────────
    logger.info("Training XGBoost model...")
    model = xgb.XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    logger.info("Evaluating model...")
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)

    logger.info("=" * 50)
    logger.info(f"RMSE : {rmse:.4f}")
    logger.info(f"R²   : {r2:.4f}")
    logger.info(f"MAE  : {mae:.4f}")
    logger.info("=" * 50)

    # ── Feature importance ────────────────────────────────────────────────────
    importance = pd.Series(
        model.feature_importances_,
        index=FEATURE_COLS
    ).sort_values(ascending=False)

    logger.info("Top 10 most important features:")
    for feat, score in importance.head(10).items():
        logger.info(f"  {feat:35} {score:.4f}")

    # ── Save metrics ──────────────────────────────────────────────────────────
    metrics = {
        "rmse": round(float(rmse), 4),
        "r2":   round(float(r2),   4),
        "mae":  round(float(mae),  4),
        "train_rows": len(df_train),
        "test_rows":  len(df_test),
        "n_features": len(FEATURE_COLS)
    }

    metrics_path = os.path.join(args.model_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved → {metrics_path}")

    # ── Save model ────────────────────────────────────────────────────────────
    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)
    logger.info(f"Model saved → {model_path}")

    logger.info("Training complete!")
