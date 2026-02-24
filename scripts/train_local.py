"""
Local Training Script — Mirrors SageMaker Workflow
----------------------------------------------------
Trains the exact same XGBoost model as src/sagemaker/training/train_rating_predictor.py
but runs locally instead of on a SageMaker managed instance.

When you have a real AWS account:
1. Run scripts/launch_training_job.py instead
2. Everything else (metrics, model artifacts, evaluation) stays identical
"""

import os
import json
import io
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import boto3
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from dotenv import load_dotenv
import time

load_dotenv()

ACCOUNT_ID       = os.getenv("AWS_ACCOUNT_ID")
REGION           = os.getenv("AWS_REGION", "us-east-1")
SAGEMAKER_BUCKET = f"imdb-sagemaker-{ACCOUNT_ID}"

s3 = boto3.client("s3", region_name=REGION)

print("=" * 60)
print("IMDb Rating Predictor — Local Training")
print("(Mirrors SageMaker training job workflow)")
print("=" * 60)

# ── STEP 1: Load data from S3 (same as SageMaker would) ───────────────────────
print("\n[1/5] Loading train/test data from S3...")
start = time.time()

def load_parquet_from_s3(bucket, prefix):
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    dfs = []
    for obj in response.get("Contents", []):
        if obj["Key"].endswith(".parquet"):
            buf = io.BytesIO()
            s3.download_fileobj(bucket, obj["Key"], buf)
            buf.seek(0)
            dfs.append(pd.read_parquet(buf))
    return pd.concat(dfs, ignore_index=True)

df_train = load_parquet_from_s3(SAGEMAKER_BUCKET, "training-data/train/")
df_test  = load_parquet_from_s3(SAGEMAKER_BUCKET, "training-data/test/")

print(f"  Train: {len(df_train):,} rows")
print(f"  Test:  {len(df_test):,} rows")
print(f"  Loaded in {time.time()-start:.1f}s")

# ── STEP 2: Prepare feature matrix ────────────────────────────────────────────
print("\n[2/5] Preparing feature matrix...")

TOP_GENRES = [
    "Drama", "Comedy", "Thriller", "Romance", "Action",
    "Horror", "Crime", "Documentary", "Adventure", "Sci-Fi",
    "Family", "Fantasy", "Mystery", "Biography", "Animation"
]
FEATURE_COLS = (
    [f"genre_{g.lower().replace('-', '_')}" for g in TOP_GENRES] +
    ["year_normalized", "is_modern", "is_classic",
     "decade_2000s", "decade_2010s", "decade_2020s",
     "runtime_normalized", "is_short_film", "is_epic",
     "votes_log", "votes_normalized", "is_well_known", "is_blockbuster"]
)
TARGET = "target_rating"

X_train = df_train[FEATURE_COLS].fillna(0)
y_train = df_train[TARGET]
X_test  = df_test[FEATURE_COLS].fillna(0)
y_test  = df_test[TARGET]

print(f"  Features: {len(FEATURE_COLS)}")
print(f"  X_train shape: {X_train.shape}")
print(f"  X_test shape:  {X_test.shape}")

# ── STEP 3: Train XGBoost ──────────────────────────────────────────────────────
print("\n[3/5] Training XGBoost model...")
start = time.time()

model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

train_time = time.time() - start
print(f"  Training complete in {train_time:.1f}s")

# ── STEP 4: Evaluate ───────────────────────────────────────────────────────────
print("\n[4/5] Evaluating model...")
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)

print("\n" + "=" * 60)
print(f"  RMSE : {rmse:.4f}  (target: < 1.0)")
print(f"  R²   : {r2:.4f}  (target: > 0.75)")
print(f"  MAE  : {mae:.4f}")
print("=" * 60)

# Feature importance
importance = pd.Series(
    model.feature_importances_,
    index=FEATURE_COLS
).sort_values(ascending=False)

print("\n  Top 10 most important features:")
for feat, score in importance.head(10).items():
    bar = "█" * int(score * 200)
    print(f"  {feat:35} {score:.4f} {bar}")

# Sample predictions vs actuals
print("\n  Sample predictions vs actuals:")
sample_idx = df_test.sample(8, random_state=42).index
sample_titles  = df_test.loc[sample_idx, "primarytitle"] if "primarytitle" in df_test.columns else ["?"] * 8
sample_actuals = y_test.loc[sample_idx].values
sample_preds   = model.predict(X_test.loc[sample_idx])

print(f"  {'Title':35} {'Actual':>8} {'Predicted':>10} {'Error':>8}")
print(f"  {'-'*65}")
for title, actual, pred in zip(sample_titles, sample_actuals, sample_preds):
    error = abs(actual - pred)
    title_str = str(title)[:33]
    print(f"  {title_str:35} {actual:>8.1f} {pred:>10.2f} {error:>8.2f}")

# ── STEP 5: Save model + metrics ──────────────────────────────────────────────
print("\n[5/5] Saving model artifacts...")
os.makedirs("data/processed/model", exist_ok=True)

# Save model locally
model_path = "data/processed/model/model.joblib"
joblib.dump(model, model_path)
print(f"  Model saved → {model_path}")

# Save metrics
metrics = {
    "rmse":        round(float(rmse), 4),
    "r2":          round(float(r2),   4),
    "mae":         round(float(mae),  4),
    "train_rows":  len(df_train),
    "test_rows":   len(df_test),
    "n_features":  len(FEATURE_COLS),
    "train_time_seconds": round(train_time, 1),
    "top_features": importance.head(5).to_dict()
}

metrics_path = "data/processed/model/metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"  Metrics saved → {metrics_path}")

# Upload model artifact to S3 (mirrors SageMaker output)
print("\n  Uploading model artifact to S3...")
s3.upload_file(
    model_path,
    SAGEMAKER_BUCKET,
    "model-artifacts/local-training/model.joblib"
)
s3.upload_file(
    metrics_path,
    SAGEMAKER_BUCKET,
    "model-artifacts/local-training/metrics.json"
)
print(f"  Uploaded → s3://{SAGEMAKER_BUCKET}/model-artifacts/local-training/")

print("\n📊 CAPTURE THIS — copy everything between the === lines for your resume")
print("=" * 60)
print(f"  Model: XGBoost Regressor (200 estimators, depth 6)")
print(f"  Dataset: {len(df_train)+len(df_test):,} movies, {len(FEATURE_COLS)} features")
print(f"  Train/Test: {len(df_train):,} / {len(df_test):,}")
print(f"  RMSE: {rmse:.4f}")
print(f"  R²:   {r2:.4f}")
print(f"  MAE:  {mae:.4f}")
print(f"  Training time: {train_time:.1f}s")
print("=" * 60)
print("\n✅ Local training complete! Model ready for deployment.")
print("🚀 To run on SageMaker: python scripts/launch_training_job.py")
