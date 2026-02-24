"""
IMDb Rating Predictor v2 — Improved Features + Tuned Hyperparameters
"""

import os, json, io, time
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import boto3
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from dotenv import load_dotenv

load_dotenv()

ACCOUNT_ID       = os.getenv("AWS_ACCOUNT_ID")
SAGEMAKER_BUCKET = f"imdb-sagemaker-{ACCOUNT_ID}"
s3 = boto3.client("s3")

print("=" * 60)
print("IMDb Rating Predictor v2 — Improved Features")
print("=" * 60)

# Load v2 data
print("\n[1/4] Loading improved feature set from S3...")
def load_parquet(bucket, prefix):
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    dfs = []
    for obj in response.get("Contents", []):
        if obj["Key"].endswith(".parquet"):
            buf = io.BytesIO()
            s3.download_fileobj(bucket, obj["Key"], buf)
            buf.seek(0)
            dfs.append(pd.read_parquet(buf))
    return pd.concat(dfs, ignore_index=True)

df_train = load_parquet(SAGEMAKER_BUCKET, "training-data-v2/train/")
df_test  = load_parquet(SAGEMAKER_BUCKET, "training-data-v2/test/")

TOP_GENRES = [
    "Drama", "Comedy", "Thriller", "Romance", "Action",
    "Horror", "Crime", "Documentary", "Adventure", "Sci-Fi",
    "Family", "Fantasy", "Mystery", "Biography", "Animation"
]
FEATURE_COLS = (
    [f"genre_{g.lower().replace('-', '_')}" for g in TOP_GENRES] +
    ["year_normalized", "is_modern", "is_classic",
     "decade_2000s", "decade_2010s", "decade_2020s"] +
    ["runtime_normalized", "is_short_film", "is_epic"] +
    ["votes_log", "votes_normalized", "is_well_known",
     "is_blockbuster", "votes_log_squared", "votes_log_binned",
     "popularity_score"] +
    ["is_drama_or_biography", "is_action_adventure", "is_horror_thriller",
     "long_drama", "classic_documentary", "modern_documentary"]
)

X_train = df_train[FEATURE_COLS].fillna(0)
y_train = df_train["target_rating"]
X_test  = df_test[FEATURE_COLS].fillna(0)
y_test  = df_test["target_rating"]

print(f"  Features: {len(FEATURE_COLS)}")
print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

# Train with better hyperparameters
print("\n[2/4] Training XGBoost v2...")
start = time.time()

model = xgb.XGBRegressor(
    n_estimators=500,       # more trees
    max_depth=5,            # slightly shallower to reduce overfitting
    learning_rate=0.05,     # slower learning = better generalization
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=5,     # prevent overfitting on rare genres
    gamma=0.1,
    reg_alpha=0.1,          # L1 regularization
    reg_lambda=1.0,         # L2 regularization
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
train_time = time.time() - start
print(f"  Done in {train_time:.1f}s")

# Evaluate
print("\n[3/4] Evaluating...")
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)

print("\n" + "=" * 60)
print(f"  v1 RMSE: 1.1299  →  v2 RMSE: {rmse:.4f}")
print(f"  v1 R²:   0.3061  →  v2 R²:   {r2:.4f}")
print(f"  v1 MAE:  0.8559  →  v2 MAE:  {mae:.4f}")
print("=" * 60)

# Feature importance
importance = pd.Series(
    model.feature_importances_, index=FEATURE_COLS
).sort_values(ascending=False)

print("\n  Top 10 features:")
for feat, score in importance.head(10).items():
    bar = "█" * int(score * 200)
    print(f"  {feat:35} {score:.4f} {bar}")

# Sample predictions
print("\n  Sample predictions:")
sample = df_test.sample(8, random_state=99)
preds  = model.predict(X_test.loc[sample.index])
print(f"  {'Title':35} {'Actual':>8} {'Predicted':>10} {'Error':>8}")
print(f"  {'-'*65}")
for (_, row), pred in zip(sample.iterrows(), preds):
    title = str(row.get("primarytitle", "?"))[:33]
    actual = row["target_rating"]
    print(f"  {title:35} {actual:>8.1f} {pred:>10.2f} {abs(actual-pred):>8.2f}")

# Save
print("\n[4/4] Saving artifacts...")
os.makedirs("data/processed/model_v2", exist_ok=True)
joblib.dump(model, "data/processed/model_v2/model.joblib")

metrics = {
    "version": "v2",
    "rmse": round(float(rmse), 4),
    "r2":   round(float(r2),   4),
    "mae":  round(float(mae),  4),
    "train_rows": len(df_train),
    "test_rows":  len(df_test),
    "n_features": len(FEATURE_COLS),
    "train_time_seconds": round(train_time, 1),
    "improvements": "added interaction features, votes_log_squared, popularity_score, tuned hyperparameters"
}
with open("data/processed/model_v2/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

s3.upload_file("data/processed/model_v2/model.joblib",
               SAGEMAKER_BUCKET, "model-artifacts/v2/model.joblib")
s3.upload_file("data/processed/model_v2/metrics.json",
               SAGEMAKER_BUCKET, "model-artifacts/v2/metrics.json")

print("\n📊 CAPTURE THIS for your resume:")
print("=" * 60)
print(f"  Model v2: XGBoost (500 estimators, depth 5, regularized)")
print(f"  Dataset:  {len(df_train)+len(df_test):,} movies, {len(FEATURE_COLS)} features")
print(f"  RMSE: {rmse:.4f}  (v1: 1.1299)")
print(f"  R²:   {r2:.4f}  (v1: 0.3061)")
print(f"  MAE:  {mae:.4f}  (v1: 0.8559)")
print(f"  Training time: {train_time:.1f}s")
print("=" * 60)
print("\n✅ v2 training complete!")
