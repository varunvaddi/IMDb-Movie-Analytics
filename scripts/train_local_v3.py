"""
IMDb Rating Predictor v3
-------------------------
Key insight from v1/v2: low-vote movies are noise.
A movie with 12 votes having 9.2 rating tells us nothing.
Filter to 50k+ votes = reliable ratings = learnable patterns.
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
print("IMDb Rating Predictor v3 — Quality-filtered Dataset")
print("=" * 60)

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

print("\n[1/5] Loading data...")
df_train = load_parquet(SAGEMAKER_BUCKET, "training-data-v2/train/")
df_test  = load_parquet(SAGEMAKER_BUCKET, "training-data-v2/test/")

df_all = pd.concat([df_train, df_test], ignore_index=True)
print(f"  Full dataset: {len(df_all):,} movies")

# ── KEY CHANGE: Filter to well-known movies only ──────────────────────────────
# votes_log > log(50000) ≈ 10.8
VOTE_THRESHOLD = np.log(50000)
df_filtered = df_all[df_all["votes_log"] >= VOTE_THRESHOLD].copy()
print(f"  After 50k+ votes filter: {len(df_filtered):,} movies")
print(f"  Rating range: {df_filtered['target_rating'].min():.1f} - {df_filtered['target_rating'].max():.1f}")
print(f"  Rating mean:  {df_filtered['target_rating'].mean():.2f}")
print(f"  Rating std:   {df_filtered['target_rating'].std():.2f}")

# Re-split on filtered data
from sklearn.model_selection import train_test_split
df_train_f, df_test_f = train_test_split(df_filtered, test_size=0.2, random_state=42)

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

X_train = df_train_f[FEATURE_COLS].fillna(0)
y_train = df_train_f["target_rating"]
X_test  = df_test_f[FEATURE_COLS].fillna(0)
y_test  = df_test_f["target_rating"]

print(f"\n[2/5] Training on {len(X_train):,} well-known movies...")
start = time.time()

model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=5,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
train_time = time.time() - start
print(f"  Done in {train_time:.1f}s")

print("\n[3/5] Evaluating...")
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)

print("\n" + "=" * 60)
print(f"  v1 RMSE: 1.1299  →  v3 RMSE: {rmse:.4f}")
print(f"  v1 R²:   0.3061  →  v3 R²:   {r2:.4f}")
print(f"  v1 MAE:  0.8559  →  v3 MAE:  {mae:.4f}")
print("=" * 60)

importance = pd.Series(
    model.feature_importances_, index=FEATURE_COLS
).sort_values(ascending=False)

print("\n  Top 10 features:")
for feat, score in importance.head(10).items():
    bar = "█" * int(score * 200)
    print(f"  {feat:35} {score:.4f} {bar}")

print("\n[4/5] Sample predictions (well-known movies):")
sample = df_test_f.sample(8, random_state=42)
preds  = model.predict(X_test.loc[sample.index])
print(f"  {'Title':35} {'Actual':>8} {'Predicted':>10} {'Error':>8}")
print(f"  {'-'*65}")
for (_, row), pred in zip(sample.iterrows(), preds):
    title = str(row.get("primarytitle", "?"))[:33]
    actual = row["target_rating"]
    print(f"  {title:35} {actual:>8.1f} {pred:>10.2f} {abs(actual-pred):>8.2f}")

print("\n[5/5] Saving...")
os.makedirs("data/processed/model_v3", exist_ok=True)
joblib.dump(model, "data/processed/model_v3/model.joblib")

metrics = {
    "version": "v3",
    "rmse": round(float(rmse), 4),
    "r2":   round(float(r2),   4),
    "mae":  round(float(mae),  4),
    "dataset_size": len(df_filtered),
    "train_rows": len(df_train_f),
    "test_rows":  len(df_test_f),
    "vote_filter": "50000+",
    "n_features": len(FEATURE_COLS),
    "train_time": round(train_time, 1),
    "insight": "Filtering to 50k+ vote movies removes noise, improves reliability"
}
with open("data/processed/model_v3/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

s3.upload_file("data/processed/model_v3/model.joblib",
               SAGEMAKER_BUCKET, "model-artifacts/v3/model.joblib")
s3.upload_file("data/processed/model_v3/metrics.json",
               SAGEMAKER_BUCKET, "model-artifacts/v3/metrics.json")

print("\n📊 CAPTURE THIS for your resume:")
print("=" * 60)
print(f"  Model v3: XGBoost, quality-filtered (50k+ votes)")
print(f"  Full dataset:    309,399 movies")
print(f"  Filtered subset: {len(df_filtered):,} well-known movies")
print(f"  RMSE: {rmse:.4f}  (v1 baseline: 1.1299)")
print(f"  R²:   {r2:.4f}  (v1 baseline: 0.3061)")
print(f"  MAE:  {mae:.4f}  (v1 baseline: 0.8559)")
print("=" * 60)
print("✅ v3 complete!")
