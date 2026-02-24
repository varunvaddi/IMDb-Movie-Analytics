"""
IMDb Rating Predictor v4 — With Director Features
"""

import os, json, io, time
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import boto3
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

load_dotenv()

ACCOUNT_ID       = os.getenv("AWS_ACCOUNT_ID")
SAGEMAKER_BUCKET = f"imdb-sagemaker-{ACCOUNT_ID}"
s3 = boto3.client("s3")

print("=" * 60)
print("IMDb Rating Predictor v4 — Director Features")
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

print("\n[1/5] Loading v4 data...")
df_train = load_parquet(SAGEMAKER_BUCKET, "training-data-v4/train/")
df_test  = load_parquet(SAGEMAKER_BUCKET, "training-data-v4/test/")
print(f"  Train: {len(df_train):,} | Test: {len(df_test):,}")

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
     "long_drama", "classic_documentary", "modern_documentary"] +
    ["director_avg_rating", "director_movie_count", "director_hit_rate",
     "director_votes_log", "has_known_director", "director_max_rating"]
)

X_train = df_train[FEATURE_COLS].fillna(0)
y_train = df_train["target_rating"]
X_test  = df_test[FEATURE_COLS].fillna(0)
y_test  = df_test["target_rating"]
print(f"  Features: {len(FEATURE_COLS)}")

# Train on ALL movies first
print("\n[2/5] Training on full dataset with director features...")
start = time.time()
model_full = xgb.XGBRegressor(
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
model_full.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
train_time = time.time() - start

y_pred_full = model_full.predict(X_test)
rmse_full = np.sqrt(mean_squared_error(y_test, y_pred_full))
r2_full   = r2_score(y_test, y_pred_full)
mae_full  = mean_absolute_error(y_test, y_pred_full)
print(f"  Done in {train_time:.1f}s")

# Also train on 50k+ votes subset
print("\n[3/5] Training on 50k+ votes subset...")
VOTE_THRESHOLD = np.log(50000)
df_all = pd.concat([df_train, df_test], ignore_index=True)
df_filtered = df_all[df_all["votes_log"] >= VOTE_THRESHOLD].copy()
df_tr_f, df_te_f = train_test_split(df_filtered, test_size=0.2, random_state=42)

X_tr_f = df_tr_f[FEATURE_COLS].fillna(0)
y_tr_f = df_tr_f["target_rating"]
X_te_f = df_te_f[FEATURE_COLS].fillna(0)
y_te_f = df_te_f["target_rating"]

model_filtered = xgb.XGBRegressor(
    n_estimators=500, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.7, min_child_weight=3,
    gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, n_jobs=-1, verbosity=0
)
model_filtered.fit(X_tr_f, y_tr_f, eval_set=[(X_te_f, y_te_f)], verbose=False)

y_pred_f = model_filtered.predict(X_te_f)
rmse_f = np.sqrt(mean_squared_error(y_te_f, y_pred_f))
r2_f   = r2_score(y_te_f, y_pred_f)
mae_f  = mean_absolute_error(y_te_f, y_pred_f)

print("\n[4/5] Results comparison:")
print("\n" + "=" * 60)
print(f"  {'Model':<35} {'RMSE':>6}  {'R²':>6}  {'MAE':>6}")
print(f"  {'-'*55}")
print(f"  {'v1 baseline (all movies)':35} {1.1299:>6.4f}  {0.3061:>6.4f}  {0.8559:>6.4f}")
print(f"  {'v3 (50k+ filter, no director)':35} {0.6929:>6.4f}  {0.4615:>6.4f}  {0.4637:>6.4f}")
print(f"  {'v4 full (all + director feats)':35} {rmse_full:>6.4f}  {r2_full:>6.4f}  {mae_full:>6.4f}")
print(f"  {'v4 filtered (50k+ + director)':35} {rmse_f:>6.4f}  {r2_f:>6.4f}  {mae_f:>6.4f}")
print("=" * 60)

# Feature importance
importance = pd.Series(
    model_filtered.feature_importances_, index=FEATURE_COLS
).sort_values(ascending=False)

print("\n  Top 10 features (filtered model):")
for feat, score in importance.head(10).items():
    bar = "█" * int(score * 200)
    print(f"  {feat:35} {score:.4f} {bar}")

# Sample predictions
print("\n  Sample predictions (50k+ votes model):")
sample = df_te_f.sample(8, random_state=42)
preds  = model_filtered.predict(X_te_f.loc[sample.index])
print(f"  {'Title':35} {'Actual':>8} {'Predicted':>10} {'Error':>8}")
print(f"  {'-'*65}")
for (_, row), pred in zip(sample.iterrows(), preds):
    title = str(row.get("primarytitle", "?"))[:33]
    actual = row["target_rating"]
    print(f"  {title:35} {actual:>8.1f} {pred:>10.2f} {abs(actual-pred):>8.2f}")

# Save best model
print("\n[5/5] Saving best model...")
best_model  = model_filtered if r2_f > r2_full else model_full
best_rmse   = rmse_f if r2_f > r2_full else rmse_full
best_r2     = r2_f if r2_f > r2_full else r2_full
best_mae    = mae_f if r2_f > r2_full else mae_full
best_label  = "v4-filtered" if r2_f > r2_full else "v4-full"

os.makedirs("data/processed/model_v4", exist_ok=True)
joblib.dump(best_model, "data/processed/model_v4/model.joblib")

metrics = {
    "version": "v4",
    "best_model": best_label,
    "rmse": round(float(best_rmse), 4),
    "r2":   round(float(best_r2),   4),
    "mae":  round(float(best_mae),  4),
    "full_dataset_r2":     round(float(r2_full), 4),
    "filtered_dataset_r2": round(float(r2_f), 4),
    "n_features": len(FEATURE_COLS),
    "director_features_added": 6,
    "improvement_over_v1": f"R² {0.3061:.4f} → {best_r2:.4f}"
}
with open("data/processed/model_v4/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

s3.upload_file("data/processed/model_v4/model.joblib",
               SAGEMAKER_BUCKET, "model-artifacts/v4/model.joblib")
s3.upload_file("data/processed/model_v4/metrics.json",
               SAGEMAKER_BUCKET, "model-artifacts/v4/metrics.json")

print("\n📊 CAPTURE THIS for your resume:")
print("=" * 60)
print(f"  Final model: XGBoost v4 with director reputation features")
print(f"  Dataset: 309,399 movies, 43 features")
print(f"  Director features: avg_rating, hit_rate, movie_count, votes_log")
print(f"  Full dataset  → RMSE: {rmse_full:.4f}, R²: {r2_full:.4f}")
print(f"  50k+ filtered → RMSE: {rmse_f:.4f}, R²: {r2_f:.4f}")
print(f"  Best R² improvement: 0.3061 → {best_r2:.4f}")
print("=" * 60)
print("✅ v4 training complete!")
