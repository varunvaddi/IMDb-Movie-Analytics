"""
Director Feature Engineering — Final Version
---------------------------------------------
Merges Phase 4 features (genre/year/runtime) with
Glue-computed director features. Both are small files.
"""

import pandas as pd
import numpy as np
import boto3
import io
import os
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

load_dotenv()

ACCOUNT_ID       = os.getenv("AWS_ACCOUNT_ID")
SAGEMAKER_BUCKET = f"imdb-sagemaker-{ACCOUNT_ID}"
FEATURES_BUCKET  = f"imdb-pipeline-features-{ACCOUNT_ID}"

s3 = boto3.client("s3")

print("=" * 60)
print("Director Feature Engineering — Merging all features")
print("=" * 60)

def load_parquet_s3(bucket, prefix):
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    dfs = []
    for obj in response.get("Contents", []):
        if obj["Key"].endswith(".parquet"):
            buf = io.BytesIO()
            s3.download_fileobj(bucket, obj["Key"], buf)
            buf.seek(0)
            dfs.append(pd.read_parquet(buf))
    return pd.concat(dfs, ignore_index=True)

# ── Load Phase 4 features (genre/year/runtime/votes) ─────────────────────────
print("\n[1/5] Loading Phase 4 features (genre/year/runtime)...")
df_phase4 = load_parquet_s3(FEATURES_BUCKET, "movies/features/")
print(f"  Loaded {len(df_phase4):,} rows, {len(df_phase4.columns)} columns")
print(f"  Columns: {list(df_phase4.columns)}")

# ── Load Glue director features ───────────────────────────────────────────────
print("\n[2/5] Loading director features from Glue output...")
df_director = load_parquet_s3(FEATURES_BUCKET, "movies/director_features/")

# Keep only director columns + tconst for joining
DIR_COLS = ["tconst", "director_avg_rating", "director_movie_count",
            "director_hit_rate", "director_votes_log",
            "has_known_director", "director_max_rating"]

available_dir_cols = [c for c in DIR_COLS if c in df_director.columns]
df_director_slim = df_director[available_dir_cols].drop_duplicates(subset="tconst")
print(f"  Loaded {len(df_director_slim):,} rows")
print(f"  Director columns: {available_dir_cols}")

# ── Merge on tconst ───────────────────────────────────────────────────────────
print("\n[3/5] Merging features on tconst...")
df_merged = df_phase4.merge(df_director_slim, on="tconst", how="left")

# Fill nulls for movies without director info
df_merged["director_avg_rating"]  = df_merged["director_avg_rating"].fillna(6.0)
df_merged["director_movie_count"] = df_merged["director_movie_count"].fillna(1)
df_merged["director_hit_rate"]    = df_merged["director_hit_rate"].fillna(0.3)
df_merged["director_votes_log"]   = df_merged["director_votes_log"].fillna(5.0)
df_merged["has_known_director"]   = df_merged["has_known_director"].fillna(0)
df_merged["director_max_rating"]  = df_merged["director_max_rating"].fillna(6.0)

print(f"  Merged: {len(df_merged):,} rows, {len(df_merged.columns)} columns")
print(f"  Movies with known director: {(df_merged['has_known_director']==1).sum():,}")

# ── Add interaction features ──────────────────────────────────────────────────
print("\n[4/5] Adding interaction features...")

# votes_log_squared and votes_log_binned (from v2 that may be missing)
if "votes_log_squared" not in df_merged.columns:
    df_merged["votes_log_squared"] = df_merged["votes_log"] ** 2

if "votes_log_binned" not in df_merged.columns:
    df_merged["votes_log_binned"] = pd.cut(
        df_merged["votes_log"], bins=10, labels=False
    ).astype(float)

if "popularity_score" not in df_merged.columns:
    df_merged["popularity_score"] = (
        df_merged["is_blockbuster"] * 3 +
        df_merged["is_well_known"] * 2 +
        (df_merged["votes_normalized"] > 0.1).astype(int)
    ).astype(float)

if "is_drama_or_biography" not in df_merged.columns:
    df_merged["is_drama_or_biography"] = (
        (df_merged["genre_drama"] == 1) | (df_merged["genre_biography"] == 1)
    ).astype(int)

if "is_action_adventure" not in df_merged.columns:
    df_merged["is_action_adventure"] = (
        (df_merged["genre_action"] == 1) & (df_merged["genre_adventure"] == 1)
    ).astype(int)

if "is_horror_thriller" not in df_merged.columns:
    df_merged["is_horror_thriller"] = (
        (df_merged["genre_horror"] == 1) | (df_merged["genre_thriller"] == 1)
    ).astype(int)

if "long_drama" not in df_merged.columns:
    df_merged["long_drama"] = (
        (df_merged["is_epic"] == 1) & (df_merged["genre_drama"] == 1)
    ).astype(int)

if "classic_documentary" not in df_merged.columns:
    df_merged["classic_documentary"] = (
        (df_merged["is_classic"] == 1) & (df_merged["genre_documentary"] == 1)
    ).astype(int)

if "modern_documentary" not in df_merged.columns:
    df_merged["modern_documentary"] = (
        (df_merged["is_modern"] == 1) & (df_merged["genre_documentary"] == 1)
    ).astype(int)

# ── Build final feature list ──────────────────────────────────────────────────
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

available = [c for c in FEATURE_COLS if c in df_merged.columns]
missing   = [c for c in FEATURE_COLS if c not in df_merged.columns]

if missing:
    print(f"  ⚠️  Still missing: {missing}")
else:
    print(f"  ✅ All {len(FEATURE_COLS)} features present!")

# ── Split and upload ──────────────────────────────────────────────────────────
print("\n[5/5] Splitting and uploading...")

TARGET_COLS = ["tconst", "primarytitle", "startyear", "target_rating", "is_high_rated"]
keep_cols   = [c for c in TARGET_COLS if c in df_merged.columns] + available
df_model    = df_merged[keep_cols].copy()

df_train, df_test = train_test_split(df_model, test_size=0.2, random_state=42)

for split_name, split_df in [("train", df_train), ("test", df_test)]:
    buf = io.BytesIO()
    split_df.to_parquet(buf, index=False)
    buf.seek(0)
    key = f"training-data-v4/{split_name}/data.parquet"
    s3.upload_fileobj(buf, SAGEMAKER_BUCKET, key)
    print(f"  Uploaded {split_name} → s3://{SAGEMAKER_BUCKET}/{key}")

print(f"\n✅ All features merged and ready!")
print(f"  Total features: {len(available)}")
print(f"  Train: {len(df_train):,} | Test: {len(df_test):,}")
