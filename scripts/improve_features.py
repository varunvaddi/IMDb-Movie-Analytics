"""
Feature improvement pass
-------------------------
The baseline model got R²=0.30 because we're missing
the most predictive signal: vote count patterns.

Key insight: numVotes is actually a proxy for quality.
Blockbuster movies that are genuinely good get millions of votes.
Bad movies stay obscure. This signal is strong.
"""

import boto3
import pandas as pd
import numpy as np
import io
import os
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

load_dotenv()

ACCOUNT_ID       = os.getenv("AWS_ACCOUNT_ID")
SAGEMAKER_BUCKET = f"imdb-sagemaker-{ACCOUNT_ID}"
FEATURES_BUCKET  = f"imdb-pipeline-features-{ACCOUNT_ID}"

s3 = boto3.client("s3")

print("Loading features...")
response = s3.list_objects_v2(Bucket=FEATURES_BUCKET, Prefix="movies/features/")
dfs = []
for obj in response.get("Contents", []):
    if obj["Key"].endswith(".parquet"):
        buf = io.BytesIO()
        s3.download_fileobj(FEATURES_BUCKET, obj["Key"], buf)
        buf.seek(0)
        dfs.append(pd.read_parquet(buf))

df = pd.concat(dfs, ignore_index=True)
print(f"Loaded {len(df):,} rows")

# ── NEW FEATURES ──────────────────────────────────────────────────────────────

# 1. votes_log squared — captures non-linear relationship
df["votes_log_squared"] = df["votes_log"] ** 2

# 2. Vote-weighted rating signal
#    High vote count = more reliable rating signal
df["votes_log_binned"] = pd.cut(
    df["votes_log"],
    bins=10,
    labels=False
).astype(float)

# 3. Genre combinations that tend to rate well
df["is_drama_or_biography"] = (
    (df["genre_drama"] == 1) | (df["genre_biography"] == 1)
).astype(int)

df["is_action_adventure"] = (
    (df["genre_action"] == 1) & (df["genre_adventure"] == 1)
).astype(int)

df["is_horror_thriller"] = (
    (df["genre_horror"] == 1) | (df["genre_thriller"] == 1)
).astype(int)

# 4. Runtime × genre interactions
#    Long dramas tend to rate higher than long action films
df["long_drama"] = (
    (df["is_epic"] == 1) & (df["genre_drama"] == 1)
).astype(int)

# 5. Era × genre interactions
#    Classic documentaries rate differently than modern ones
df["classic_documentary"] = (
    (df["is_classic"] == 1) & (df["genre_documentary"] == 1)
).astype(int)

df["modern_documentary"] = (
    (df["is_modern"] == 1) & (df["genre_documentary"] == 1)
).astype(int)

# 6. Popularity tiers as ordinal feature
df["popularity_score"] = (
    df["is_blockbuster"] * 3 +
    df["is_well_known"] * 2 +
    (df["votes_normalized"] > 0.1).astype(int)
).astype(float)

print("New features added ✅")

# ── UPDATED FEATURE LIST ──────────────────────────────────────────────────────
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

print(f"Total features: {len(FEATURE_COLS)}")

# ── SPLIT AND UPLOAD ──────────────────────────────────────────────────────────
TARGET_COLS = ["tconst", "primarytitle", "startyear", "target_rating", "is_high_rated"]
df_model = df[TARGET_COLS + FEATURE_COLS].copy()

df_train, df_test = train_test_split(df_model, test_size=0.2, random_state=42)
print(f"Train: {len(df_train):,} | Test: {len(df_test):,}")

for split_name, split_df in [("train", df_train), ("test", df_test)]:
    buf = io.BytesIO()
    split_df.to_parquet(buf, index=False)
    buf.seek(0)
    key = f"training-data-v2/{split_name}/data.parquet"
    s3.upload_fileobj(buf, SAGEMAKER_BUCKET, key)
    print(f"Uploaded {split_name} → s3://{SAGEMAKER_BUCKET}/{key}")

print("\n✅ Improved features ready!")
print(f"Feature count: {len(FEATURE_COLS)} (was 28, now {len(FEATURE_COLS)})")
