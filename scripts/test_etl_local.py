"""
Local ETL test script
---------------------
Mimics what the Glue job does but runs locally with Pandas.
Use this to validate logic before deploying to AWS Glue.
"""

import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

RAW_PATH = "data/raw"

print("=" * 60)
print("IMDb ETL — Local Test Run")
print("=" * 60)

# ── EXTRACT ───────────────────────────────────────────────────────────────────
print("\n[1/5] Reading raw files...")
df_basics = pd.read_csv(
    f"{RAW_PATH}/title.basics.tsv.gz",
    sep="\t",
    na_values="\\N",
    low_memory=False
)
df_ratings = pd.read_csv(
    f"{RAW_PATH}/title.ratings.tsv.gz",
    sep="\t",
    na_values="\\N"
)

print(f"  basics:  {len(df_basics):>10,} rows | columns: {list(df_basics.columns)}")
print(f"  ratings: {len(df_ratings):>10,} rows | columns: {list(df_ratings.columns)}")

# ── TRANSFORM ─────────────────────────────────────────────────────────────────
print("\n[2/5] Filtering movies only...")
df_movies = df_basics[df_basics["titleType"] == "movie"].copy()
print(f"  movies: {len(df_movies):,} (from {len(df_basics):,} total titles)")

print("\n[3/5] Joining basics + ratings...")
df_joined = df_movies.merge(df_ratings, on="tconst", how="inner")
print(f"  joined: {len(df_joined):,} movies with ratings")

print("\n[4/5] Cleaning and enriching...")
# Cast types
df_joined["startYear"] = pd.to_numeric(df_joined["startYear"], errors="coerce")
df_joined["runtimeMinutes"] = pd.to_numeric(df_joined["runtimeMinutes"], errors="coerce")
df_joined["averageRating"] = pd.to_numeric(df_joined["averageRating"], errors="coerce")
df_joined["numVotes"] = pd.to_numeric(df_joined["numVotes"], errors="coerce")

# Drop nulls in critical columns
df_clean = df_joined.dropna(subset=["primaryTitle", "startYear", "averageRating", "numVotes"])

# Apply filters
df_clean = df_clean[
    (df_clean["numVotes"] >= 10) &
    (df_clean["startYear"] >= 1900) &
    (df_clean["startYear"] <= 2024) &
    (df_clean["averageRating"] >= 1.0) &
    (df_clean["averageRating"] <= 10.0)
]
print(f"  clean:  {len(df_clean):,} movies after filtering")

# Feature engineering
df_clean["decade"] = (df_clean["startYear"] // 10 * 10).astype(int)

df_clean["rating_tier"] = pd.cut(
    df_clean["averageRating"],
    bins=[0, 5, 7, 8, 10],
    labels=["poor", "average", "good", "excellent"]
)

df_clean["runtime_bucket"] = pd.cut(
    df_clean["runtimeMinutes"],
    bins=[0, 90, 120, 9999],
    labels=["short", "standard", "long"]
)

df_clean["is_high_rated"] = (df_clean["averageRating"] >= 7.0).astype(int)

df_clean["vote_tier"] = pd.cut(
    df_clean["numVotes"],
    bins=[0, 1000, 10000, 100000, float("inf")],
    labels=["obscure", "known", "popular", "blockbuster"]
)

# ── ANALYZE ───────────────────────────────────────────────────────────────────
print("\n[5/5] Quick analysis...")
print(f"\n  Rating distribution:")
print(df_clean["rating_tier"].value_counts().to_string())

print(f"\n  Top 5 movies by votes:")
top5 = df_clean.nlargest(5, "numVotes")[["primaryTitle", "startYear", "averageRating", "numVotes"]]
print(top5.to_string(index=False))

print(f"\n  Movies per decade (last 5 decades):")
decade_counts = df_clean[df_clean["decade"] >= 1980]["decade"].value_counts().sort_index()
print(decade_counts.to_string())

print("\n" + "=" * 60)
print(f"✅ Local ETL test passed — {len(df_clean):,} movies ready")
print("=" * 60)

# Save a small sample for testing
sample = df_clean.sample(n=min(1000, len(df_clean)), random_state=42)
sample.to_csv("data/sample/movies_sample.csv", index=False)
print(f"\n💾 Saved 1,000 row sample → data/sample/movies_sample.csv")
