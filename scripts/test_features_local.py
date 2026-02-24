"""
Local feature engineering test
--------------------------------
Tests all feature logic locally with Pandas before
deploying to Glue. Fast feedback loop.
"""

import pandas as pd
import numpy as np

print("=" * 60)
print("IMDb Feature Engineering — Local Test")
print("=" * 60)

# Load the clean sample we saved in Phase 2
df = pd.read_csv("data/sample/movies_sample.csv")
print(f"\nLoaded {len(df):,} sample movies")

# ── FEATURE 1: Genre one-hot encoding ─────────────────────────────────────────
print("\n[1/5] Genre encoding...")
TOP_GENRES = [
    "Drama", "Comedy", "Thriller", "Romance", "Action",
    "Horror", "Crime", "Documentary", "Adventure", "Sci-Fi",
    "Family", "Fantasy", "Mystery", "Biography", "Animation"
]

for genre in TOP_GENRES:
    col = f"genre_{genre.lower().replace('-', '_')}"
    df[col] = df["genres"].str.contains(genre, na=False).astype(int)

genre_cols = [f"genre_{g.lower().replace('-','_')}" for g in TOP_GENRES]
print(f"  Genre distribution (sample):")
print(f"  {df[genre_cols].sum().sort_values(ascending=False).head(8).to_string()}")

# ── FEATURE 2: Year features ──────────────────────────────────────────────────
print("\n[2/5] Year features...")
df["year_normalized"] = (df["startYear"] - 1900) / (2024 - 1900)
df["is_modern"]       = (df["startYear"] >= 2000).astype(int)
df["is_classic"]      = (df["startYear"] < 1970).astype(int)
df["decade_2000s"]    = ((df["startYear"] >= 2000) & (df["startYear"] < 2010)).astype(int)
df["decade_2010s"]    = ((df["startYear"] >= 2010) & (df["startYear"] < 2020)).astype(int)
df["decade_2020s"]    = (df["startYear"] >= 2020).astype(int)
print(f"  Modern movies (2000+): {df['is_modern'].sum():,}")
print(f"  Classic movies (<1970): {df['is_classic'].sum():,}")

# ── FEATURE 3: Runtime features ───────────────────────────────────────────────
print("\n[3/5] Runtime features...")
MEDIAN_RUNTIME = 95.0
df["runtime_filled"]     = df["runtimeMinutes"].fillna(MEDIAN_RUNTIME)
df["runtime_normalized"] = (df["runtime_filled"] / 180.0).clip(upper=1.0)
df["is_short_film"]      = (df["runtime_filled"] < 60).astype(int)
df["is_epic"]            = (df["runtime_filled"] > 150).astype(int)
print(f"  Short films (<60min): {df['is_short_film'].sum():,}")
print(f"  Epics (>150min):      {df['is_epic'].sum():,}")
print(f"  Null runtimes filled: {df['runtimeMinutes'].isna().sum():,}")

# ── FEATURE 4: Vote features ───────────────────────────────────────────────────
print("\n[4/5] Vote features...")
df["votes_log"]        = np.log(df["numVotes"] + 1)
df["votes_normalized"] = (df["numVotes"] / 100000.0).clip(upper=1.0)
df["is_well_known"]    = (df["numVotes"] >= 10000).astype(int)
df["is_blockbuster"]   = (df["numVotes"] >= 100000).astype(int)
print(f"  votes_log range: {df['votes_log'].min():.1f} → {df['votes_log'].max():.1f}")
print(f"  Well known (10k+ votes):     {df['is_well_known'].sum():,}")
print(f"  Blockbuster (100k+ votes):   {df['is_blockbuster'].sum():,}")

# ── FEATURE 5: Target variable ────────────────────────────────────────────────
print("\n[5/5] Target variable...")
df["target_rating"] = df["averageRating"].astype(float)
print(f"  target_rating range: {df['target_rating'].min()} → {df['target_rating'].max()}")
print(f"  target_rating mean:  {df['target_rating'].mean():.2f}")

# ── FINAL FEATURE MATRIX ──────────────────────────────────────────────────────
FEATURE_COLS = (
    ["tconst", "primaryTitle", "startYear", "target_rating", "is_high_rated"] +
    genre_cols +
    ["year_normalized", "is_modern", "is_classic",
     "decade_2000s", "decade_2010s", "decade_2020s"] +
    ["runtime_normalized", "is_short_film", "is_epic"] +
    ["votes_log", "votes_normalized", "is_well_known", "is_blockbuster"]
)

df_features = df[FEATURE_COLS]

print(f"\n{'=' * 60}")
print(f"Feature matrix: {len(df_features):,} rows × {len(FEATURE_COLS)} columns")
print(f"ML features (excluding ID cols): {len(FEATURE_COLS) - 3}")
print(f"\nSample row (The Dark Knight equivalent):")
sample = df_features[df_features["primaryTitle"].str.contains("Dark", na=False)].head(1)
if len(sample) > 0:
    for col in sample.columns:
        print(f"  {col:30} = {sample[col].values[0]}")

# Save features sample
df_features.to_parquet("data/processed/features_sample.parquet", index=False)
print(f"\n💾 Saved → data/processed/features_sample.parquet")
print("=" * 60)
print("✅ Feature engineering test passed!")
print("=" * 60)
