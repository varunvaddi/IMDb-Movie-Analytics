"""
Test SageMaker Endpoint Predictions
-------------------------------------
Sends real movie features to the live endpoint
and gets predicted ratings back.
"""

import boto3
import json
import io
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

ACCOUNT_ID       = os.getenv("AWS_ACCOUNT_ID")
REGION           = os.getenv("AWS_REGION", "us-east-1")
SAGEMAKER_BUCKET = f"imdb-sagemaker-{ACCOUNT_ID}"

sm_runtime = boto3.client("sagemaker-runtime", region_name=REGION)
s3         = boto3.client("s3", region_name=REGION)

ENDPOINT_NAME = open(".endpoint_name").read().strip()

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

def predict_rating(features_dict):
    """Send features to endpoint, get predicted rating."""
    row = [features_dict.get(col, 0) for col in FEATURE_COLS]
    csv_row = ",".join(str(v) for v in row)

    response = sm_runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="text/csv",
        Body=csv_row
    )
    predicted = float(response["Body"].read().decode())
    return round(predicted, 2)


def build_features(
    genres,           # e.g. "Drama,Crime"
    year,             # e.g. 1994
    runtime,          # e.g. 142
    num_votes,        # e.g. 2800000
    director_avg=6.5, # director's historical avg rating
    director_hits=0.6 # director's hit rate
):
    """Build feature vector from human-readable inputs."""
    f = {col: 0 for col in FEATURE_COLS}

    # Genre features
    for genre in TOP_GENRES:
        col = f"genre_{genre.lower().replace('-', '_')}"
        f[col] = 1 if genre in genres else 0

    # Year features
    f["year_normalized"]  = (year - 1900) / (2024 - 1900)
    f["is_modern"]        = 1 if year >= 2000 else 0
    f["is_classic"]       = 1 if year < 1970 else 0
    f["decade_2000s"]     = 1 if 2000 <= year < 2010 else 0
    f["decade_2010s"]     = 1 if 2010 <= year < 2020 else 0
    f["decade_2020s"]     = 1 if year >= 2020 else 0

    # Runtime features
    f["runtime_normalized"] = min(runtime / 180.0, 1.0)
    f["is_short_film"]      = 1 if runtime < 60 else 0
    f["is_epic"]            = 1 if runtime > 150 else 0

    # Vote features
    votes_log = np.log(num_votes + 1)
    f["votes_log"]         = votes_log
    f["votes_normalized"]  = min(num_votes / 100000.0, 1.0)
    f["is_well_known"]     = 1 if num_votes >= 10000 else 0
    f["is_blockbuster"]    = 1 if num_votes >= 100000 else 0
    f["votes_log_squared"] = votes_log ** 2
    f["votes_log_binned"]  = min(int(votes_log), 9)
    f["popularity_score"]  = (
        (3 if num_votes >= 100000 else 0) +
        (2 if num_votes >= 10000 else 0) +
        (1 if num_votes >= 10000 else 0)
    )

    # Interaction features
    f["is_drama_or_biography"] = 1 if ("Drama" in genres or "Biography" in genres) else 0
    f["is_action_adventure"]   = 1 if ("Action" in genres and "Adventure" in genres) else 0
    f["is_horror_thriller"]    = 1 if ("Horror" in genres or "Thriller" in genres) else 0
    f["long_drama"]            = 1 if (runtime > 150 and "Drama" in genres) else 0
    f["classic_documentary"]   = 1 if (year < 1970 and "Documentary" in genres) else 0
    f["modern_documentary"]    = 1 if (year >= 2000 and "Documentary" in genres) else 0

    # Director features
    f["director_avg_rating"]  = director_avg
    f["director_movie_count"] = 10
    f["director_hit_rate"]    = director_hits
    f["director_votes_log"]   = np.log(500000 + 1)
    f["has_known_director"]   = 1
    f["director_max_rating"]  = min(director_avg + 1.0, 10.0)

    return f


print("=" * 60)
print("IMDb Rating Predictor — Live Endpoint Test")
print(f"Endpoint: {ENDPOINT_NAME}")
print("=" * 60)

# ── Test with famous movies ───────────────────────────────────────────────────
test_movies = [
    {
        "title":       "The Dark Knight (2008)",
        "actual":      9.0,
        "genres":      "Action,Crime,Drama",
        "year":        2008,
        "runtime":     152,
        "num_votes":   2800000,
        "director_avg": 8.4,
        "director_hits": 0.9
    },
    {
        "title":       "The Notebook (2004)",
        "actual":      7.8,
        "genres":      "Drama,Romance",
        "year":        2004,
        "runtime":     123,
        "num_votes":   600000,
        "director_avg": 6.8,
        "director_hits": 0.5
    },
    {
        "title":       "Transformers (2007)",
        "actual":      7.0,
        "genres":      "Action,Adventure,Sci-Fi",
        "year":        2007,
        "runtime":     144,
        "num_votes":   750000,
        "director_avg": 6.5,
        "director_hits": 0.4
    },
    {
        "title":       "Paranormal Activity (2007)",
        "actual":      6.3,
        "genres":      "Horror",
        "year":        2007,
        "runtime":     86,
        "num_votes":   200000,
        "director_avg": 5.8,
        "director_hits": 0.3
    },
    {
        "title":       "Schindler's List (1993)",
        "actual":      9.0,
        "genres":      "Biography,Drama,History",
        "year":        1993,
        "runtime":     195,
        "num_votes":   1400000,
        "director_avg": 8.2,
        "director_hits": 0.85
    },
    {
        "title":       "Hypothetical Indie Drama (2023)",
        "actual":      None,
        "genres":      "Drama",
        "year":        2023,
        "runtime":     105,
        "num_votes":   500,
        "director_avg": 6.0,
        "director_hits": 0.3
    }
]

print(f"\n{'Title':42} {'Actual':>8} {'Predicted':>10} {'Error':>8}")
print(f"{'-' * 72}")

for movie in test_movies:
    features = build_features(
        genres=movie["genres"],
        year=movie["year"],
        runtime=movie["runtime"],
        num_votes=movie["num_votes"],
        director_avg=movie.get("director_avg", 6.5),
        director_hits=movie.get("director_hits", 0.4)
    )
    predicted = predict_rating(features)
    actual    = movie["actual"]

    if actual:
        error = abs(actual - predicted)
        print(f"  {movie['title']:40} {actual:>8.1f} {predicted:>10.2f} {error:>8.2f}")
    else:
        print(f"  {movie['title']:40} {'N/A':>8} {predicted:>10.2f} {'':>8}")

print(f"\n{'=' * 60}")
print(f"✅ Endpoint responding successfully!")
print(f"⚠️  Delete endpoint when done: python scripts/delete_endpoint.py")
