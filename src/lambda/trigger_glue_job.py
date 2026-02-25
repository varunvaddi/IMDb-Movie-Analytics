"""
Lambda Function — IMDb Rating Predictor API
--------------------------------------------
Wraps the SageMaker endpoint as a simple API.
EventBridge or API Gateway calls this Lambda,
Lambda calls SageMaker, returns prediction.

Why Lambda in front of SageMaker?
- Adds input validation
- Handles feature engineering before calling endpoint
- Can be called from any AWS service
- Adds logging and error handling
- Abstracts endpoint name from callers
"""

import json
import boto3
import numpy as np
import os

ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "imdb-rating-predictor")
REGION        = os.environ.get("AWS_REGION", "us-east-1")

sm_runtime = boto3.client("sagemaker-runtime", region_name=REGION)

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


def build_features(event):
    """Build feature vector from API input."""
    genres   = event.get("genres", "")
    year     = int(event.get("year", 2000))
    runtime  = int(event.get("runtime", 95))
    votes    = int(event.get("num_votes", 1000))
    dir_avg  = float(event.get("director_avg_rating", 6.0))
    dir_hits = float(event.get("director_hit_rate", 0.3))

    f = {col: 0 for col in FEATURE_COLS}

    for genre in TOP_GENRES:
        col = f"genre_{genre.lower().replace('-', '_')}"
        f[col] = 1 if genre in genres else 0

    votes_log = np.log(votes + 1)
    f.update({
        "year_normalized":      (year - 1900) / (2024 - 1900),
        "is_modern":            1 if year >= 2000 else 0,
        "is_classic":           1 if year < 1970 else 0,
        "decade_2000s":         1 if 2000 <= year < 2010 else 0,
        "decade_2010s":         1 if 2010 <= year < 2020 else 0,
        "decade_2020s":         1 if year >= 2020 else 0,
        "runtime_normalized":   min(runtime / 180.0, 1.0),
        "is_short_film":        1 if runtime < 60 else 0,
        "is_epic":              1 if runtime > 150 else 0,
        "votes_log":            votes_log,
        "votes_normalized":     min(votes / 100000.0, 1.0),
        "is_well_known":        1 if votes >= 10000 else 0,
        "is_blockbuster":       1 if votes >= 100000 else 0,
        "votes_log_squared":    votes_log ** 2,
        "votes_log_binned":     min(int(votes_log), 9),
        "popularity_score":     (3 if votes >= 100000 else 0) + (2 if votes >= 10000 else 0),
        "is_drama_or_biography": 1 if ("Drama" in genres or "Biography" in genres) else 0,
        "is_action_adventure":  1 if ("Action" in genres and "Adventure" in genres) else 0,
        "is_horror_thriller":   1 if ("Horror" in genres or "Thriller" in genres) else 0,
        "long_drama":           1 if (runtime > 150 and "Drama" in genres) else 0,
        "classic_documentary":  1 if (year < 1970 and "Documentary" in genres) else 0,
        "modern_documentary":   1 if (year >= 2000 and "Documentary" in genres) else 0,
        "director_avg_rating":  dir_avg,
        "director_movie_count": 10,
        "director_hit_rate":    dir_hits,
        "director_votes_log":   np.log(500000 + 1),
        "has_known_director":   1,
        "director_max_rating":  min(dir_avg + 1.0, 10.0)
    })
    return f


def lambda_handler(event, context):
    """
    Expected input:
    {
        "genres": "Action,Crime,Drama",
        "year": 2008,
        "runtime": 152,
        "num_votes": 2800000,
        "director_avg_rating": 8.4,
        "director_hit_rate": 0.9
    }
    """
    try:
        # Validate input
        if "genres" not in event:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "genres field required"})
            }

        # Build features
        features = build_features(event)
        csv_row  = ",".join(str(features[col]) for col in FEATURE_COLS)

        # Call SageMaker endpoint
        response = sm_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="text/csv",
            Body=csv_row
        )
        predicted_rating = float(response["Body"].read().decode())
        predicted_rating = round(max(1.0, min(10.0, predicted_rating)), 2)

        return {
            "statusCode": 200,
            "body": json.dumps({
                "predicted_rating": predicted_rating,
                "is_high_rated":    predicted_rating >= 7.0,
                "confidence_tier":  (
                    "excellent" if predicted_rating >= 8.0 else
                    "good"      if predicted_rating >= 7.0 else
                    "average"   if predicted_rating >= 5.0 else
                    "poor"
                ),
                "input": event
            })
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
