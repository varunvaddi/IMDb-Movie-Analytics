"""
Launch SageMaker Training Job — using native XGBoost container
"""

import boto3
import io
import os
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

ACCOUNT_ID       = os.getenv("AWS_ACCOUNT_ID")
REGION           = os.getenv("AWS_REGION", "us-east-1")
SAGEMAKER_BUCKET = f"imdb-sagemaker-{ACCOUNT_ID}"
ROLE_ARN         = f"arn:aws:iam::{ACCOUNT_ID}:role/IMDbSageMakerRole"

sm = boto3.client("sagemaker", region_name=REGION)
s3 = boto3.client("s3",        region_name=REGION)

job_name = f"imdb-xgb-v4-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
print(f"Launching: {job_name}")

# ── Convert parquet → CSV (XGBoost built-in container requires CSV) ───────────
print("Converting parquet → CSV for SageMaker XGBoost container...")

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

for split in ["train", "test"]:
    buf = io.BytesIO()
    s3.download_fileobj(
        SAGEMAKER_BUCKET,
        f"training-data-v4/{split}/data.parquet",
        buf
    )
    buf.seek(0)
    df = pd.read_parquet(buf)

    # XGBoost built-in expects: label first, then features, no header
    df_csv = df[["target_rating"] + FEATURE_COLS].fillna(0)

    csv_buf = io.StringIO()
    df_csv.to_csv(csv_buf, index=False, header=False)
    csv_buf.seek(0)

    s3.put_object(
        Bucket=SAGEMAKER_BUCKET,
        Key=f"training-data-v4-csv/{split}/data.csv",
        Body=csv_buf.getvalue().encode()
    )
    print(f"  Uploaded {split} CSV → {df_csv.shape}")

# ── XGBoost built-in container ────────────────────────────────────────────────
xgb_image = f"683313688378.dkr.ecr.{REGION}.amazonaws.com/sagemaker-xgboost:1.7-1"

response = sm.create_training_job(
    TrainingJobName=job_name,
    AlgorithmSpecification={
        "TrainingImage":     xgb_image,
        "TrainingInputMode": "File"
    },
    HyperParameters={
        "objective":        "reg:squarederror",
        "num_round":        "500",
        "max_depth":        "5",
        "eta":              "0.05",
        "subsample":        "0.8",
        "colsample_bytree": "0.7",
        "min_child_weight": "5",
        "gamma":            "0.1",
        "alpha":            "0.1",
        "lambda":           "1.0",
        "eval_metric":      "rmse"
    },
    InputDataConfig=[
        {
            "ChannelName": "train",
            "DataSource": {"S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": f"s3://{SAGEMAKER_BUCKET}/training-data-v4-csv/train/",
                "S3DataDistributionType": "FullyReplicated"
            }},
            "ContentType": "text/csv"
        },
        {
            "ChannelName": "validation",
            "DataSource": {"S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": f"s3://{SAGEMAKER_BUCKET}/training-data-v4-csv/test/",
                "S3DataDistributionType": "FullyReplicated"
            }},
            "ContentType": "text/csv"
        }
    ],
    OutputDataConfig={
        "S3OutputPath": f"s3://{SAGEMAKER_BUCKET}/model-artifacts/sagemaker-v4/"
    },
    ResourceConfig={
        "InstanceType":   "ml.m5.xlarge",
        "InstanceCount":  1,
        "VolumeSizeInGB": 10
    },
    RoleArn=ROLE_ARN,
    StoppingCondition={"MaxRuntimeInSeconds": 3600}
)

print(f"\nJob submitted ✅")
print(f"Job name: {job_name}")
print(f"\nCheck status:")
print(f'aws sagemaker describe-training-job --training-job-name "{job_name}" --query "TrainingJobStatus" --output text')

with open(".sagemaker_job_name", "w") as f:
    f.write(job_name)
