"""
Prepare training data for SageMaker
-------------------------------------
Reads features from S3, splits 80/20 train/test,
uploads both splits back to S3 for SageMaker to consume.
"""

import boto3
import pandas as pd
import numpy as np
import os
import io
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

load_dotenv()

ACCOUNT_ID  = os.getenv("AWS_ACCOUNT_ID")
REGION      = os.getenv("AWS_REGION", "us-east-1")
FEATURES_BUCKET = f"imdb-pipeline-features-{ACCOUNT_ID}"
SAGEMAKER_BUCKET = f"imdb-sagemaker-{ACCOUNT_ID}"

s3 = boto3.client("s3", region_name=REGION)

# ── Create SageMaker bucket if needed ─────────────────────────────────────────
print(f"Creating SageMaker bucket: {SAGEMAKER_BUCKET}")
try:
    s3.create_bucket(Bucket=SAGEMAKER_BUCKET)
    print("  Created ✅")
except s3.exceptions.BucketAlreadyOwnedByYou:
    print("  Already exists ✅")

# ── Load features from S3 ─────────────────────────────────────────────────────
print("\nLoading features from S3...")
response = boto3.client("s3").list_objects_v2(
    Bucket=FEATURES_BUCKET,
    Prefix="movies/features/"
)

dfs = []
for obj in response["Contents"]:
    if obj["Key"].endswith(".parquet"):
        buf = io.BytesIO()
        s3.download_fileobj(FEATURES_BUCKET, obj["Key"], buf)
        buf.seek(0)
        dfs.append(pd.read_parquet(buf))

df = pd.concat(dfs, ignore_index=True)
print(f"Loaded {len(df):,} rows × {len(df.columns)} columns")

# ── Split train/test ───────────────────────────────────────────────────────────
print("\nSplitting 80/20 train/test...")
df_train, df_test = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)
print(f"  Train: {len(df_train):,} rows")
print(f"  Test:  {len(df_test):,} rows")

# ── Upload to S3 ───────────────────────────────────────────────────────────────
print("\nUploading to S3...")

for split_name, split_df in [("train", df_train), ("test", df_test)]:
    buf = io.BytesIO()
    split_df.to_parquet(buf, index=False)
    buf.seek(0)

    key = f"training-data/{split_name}/data.parquet"
    s3.upload_fileobj(buf, SAGEMAKER_BUCKET, key)
    print(f"  Uploaded {split_name} → s3://{SAGEMAKER_BUCKET}/{key}")

print("\n📊 CAPTURE THIS — copy the output above, we'll use it for your resume bullet")
print(f"\nTrain path: s3://{SAGEMAKER_BUCKET}/training-data/train/")
print(f"Test path:  s3://{SAGEMAKER_BUCKET}/training-data/test/")
print("\n✅ Training data ready for SageMaker!")
