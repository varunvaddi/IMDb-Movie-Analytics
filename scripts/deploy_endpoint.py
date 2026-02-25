"""
Deploy SageMaker Endpoint
--------------------------
Takes the trained model artifact from S3 and deploys
it as a live endpoint that accepts movie features
and returns predicted ratings.
"""

import boto3
import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

ACCOUNT_ID       = os.getenv("AWS_ACCOUNT_ID")
REGION           = os.getenv("AWS_REGION", "us-east-1")
SAGEMAKER_BUCKET = f"imdb-sagemaker-{ACCOUNT_ID}"
ROLE_ARN         = f"arn:aws:iam::{ACCOUNT_ID}:role/IMDbSageMakerRole"

sm = boto3.client("sagemaker", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)

timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
MODEL_NAME    = f"imdb-xgb-model-{timestamp}"
CONFIG_NAME   = f"imdb-xgb-config-{timestamp}"
ENDPOINT_NAME = f"imdb-rating-predictor"

# ── STEP 1: Find model artifact ───────────────────────────────────────────────
print("Finding model artifact in S3...")
response = s3.list_objects_v2(
    Bucket=SAGEMAKER_BUCKET,
    Prefix="model-artifacts/sagemaker-v4/"
)

model_artifact = None
for obj in response.get("Contents", []):
    if obj["Key"].endswith("model.tar.gz"):
        model_artifact = f"s3://{SAGEMAKER_BUCKET}/{obj['Key']}"
        break

if not model_artifact:
    raise ValueError("model.tar.gz not found! Check your S3 bucket.")

print(f"  Found: {model_artifact}")

# ── STEP 2: Register model with SageMaker ────────────────────────────────────
# This tells SageMaker: here's the artifact + which container to use for inference
print(f"\nRegistering model: {MODEL_NAME}")

xgb_image = f"683313688378.dkr.ecr.{REGION}.amazonaws.com/sagemaker-xgboost:1.7-1"

sm.create_model(
    ModelName=MODEL_NAME,
    PrimaryContainer={
        "Image":           xgb_image,
        "ModelDataUrl":    model_artifact,
        "Environment": {
            "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
            "SAGEMAKER_REGION": REGION
        }
    },
    ExecutionRoleArn=ROLE_ARN
)
print(f"  Model registered ✅")

# ── STEP 3: Create endpoint config ────────────────────────────────────────────
# Defines what instance type the endpoint runs on
# ml.t2.medium = smallest/cheapest inference instance (~$0.05/hr)
print(f"\nCreating endpoint config: {CONFIG_NAME}")

sm.create_endpoint_config(
    EndpointConfigName=CONFIG_NAME,
    ProductionVariants=[
        {
            "VariantName":           "primary",
            "ModelName":             MODEL_NAME,
            "InitialInstanceCount":  1,
            "InstanceType":          "ml.t2.medium",
            "InitialVariantWeight":  1.0
        }
    ]
)
print(f"  Config created ✅")

# ── STEP 4: Deploy endpoint ───────────────────────────────────────────────────
print(f"\nDeploying endpoint: {ENDPOINT_NAME}")
print("  This takes 3-5 minutes...")

sm.create_endpoint(
    EndpointName=ENDPOINT_NAME,
    EndpointConfigName=CONFIG_NAME
)

# Poll until InService
while True:
    status = sm.describe_endpoint(
        EndpointName=ENDPOINT_NAME
    )["EndpointStatus"]
    print(f"  Status: {status}")
    if status == "InService":
        break
    elif status == "Failed":
        reason = sm.describe_endpoint(
            EndpointName=ENDPOINT_NAME
        ).get("FailureReason")
        raise RuntimeError(f"Endpoint failed: {reason}")
    time.sleep(30)

print(f"\n✅ Endpoint is live: {ENDPOINT_NAME}")

# Save endpoint name
with open(".endpoint_name", "w") as f:
    f.write(ENDPOINT_NAME)

print(f"\nEndpoint name saved → .endpoint_name")
print(f"⚠️  Remember to delete endpoint when done to avoid charges!")
print(f"   Run: python scripts/delete_endpoint.py")
