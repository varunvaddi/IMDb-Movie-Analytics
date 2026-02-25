"""
Delete SageMaker Endpoint
--------------------------
Always run this when done testing.
A running endpoint costs ~$0.05/hr = $1.20/day.
"""

import boto3
import os
from dotenv import load_dotenv

load_dotenv()

REGION = os.getenv("AWS_REGION", "us-east-1")
sm     = boto3.client("sagemaker", region_name=REGION)

ENDPOINT_NAME = open(".endpoint_name").read().strip()

print(f"Deleting endpoint: {ENDPOINT_NAME}")
sm.delete_endpoint(EndpointName=ENDPOINT_NAME)
print(f"✅ Endpoint deleted — no more charges!")
print(f"   Model artifact still safe in S3.")
