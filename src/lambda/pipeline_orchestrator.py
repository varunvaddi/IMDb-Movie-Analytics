"""
IMDb Pipeline Orchestrator — Lambda Function
---------------------------------------------
Triggered by EventBridge on a schedule.
Runs all Glue jobs in sequence.
Sends SNS notifications on success or failure.

Flow:
EventBridge (daily schedule)
    → Lambda (this file)
        → Glue ETL job
        → Glue Feature Engineering job  
        → Glue Director Features job
        → SNS success/failure notification
"""

import boto3
import json
import os
import time
import logging
from datetime import datetime

logger = logging.getLogger()
logger.setLevel(logging.INFO)

REGION    = os.environ.get("AWS_REGION", "us-east-1")
TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN")

glue = boto3.client("glue", region_name=REGION)
sns  = boto3.client("sns",  region_name=REGION)

# Jobs run in this exact order — each must succeed before next starts
PIPELINE_JOBS = [
    {
        "name":        "imdb-etl-movies",
        "description": "ETL: raw TSV → clean Parquet",
        "timeout":     30
    },
    {
        "name":        "imdb-feature-engineering",
        "description": "Feature engineering: 28 ML features",
        "timeout":     30
    },
    {
        "name":        "imdb-director-features",
        "description": "Director features: joins principals + ratings",
        "timeout":     30
    }
]


def run_glue_job(job_name, timeout_minutes=30):
    """
    Start a Glue job and wait for it to complete.
    Returns (success: bool, duration: int, error: str)
    """
    logger.info(f"Starting Glue job: {job_name}")

    # Start the job
    response  = glue.start_job_run(JobName=job_name)
    run_id    = response["JobRunId"]
    start     = time.time()
    timeout   = timeout_minutes * 60

    logger.info(f"Job run ID: {run_id}")

    # Poll until complete
    while True:
        elapsed = time.time() - start

        if elapsed > timeout:
            glue.batch_stop_job_run(
                JobName=job_name,
                JobRunIds=[run_id]
            )
            return False, int(elapsed), f"Timed out after {timeout_minutes} minutes"

        run_info = glue.get_job_run(
            JobName=job_name,
            RunId=run_id
        )["JobRun"]

        status = run_info["JobRunState"]
        logger.info(f"  {job_name}: {status} ({int(elapsed)}s)")

        if status == "SUCCEEDED":
            return True, int(elapsed), None
        elif status in ["FAILED", "ERROR", "STOPPED"]:
            error = run_info.get("ErrorMessage", "Unknown error")
            return False, int(elapsed), error

        time.sleep(30)


def send_notification(subject, message):
    """Send SNS notification."""
    if TOPIC_ARN:
        sns.publish(
            TopicArn=TOPIC_ARN,
            Subject=subject,
            Message=message
        )
        logger.info(f"Notification sent: {subject}")


def lambda_handler(event, context):
    """
    Main Lambda handler.
    Called by EventBridge on schedule or manually.
    """
    run_date  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Pipeline started: {run_date}")

    results   = []
    pipeline_start = time.time()

    # ── Run each job in sequence ──────────────────────────────────────────────
    for job_config in PIPELINE_JOBS:
        job_name = job_config["name"]
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {job_name}")
        logger.info(f"Description: {job_config['description']}")

        success, duration, error = run_glue_job(
            job_name,
            job_config["timeout"]
        )

        results.append({
            "job":      job_name,
            "success":  success,
            "duration": duration,
            "error":    error
        })

        if not success:
            # Pipeline failed — notify and stop
            logger.error(f"Job failed: {job_name} — {error}")

            message = f"""
IMDb Pipeline FAILED ❌
Date: {run_date}
Failed Job: {job_name}
Error: {error}
Duration before failure: {duration}s

Previous jobs:
{json.dumps(results[:-1], indent=2)}

Action required: Check AWS Glue console for details.
https://console.aws.amazon.com/glue/home#/etl/jobs
"""
            send_notification(
                f"IMDb Pipeline FAILED — {job_name}",
                message
            )

            return {
                "statusCode": 500,
                "body": json.dumps({
                    "status":  "FAILED",
                    "job":     job_name,
                    "error":   error,
                    "results": results
                })
            }

        logger.info(f"✅ {job_name} completed in {duration}s")

    # ── All jobs succeeded ────────────────────────────────────────────────────
    total_duration = int(time.time() - pipeline_start)
    logger.info(f"\nAll jobs completed in {total_duration}s")

    message = f"""
IMDb Pipeline SUCCEEDED ✅
Date: {run_date}
Total Duration: {total_duration}s

Job Results:
{chr(10).join([f"  ✅ {r['job']}: {r['duration']}s" for r in results])}

Data processed:
- 12.3M raw IMDb records
- 309,399 clean movies
- 43 ML features engineered

Dashboard: https://varunvaddi.grafana.net
"""
    send_notification(
        "IMDb Pipeline Completed Successfully ✅",
        message
    )

    return {
        "statusCode": 200,
        "body": json.dumps({
            "status":         "SUCCEEDED",
            "total_duration": total_duration,
            "results":        results
        })
    }
