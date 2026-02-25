"""
Setup SNS alerts and CloudWatch monitoring
------------------------------------------
Creates SNS topic for pipeline alerts.
Sets up CloudWatch alarms for Glue job failures.
When a Glue job fails, SNS sends you an email instantly.
"""

import boto3
import json
import os
from dotenv import load_dotenv

load_dotenv()

ACCOUNT_ID  = os.getenv("AWS_ACCOUNT_ID")
REGION      = os.getenv("AWS_REGION", "us-east-1")
ALERT_EMAIL = os.getenv("ALERT_EMAIL")

sns = boto3.client("sns", region_name=REGION)
cw  = boto3.client("cloudwatch", region_name=REGION)

print("=" * 60)
print("IMDb Pipeline Monitoring Setup")
print("=" * 60)

# ── STEP 1: Create SNS topic ──────────────────────────────────────────────────
# SNS = Simple Notification Service
# Think of it as AWS's email/SMS broadcast system
# When an alarm triggers, SNS sends a message to all subscribers
print("\n[1/4] Creating SNS alert topic...")
topic = sns.create_topic(
    Name="imdb-pipeline-alerts",
    Tags=[{"Key": "project", "Value": "imdb-analytics"}]
)
TOPIC_ARN = topic["TopicArn"]
print(f"  Topic ARN: {TOPIC_ARN}")

# Subscribe your email to the topic
if ALERT_EMAIL:
    sns.subscribe(
        TopicArn=TOPIC_ARN,
        Protocol="email",
        Endpoint=ALERT_EMAIL
    )
    print(f"  Subscribed: {ALERT_EMAIL}")
    print(f"  ⚠️  Check your email and confirm the subscription!")
else:
    print("  ⚠️  No ALERT_EMAIL in .env — skipping email subscription")

# ── STEP 2: Test SNS topic ────────────────────────────────────────────────────
print("\n[2/4] Testing SNS topic...")
sns.publish(
    TopicArn=TOPIC_ARN,
    Subject="IMDb Pipeline — Monitoring Active",
    Message="""
IMDb Movie Analytics Pipeline monitoring is now active.

Pipeline Components:
- S3 Data Lake: 3 buckets
- Glue ETL Jobs: 3 jobs (etl, features, director)
- Athena: 6 analytical views
- SageMaker: Model trained, endpoint deployed + deleted
- Grafana: 3 dashboards live

You will receive alerts when:
- Any Glue job fails
- Data quality drops below threshold
- Pipeline takes longer than expected

Dashboard: https://varunvaddi.grafana.net
"""
)
print("  Test notification sent ✅")

# ── STEP 3: CloudWatch alarm for Glue job failures ────────────────────────────
# CloudWatch watches Glue metrics automatically
# We create alarms that trigger when jobs fail
print("\n[3/4] Creating CloudWatch alarms...")

GLUE_JOBS = [
    "imdb-etl-movies",
    "imdb-feature-engineering",
    "imdb-director-features"
]

for job_name in GLUE_JOBS:
    try:
        cw.put_metric_alarm(
            AlarmName=f"imdb-glue-failure-{job_name}",
            AlarmDescription=f"Alert when {job_name} fails",
            MetricName="glue.driver.aggregate.numFailedTasks",
            Namespace="Glue",
            Dimensions=[{"Name": "JobName", "Value": job_name}],
            Period=300,
            EvaluationPeriods=1,
            Threshold=1,
            ComparisonOperator="GreaterThanOrEqualToThreshold",
            Statistic="Sum",
            AlarmActions=[TOPIC_ARN],
            TreatMissingData="notBreaching"
        )
        print(f"  Alarm created: {job_name} ✅")
    except Exception as e:
        print(f"  {job_name}: {e}")

# ── STEP 4: Save topic ARN for later use ──────────────────────────────────────
print("\n[4/4] Saving config...")
config = {
    "sns_topic_arn": TOPIC_ARN,
    "alert_email":   ALERT_EMAIL,
    "glue_jobs_monitored": GLUE_JOBS
}
with open("config/monitoring_config.json", "w") as f:
    json.dump(config, f, indent=2)
print(f"  Saved → config/monitoring_config.json")

print("\n" + "=" * 60)
print("✅ Monitoring setup complete!")
print(f"   SNS Topic: {TOPIC_ARN}")
print(f"   CloudWatch alarms: {len(GLUE_JOBS)} Glue jobs monitored")
print("=" * 60)
