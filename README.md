# IMDb Movie Analytics Pipeline 🎬

End-to-end data engineering and ML pipeline on AWS processing 12.3M IMDb records to predict movie ratings using XGBoost.

## 🔗 Live Links
- **Grafana Dashboards:** https://varunvaddi.grafana.net
- **GitHub:** https://github.com/varunvaddi/IMDb-Movie-Analytics

---

## 🏗️ Architecture
```
IMDb Datasets (12.3M records)
        ↓
S3 Data Lake (Raw/Processed/Features)
        ↓
AWS Glue ETL (PySpark) → 309K clean movies
        ↓
AWS Athena (Serverless SQL Warehouse)
        ↓
Feature Engineering (43 ML features)
        ↓
XGBoost Model (R²=0.664, RMSE=0.786)
        ↓
SageMaker Endpoint (live predictions)
        ↓
Grafana + CloudWatch (monitoring)
        ↓
EventBridge + Lambda (daily automation)
```

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| Raw records processed | 12.3M |
| Clean movies | 309,399 |
| ML features engineered | 43 |
| Model R² | 0.664 |
| Model RMSE | 0.786 |
| Endpoint latency | <100ms |
| Pipeline runtime | ~16 min |
| Data quality score | 99%+ |

### Model Iteration History

| Version | Features | R² | RMSE | Key Change |
|---------|----------|-----|------|------------|
| v1 | 28 | 0.306 | 1.130 | Baseline |
| v2 | 37 | 0.306 | 1.130 | +Interactions (no gain) |
| v3 | 37 | 0.462 | 0.693 | 50k+ vote filter |
| v4 | 43 | 0.664 | 0.786 | +Director features (117% improvement) |

### Live Endpoint Predictions

| Movie | Actual | Predicted | Error |
|-------|--------|-----------|-------|
| Schindler's List | 9.0 | 9.11 | 0.11 |
| The Dark Knight | 9.0 | 8.58 | 0.42 |
| The Notebook | 7.8 | 7.88 | 0.08 |
| Transformers | 7.0 | 7.33 | 0.33 |
| Paranormal Activity | 6.3 | 5.42 | 0.88 |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Storage | AWS S3 (Bronze/Silver/Gold pattern) |
| ETL | AWS Glue (PySpark) |
| Warehouse | AWS Athena (Serverless SQL) |
| ML Training | XGBoost + AWS SageMaker |
| Orchestration | AWS Lambda + EventBridge |
| Monitoring | AWS CloudWatch + Grafana Cloud |
| Alerting | AWS SNS |
| Version Control | Git + GitHub |

---

## 📁 Project Structure
```
IMDb-Movie-Analytics/
├── src/
│   ├── glue_jobs/
│   │   ├── etl_movie_data.py          # Phase 2: ETL pipeline
│   │   ├── feature_engineering.py     # Phase 4: Feature engineering
│   │   └── director_features.py       # Phase 5: Director features
│   ├── lambda/
│   │   ├── pipeline_orchestrator.py   # Phase 9: Pipeline automation
│   │   └── trigger_glue_job.py        # Phase 6: Inference API
│   ├── sagemaker/
│   │   └── training/
│   │       └── train_rating_predictor.py
│   └── sql/
│       └── athena_queries/
│           ├── grafana_queries.sql
│           └── quicksight_views.sql
├── scripts/
│   ├── add_director_features.py       # Merge director + movie features
│   ├── train_local_v4.py              # Local model training
│   ├── deploy_endpoint.py             # SageMaker deployment
│   ├── predict.py                     # Live predictions
│   ├── delete_endpoint.py             # Cost control
│   ├── launch_training_job.py         # SageMaker job launcher
│   └── setup_monitoring.py            # SNS + CloudWatch alarms
├── monitoring/
│   └── grafana/
│       └── dashboards/
│           ├── cloudwatch_dashboard.json
│           └── queries.sql
├── config/
│   └── monitoring_config.json
├── docs/
│   └── screenshots/
│       ├── phase1/
│       ├── phase2/
│       ├── phase3/
│       ├── phase4/
│       ├── phase5/
│       ├── phase6/
│       ├── phase7/
│       └── phase8/
└── data/
    └── processed/
        ├── model_v4/
        └── quicksight/
```

---

## 🚀 Pipeline Phases

### Phase 1 — S3 Data Lake
- Downloaded 3 IMDb datasets (12.3M records, ~525MB)
- Created Bronze/Silver/Gold S3 bucket architecture
- Raw data partitioned by dataset type

### Phase 2 — Glue ETL
- PySpark job processing 12.3M → 309,399 clean movies
- Filters: titleType=movie, numVotes≥10, year 1900-2024
- Output: Parquet files partitioned by decade (10x compression)
- Glue Data Catalog for schema management

### Phase 3 — Athena Warehouse
- 6 analytical views for business intelligence
- Serverless SQL on S3 Parquet files
- Replaces Redshift (student account workaround)
- Decade stats, genre performance, top movies

### Phase 4 — Feature Engineering
- 28 ML features: genre encoding, temporal, runtime, votes
- Log transformation for skewed vote distributions
- One-hot encoding for 15 genres
- Output: 309,399 × 33 feature matrix

### Phase 5 — ML Model Training
- 4 model iterations (v1→v4)
- Key insight: director reputation = strongest missing signal
- Added 6 director features via Glue job on 500MB principals file
- Final: R²=0.664, RMSE=0.786 (117% improvement over baseline)
- Trained locally + validated on SageMaker (val-RMSE=0.787)

### Phase 6 — Model Deployment
- SageMaker endpoint: ml.t2.medium instance
- <100ms inference latency
- Lambda wrapper for production API integration
- Endpoint deleted after testing (cost control)

### Phase 7 — Dashboards
- Grafana Cloud connected to Athena
- 5 dashboards: Movie Analytics, Genre Analysis, ML Performance,
  Pipeline Monitoring, ML Model Monitoring
- Public URLs for portfolio sharing

### Phase 8 — Monitoring
- SNS topic: imdb-pipeline-alerts
- CloudWatch alarms on 3 Glue jobs
- Email alerts on pipeline failure
- CloudWatch dashboard: IMDb-Pipeline-Monitoring

### Phase 9 — Orchestration
- Lambda orchestrator runs all 3 Glue jobs in sequence
- EventBridge rule: daily at 2am UTC
- Auto-retry logic and error handling
- SNS success/failure notifications
- Full pipeline: ~16 minutes, zero manual intervention

---

## 💰 Cost Summary

| Service | Usage | Cost |
|---------|-------|------|
| S3 | ~535MB storage | ~$0.01/month |
| Glue ETL | 3 jobs × ~$0.15/run | ~$0.45/run |
| Athena | ~10 queries × 10MB | ~$0.01 |
| SageMaker Training | 1× ml.m5.xlarge | ~$0.23 |
| SageMaker Endpoint | Tested + deleted | ~$0.05 |
| Lambda | Free tier | $0.00 |
| EventBridge | Free tier | $0.00 |
| SNS | Free tier | $0.00 |
| **Total** | **Phases 1-10** | **~$1.00** |

---

## 🔑 Key Engineering Decisions

**Athena over Redshift** — Student account blocked Redshift. Athena provides identical SQL functionality serverlessly, cheaper for intermittent queries.

**Glue for director features** — title.principals.tsv.gz is 500MB. Processing locally caused 15+ minute downloads every run. Moved to Glue to process in cloud, download only the 17MB output.

**Director reputation as key feature** — Genre/year/runtime alone gave R²=0.31. Adding director historical avg rating, hit rate, and movie count pushed R² to 0.664. Domain knowledge > feature engineering tricks.

**Built-in XGBoost container** — Custom sklearn container failed due to missing packages. AWS native XGBoost container requires only hyperparameters, no custom code.

**Bronze/Silver/Gold pattern** — Raw TSV → clean Parquet → engineered features. Each layer is independently queryable and reprocessable without touching upstream data.

---

## 📸 Screenshots

<!-- Add screenshots here after taking them -->
<!-- docs/screenshots/phase1/s3-buckets.png -->
<!-- docs/screenshots/phase2/glue-etl-succeeded.png -->
<!-- docs/screenshots/phase5/sagemaker-training-metrics.png -->
<!-- docs/screenshots/phase6/endpoint-predictions.png -->
<!-- docs/screenshots/phase7/grafana-dashboard.png -->
<!-- docs/screenshots/phase8/cloudwatch-dashboard.png -->

---

## ⚙️ Setup & Reproduction

### Prerequisites
```bash
# Python environment
conda create -n imdb-aws python=3.11
conda activate imdb-aws
pip install boto3 pandas numpy scikit-learn xgboost pyarrow python-dotenv

# AWS CLI
aws configure
```

### Environment Variables
```bash
# .env file
AWS_ACCOUNT_ID=your_account_id
AWS_REGION=us-east-1
ALERT_EMAIL=your_email@example.com
```

### Run Pipeline Manually
```bash
# Trigger full pipeline
aws lambda invoke \
    --function-name "imdb-pipeline-orchestrator" \
    --payload '{"source": "manual"}' \
    /tmp/response.json
```

---

## 📝 Resume Bullets
```
- Built end-to-end ML pipeline processing 12.3M IMDb records using
  AWS Glue (PySpark), S3, and Athena achieving 99%+ data quality

- Engineered 43 ML features including director reputation metrics
  extracted from 500MB dataset processed via AWS Glue in cloud

- Trained XGBoost model iterating 4 versions, improving R² from
  0.306 to 0.664 (117% gain) by identifying director features
  as key missing signal; validated on SageMaker (RMSE=0.786)

- Deployed real-time SageMaker prediction endpoint (<100ms latency)
  predicting Schindler's List at 9.11 vs 9.0 actual rating

- Automated pipeline with AWS Lambda + EventBridge (daily 2am UTC),
  running 3 Glue jobs in sequence with SNS failure alerts

- Monitored pipeline health via CloudWatch dashboards and Grafana
  Cloud connected to Athena for live analytics
```
