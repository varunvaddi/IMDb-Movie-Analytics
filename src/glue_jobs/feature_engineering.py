"""
IMDb Feature Engineering Job
-----------------------------
Reads clean Parquet from S3, engineers ML features,
writes feature matrix back to S3 features bucket.

Runs in AWS Glue (PySpark) but logic is testable locally.
"""

import sys
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, FloatType
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

args = getResolvedOptions(sys.argv, [
    'JOB_NAME',
    'PROCESSED_BUCKET',
    'FEATURES_BUCKET'
])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# ── EXTRACT ───────────────────────────────────────────────────────────────────
logger.info("Reading clean movies from S3...")
df = spark.read.parquet(
    f"s3://{args['PROCESSED_BUCKET']}/movies/cleaned/"
)
logger.info(f"Loaded {df.count():,} movies")

# ── FEATURE 1: Genre one-hot encoding ─────────────────────────────────────────
# "Action,Crime,Drama" → genre_action=1, genre_crime=1, genre_drama=1
# We take top 15 genres — covers 95%+ of all movies
TOP_GENRES = [
    "Drama", "Comedy", "Thriller", "Romance", "Action",
    "Horror", "Crime", "Documentary", "Adventure", "Sci-Fi",
    "Family", "Fantasy", "Mystery", "Biography", "Animation"
]

for genre in TOP_GENRES:
    col_name = f"genre_{genre.lower().replace('-', '_')}"
    df = df.withColumn(
        col_name,
        F.when(F.col("genres").contains(genre), 1).otherwise(0)
    )

# ── FEATURE 2: Year/decade features ───────────────────────────────────────────
# Normalize year to 0-1 range (1900=0, 2024=1)
# Models learn better from normalized inputs
df = df \
    .withColumn("year_normalized",
        (F.col("startyear") - 1900) / (2024 - 1900)
    ) \
    .withColumn("is_modern",
        F.when(F.col("startyear") >= 2000, 1).otherwise(0)
    ) \
    .withColumn("is_classic",
        F.when(F.col("startyear") < 1970, 1).otherwise(0)
    ) \
    .withColumn("decade_2000s",
        F.when((F.col("startyear") >= 2000) & (F.col("startyear") < 2010), 1).otherwise(0)
    ) \
    .withColumn("decade_2010s",
        F.when((F.col("startyear") >= 2010) & (F.col("startyear") < 2020), 1).otherwise(0)
    ) \
    .withColumn("decade_2020s",
        F.when(F.col("startyear") >= 2020, 1).otherwise(0)
    )

# ── FEATURE 3: Runtime features ───────────────────────────────────────────────
# Normalize runtime, handle nulls with median (95 minutes)
MEDIAN_RUNTIME = 95.0

df = df \
    .withColumn("runtime_filled",
        F.when(F.col("runtimeminutes").isNull(), MEDIAN_RUNTIME)
         .otherwise(F.col("runtimeminutes").cast(FloatType()))
    ) \
    .withColumn("runtime_normalized",
        F.least(F.col("runtime_filled") / 180.0, F.lit(1.0))
    ) \
    .withColumn("is_short_film",
        F.when(F.col("runtime_filled") < 60, 1).otherwise(0)
    ) \
    .withColumn("is_epic",
        F.when(F.col("runtime_filled") > 150, 1).otherwise(0)
    )

# ── FEATURE 4: Vote count features ────────────────────────────────────────────
# Log transform compresses 10 → 3,000,000 range to 2.3 → 14.9
# This prevents high-vote movies from dominating the model
df = df \
    .withColumn("votes_log",
        F.log(F.col("numvotes").cast(FloatType()) + 1)
    ) \
    .withColumn("votes_normalized",
        F.least(F.col("numvotes").cast(FloatType()) / 100000.0, F.lit(1.0))
    ) \
    .withColumn("is_well_known",
        F.when(F.col("numvotes") >= 10000, 1).otherwise(0)
    ) \
    .withColumn("is_blockbuster",
        F.when(F.col("numvotes") >= 100000, 1).otherwise(0)
    )

# ── FEATURE 5: Target variable ────────────────────────────────────────────────
# averagerating IS our target — just rename for clarity
df = df.withColumn("target_rating", F.col("averagerating").cast(FloatType()))

# ── SELECT final feature matrix ───────────────────────────────────────────────
FEATURE_COLS = (
    ["tconst", "primarytitle", "startyear", "target_rating", "is_high_rated"] +
    [f"genre_{g.lower().replace('-', '_')}" for g in TOP_GENRES] +
    ["year_normalized", "is_modern", "is_classic",
     "decade_2000s", "decade_2010s", "decade_2020s"] +
    ["runtime_normalized", "is_short_film", "is_epic"] +
    ["votes_log", "votes_normalized", "is_well_known", "is_blockbuster"]
)

df_features = df.select(FEATURE_COLS)

logger.info(f"Feature matrix shape: {df_features.count():,} rows x {len(FEATURE_COLS)} columns")
logger.info(f"Features: {FEATURE_COLS}")

# ── WRITE to S3 features bucket ───────────────────────────────────────────────
output_path = f"s3://{args['FEATURES_BUCKET']}/movies/features/"

logger.info(f"Writing features to {output_path}...")
df_features.write \
    .mode("overwrite") \
    .parquet(output_path)

logger.info("Feature engineering complete!")
job.commit()
