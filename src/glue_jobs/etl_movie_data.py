"""
IMDb Movie ETL Job
------------------
Reads raw IMDb TSV files from S3, cleans and joins them,
then writes clean Parquet files back to S3 processed bucket.

This script runs inside AWS Glue using PySpark.
"""

import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, FloatType
import logging

# ── Logging setup ──────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ── Job parameters ─────────────────────────────────────────────────────────────
# These are passed in when the Glue job runs (we set them in the AWS console)
args = getResolvedOptions(sys.argv, [
    'JOB_NAME',
    'RAW_BUCKET',
    'PROCESSED_BUCKET'
])

# ── Initialize Spark + Glue context ───────────────────────────────────────────
# SparkContext = the entry point to all Spark functionality
# GlueContext = AWS wrapper around Spark with extra Glue features
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

logger.info("Glue job started: %s", args['JOB_NAME'])

# ── EXTRACT: Read raw files from S3 ───────────────────────────────────────────
# IMDb uses \t (tab) as separator and "\\N" to represent NULL values
RAW_BUCKET = args['RAW_BUCKET']

logger.info("Reading title.basics from S3...")
df_basics = spark.read \
    .option("sep", "\t") \
    .option("header", "true") \
    .option("nullValue", "\\N") \
    .csv(f"s3://{RAW_BUCKET}/imdb/title.basics.tsv.gz")

logger.info("Reading title.ratings from S3...")
df_ratings = spark.read \
    .option("sep", "\t") \
    .option("header", "true") \
    .option("nullValue", "\\N") \
    .csv(f"s3://{RAW_BUCKET}/imdb/title.ratings.tsv.gz")

logger.info(f"Basics row count: {df_basics.count():,}")
logger.info(f"Ratings row count: {df_ratings.count():,}")

# ── TRANSFORM Step 1: Filter movies only ──────────────────────────────────────
# IMDb has movies, TV shows, shorts, video games, podcasts etc.
# titleType = 'movie' gives us theatrical releases only
logger.info("Filtering movies only...")
df_movies = df_basics.filter(F.col("titleType") == "movie")
logger.info(f"Movies after filter: {df_movies.count():,}")

# ── TRANSFORM Step 2: Cast column types ───────────────────────────────────────
# Everything comes in as string from CSV — cast to proper types
df_movies = df_movies \
    .withColumn("startYear", F.col("startYear").cast(IntegerType())) \
    .withColumn("runtimeMinutes", F.col("runtimeMinutes").cast(IntegerType())) \
    .withColumn("isAdult", F.col("isAdult").cast(IntegerType()))

df_ratings = df_ratings \
    .withColumn("averageRating", F.col("averageRating").cast(FloatType())) \
    .withColumn("numVotes", F.col("numVotes").cast(IntegerType()))

# ── TRANSFORM Step 3: Join basics + ratings ───────────────────────────────────
# tconst is the unique movie ID (e.g. "tt0111161" = Shawshank Redemption)
# Inner join = only keep movies that HAVE ratings (drops unrated movies)
logger.info("Joining basics and ratings...")
df_joined = df_movies.join(df_ratings, on="tconst", how="inner")
logger.info(f"Joined row count: {df_joined.count():,}")

# ── TRANSFORM Step 4: Clean nulls ─────────────────────────────────────────────
# Drop rows missing critical fields
df_clean = df_joined.dropna(subset=[
    "tconst",
    "primaryTitle",
    "startYear",
    "averageRating",
    "numVotes"
])

# Filter out movies with very few votes (unreliable ratings)
# and unrealistic years
df_clean = df_clean \
    .filter(F.col("numVotes") >= 10) \
    .filter(F.col("startYear") >= 1900) \
    .filter(F.col("startYear") <= 2024) \
    .filter(F.col("averageRating") >= 1.0) \
    .filter(F.col("averageRating") <= 10.0)

logger.info(f"Clean row count after null/filter removal: {df_clean.count():,}")

# ── TRANSFORM Step 5: Feature engineering ─────────────────────────────────────
# Add derived columns useful for analysis and ML later

df_enriched = df_clean \
    .withColumn("decade",
        (F.floor(F.col("startYear") / 10) * 10).cast(IntegerType())
    ) \
    .withColumn("rating_tier",
        F.when(F.col("averageRating") >= 8.0, "excellent")
         .when(F.col("averageRating") >= 7.0, "good")
         .when(F.col("averageRating") >= 5.0, "average")
         .otherwise("poor")
    ) \
    .withColumn("runtime_bucket",
        F.when(F.col("runtimeMinutes") < 90, "short")
         .when(F.col("runtimeMinutes") <= 120, "standard")
         .when(F.col("runtimeMinutes") > 120, "long")
         .otherwise("unknown")
    ) \
    .withColumn("is_high_rated",
        F.when(F.col("averageRating") >= 7.0, 1).otherwise(0)
    ) \
    .withColumn("vote_tier",
        F.when(F.col("numVotes") >= 100000, "blockbuster")
         .when(F.col("numVotes") >= 10000, "popular")
         .when(F.col("numVotes") >= 1000, "known")
         .otherwise("obscure")
    )

# Select and order final columns cleanly
df_final = df_enriched.select(
    "tconst",
    "primaryTitle",
    "originalTitle",
    "startYear",
    "decade",
    "runtimeMinutes",
    "runtime_bucket",
    "genres",
    "isAdult",
    "averageRating",
    "numVotes",
    "rating_tier",
    "is_high_rated",
    "vote_tier"
)

logger.info("Sample output:")
df_final.show(5, truncate=False)
logger.info(f"Final schema: {df_final.dtypes}")

# ── LOAD: Write Parquet to S3 processed bucket ────────────────────────────────
# Parquet is columnar — queries only read columns they need (much faster than CSV)
# partitionBy("decade") = splits files by decade folder, queries filter faster
PROCESSED_BUCKET = args['PROCESSED_BUCKET']
output_path = f"s3://{PROCESSED_BUCKET}/movies/cleaned/"

logger.info(f"Writing Parquet to {output_path}...")
df_final.write \
    .mode("overwrite") \
    .partitionBy("decade") \
    .parquet(output_path)

logger.info("ETL job complete!")
job.commit()
