"""
Director Feature Engineering — Glue Job
-----------------------------------------
Processes title.principals (500MB) in the cloud.
Computes director reputation stats and joins to movie features.
Writes small director_features.parquet back to S3.
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
    'RAW_BUCKET',
    'PROCESSED_BUCKET',
    'FEATURES_BUCKET'
])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

RAW_BUCKET       = args['RAW_BUCKET']
PROCESSED_BUCKET = args['PROCESSED_BUCKET']
FEATURES_BUCKET  = args['FEATURES_BUCKET']

# ── STEP 1: Load title.principals ─────────────────────────────────────────────
logger.info("Loading title.principals...")
df_principals = spark.read \
    .option("sep", "\t") \
    .option("header", "true") \
    .option("nullValue", "\\N") \
    .csv(f"s3://{RAW_BUCKET}/imdb/title.principals.tsv.gz")

# Filter to directors only
df_directors = df_principals \
    .filter(F.col("category") == "director") \
    .select("tconst", F.col("nconst").alias("director_id"))

logger.info(f"Director entries: {df_directors.count():,}")

# ── STEP 2: Load ratings ───────────────────────────────────────────────────────
logger.info("Loading ratings...")
df_ratings = spark.read \
    .option("sep", "\t") \
    .option("header", "true") \
    .option("nullValue", "\\N") \
    .csv(f"s3://{RAW_BUCKET}/imdb/title.ratings.tsv.gz") \
    .withColumn("averageRating", F.col("averageRating").cast(FloatType())) \
    .withColumn("numVotes", F.col("numVotes").cast(IntegerType()))

# ── STEP 3: Compute director stats ────────────────────────────────────────────
logger.info("Computing director reputation stats...")
df_dir_ratings = df_directors.join(df_ratings, on="tconst", how="inner")

director_stats = df_dir_ratings.groupBy("director_id").agg(
    F.count("tconst").alias("director_movie_count"),
    F.round(F.avg("averageRating"), 3).alias("director_avg_rating"),
    F.round(F.avg("numVotes"), 0).alias("director_avg_votes"),
    F.max("averageRating").alias("director_max_rating"),
    F.round(
        F.avg(F.when(F.col("averageRating") >= 7.0, 1).otherwise(0)), 3
    ).alias("director_hit_rate")
) \
.withColumn("director_votes_log",
    F.log(F.col("director_avg_votes") + 1)
) \
.withColumn("has_known_director",
    F.when(F.col("director_movie_count") >= 3, 1).otherwise(0)
)

logger.info(f"Director stats computed for {director_stats.count():,} directors")

# ── STEP 4: Load clean movies + join director stats ───────────────────────────
logger.info("Loading clean movies...")
df_movies = spark.read.parquet(
    f"s3://{PROCESSED_BUCKET}/movies/cleaned/"
)

# Get primary director per movie
df_primary_director = df_directors \
    .dropDuplicates(["tconst"])

df_enriched = df_movies \
    .join(df_primary_director, on="tconst", how="left") \
    .join(director_stats, on="director_id", how="left")

# Fill nulls for unknown directors with median values
df_enriched = df_enriched \
    .fillna({
        "director_avg_rating":  6.0,
        "director_movie_count": 1,
        "director_hit_rate":    0.3,
        "director_votes_log":   5.0,
        "has_known_director":   0,
        "director_max_rating":  6.0
    })

logger.info(f"Enriched dataset: {df_enriched.count():,} movies")

# ── STEP 5: Write output ──────────────────────────────────────────────────────
output_path = f"s3://{FEATURES_BUCKET}/movies/director_features/"

logger.info(f"Writing to {output_path}...")
df_enriched.write \
    .mode("overwrite") \
    .parquet(output_path)

logger.info("Director features job complete!")
job.commit()
