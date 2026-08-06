"""Build a reusable Gold ML feature table for consumer complaints.

This script prepares a model-ready feature dataset for predicting whether a
complaint will receive an untimely company response. It reads the curated Gold
warehouse model, keeps one feature row per complaint, derives intake-safe
features, and writes a managed Delta table for downstream training workflows.

Feature-engineering flow:
1. Validate that the Gold fact and supporting dimensions exist and contain rows.
2. Join the fact table to the conformed dimensions for business-friendly labels.
3. Derive model features that are safe to use at complaint intake time.
4. Create deterministic train/validation/test split assignments.
5. Overwrite the target feature table idempotently.
6. Validate row counts, uniqueness, null handling, and split coverage.
"""

from __future__ import annotations

import logging

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


CATALOG = "fintech_lakehouse_dev"
GOLD_SCHEMA = "gold"

FACT_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.fact_consumer_complaints"
DIM_DATE_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.dim_date"
DIM_PRODUCT_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.dim_product"
DIM_COMPANY_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.dim_company"
DIM_STATE_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.dim_state"

FEATURE_TABLE_NAME = "consumer_complaints_timely_response_features"
TARGET_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.{FEATURE_TABLE_NAME}"

MISSING_MEMBER_LABEL = "NOT_PROVIDED"
TRAIN_SPLIT_LABEL = "TRAIN"
VALIDATION_SPLIT_LABEL = "VALIDATION"
TEST_SPLIT_LABEL = "TEST"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

LOGGER = logging.getLogger(__name__)


# ------------------------------------------------------------
# Confirm that the Gold warehouse model exists before building
# ML features on top of it. The feature table should only be
# created from validated warehouse outputs.
# ------------------------------------------------------------
def validate_gold_sources(spark: SparkSession) -> int:
    """Validate the Gold fact and dimensions and return the fact row count."""

    required_tables = [
        FACT_TABLE,
        DIM_DATE_TABLE,
        DIM_PRODUCT_TABLE,
        DIM_COMPANY_TABLE,
        DIM_STATE_TABLE,
    ]

    for table_name in required_tables:
        if not spark.catalog.tableExists(table_name):
            raise RuntimeError(f"Required Gold table does not exist: {table_name}")

    fact_row_count = spark.table(FACT_TABLE).count()
    if fact_row_count == 0:
        raise RuntimeError(f"Gold fact table is empty: {FACT_TABLE}")

    LOGGER.info(
        "Validated Gold warehouse sources. Fact table %s contains %s rows.",
        FACT_TABLE,
        f"{fact_row_count:,}",
    )
    return fact_row_count


# ------------------------------------------------------------
# Read the complaint fact and attach descriptive dimension
# attributes so feature engineering can work with business-
# readable labels instead of surrogate keys alone.
# ------------------------------------------------------------
def read_feature_base(spark: SparkSession) -> DataFrame:
    """Read the Gold fact and join the supporting dimensions."""

    fact_dataframe = spark.table(FACT_TABLE).alias("f")
    dim_date_dataframe = (
        spark.table(DIM_DATE_TABLE)
        .select(
            F.col("date_key").alias("received_date_key"),
            F.col("full_date").alias("received_full_date"),
            F.col("calendar_year").alias("received_year"),
            F.col("calendar_quarter").alias("received_quarter"),
            F.col("calendar_month").alias("received_month"),
            F.col("calendar_month_name").alias("received_month_name"),
            F.col("calendar_day").alias("received_day"),
            F.col("day_of_week").alias("received_day_of_week"),
            F.col("day_name").alias("received_day_name"),
            F.col("is_weekend").alias("received_is_weekend"),
        )
        .alias("dd")
    )
    dim_product_dataframe = spark.table(DIM_PRODUCT_TABLE).alias("dp")
    dim_company_dataframe = spark.table(DIM_COMPANY_TABLE).alias("dc")
    dim_state_dataframe = spark.table(DIM_STATE_TABLE).alias("ds")

    return (
        fact_dataframe
        .join(
            dim_date_dataframe,
            on="received_date_key",
            how="left",
        )
        .join(
            dim_product_dataframe,
            on="product_key",
            how="left",
        )
        .join(
            dim_company_dataframe,
            on="company_key",
            how="left",
        )
        .join(
            dim_state_dataframe,
            on="state_key",
            how="left",
        )
    )


# ------------------------------------------------------------
# Keep only features that are available at or very near intake
# time. This avoids leaking outcome information such as post-
# response process attributes into the predictive dataset.
# ------------------------------------------------------------
def build_feature_dataframe(base_dataframe: DataFrame) -> DataFrame:
    """Create a model-ready feature DataFrame from the Gold warehouse base."""

    feature_dataframe = (
        base_dataframe
        .filter(F.col("timely_response").isin("YES", "NO"))
        .withColumn(
            "label_untimely_response",
            F.when(F.col("timely_response") == "NO", F.lit(1)).otherwise(F.lit(0)),
        )
        .withColumn(
            "complaint_received_date",
            F.col("received_full_date"),
        )
        .withColumn(
            "zip_code_prefix3",
            F.when(
                F.col("zip_code").isNull(),
                F.lit(MISSING_MEMBER_LABEL),
            ).otherwise(F.substring(F.col("zip_code"), 1, 3)),
        )
        .withColumn(
            "product",
            F.coalesce(F.col("product"), F.lit(MISSING_MEMBER_LABEL)),
        )
        .withColumn(
            "sub_product",
            F.coalesce(F.col("sub_product"), F.lit(MISSING_MEMBER_LABEL)),
        )
        .withColumn(
            "company",
            F.coalesce(F.col("company"), F.lit(MISSING_MEMBER_LABEL)),
        )
        .withColumn(
            "state",
            F.coalesce(F.col("state"), F.lit(MISSING_MEMBER_LABEL)),
        )
        .withColumn(
            "issue",
            F.coalesce(F.col("issue"), F.lit(MISSING_MEMBER_LABEL)),
        )
        .withColumn(
            "sub_issue",
            F.coalesce(F.col("sub_issue"), F.lit(MISSING_MEMBER_LABEL)),
        )
        .withColumn(
            "submitted_via",
            F.coalesce(F.col("submitted_via"), F.lit(MISSING_MEMBER_LABEL)),
        )
        .withColumn(
            "received_month_name",
            F.coalesce(F.col("received_month_name"), F.lit(MISSING_MEMBER_LABEL)),
        )
        .withColumn(
            "received_day_name",
            F.coalesce(F.col("received_day_name"), F.lit(MISSING_MEMBER_LABEL)),
        )
        .withColumn(
            "received_is_weekend",
            F.coalesce(F.col("received_is_weekend").cast("int"), F.lit(0)),
        )
        .withColumn(
            "has_consumer_narrative",
            F.col("has_consumer_narrative").cast("int"),
        )
        .withColumn(
            "has_tags",
            F.col("has_tags").cast("int"),
        )
        .withColumn(
            "complaint_received_year",
            F.coalesce(F.col("received_year"), F.lit(0)),
        )
        .withColumn(
            "complaint_received_quarter",
            F.coalesce(F.col("received_quarter"), F.lit(0)),
        )
        .withColumn(
            "complaint_received_month",
            F.coalesce(F.col("received_month"), F.lit(0)),
        )
        .withColumn(
            "complaint_received_day",
            F.coalesce(F.col("received_day"), F.lit(0)),
        )
        .withColumn(
            "complaint_received_day_of_week",
            F.coalesce(F.col("received_day_of_week"), F.lit(0)),
        )
        .withColumn(
            "split_bucket",
            F.pmod(F.xxhash64(F.col("complaint_id")), F.lit(100)),
        )
        .withColumn(
            "dataset_split",
            F.when(F.col("split_bucket") < 70, F.lit(TRAIN_SPLIT_LABEL))
            .when(F.col("split_bucket") < 85, F.lit(VALIDATION_SPLIT_LABEL))
            .otherwise(F.lit(TEST_SPLIT_LABEL)),
        )
        .withColumn("_feature_processed_at", F.current_timestamp())
        .select(
            "complaint_id",
            "label_untimely_response",
            "timely_response",
            "complaint_received_date",
            "complaint_received_year",
            "complaint_received_quarter",
            "complaint_received_month",
            "complaint_received_day",
            "complaint_received_day_of_week",
            "received_month_name",
            "received_day_name",
            "received_is_weekend",
            "product",
            "sub_product",
            "company",
            "state",
            "issue",
            "sub_issue",
            "submitted_via",
            "zip_code_prefix3",
            "has_consumer_narrative",
            "has_tags",
            "dataset_split",
            "split_bucket",
            "_ingestion_date",
            "_ingested_at",
            "_source_zip_path",
            "_source_csv_name",
            "_silver_processed_at",
            "_gold_processed_at",
            "_feature_processed_at",
        )
    )

    LOGGER.info("Built ML feature DataFrame for timely-response prediction.")
    return feature_dataframe


# ------------------------------------------------------------
# Persist the feature dataset as a managed Delta table so it can
# be reused by training, validation, and future experimentation
# without recomputing the feature logic each time.
# ------------------------------------------------------------
def write_feature_table(dataframe: DataFrame) -> None:
    """Write the model feature table idempotently."""

    LOGGER.info("Writing ML feature table to %s", TARGET_TABLE)

    (
        dataframe.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(TARGET_TABLE)
    )

    LOGGER.info("ML feature table write completed.")


# ------------------------------------------------------------
# Validate that the feature dataset preserves one row per
# complaint, keeps a non-null binary label, and covers all
# deterministic split buckets needed for downstream training.
# ------------------------------------------------------------
def validate_feature_table(spark: SparkSession, source_row_count: int) -> None:
    """Run post-write validations on the feature table."""

    feature_dataframe = spark.table(TARGET_TABLE)
    feature_row_count = feature_dataframe.count()

    if feature_row_count == 0:
        raise RuntimeError("Feature validation failed: target row count is zero.")

    if feature_row_count > source_row_count:
        raise RuntimeError(
            "Feature validation failed: feature row count exceeds the Gold fact row count."
        )

    null_complaint_ids = feature_dataframe.filter(F.col("complaint_id").isNull()).count()
    if null_complaint_ids > 0:
        raise RuntimeError(
            f"Feature validation failed: {null_complaint_ids} null complaint IDs found."
        )

    duplicate_complaint_ids = (
        feature_dataframe.groupBy("complaint_id")
        .count()
        .filter(F.col("count") > 1)
        .count()
    )
    if duplicate_complaint_ids > 0:
        raise RuntimeError(
            f"Feature validation failed: {duplicate_complaint_ids} duplicate complaint IDs found."
        )

    invalid_labels = feature_dataframe.filter(
        ~F.col("label_untimely_response").isin(0, 1)
        | F.col("label_untimely_response").isNull()
    ).count()
    if invalid_labels > 0:
        raise RuntimeError(
            f"Feature validation failed: {invalid_labels} invalid label values found."
        )

    split_counts = {
        row["dataset_split"]: row["row_count"]
        for row in feature_dataframe.groupBy("dataset_split").count().withColumnRenamed("count", "row_count").collect()
    }
    required_splits = [
        TRAIN_SPLIT_LABEL,
        VALIDATION_SPLIT_LABEL,
        TEST_SPLIT_LABEL,
    ]
    missing_splits = [
        split_name
        for split_name in required_splits
        if split_counts.get(split_name, 0) == 0
    ]
    if missing_splits:
        raise RuntimeError(
            f"Feature validation failed: missing dataset splits {missing_splits}."
        )

    LOGGER.info(
        "Feature validation successful: %s rows written. Split sizes: train=%s, validation=%s, test=%s.",
        f"{feature_row_count:,}",
        f"{split_counts.get(TRAIN_SPLIT_LABEL, 0):,}",
        f"{split_counts.get(VALIDATION_SPLIT_LABEL, 0):,}",
        f"{split_counts.get(TEST_SPLIT_LABEL, 0):,}",
    )


# ------------------------------------------------------------
# Run the end-to-end ML feature build after the Gold warehouse
# layer has been validated and published.
# ------------------------------------------------------------
def main() -> None:
    """Build and validate the consumer complaints ML feature table."""

    spark = SparkSession.builder.getOrCreate()

    LOGGER.info("Starting Consumer Complaints ML feature engineering.")

    fact_row_count = validate_gold_sources(spark)
    feature_base = read_feature_base(spark)
    feature_dataframe = build_feature_dataframe(feature_base)

    write_feature_table(feature_dataframe)
    validate_feature_table(spark, fact_row_count)

    LOGGER.info("Consumer Complaints ML feature engineering finished successfully.")


if __name__ == "__main__":
    main()
