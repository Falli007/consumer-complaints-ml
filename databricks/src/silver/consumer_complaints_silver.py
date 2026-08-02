"""Build the Silver Consumer Complaints Delta table.

This job refines the Bronze Consumer Complaints dataset into a cleaner Silver
table for downstream analytics, reporting, and modelling. The script is
designed for Databricks serverless compute and Databricks Asset Bundles, using
Spark DataFrames end to end without pandas.

Silver processing flow:
1. Validate that the Bronze source table exists and contains rows.
2. Trim all string columns and convert blank strings to null.
3. Parse complaint dates into timestamps and cast complaint IDs to long.
4. Standardise selected categorical values while preserving the useful source
   complaint fields.
5. Remove records with invalid complaint IDs.
6. Deduplicate complaint IDs, keeping the most recently ingested record.
7. Overwrite the Silver Delta table idempotently.
8. Run post-write data-quality checks and raise clear errors if validation fails.
"""

from __future__ import annotations

import logging

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window


CATALOG = "fintech_lakehouse_dev"
BRONZE_SCHEMA = "bronze"
SILVER_SCHEMA = "silver"
BRONZE_TABLE_NAME = "bronze_consumer_complaints"
SILVER_TABLE_NAME = "silver_consumer_complaints"

SOURCE_TABLE = f"{CATALOG}.{BRONZE_SCHEMA}.{BRONZE_TABLE_NAME}"
TARGET_TABLE = f"{CATALOG}.{SILVER_SCHEMA}.{SILVER_TABLE_NAME}"

DATE_PARSE_PATTERNS = [
    "MM/dd/yyyy",
    "M/d/yyyy",
    "MM/dd/yy",
    "M/d/yy",
    "yyyy-MM-dd",
    "yyyy-MM-dd HH:mm:ss",
]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

LOGGER = logging.getLogger(__name__)


def validate_source_table(spark: SparkSession) -> int:
    """Validate that the Bronze source table exists and contains rows."""

    if not spark.catalog.tableExists(SOURCE_TABLE):
        raise RuntimeError(f"Bronze source table does not exist: {SOURCE_TABLE}")

    bronze_row_count = spark.table(SOURCE_TABLE).count()

    if bronze_row_count == 0:
        raise RuntimeError(f"Bronze source table is empty: {SOURCE_TABLE}")

    LOGGER.info(
        "Validated Bronze source table %s with %s rows.",
        SOURCE_TABLE,
        f"{bronze_row_count:,}",
    )
    return bronze_row_count


def get_string_columns(dataframe: DataFrame) -> list[str]:
    """Return the list of string column names from the DataFrame schema."""

    return [
        field.name
        for field in dataframe.schema.fields
        if isinstance(field.dataType, T.StringType)
    ]


def clean_string_columns(dataframe: DataFrame) -> DataFrame:
    """Trim string fields and convert blank or whitespace-only values to null."""

    cleaned_dataframe = dataframe
    string_columns = get_string_columns(dataframe)

    for column_name in string_columns:
        cleaned_dataframe = cleaned_dataframe.withColumn(
            column_name,
            F.when(
                F.trim(F.col(column_name)) == "",
                F.lit(None),
            ).otherwise(F.trim(F.col(column_name))),
        )

    LOGGER.info(
        "Trimmed and normalised blank values across %d string columns.",
        len(string_columns),
    )
    return cleaned_dataframe


def parse_timestamp_column(column_name: str) -> F.Column:
    """Create a resilient timestamp expression for a source date column."""

    escaped_column_name = column_name.replace("`", "``")
    escaped_patterns = [
        pattern.replace("'", "''")
        for pattern in DATE_PARSE_PATTERNS
    ]

    return F.coalesce(
        F.expr(
            f"try_to_timestamp(`{escaped_column_name}`)"
        ),
        *[
            F.expr(
                f"try_to_timestamp(`{escaped_column_name}`, '{pattern}')"
            )
            for pattern in escaped_patterns
        ]
    )


def standardise_flag_column(dataframe: DataFrame, column_name: str) -> DataFrame:
    """Standardise Yes/No style fields into consistent values where present."""

    if column_name not in dataframe.columns:
        return dataframe

    normalised_value = F.lower(F.trim(F.col(column_name)))

    return dataframe.withColumn(
        column_name,
        F.when(F.col(column_name).isNull(), F.lit(None))
        .when(normalised_value.isin("yes", "y", "true", "1"), F.lit("YES"))
        .when(normalised_value.isin("no", "n", "false", "0"), F.lit("NO"))
        .otherwise(F.upper(F.col(column_name))),
    )


def transform_silver_dataframe(bronze_dataframe: DataFrame) -> DataFrame:
    """Apply Silver-layer cleansing and standardisation to the Bronze data."""

    silver_dataframe = clean_string_columns(bronze_dataframe)

    silver_dataframe = (
        silver_dataframe
        .withColumn("date_received", parse_timestamp_column("date_received"))
        .withColumn(
            "date_sent_to_company",
            parse_timestamp_column("date_sent_to_company"),
        )
        .withColumn(
            "complaint_id",
            F.when(
                F.col("complaint_id").rlike(r"^[0-9]+$"),
                F.col("complaint_id").cast(T.LongType()),
            ).otherwise(F.lit(None).cast(T.LongType())),
        )
    )

    if "state" in silver_dataframe.columns:
        silver_dataframe = silver_dataframe.withColumn(
            "state",
            F.when(F.col("state").isNull(), F.lit(None)).otherwise(F.upper(F.col("state"))),
        )

    silver_dataframe = standardise_flag_column(silver_dataframe, "timely_response")
    silver_dataframe = standardise_flag_column(silver_dataframe, "consumer_disputed")

    silver_dataframe = (
        silver_dataframe
        .filter(F.col("complaint_id").isNotNull())
        .withColumn("_silver_processed_at", F.current_timestamp())
        .withColumn("_silver_record_status", F.lit("VALID"))
    )

    LOGGER.info(
        "Applied Silver transformations and removed records with invalid complaint IDs."
    )
    return silver_dataframe


def deduplicate_complaints(dataframe: DataFrame) -> DataFrame:
    """Remove duplicate complaint IDs, keeping the most recent ingested record."""

    deduplication_window = Window.partitionBy("complaint_id").orderBy(
        F.col("_ingested_at").desc(),
        F.col("_ingestion_date").desc(),
    )

    deduplicated_dataframe = (
        dataframe
        .withColumn("_row_number", F.row_number().over(deduplication_window))
        .filter(F.col("_row_number") == 1)
        .drop("_row_number")
    )

    LOGGER.info("Deduplicated complaint records by complaint_id.")
    return deduplicated_dataframe


def ensure_silver_schema_exists(spark: SparkSession) -> None:
    """Create the Silver schema when it is missing."""

    spark.sql(
        f"""
        CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SILVER_SCHEMA}
        COMMENT 'Cleaned and deduplicated consumer complaints data for downstream analytics.'
        """
    )


def write_silver_table(spark: SparkSession, dataframe: DataFrame) -> None:
    """Write the full refreshed Silver table idempotently."""

    ensure_silver_schema_exists(spark)

    LOGGER.info("Writing Silver table to %s", TARGET_TABLE)

    (
        dataframe.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(TARGET_TABLE)
    )

    LOGGER.info("Silver table write completed.")


def count_parseable_values(dataframe: DataFrame, source_column: str) -> int:
    """Count non-null source values that can be parsed into timestamps."""

    return dataframe.filter(
        F.col(source_column).isNotNull()
        & parse_timestamp_column(source_column).isNotNull()
    ).count()


def filter_valid_complaint_ids(dataframe: DataFrame) -> DataFrame:
    """Keep only rows whose complaint_id can be safely treated as a numeric key."""

    return dataframe.filter(
        F.col("complaint_id").isNotNull()
        & F.trim(F.col("complaint_id")).rlike(r"^[0-9]+$")
    )


def validate_silver_table(
    spark: SparkSession,
    bronze_dataframe: DataFrame,
    bronze_row_count: int,
) -> None:
    """Run post-write data-quality validations on the Silver table."""

    silver_dataframe = spark.table(TARGET_TABLE)
    silver_row_count = silver_dataframe.count()

    if silver_row_count == 0:
        raise RuntimeError("Silver validation failed: target row count is zero.")

    null_complaint_ids = silver_dataframe.filter(F.col("complaint_id").isNull()).count()
    if null_complaint_ids > 0:
        raise RuntimeError(
            f"Silver validation failed: {null_complaint_ids} null complaint_id values found."
        )

    duplicate_complaint_ids = (
        silver_dataframe.groupBy("complaint_id")
        .count()
        .filter(F.col("count") > 1)
        .count()
    )
    if duplicate_complaint_ids > 0:
        raise RuntimeError(
            f"Silver validation failed: {duplicate_complaint_ids} duplicate complaint_id values found."
        )

    retained_bronze_dataframe = filter_valid_complaint_ids(bronze_dataframe)
    parseable_date_received_count = count_parseable_values(
        retained_bronze_dataframe,
        "date_received",
    )
    silver_date_received_count = silver_dataframe.filter(
        F.col("date_received").isNotNull()
    ).count()
    if silver_date_received_count < parseable_date_received_count:
        raise RuntimeError(
            "Silver validation failed: valid source date_received values were not fully parsed."
        )

    if silver_row_count > bronze_row_count:
        raise RuntimeError(
            "Silver validation failed: Silver row count exceeds Bronze row count."
        )

    LOGGER.info(
        "Silver validation successful: %s rows with unique non-null complaint IDs.",
        f"{silver_row_count:,}",
    )


def main() -> None:
    """Run the full Silver Consumer Complaints processing workflow."""

    spark = SparkSession.builder.getOrCreate()

    LOGGER.info("Starting Consumer Complaints Silver processing.")

    bronze_row_count = validate_source_table(spark)
    bronze_dataframe = spark.table(SOURCE_TABLE)

    silver_dataframe = transform_silver_dataframe(bronze_dataframe)
    silver_dataframe = deduplicate_complaints(silver_dataframe)

    write_silver_table(spark, silver_dataframe)
    validate_silver_table(spark, bronze_dataframe, bronze_row_count)

    LOGGER.info("Consumer Complaints Silver processing finished successfully: %s", TARGET_TABLE)


if __name__ == "__main__":
    main()
