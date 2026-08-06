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

import logging  # standard structured logging for job run output

from pyspark.sql import DataFrame, SparkSession  # core Spark types used throughout
from pyspark.sql import functions as F  # Spark column expressions (F.col, F.when, etc.)
from pyspark.sql import types as T  # Spark data types, used for casts and schema checks
from pyspark.sql.window import Window  # used for the deduplication ranking window


CATALOG = "fintech_lakehouse_dev"  # Unity Catalog catalog name shared across all layers
BRONZE_SCHEMA = "bronze"  # schema this job reads from
SILVER_SCHEMA = "silver"  # schema this job writes to
BRONZE_TABLE_NAME = "bronze_consumer_complaints"  # source table name
SILVER_TABLE_NAME = "silver_consumer_complaints"  # target table name

SOURCE_TABLE = f"{CATALOG}.{BRONZE_SCHEMA}.{BRONZE_TABLE_NAME}"  # fully qualified source table
TARGET_TABLE = f"{CATALOG}.{SILVER_SCHEMA}.{SILVER_TABLE_NAME}"  # fully qualified target table

DATE_PARSE_PATTERNS = [
    "MM/dd/yyyy",  # e.g. 07/30/2026
    "M/d/yyyy",  # e.g. 7/30/2026
    "MM/dd/yy",  # e.g. 07/30/26
    "M/d/yy",  # e.g. 7/30/26
    "yyyy-MM-dd",  # e.g. 2026-07-30
    "yyyy-MM-dd HH:mm:ss",  # e.g. 2026-07-30 00:00:00
]


logging.basicConfig(
    level=logging.INFO,  # default verbosity for this job
    format="%(asctime)s %(levelname)s %(message)s",  # timestamped, single-line log format
)

LOGGER = logging.getLogger(__name__)  # module-level logger used by every function below


def validate_source_table(spark: SparkSession) -> int:
    """Validate that the Bronze source table exists and contains rows."""

    if not spark.catalog.tableExists(SOURCE_TABLE):  # fail fast if Bronze hasn't run yet
        raise RuntimeError(f"Bronze source table does not exist: {SOURCE_TABLE}")

    bronze_row_count = spark.table(SOURCE_TABLE).count()  # full-table count, used for later validation too

    if bronze_row_count == 0:  # an existing-but-empty table is also not usable
        raise RuntimeError(f"Bronze source table is empty: {SOURCE_TABLE}")

    LOGGER.info(
        "Validated Bronze source table %s with %s rows.",
        SOURCE_TABLE,
        f"{bronze_row_count:,}",  # comma-formatted for readability in logs
    )
    return bronze_row_count  # reused later to bound the Silver row count


def get_string_columns(dataframe: DataFrame) -> list[str]:
    """Return the list of string column names from the DataFrame schema."""

    return [
        field.name  # column name
        for field in dataframe.schema.fields  # every field in the schema
        if isinstance(field.dataType, T.StringType)  # keep only string-typed columns
    ]


def clean_string_columns(dataframe: DataFrame) -> DataFrame:
    """Trim string fields and convert blank or whitespace-only values to null."""

    cleaned_dataframe = dataframe  # accumulator rebound on each loop iteration below
    string_columns = get_string_columns(dataframe)  # only string columns need trimming

    for column_name in string_columns:
        cleaned_dataframe = cleaned_dataframe.withColumn(
            column_name,
            F.when(
                F.trim(F.col(column_name)) == "",  # whitespace-only after trimming counts as blank
                F.lit(None),  # blank becomes a real null, not an empty string
            ).otherwise(F.trim(F.col(column_name))),  # otherwise keep the trimmed value
        )

    LOGGER.info(
        "Trimmed and normalised blank values across %d string columns.",
        len(string_columns),
    )
    return cleaned_dataframe


def parse_timestamp_column(column_name: str) -> F.Column:
    """Create a resilient timestamp expression for a source date column."""

    escaped_column_name = column_name.replace("`", "``")  # escape backticks for the SQL identifier
    escaped_patterns = [
        pattern.replace("'", "''")  # escape single quotes for the SQL string literal
        for pattern in DATE_PARSE_PATTERNS
    ]

    return F.coalesce(
        F.expr(
            f"try_to_timestamp(`{escaped_column_name}`)"  # first try Spark's default format inference
        ),
        *[
            F.expr(
                f"try_to_timestamp(`{escaped_column_name}`, '{pattern}')"  # then try each known source format in order
            )
            for pattern in escaped_patterns
        ]
    )


def standardise_flag_column(dataframe: DataFrame, column_name: str) -> DataFrame:
    """Standardise Yes/No style fields into consistent values where present."""

    if column_name not in dataframe.columns:  # some source extracts may not include this column
        return dataframe

    normalised_value = F.lower(F.trim(F.col(column_name)))  # case/whitespace-insensitive comparison basis

    return dataframe.withColumn(
        column_name,
        F.when(F.col(column_name).isNull(), F.lit(None))  # preserve nulls as nulls
        .when(normalised_value.isin("yes", "y", "true", "1"), F.lit("YES"))  # collapse all truthy spellings
        .when(normalised_value.isin("no", "n", "false", "0"), F.lit("NO"))  # collapse all falsy spellings
        .otherwise(F.upper(F.col(column_name))),  # anything unrecognised is just upper-cased, not dropped
    )


def transform_silver_dataframe(bronze_dataframe: DataFrame) -> DataFrame:
    """Apply Silver-layer cleansing and standardisation to the Bronze data."""

    silver_dataframe = clean_string_columns(bronze_dataframe)  # trim/blank-to-null pass first

    silver_dataframe = (
        silver_dataframe
        .withColumn("date_received", parse_timestamp_column("date_received"))  # parse into a real timestamp
        .withColumn(
            "date_sent_to_company",
            parse_timestamp_column("date_sent_to_company"),  # same parsing logic, different column
        )
        .withColumn(
            "complaint_id",
            F.when(
                F.col("complaint_id").rlike(r"^[0-9]+$"),  # only digit-only strings are safe to cast
                F.col("complaint_id").cast(T.LongType()),  # cast to a numeric primary key
            ).otherwise(F.lit(None).cast(T.LongType())),  # anything else becomes null, filtered out below
        )
    )

    if "state" in silver_dataframe.columns:  # defensive check, same reasoning as standardise_flag_column
        silver_dataframe = silver_dataframe.withColumn(
            "state",
            F.when(F.col("state").isNull(), F.lit(None)).otherwise(F.upper(F.col("state"))),  # normalise casing only
        )

    silver_dataframe = standardise_flag_column(silver_dataframe, "timely_response")  # normalise Yes/No values
    silver_dataframe = standardise_flag_column(silver_dataframe, "consumer_disputed")  # same treatment, different column

    silver_dataframe = (
        silver_dataframe
        .filter(F.col("complaint_id").isNotNull())  # drop rows that failed the numeric complaint_id cast
        .withColumn("_silver_processed_at", F.current_timestamp())  # record when this row was processed
        .withColumn("_silver_record_status", F.lit("VALID"))  # reserved for future soft-reject handling
    )

    LOGGER.info(
        "Applied Silver transformations and removed records with invalid complaint IDs."
    )
    return silver_dataframe


def deduplicate_complaints(dataframe: DataFrame) -> DataFrame:
    """Remove duplicate complaint IDs, keeping the most recent ingested record."""

    deduplication_window = Window.partitionBy("complaint_id").orderBy(
        F.col("_ingested_at").desc(),  # most recently ingested row wins first
        F.col("_ingestion_date").desc(),  # tie-breaker if ingestion timestamps collide
    )

    deduplicated_dataframe = (
        dataframe
        .withColumn("_row_number", F.row_number().over(deduplication_window))  # rank duplicates within each complaint_id
        .filter(F.col("_row_number") == 1)  # keep only the top-ranked (most recent) row
        .drop("_row_number")  # ranking column was only needed for the filter above
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

    ensure_silver_schema_exists(spark)  # create the schema on first run, no-op afterwards

    LOGGER.info("Writing Silver table to %s", TARGET_TABLE)

    (
        dataframe.write
        .format("delta")  # managed Delta table, matching Bronze and Gold
        .mode("overwrite")  # full-refresh write, not incremental
        .option("overwriteSchema", "true")  # allow schema evolution across runs
        .saveAsTable(TARGET_TABLE)
    )

    LOGGER.info("Silver table write completed.")


def count_parseable_values(dataframe: DataFrame, source_column: str) -> int:
    """Count non-null source values that can be parsed into timestamps."""

    return dataframe.filter(
        F.col(source_column).isNotNull()  # only consider values that existed in the source
        & parse_timestamp_column(source_column).isNotNull()  # and that survive timestamp parsing
    ).count()


def filter_valid_complaint_ids(dataframe: DataFrame) -> DataFrame:
    """Keep only rows whose complaint_id can be safely treated as a numeric key."""

    return dataframe.filter(
        F.col("complaint_id").isNotNull()  # exclude missing IDs
        & F.trim(F.col("complaint_id")).rlike(r"^[0-9]+$")  # exclude non-numeric IDs
    )


def validate_silver_table(
    spark: SparkSession,
    bronze_dataframe: DataFrame,
    bronze_row_count: int,
) -> None:
    """Run post-write data-quality validations on the Silver table."""

    silver_dataframe = spark.table(TARGET_TABLE)  # read back what was actually written
    silver_row_count = silver_dataframe.count()  # used in multiple checks below

    if silver_row_count == 0:  # a successful write with zero rows still indicates a bug upstream
        raise RuntimeError("Silver validation failed: target row count is zero.")

    null_complaint_ids = silver_dataframe.filter(F.col("complaint_id").isNull()).count()  # should be impossible after the transform filter
    if null_complaint_ids > 0:
        raise RuntimeError(
            f"Silver validation failed: {null_complaint_ids} null complaint_id values found."
        )

    duplicate_complaint_ids = (
        silver_dataframe.groupBy("complaint_id")
        .count()
        .filter(F.col("count") > 1)  # any complaint_id appearing more than once means dedup failed
        .count()
    )
    if duplicate_complaint_ids > 0:
        raise RuntimeError(
            f"Silver validation failed: {duplicate_complaint_ids} duplicate complaint_id values found."
        )

    retained_bronze_dataframe = filter_valid_complaint_ids(bronze_dataframe)  # same filter Silver itself applies
    parseable_date_received_count = count_parseable_values(
        retained_bronze_dataframe,
        "date_received",  # how many Bronze rows should have produced a parsed date
    )
    silver_date_received_count = silver_dataframe.filter(
        F.col("date_received").isNotNull()  # how many Silver rows actually have a parsed date
    ).count()
    if silver_date_received_count < parseable_date_received_count:  # Silver should never have fewer valid dates than Bronze could produce
        raise RuntimeError(
            "Silver validation failed: valid source date_received values were not fully parsed."
        )

    if silver_row_count > bronze_row_count:  # Silver only ever removes rows, it never adds them
        raise RuntimeError(
            "Silver validation failed: Silver row count exceeds Bronze row count."
        )

    LOGGER.info(
        "Silver validation successful: %s rows with unique non-null complaint IDs.",
        f"{silver_row_count:,}",
    )


def main() -> None:
    """Run the full Silver Consumer Complaints processing workflow."""

    spark = SparkSession.builder.getOrCreate()  # reuse the active serverless Spark session

    LOGGER.info("Starting Consumer Complaints Silver processing.")

    bronze_row_count = validate_source_table(spark)  # fail fast before doing any transformation work
    bronze_dataframe = spark.table(SOURCE_TABLE)  # load once, reused for both the transform and validation

    silver_dataframe = transform_silver_dataframe(bronze_dataframe)  # cleaning, parsing, standardisation
    silver_dataframe = deduplicate_complaints(silver_dataframe)  # collapse repeat complaint_id rows

    write_silver_table(spark, silver_dataframe)  # overwrite the Delta table
    validate_silver_table(spark, bronze_dataframe, bronze_row_count)  # post-write data-quality gate

    LOGGER.info("Consumer Complaints Silver processing finished successfully: %s", TARGET_TABLE)


if __name__ == "__main__":
    main()  # production entry point, invoked by the Databricks Job task
