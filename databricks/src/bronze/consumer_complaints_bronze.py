"""Create the Bronze Consumer Complaints Delta table.

This script:
1. Reads the CFPB ZIP file from a Unity Catalog external volume.
2. Extracts the CSV to a temporary staging directory in the volume.
3. Loads all source columns as strings.
4. Adds ingestion metadata.
5. Writes an idempotent Bronze Delta table.
6. Removes the temporary extracted CSV.
"""

from __future__ import annotations

import logging  # structured logging for job run output
import re  # used to normalise raw CSV headers into safe column names
import shutil  # used to stream-copy the extracted CSV and to clean up staging
import zipfile  # used to read the source ZIP without extracting it up front
from datetime import datetime, timezone  # imported for potential timestamp use, current logic uses INGESTION_DATE directly
from pathlib import Path  # filesystem paths on the Unity Catalog volume

from pyspark.sql import DataFrame, SparkSession  # core Spark types used throughout
from pyspark.sql import functions as F  # Spark column expressions (F.col, F.lit, etc.)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CATALOG = "fintech_lakehouse_dev"  # Unity Catalog catalog name shared across all layers
SCHEMA = "bronze"  # schema this job writes to
TABLE_NAME = "bronze_consumer_complaints"  # target table name
TARGET_TABLE = f"{CATALOG}.{SCHEMA}.{TABLE_NAME}"  # fully qualified target table

INGESTION_DATE = "2026-07-30"  # partition value for this run; matches the ingestion script's date

VOLUME_ROOT = Path(
    "/Volumes/fintech_lakehouse_dev/"
    "bronze/consumer_complaints_raw_dev"  # Unity Catalog volume holding the raw ZIP and staging area
)

SOURCE_ZIP = Path("/Volumes/fintech_lakehouse_dev/bronze/consumer_complaints_raw_dev/consumer_complaints/complaints.csv.zip")  # exact ZIP uploaded by the ingestion step

STAGING_DIRECTORY = (
    VOLUME_ROOT
    / "_staging"
    / f"ingestion_date={INGESTION_DATE}"  # date-partitioned so concurrent runs never collide
)

EXTRACTED_CSV = STAGING_DIRECTORY / "complaints.csv"  # temporary extracted file, deleted at the end of the run


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,  # default verbosity for this job
    format="%(asctime)s %(levelname)s %(message)s",  # timestamped, single-line log format
)

LOGGER = logging.getLogger(__name__)  # module-level logger used by every function below


def validate_source() -> None:
    """Confirm that the source ZIP exists and contains a CSV file."""

    if not SOURCE_ZIP.exists():  # fail fast if ingestion hasn't uploaded the ZIP yet
        raise FileNotFoundError(
            f"Source ZIP does not exist: {SOURCE_ZIP}"
        )

    if SOURCE_ZIP.stat().st_size == 0:  # catches a truncated/failed upload
        raise ValueError(f"Source ZIP is empty: {SOURCE_ZIP}")

    if not zipfile.is_zipfile(SOURCE_ZIP):  # catches a corrupted or non-ZIP file at that path
        raise ValueError(f"Source file is not a valid ZIP: {SOURCE_ZIP}")

    LOGGER.info(
        "Source ZIP validated: %s (%.2f GB)",
        SOURCE_ZIP,
        SOURCE_ZIP.stat().st_size / (1024**3),  # bytes to GB for a readable log line
    )


def extract_csv() -> Path:
    """Extract the complaints CSV into temporary volume storage."""

    STAGING_DIRECTORY.mkdir(parents=True, exist_ok=True)  # create the date-partitioned staging folder

    if EXTRACTED_CSV.exists():  # clean up a leftover file from a previous failed/partial run
        LOGGER.info(
            "Removing existing staged CSV: %s",
            EXTRACTED_CSV,
        )
        EXTRACTED_CSV.unlink()

    LOGGER.info("Opening ZIP file: %s", SOURCE_ZIP)

    with zipfile.ZipFile(SOURCE_ZIP, mode="r") as archive:
        csv_members = [
            name
            for name in archive.namelist()  # every entry in the ZIP
            if name.lower().endswith(".csv")  # only interested in the CSV payload
        ]

        if not csv_members:  # the ZIP exists but has no CSV inside, still a hard failure
            raise ValueError(
                f"No CSV file was found inside {SOURCE_ZIP}"
            )

        if len(csv_members) > 1:  # defensive: the CFPB export is expected to contain exactly one CSV
            LOGGER.warning(
                "Multiple CSV files found. Using: %s",
                csv_members[0],
            )

        source_member = csv_members[0]  # the (only, or first) CSV entry to extract

        LOGGER.info(
            "Extracting %s to %s",
            source_member,
            EXTRACTED_CSV,
        )

        with archive.open(source_member) as source:
            with EXTRACTED_CSV.open("wb") as destination:
                shutil.copyfileobj(
                    source,
                    destination,
                    length=16 * 1024 * 1024,  # 16 MB chunks, avoids loading the whole CSV into memory at once
                )

    LOGGER.info(
        "CSV extraction complete: %.2f GB",
        EXTRACTED_CSV.stat().st_size / (1024**3),
    )

    return EXTRACTED_CSV  # path handed to read_bronze_dataframe


def read_bronze_dataframe(
    spark: SparkSession,
    csv_path: Path,
) -> DataFrame:
    """Read the source CSV with minimal Bronze-layer transformation."""

    LOGGER.info("Reading CSV with Spark: %s", csv_path)

    source_df = (
        spark.read
        .format("csv")
        .option("header", "true")  # first row is column names
        .option("inferSchema", "false")  # keep everything as strings; typing happens in Silver
        .option("multiLine", "true")  # complaint narratives can contain embedded newlines
        .option("quote", '"')  # standard CSV quoting
        .option("escape", '"')  # doubled-quote escaping within quoted fields
        .option("encoding", "UTF-8")  # source file encoding
        .option("mode", "PERMISSIVE")  # keep malformed rows rather than dropping the whole read
        .load(str(csv_path))
    )

    bronze_df = (
        source_df.select(
            *[
                F.col(column_name).alias(normalise_column_name(column_name))  # rename to a Delta-safe column name
                for column_name in source_df.columns
            ]
        )
        .withColumn(
            "_ingestion_date",
            F.to_date(F.lit(INGESTION_DATE)),  # partition column, used by write_bronze_table for replace-on-rerun
        )
        .withColumn(
            "_ingested_at",
            F.current_timestamp(),  # exact write time, used later by Silver's deduplication ordering
        )
        .withColumn(
            "_source_zip_path",
            F.lit(str(SOURCE_ZIP)),  # provenance: which source file this row came from
        )
        .withColumn(
            "_source_csv_name",
            F.lit(csv_path.name),  # provenance: which CSV member inside the ZIP
        )
    )

    LOGGER.info(
        "Source schema contains %s columns.",
        len(source_df.columns),
    )

    return bronze_df


def normalise_column_name(column_name: str) -> str:
    """Convert source headers into Delta-safe Bronze column names."""

    cleaned_name = column_name.strip().lower()  # case-insensitive, no leading/trailing whitespace
    cleaned_name = re.sub(r"[^a-z0-9]+", "_", cleaned_name)  # replace any run of non-alphanumeric characters with one underscore
    cleaned_name = re.sub(r"_+", "_", cleaned_name)  # collapse any resulting duplicate underscores
    return cleaned_name.strip("_")  # drop a leading/trailing underscore left over from the substitutions


def write_bronze_table(
    spark: SparkSession,
    dataframe: DataFrame,
) -> None:
    """Write one ingestion-date snapshot into the Bronze table."""

    spark.sql(
        f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}"  # no-op after the first run
    )

    if spark.catalog.tableExists(TARGET_TABLE):  # table already exists: replace just this date's partition
        LOGGER.info(
            "Removing any existing records for %s",
            INGESTION_DATE,
        )

        spark.sql(
            f"""
            DELETE FROM {TARGET_TABLE}
            WHERE _ingestion_date = DATE '{INGESTION_DATE}'
            """  # makes a rerun for the same date idempotent instead of duplicating rows
        )

        LOGGER.info("Appending snapshot to %s", TARGET_TABLE)

        (
            dataframe.write
            .format("delta")
            .mode("append")  # only this date's rows were deleted above, so append is safe here
            .saveAsTable(TARGET_TABLE)
        )

    else:  # first run ever: create the table from scratch
        LOGGER.info("Creating Bronze table: %s", TARGET_TABLE)

        (
            dataframe.write
            .format("delta")
            .mode("overwrite")
            .partitionBy("_ingestion_date")  # partition layout established on table creation
            .saveAsTable(TARGET_TABLE)
        )

    LOGGER.info("Bronze table write completed.")


def validate_target(spark: SparkSession) -> None:
    """Validate that records were written for this ingestion date."""

    result = spark.sql(
        f"""
        SELECT COUNT(*) AS row_count
        FROM {TARGET_TABLE}
        WHERE _ingestion_date = DATE '{INGESTION_DATE}'
        """
    ).first()

    row_count = result["row_count"]  # count for just this run's partition, not the whole table

    if row_count == 0:  # a successful write with zero rows still indicates a bug upstream
        raise RuntimeError(
            "Bronze validation failed: zero rows were written."
        )

    LOGGER.info(
        "Bronze validation successful. Rows written: %s",
        f"{row_count:,}",  # comma-formatted for readability in logs
    )


def clean_staging() -> None:
    """Delete the temporary extracted CSV after successful ingestion."""

    if STAGING_DIRECTORY.exists():  # nothing to clean up if extraction never ran
        LOGGER.info(
            "Removing temporary staging directory: %s",
            STAGING_DIRECTORY,
        )
        shutil.rmtree(STAGING_DIRECTORY)


def main() -> None:
    """Run the Bronze ingestion pipeline."""

    spark = SparkSession.builder.getOrCreate()  # reuse the active serverless Spark session

    LOGGER.info("Starting Consumer Complaints Bronze ingestion.")

    validate_source()  # fail fast before touching the volume's staging area
    csv_path = extract_csv()

    try:
        bronze_df = read_bronze_dataframe(
            spark=spark,
            csv_path=csv_path,
        )

        write_bronze_table(
            spark=spark,
            dataframe=bronze_df,
        )

        validate_target(spark)  # post-write check that this run's partition actually has rows

    finally:
        clean_staging()  # always remove the staged CSV, even if the pipeline failed above

    LOGGER.info(
        "Bronze ingestion finished successfully: %s",
        TARGET_TABLE,
    )


if __name__ == "__main__":
    main()  # production entry point, invoked by the Databricks Job task
