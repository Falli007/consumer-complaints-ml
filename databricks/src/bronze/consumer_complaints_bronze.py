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

import logging
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CATALOG = "fintech_lakehouse_dev"
SCHEMA = "bronze"
TABLE_NAME = "bronze_consumer_complaints"
TARGET_TABLE = f"{CATALOG}.{SCHEMA}.{TABLE_NAME}"

INGESTION_DATE = "2026-07-30"

VOLUME_ROOT = Path(
    "/Volumes/fintech_lakehouse_dev/"
    "bronze/consumer_complaints_raw_dev"
)

SOURCE_ZIP = Path("/Volumes/fintech_lakehouse_dev/bronze/consumer_complaints_raw_dev/consumer_complaints/complaints.csv.zip")

STAGING_DIRECTORY = (
    VOLUME_ROOT
    / "_staging"
    / f"ingestion_date={INGESTION_DATE}"
)

EXTRACTED_CSV = STAGING_DIRECTORY / "complaints.csv"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

LOGGER = logging.getLogger(__name__)


def validate_source() -> None:
    """Confirm that the source ZIP exists and contains a CSV file."""

    if not SOURCE_ZIP.exists():
        raise FileNotFoundError(
            f"Source ZIP does not exist: {SOURCE_ZIP}"
        )

    if SOURCE_ZIP.stat().st_size == 0:
        raise ValueError(f"Source ZIP is empty: {SOURCE_ZIP}")

    if not zipfile.is_zipfile(SOURCE_ZIP):
        raise ValueError(f"Source file is not a valid ZIP: {SOURCE_ZIP}")

    LOGGER.info(
        "Source ZIP validated: %s (%.2f GB)",
        SOURCE_ZIP,
        SOURCE_ZIP.stat().st_size / (1024**3),
    )


def extract_csv() -> Path:
    """Extract the complaints CSV into temporary volume storage."""

    STAGING_DIRECTORY.mkdir(parents=True, exist_ok=True)

    if EXTRACTED_CSV.exists():
        LOGGER.info(
            "Removing existing staged CSV: %s",
            EXTRACTED_CSV,
        )
        EXTRACTED_CSV.unlink()

    LOGGER.info("Opening ZIP file: %s", SOURCE_ZIP)

    with zipfile.ZipFile(SOURCE_ZIP, mode="r") as archive:
        csv_members = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".csv")
        ]

        if not csv_members:
            raise ValueError(
                f"No CSV file was found inside {SOURCE_ZIP}"
            )

        if len(csv_members) > 1:
            LOGGER.warning(
                "Multiple CSV files found. Using: %s",
                csv_members[0],
            )

        source_member = csv_members[0]

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
                    length=16 * 1024 * 1024,
                )

    LOGGER.info(
        "CSV extraction complete: %.2f GB",
        EXTRACTED_CSV.stat().st_size / (1024**3),
    )

    return EXTRACTED_CSV


def read_bronze_dataframe(
    spark: SparkSession,
    csv_path: Path,
) -> DataFrame:
    """Read the source CSV with minimal Bronze-layer transformation."""

    LOGGER.info("Reading CSV with Spark: %s", csv_path)

    source_df = (
        spark.read
        .format("csv")
        .option("header", "true")
        .option("inferSchema", "false")
        .option("multiLine", "true")
        .option("quote", '"')
        .option("escape", '"')
        .option("encoding", "UTF-8")
        .option("mode", "PERMISSIVE")
        .load(str(csv_path))
    )

    bronze_df = (
        source_df.select(
            *[
                F.col(column_name).alias(normalise_column_name(column_name))
                for column_name in source_df.columns
            ]
        )
        .withColumn(
            "_ingestion_date",
            F.to_date(F.lit(INGESTION_DATE)),
        )
        .withColumn(
            "_ingested_at",
            F.current_timestamp(),
        )
        .withColumn(
            "_source_zip_path",
            F.lit(str(SOURCE_ZIP)),
        )
        .withColumn(
            "_source_csv_name",
            F.lit(csv_path.name),
        )
    )

    LOGGER.info(
        "Source schema contains %s columns.",
        len(source_df.columns),
    )

    return bronze_df


def normalise_column_name(column_name: str) -> str:
    """Convert source headers into Delta-safe Bronze column names."""

    cleaned_name = column_name.strip().lower()
    cleaned_name = re.sub(r"[^a-z0-9]+", "_", cleaned_name)
    cleaned_name = re.sub(r"_+", "_", cleaned_name)
    return cleaned_name.strip("_")


def write_bronze_table(
    spark: SparkSession,
    dataframe: DataFrame,
) -> None:
    """Write one ingestion-date snapshot into the Bronze table."""

    spark.sql(
        f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}"
    )

    if spark.catalog.tableExists(TARGET_TABLE):
        LOGGER.info(
            "Removing any existing records for %s",
            INGESTION_DATE,
        )

        spark.sql(
            f"""
            DELETE FROM {TARGET_TABLE}
            WHERE _ingestion_date = DATE '{INGESTION_DATE}'
            """
        )

        LOGGER.info("Appending snapshot to %s", TARGET_TABLE)

        (
            dataframe.write
            .format("delta")
            .mode("append")
            .saveAsTable(TARGET_TABLE)
        )

    else:
        LOGGER.info("Creating Bronze table: %s", TARGET_TABLE)

        (
            dataframe.write
            .format("delta")
            .mode("overwrite")
            .partitionBy("_ingestion_date")
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

    row_count = result["row_count"]

    if row_count == 0:
        raise RuntimeError(
            "Bronze validation failed: zero rows were written."
        )

    LOGGER.info(
        "Bronze validation successful. Rows written: %s",
        f"{row_count:,}",
    )


def clean_staging() -> None:
    """Delete the temporary extracted CSV after successful ingestion."""

    if STAGING_DIRECTORY.exists():
        LOGGER.info(
            "Removing temporary staging directory: %s",
            STAGING_DIRECTORY,
        )
        shutil.rmtree(STAGING_DIRECTORY)


def main() -> None:
    """Run the Bronze ingestion pipeline."""

    spark = SparkSession.builder.getOrCreate()

    LOGGER.info("Starting Consumer Complaints Bronze ingestion.")

    validate_source()
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

        validate_target(spark)

    finally:
        clean_staging()

    LOGGER.info(
        "Bronze ingestion finished successfully: %s",
        TARGET_TABLE,
    )


if __name__ == "__main__":
    main()
