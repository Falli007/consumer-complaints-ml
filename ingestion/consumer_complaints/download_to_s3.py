from __future__ import annotations

# Standard library imports for paths, runtime configuration, timestamps, and logging.
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import logging
import os

# Third-party libraries for AWS S3 upload support and streamed HTTP downloads.
import boto3
import requests
from boto3.s3.transfer import TransferConfig


# Source dataset location and local runtime defaults for this ingestion job.
DATASET_URL = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"
LOCAL_FILE = Path("data/raw/consumer_complaints/complaints.csv.zip")
AWS_REGION = "eu-west-2"
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionConfig:
    """Runtime configuration resolved once and reused across the ingestion steps."""

    bucket_name: str
    ingestion_date: str
    s3_key: str
    local_file: Path


def get_current_utc_date() -> str:
    """Return the current UTC date in ISO format."""

    return datetime.now(timezone.utc).date().isoformat()


def build_s3_key(ingestion_date: str) -> str:
    """Build the S3 object key for a given ingestion date."""

    return (
        "raw/consumer_complaints/"
        f"ingestion_date={ingestion_date}/"
        "complaints.csv.zip"
    )


def load_config() -> IngestionConfig:
    """Load and validate runtime configuration."""

    bucket_name = os.getenv("S3_BUCKET_NAME", "").strip()
    if not bucket_name:
        raise ValueError("Missing required environment variable: S3_BUCKET_NAME")

    # Capture the ingestion date once per run so the S3 path and metadata stay aligned.
    ingestion_date = get_current_utc_date()
    return IngestionConfig(
        bucket_name=bucket_name,
        ingestion_date=ingestion_date,
        s3_key=build_s3_key(ingestion_date),
        local_file=LOCAL_FILE,
    )


def download_dataset(local_file: Path) -> None:
    """Download the dataset to the local project folder."""

    local_file.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temporary file first so a failed download never leaves a partial ZIP at the final path.
    temp_file = local_file.with_suffix(local_file.suffix + ".part")

    LOGGER.info("Downloading dataset to %s", local_file)
    last_logged_percentage = -10

    try:
        with requests.get(
            DATASET_URL,
            stream=True,
            timeout=120,
        ) as response:
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            with temp_file.open("wb") as output_file:
                for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                    if not chunk:
                        continue

                    output_file.write(chunk)
                    downloaded += len(chunk)

                    if total_size:
                        # Log progress in coarse steps to keep long-running downloads readable.
                        percentage = int(downloaded / total_size * 100)
                        if percentage >= last_logged_percentage + 10:
                            LOGGER.info("Download progress: %s%%", percentage)
                            last_logged_percentage = percentage
    except Exception:
        if temp_file.exists():
            temp_file.unlink()
        raise

    temp_file.replace(local_file)
    LOGGER.info("Dataset download complete.")


def upload_to_s3(config: IngestionConfig) -> None:
    """Upload the downloaded ZIP file to Amazon S3."""

    if not config.local_file.exists():
        raise FileNotFoundError(
            f"Local dataset file not found: {config.local_file}"
        )

    # Keep the existing multipart upload settings so large files can upload reliably.
    s3_client = boto3.client("s3", region_name=AWS_REGION)

    transfer_config = TransferConfig(
        multipart_threshold=64 * 1024 * 1024,
        multipart_chunksize=64 * 1024 * 1024,
        max_concurrency=6,
        use_threads=True,
    )

    LOGGER.info("Uploading to s3://%s/%s", config.bucket_name, config.s3_key)

    s3_client.upload_file(
        Filename=str(config.local_file),
        Bucket=config.bucket_name,
        Key=config.s3_key,
        Config=transfer_config,
        ExtraArgs={
            "ContentType": "application/zip",
            "Metadata": {
                "source": "cfpb-consumer-complaints",
                # Persist the runtime ingestion date with the object for traceability.
                "ingestion-date": config.ingestion_date,
            },
        },
    )

    LOGGER.info("S3 upload complete.")


def verify_upload(config: IngestionConfig) -> None:
    """Confirm the uploaded object exists in S3."""

    # Use a HEAD request so verification does not re-download the uploaded object.
    s3_client = boto3.client("s3", region_name=AWS_REGION)
    response = s3_client.head_object(
        Bucket=config.bucket_name,
        Key=config.s3_key,
    )

    size_mb = response["ContentLength"] / (1024 ** 2)
    LOGGER.info("Upload verification successful.")
    LOGGER.info("S3 location: s3://%s/%s", config.bucket_name, config.s3_key)
    LOGGER.info("File size: %.2f MB", size_mb)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def main() -> int:
    """Run the ingestion workflow and return a shell-friendly status code."""

    configure_logging()

    try:
        config = load_config()
        download_dataset(config.local_file)
        upload_to_s3(config)
        verify_upload(config)
    except Exception as exc:
        LOGGER.error("Ingestion failed: %s", exc)
        LOGGER.debug("Detailed ingestion failure", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
