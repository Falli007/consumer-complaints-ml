# Consumer Complaints Ingestion

This repository contains a small ingestion script that downloads the CFPB Consumer Complaint Database ZIP file, uploads it to Amazon S3 with multipart transfer, and verifies the uploaded object.

## Prerequisites

- Python 3.11+
- AWS credentials available to `boto3`
- PowerShell environment variable `S3_BUCKET_NAME`

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run tests

```powershell
python -m unittest discover -s tests -v
```

## Run the ingestion

The script generates the ingestion date at runtime using the current UTC date and uses it in both the S3 key and the S3 object metadata.

```powershell
$env:S3_BUCKET_NAME = "your-bucket-name"
python ingestion\consumer_complaints\download_to_s3.py
```

## Notes

- Downloaded data is stored locally under `data/` before upload.
- Local data files, ZIP files, virtual environments, and Python cache files are excluded from Git.
- The test suite covers runtime date/path generation and configuration validation without downloading the dataset.
