from datetime import date
from pathlib import Path
import importlib.util
import os
import sys
import types
import unittest
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "ingestion" / "consumer_complaints" / "download_to_s3.py"
SPEC = importlib.util.spec_from_file_location("download_to_s3", MODULE_PATH)
download_to_s3 = importlib.util.module_from_spec(SPEC)

boto3_stub = types.ModuleType("boto3")
boto3_stub.client = lambda *args, **kwargs: None
transfer_module = types.ModuleType("boto3.s3.transfer")


class DummyTransferConfig:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs


transfer_module.TransferConfig = DummyTransferConfig

sys.modules.setdefault("boto3", boto3_stub)
sys.modules.setdefault("boto3.s3", types.ModuleType("boto3.s3"))
sys.modules.setdefault("boto3.s3.transfer", transfer_module)
sys.modules.setdefault("requests", types.ModuleType("requests"))
sys.modules["download_to_s3"] = download_to_s3

assert SPEC.loader is not None
SPEC.loader.exec_module(download_to_s3)


class DownloadToS3Tests(unittest.TestCase):
    def test_build_s3_key_uses_runtime_date(self) -> None:
        self.assertEqual(
            download_to_s3.build_s3_key("2026-07-30"),
            "raw/consumer_complaints/ingestion_date=2026-07-30/complaints.csv.zip",
        )

    @patch.object(download_to_s3, "get_current_utc_date", return_value="2026-07-30")
    @patch.dict(os.environ, {"S3_BUCKET_NAME": "test-bucket"}, clear=True)
    def test_load_config_uses_dynamic_date_in_key(self, _: object) -> None:
        config = download_to_s3.load_config()

        self.assertEqual(config.bucket_name, "test-bucket")
        self.assertEqual(config.ingestion_date, "2026-07-30")
        self.assertEqual(
            config.s3_key,
            "raw/consumer_complaints/ingestion_date=2026-07-30/complaints.csv.zip",
        )
        self.assertEqual(
            config.local_file,
            Path("data/raw/consumer_complaints/complaints.csv.zip"),
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_load_config_requires_bucket_name(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "Missing required environment variable: S3_BUCKET_NAME"
        ):
            download_to_s3.load_config()

    def test_get_current_utc_date_returns_iso_date(self) -> None:
        value = download_to_s3.get_current_utc_date()
        self.assertEqual(value, date.fromisoformat(value).isoformat())


if __name__ == "__main__":
    unittest.main()
