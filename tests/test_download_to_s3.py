from datetime import date  # used to validate the ISO date format returned by get_current_utc_date
from pathlib import Path  # used to build a filesystem path to the module under test
import importlib.util  # used to load download_to_s3.py by path instead of via a package import
import os  # used to patch os.environ for config-loading tests
import sys  # used to register stub modules in sys.modules before the target module imports them
import types  # used to build lightweight stub modules for boto3/requests
import unittest  # test framework
from unittest.mock import patch  # used to mock get_current_utc_date and os.environ


MODULE_PATH = Path(__file__).resolve().parents[1] / "ingestion" / "consumer_complaints" / "download_to_s3.py"  # absolute path to the script under test
SPEC = importlib.util.spec_from_file_location("download_to_s3", MODULE_PATH)  # module spec built from that path
download_to_s3 = importlib.util.module_from_spec(SPEC)  # empty module object, not yet executed

boto3_stub = types.ModuleType("boto3")  # fake boto3 module so download_to_s3 can import it without the real AWS SDK
boto3_stub.client = lambda *args, **kwargs: None  # fake boto3.client(), never actually called in these tests
transfer_module = types.ModuleType("boto3.s3.transfer")  # fake boto3.s3.transfer submodule


class DummyTransferConfig:  # stand-in for boto3.s3.transfer.TransferConfig
    def __init__(self, *args, **kwargs) -> None:
        self.args = args  # keep constructor args around in case a test wants to inspect them
        self.kwargs = kwargs  # keep constructor kwargs around in case a test wants to inspect them


transfer_module.TransferConfig = DummyTransferConfig  # attach the dummy class to the fake submodule

sys.modules.setdefault("boto3", boto3_stub)  # register the fake boto3 before download_to_s3 is executed
sys.modules.setdefault("boto3.s3", types.ModuleType("boto3.s3"))  # register the fake boto3.s3 namespace package
sys.modules.setdefault("boto3.s3.transfer", transfer_module)  # register the fake boto3.s3.transfer module
sys.modules.setdefault("requests", types.ModuleType("requests"))  # fake requests module, also not exercised here
sys.modules["download_to_s3"] = download_to_s3  # pre-register the module so relative lookups resolve correctly

assert SPEC.loader is not None  # narrows the type for the exec_module call below
SPEC.loader.exec_module(download_to_s3)  # actually run download_to_s3.py, populating the module object


class DownloadToS3Tests(unittest.TestCase):
    def test_build_s3_key_uses_runtime_date(self) -> None:
        self.assertEqual(
            download_to_s3.build_s3_key("2026-07-30"),  # build the key for a fixed, known date
            "raw/consumer_complaints/ingestion_date=2026-07-30/complaints.csv.zip",  # expected Hive-style partitioned key
        )

    @patch.object(download_to_s3, "get_current_utc_date", return_value="2026-07-30")  # freeze "today" for a deterministic key
    @patch.dict(os.environ, {"S3_BUCKET_NAME": "test-bucket"}, clear=True)  # provide the only required env var
    def test_load_config_uses_dynamic_date_in_key(self, _: object) -> None:
        config = download_to_s3.load_config()  # build config using the mocked date and env var

        self.assertEqual(config.bucket_name, "test-bucket")  # bucket name should come straight from the env var
        self.assertEqual(config.ingestion_date, "2026-07-30")  # ingestion date should come from the mocked clock
        self.assertEqual(
            config.s3_key,
            "raw/consumer_complaints/ingestion_date=2026-07-30/complaints.csv.zip",  # key should embed that same date
        )
        self.assertEqual(
            config.local_file,
            Path("data/raw/consumer_complaints/complaints.csv.zip"),  # local download path is fixed, not date-partitioned
        )

    @patch.dict(os.environ, {}, clear=True)  # simulate a completely empty environment
    def test_load_config_requires_bucket_name(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "Missing required environment variable: S3_BUCKET_NAME"  # config loading must fail loudly, not silently
        ):
            download_to_s3.load_config()

    def test_get_current_utc_date_returns_iso_date(self) -> None:
        value = download_to_s3.get_current_utc_date()  # call the real (unmocked) clock function
        self.assertEqual(value, date.fromisoformat(value).isoformat())  # round-tripping through ISO parsing should be a no-op


if __name__ == "__main__":
    unittest.main()  # allow running this file directly, in addition to `python -m unittest discover`
