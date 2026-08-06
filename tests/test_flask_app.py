"""Unit tests for the pure request-encoding functions in flask_app/app.py.

Skipped automatically (not failed) when Flask/pandas aren't installed, since the Flask app's
dependencies (flask_app/requirements.txt) are a separate environment from the ingestion tests'
(root requirements.txt) - see README.md "Setup". Run these with the flask_app virtual
environment's interpreter to actually execute them:

    flask_app\\.venv\\Scripts\\python.exe -m unittest tests.test_flask_app -v
"""

from pathlib import Path  # used to build a filesystem path to the module under test
import importlib.util  # used to load flask_app/app.py by path
import unittest  # test framework

try:
    import pandas as pd  # noqa: F401 - only imported to detect availability
    import flask  # noqa: F401 - only imported to detect availability
    _FLASK_DEPENDENCIES_AVAILABLE = True  # both imported cleanly
except ImportError:
    _FLASK_DEPENDENCIES_AVAILABLE = False  # Flask/pandas not installed in this interpreter


def _load_app_module():
    """Load flask_app/app.py by path. Safe to import without model files present: model/serving
    metadata are only loaded lazily by get_model()/get_serving_metadata(), never at import time."""

    module_path = Path(__file__).resolve().parents[1] / "flask_app" / "app.py"  # path to the script under test
    spec = importlib.util.spec_from_file_location("flask_app_app", module_path)  # module spec built from that path
    module = importlib.util.module_from_spec(spec)  # empty module object, not yet executed
    spec.loader.exec_module(module)  # actually run app.py, populating the module object
    return module  # handed back to each test class's setUpClass


# A minimal but realistic serving_metadata.json fixture, covering one target-encoded categorical
# column and one numeric column, enough to exercise both branches of encode_request/raw_input_fields
# without needing the real (much larger) champion model's metadata.
SAMPLE_SERVING_METADATA = {
    "categorical_feature_columns": ["company"],  # raw categorical inputs the form should ask for
    "numeric_feature_columns": ["complaint_received_year"],  # raw numeric inputs the form should ask for
    "selected_feature_columns": ["company_indexed", "complaint_received_year"],  # exact columns/order the model expects
    "encoding_maps": {"company": {"EQUIFAX, INC.": 0.02, "OTHER CO": 0.01}},  # per-category smoothed positive rates
    "global_positive_rate": 0.006,  # fallback for a company not present in encoding_maps
}


@unittest.skipUnless(
    _FLASK_DEPENDENCIES_AVAILABLE,  # only run when flask/pandas are importable
    "requires flask/pandas - run with flask_app/.venv's interpreter",
)
class RawInputFieldsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app_module = _load_app_module()  # loaded once, reused by every test below

    def test_categorical_column_is_reported_by_its_raw_name_not_indexed_name(self) -> None:
        fields = self.app_module.raw_input_fields(SAMPLE_SERVING_METADATA)

        field_by_name = {field["name"]: field for field in fields}  # index by field name for easy lookup
        self.assertIn("company", field_by_name)  # not "company_indexed" - that's an implementation detail
        self.assertEqual(field_by_name["company"]["type"], "categorical")  # reported type matches its source column

    def test_numeric_column_is_reported_as_numeric(self) -> None:
        fields = self.app_module.raw_input_fields(SAMPLE_SERVING_METADATA)

        field_by_name = {field["name"]: field for field in fields}  # index by field name for easy lookup
        self.assertEqual(field_by_name["complaint_received_year"]["type"], "numeric")  # numeric columns pass through unchanged

    def test_only_selected_feature_columns_are_requested(self) -> None:
        # The champion model may only use ~10 of 18 candidate columns; the form must not ask for
        # inputs the model doesn't actually need.
        fields = self.app_module.raw_input_fields(SAMPLE_SERVING_METADATA)
        self.assertEqual(len(fields), 2)  # exactly the two columns in selected_feature_columns, nothing more


@unittest.skipUnless(
    _FLASK_DEPENDENCIES_AVAILABLE,  # only run when flask/pandas are importable
    "requires flask/pandas - run with flask_app/.venv's interpreter",
)
class EncodeRequestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app_module = _load_app_module()  # loaded once, reused by every test below

    def test_known_category_uses_its_target_encoded_value(self) -> None:
        raw_complaint = {"company": "EQUIFAX, INC.", "complaint_received_year": 2024}  # a company present in encoding_maps

        encoded = self.app_module.encode_request(raw_complaint, SAMPLE_SERVING_METADATA)

        self.assertEqual(encoded["company_indexed"].iloc[0], 0.02)  # exact value from encoding_maps, not the fallback

    def test_unseen_category_falls_back_to_global_positive_rate(self) -> None:
        # Mirrors training-time behaviour for a company the model never saw - see
        # _target_encode_column's fillna(global_positive_rate) in consumer_complaints_ml_train.py.
        raw_complaint = {"company": "A BRAND NEW COMPANY LLC", "complaint_received_year": 2024}  # absent from encoding_maps

        encoded = self.app_module.encode_request(raw_complaint, SAMPLE_SERVING_METADATA)

        self.assertEqual(encoded["company_indexed"].iloc[0], SAMPLE_SERVING_METADATA["global_positive_rate"])  # fallback used

    def test_numeric_field_submitted_as_string_is_cast_to_float(self) -> None:
        # Regression test: HTML form submissions arrive as strings (FormData has no numeric type),
        # and XGBoost rejects object-dtype columns outright rather than coercing them. This exact
        # bug was hit and fixed during manual browser testing.
        raw_complaint = {"company": "EQUIFAX, INC.", "complaint_received_year": "2024"}  # string, as a real HTML form would send it

        encoded = self.app_module.encode_request(raw_complaint, SAMPLE_SERVING_METADATA)

        self.assertEqual(encoded["complaint_received_year"].dtype.kind, "f")  # cast to a real float dtype, not left as object
        self.assertEqual(encoded["complaint_received_year"].iloc[0], 2024.0)  # value survived the cast unchanged

    def test_missing_numeric_field_defaults_to_zero_instead_of_raising(self) -> None:
        raw_complaint = {"company": "EQUIFAX, INC."}  # complaint_received_year omitted entirely

        encoded = self.app_module.encode_request(raw_complaint, SAMPLE_SERVING_METADATA)

        self.assertEqual(encoded["complaint_received_year"].iloc[0], 0.0)  # missing field defaults to 0.0, does not raise

    def test_encoded_row_only_contains_selected_feature_columns_in_order(self) -> None:
        raw_complaint = {"company": "EQUIFAX, INC.", "complaint_received_year": 2024}

        encoded = self.app_module.encode_request(raw_complaint, SAMPLE_SERVING_METADATA)

        self.assertEqual(list(encoded.columns), SAMPLE_SERVING_METADATA["selected_feature_columns"])  # exact column order the model needs


_MODEL_DIR = Path(__file__).resolve().parents[1] / "flask_app" / "model"  # local (gitignored) model directory
_MODEL_PATH = _MODEL_DIR / "champion_model.joblib"  # the calibrated estimator
_SERVING_METADATA_PATH = _MODEL_DIR / "serving_metadata.json"  # the serving contract
_MODEL_ARTIFACTS_AVAILABLE = _MODEL_PATH.exists() and _SERVING_METADATA_PATH.exists()  # both files must be present to run


@unittest.skipUnless(
    _FLASK_DEPENDENCIES_AVAILABLE,  # only run when flask/pandas/joblib are importable
    "requires flask/pandas/joblib - run with flask_app/.venv's interpreter",
)
@unittest.skipUnless(
    _MODEL_ARTIFACTS_AVAILABLE,  # only run when the local model files actually exist
    "flask_app/model/ is gitignored - run flask_app/refresh_model.ps1 first",
)
class ModelServingContractTests(unittest.TestCase):
    """Guards against the exact bug hit once during development: a stale serving_metadata.json
    (from a losing model run) getting paired with a different model's champion_model.joblib,
    which crashes Flask with a feature_names mismatch only at prediction time, not at startup."""

    def test_model_feature_names_match_serving_metadata_selected_columns(self) -> None:
        import joblib  # imported here, not at module top, since it is only needed by this one test
        import json  # imported here, not at module top, since it is only needed by this one test

        model = joblib.load(_MODEL_PATH)  # the actual champion model currently in flask_app/model/
        with open(_SERVING_METADATA_PATH, encoding="utf-8") as f:
            serving_metadata = json.load(f)  # the actual serving contract currently in flask_app/model/

        self.assertEqual(
            list(model.feature_names_in_),  # what the model object itself was fit on
            serving_metadata["selected_feature_columns"],  # what the serving contract claims to feed it
        )


if __name__ == "__main__":
    unittest.main()  # allow running this file directly, in addition to unittest discovery
