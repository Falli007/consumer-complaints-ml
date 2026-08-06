"""Unit tests for the pure pandas/numpy/sklearn functions in consumer_complaints_ml_train.py.

Skipped automatically (not failed) when scikit-learn/xgboost aren't installed, since the
training script's dependencies (flask_app/requirements.txt) are a separate environment from
the ingestion tests' (root requirements.txt) - see README.md "Setup". Run these with the
flask_app virtual environment's interpreter to actually execute them:

    flask_app\\.venv\\Scripts\\python.exe -m unittest tests.test_consumer_complaints_ml_train -v
"""

from pathlib import Path  # used to build a filesystem path to the module under test
import importlib.util  # used to load consumer_complaints_ml_train.py by path
import sys  # used to register stub pyspark modules before the target module imports them
import types  # used to build lightweight stub modules for pyspark
import unittest  # test framework

try:
    import numpy as np  # real dependency, required to exercise the functions under test
    import pandas as pd  # real dependency, required to exercise the functions under test

    import sklearn  # noqa: F401 - only imported to detect availability, not used directly
    _ML_DEPENDENCIES_AVAILABLE = True  # sklearn (and therefore the target module) imported cleanly
except ImportError:
    _ML_DEPENDENCIES_AVAILABLE = False  # scikit-learn/pandas/numpy not installed in this interpreter


def _load_module_with_pyspark_stubbed():
    """Load consumer_complaints_ml_train.py with pyspark faked out, since it is only used here
    for reading/writing Spark tables, never by the pure pandas functions under test."""

    module_path = (
        Path(__file__).resolve().parents[1]  # repo root
        / "databricks" / "src" / "ml" / "consumer_complaints_ml_train.py"  # path to the script under test
    )
    spec = importlib.util.spec_from_file_location("consumer_complaints_ml_train", module_path)  # module spec built from that path
    module = importlib.util.module_from_spec(spec)  # empty module object, not yet executed

    pyspark_stub = types.ModuleType("pyspark")  # fake top-level pyspark package
    sql_stub = types.ModuleType("pyspark.sql")  # fake pyspark.sql submodule
    sql_stub.DataFrame = type("DataFrame", (), {})  # only referenced as a type hint at runtime
    sql_stub.SparkSession = type("SparkSession", (), {})  # only referenced as a type hint at runtime
    functions_stub = types.ModuleType("pyspark.sql.functions")  # fake pyspark.sql.functions (F)
    types_stub = types.ModuleType("pyspark.sql.types")  # fake pyspark.sql.types
    for type_name in ("BooleanType", "StringType", "StructField", "StructType"):
        setattr(types_stub, type_name, type(type_name, (), {}))  # dummy class per imported name

    sys.modules["pyspark"] = pyspark_stub  # register the fake pyspark before the target module is executed
    sys.modules["pyspark.sql"] = sql_stub  # register the fake pyspark.sql submodule
    sys.modules["pyspark.sql.functions"] = functions_stub  # register the fake pyspark.sql.functions submodule
    sys.modules["pyspark.sql.types"] = types_stub  # register the fake pyspark.sql.types submodule

    spec.loader.exec_module(module)  # actually run the script, populating the module object
    return module  # handed back to each test class's setUpClass


@unittest.skipUnless(
    _ML_DEPENDENCIES_AVAILABLE,  # only run when sklearn/pandas/numpy are importable
    "requires scikit-learn/pandas/numpy - run with flask_app/.venv's interpreter",
)
class TargetEncodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ml_train = _load_module_with_pyspark_stubbed()  # loaded once, reused by every test below

    def test_target_encode_column_uses_smoothed_positive_rate(self) -> None:
        fit_pdf = pd.DataFrame(
            {
                "company": ["A", "A", "A", "B", "B", "B"],  # category "A" is all positive, "B" all negative
                self.ml_train.LABEL_COLUMN: [1, 1, 1, 0, 0, 0],  # mean=1.0 for "A", mean=0.0 for "B"
            }
        )
        transform_series = pd.Series(["A", "B"])  # the values to encode, one per category
        global_positive_rate = 0.5  # smoothing target both categories are pulled toward

        encoded = self.ml_train._target_encode_column(
            fit_pdf, transform_series, "company", global_positive_rate
        )

        # With TARGET_ENCODING_SMOOTHING pulling toward the global rate, neither category should
        # land exactly at its raw 1.0/0.0 mean, but "A" must still score higher than "B".
        self.assertGreater(encoded.iloc[0], encoded.iloc[1])  # "A" (positive) outranks "B" (negative)
        self.assertLess(encoded.iloc[0], 1.0)  # smoothing pulled "A" down from its raw 1.0 mean
        self.assertGreater(encoded.iloc[1], 0.0)  # smoothing pulled "B" up from its raw 0.0 mean

    def test_target_encode_column_falls_back_to_global_rate_for_unseen_category(self) -> None:
        fit_pdf = pd.DataFrame(
            {
                "company": ["A", "A"],  # only category "A" exists in the fit data
                self.ml_train.LABEL_COLUMN: [1, 1],
            }
        )
        transform_series = pd.Series(["NEVER_SEEN_BEFORE"])  # a category absent from fit_pdf entirely
        global_positive_rate = 0.42  # the fallback value an unseen category must resolve to

        encoded = self.ml_train._target_encode_column(
            fit_pdf, transform_series, "company", global_positive_rate
        )

        # map() leaves unseen categories as NaN, which _target_encode_column must fill with the
        # global rate - this is exactly what happens in Flask's encode_request() for a company the
        # model never saw in training.
        self.assertEqual(encoded.iloc[0], global_positive_rate)  # NaN was filled, not left/dropped

    def test_target_encode_categorical_columns_never_leaks_a_train_rows_own_label(self) -> None:
        ml_train = self.ml_train  # local alias, shorter than self.ml_train everywhere below
        rows = 60  # small enough to run fast, large enough for 5-fold KFold to behave sensibly
        feature_pandas_df = pd.DataFrame(
            {
                "product": [f"product_{i}" for i in range(rows)],  # every row is its own unique category
                ml_train.LABEL_COLUMN: [i % 2 for i in range(rows)],  # alternating 0/1 labels
                ml_train.SPLIT_COLUMN: [ml_train.TRAIN_SPLIT_LABEL] * rows,  # every row is TRAIN, exercising the out-of-fold path
            }
        )
        for column_name in ml_train.CATEGORICAL_FEATURE_COLUMNS:
            if column_name != "product":
                feature_pandas_df[column_name] = "constant"  # other categoricals are irrelevant here, keep them fixed

        training_only_pdf = feature_pandas_df  # every row is TRAIN, so this is the same frame

        encoded_df, encoding_maps, global_positive_rate = ml_train._target_encode_categorical_columns(
            feature_pandas_df, training_only_pdf
        )

        # Every category here is unique (one row each), so a leak-free encoding must fall back to
        # something close to the global rate for every single row, not the row's own 0/1 label.
        self.assertTrue(
            (encoded_df["product_indexed"] - global_positive_rate).abs().lt(0.5).all()  # no row is close to its own 0/1 label
        )
        self.assertIn("product", encoding_maps)  # the per-category encoding map was returned too


@unittest.skipUnless(
    _ML_DEPENDENCIES_AVAILABLE,  # only run when sklearn/pandas/numpy are importable
    "requires scikit-learn/pandas/numpy - run with flask_app/.venv's interpreter",
)
class ThresholdSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ml_train = _load_module_with_pyspark_stubbed()  # loaded once, reused by every test below

    def test_selects_a_threshold_that_meets_the_recall_floor(self) -> None:
        labels = pd.Series([0] * 50 + [1] * 50)  # 50 negatives followed by 50 positives
        probabilities = np.concatenate([np.linspace(0.0, 0.4, 50), np.linspace(0.6, 1.0, 50)])  # a separable toy problem

        threshold = self.ml_train._select_threshold_for_recall_floor(labels, probabilities, recall_floor=0.75)

        predictions = (probabilities >= threshold).astype(int)  # apply the selected threshold
        achieved_recall = predictions[labels == 1].sum() / (labels == 1).sum()  # recall among the true positives
        self.assertGreaterEqual(achieved_recall, 0.75)  # the whole point of a recall-floor threshold

    def test_falls_back_to_max_recall_threshold_when_floor_is_unreachable(self) -> None:
        labels = pd.Series([0, 0, 1, 1])  # two negatives, two positives
        probabilities = np.array([0.9, 0.8, 0.7, 0.1])  # only one of the two positives is separable from the negatives

        threshold = self.ml_train._select_threshold_for_recall_floor(labels, probabilities, recall_floor=0.99)  # a floor no threshold can reach here

        self.assertIsInstance(threshold, float)  # must not raise, must fall back to a usable float instead


if __name__ == "__main__":
    unittest.main()  # allow running this file directly, in addition to unittest discovery
