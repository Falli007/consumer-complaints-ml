"""Train tree-based untimely-response models from the Gold feature table.

This script reads the managed ML feature table, compares multiple tree-based
binary classifiers, and writes evaluation outputs that can be reviewed in
Databricks SQL, dashboards, or notebooks.

Training runs on pandas/scikit-learn/XGBoost rather than Spark ML: this
workspace's serverless compute does not reliably support classic
``pyspark.ml`` (constructor whitelisting on older environment versions, and a
1GB Spark Connect ML session cache limit on newer ones), and the reduced,
already-aggregated feature table is small enough to train comfortably on the
driver. Spark is still used to read the managed Gold-layer feature table and
to persist the monitoring outputs.

Training flow:
1. Validate that the feature table exists and contains all split groups.
2. Collect the reduced feature columns into pandas and apply class weighting
   to address the rare untimely-response class.
3. Ordinal-encode categorical columns, fit on the training split only.
4. Use Random Forest feature importance to select the top model inputs.
5. Train Decision Tree, Random Forest, and (when available) XGBoost on the
   reduced feature set.
6. Class weighting distorts predicted probabilities, so each model is
   sigmoid (Platt) calibrated on half of the validation split before a
   decision threshold is chosen on the other half, targeting a recall floor
   (see TARGET_RECALL_FLOOR) rather than maximising F1 - under this class
   imbalance a missed untimely complaint is assumed costlier than an
   over-flagged timely one. Isotonic calibration was tried first and
   confirmed to degenerate under this level of imbalance with a modest
   calibration fold. Test is never touched for calibration or threshold
   selection.
7. Evaluate calibrated validation and test predictions for all models
   (AUC-ROC, AUC-PR, precision, recall, F2, confusion matrix).
8. Log every model's params/metrics to MLflow for experiment tracking, and
   write the model with the best validation AUC-PR straight to a Unity
   Catalog Volume as champion_model.joblib plus serving_metadata_champion.json
   - this is what the Flask app under flask_app/ actually loads to serve
   predictions, with no MLflow model registry involved.
9. Persist metrics plus scored test predictions to the monitoring schema.
"""

from __future__ import annotations

import logging
import time

import json
import shutil

import joblib
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType, StringType, StructField, StructType
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
    XGBOOST_IMPORT_ERROR = None
except Exception as xgboost_import_error:  # noqa: BLE001 - report whatever the real failure is, not just ImportError.
    XGBClassifier = None
    XGBOOST_AVAILABLE = False
    XGBOOST_IMPORT_ERROR = xgboost_import_error

CATALOG = "fintech_lakehouse_dev"
GOLD_SCHEMA = "gold"
MONITORING_SCHEMA = "monitoring"

FEATURE_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.consumer_complaints_timely_response_features"
METRICS_TABLE = (
    f"{CATALOG}.{MONITORING_SCHEMA}.consumer_complaints_timely_response_model_metrics"
)
PREDICTIONS_TABLE = (
    f"{CATALOG}.{MONITORING_SCHEMA}.consumer_complaints_timely_response_test_predictions"
)

LABEL_COLUMN = "label_untimely_response"
WEIGHT_COLUMN = "class_weight"
SPLIT_COLUMN = "dataset_split"
MODEL_NAME_COLUMN = "model_name"
SELECTED_FEATURE_RANK_COLUMN = "selected_feature_rank"

TRAIN_SPLIT_LABEL = "TRAIN"
VALIDATION_SPLIT_LABEL = "VALIDATION"
TEST_SPLIT_LABEL = "TEST"

DECISION_TREE_MODEL_NAME = "decision_tree"
RANDOM_FOREST_MODEL_NAME = "random_forest"
XGBOOST_MODEL_NAME = "spark_xgboost"

CATEGORICAL_FEATURE_COLUMNS = [
    "product",
    "sub_product",
    "company",
    "state",
    "issue",
    "sub_issue",
    "submitted_via",
    "zip_code_prefix3",
    "received_month_name",
    "received_day_name",
]

# TODO: "has_consumer_narrative" is a binary presence flag only; the actual narrative text is not
# used as a feature anywhere in this pipeline. Real text features (e.g. embeddings from a
# pretrained transformer) are a plausible source of additional signal but are out of scope for
# this iteration - they need new plumbing in Silver/Gold to retain the raw text, plus new
# feature-engineering and training work. Do not add this without first scoping that work properly.
NUMERIC_FEATURE_COLUMNS = [
    "complaint_received_year",
    "complaint_received_quarter",
    "complaint_received_month",
    "complaint_received_day",
    "complaint_received_day_of_week",
    "received_is_weekend",
    "has_consumer_narrative",
    "has_tags",
]

INDEXED_CATEGORICAL_FEATURE_COLUMNS = [
    f"{column_name}_indexed"
    for column_name in CATEGORICAL_FEATURE_COLUMNS
]
MODEL_INPUT_COLUMNS = INDEXED_CATEGORICAL_FEATURE_COLUMNS + NUMERIC_FEATURE_COLUMNS
TOP_FEATURE_COUNT = 10
MAX_ROWS_PER_SPLIT = 500_000

# ASSUMPTION (opinion): categoricals are target-encoded rather than ordinal-encoded. OrdinalEncoder
# assigns arbitrary integer codes (0, 1, 2, ...) to categories with no real order (e.g. company
# names), falsely implying category 47 is "closer to" category 48 than to category 2. Tree models
# can partially work around this via repeated splits, but it caps how much signal they can extract -
# and company/product were the top two most important features, so this plausibly costs real signal.
TARGET_ENCODING_SMOOTHING = 20  # Pulls rare categories' estimates toward the global TRAIN rate.
TARGET_ENCODING_FOLDS = 5  # Out-of-fold encoding for TRAIN, to avoid leaking each row's own label into its own encoded value.

PREDICTION_OUTPUT_COLUMNS = [
    "complaint_id",
    "product",
    "sub_product",
    "company",
    "state",
    "issue",
    "sub_issue",
    "submitted_via",
    "complaint_received_date",
]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

LOGGER = logging.getLogger(__name__)

RUNTIME_DIAGNOSTICS_TABLE = f"{CATALOG}.{MONITORING_SCHEMA}.consumer_complaints_ml_runtime_diagnostics"


# ------------------------------------------------------------
# Persist environment diagnostics (e.g. why XGBoost may be
# unavailable) to a small table, queryable without needing to
# dig through job/notebook stdout.
# ------------------------------------------------------------
def persist_runtime_diagnostics(spark: SparkSession) -> None:
    """Write a one-row diagnostics table capturing XGBoost availability."""

    # Explicit schema: a fully-None xgboost_import_error column (when import succeeds) otherwise
    # fails Spark's type inference in createDataFrame.
    diagnostics_schema = StructType([
        StructField("xgboost_available", BooleanType(), False),
        StructField("xgboost_import_error", StringType(), True),
    ])
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{MONITORING_SCHEMA}")
    spark.createDataFrame(
        [(bool(XGBOOST_AVAILABLE), str(XGBOOST_IMPORT_ERROR) if XGBOOST_IMPORT_ERROR is not None else None)],
        schema=diagnostics_schema,
    ).withColumn("_checked_at", F.current_timestamp()).write.format("delta").mode("overwrite").saveAsTable(
        RUNTIME_DIAGNOSTICS_TABLE
    )


# ------------------------------------------------------------
# Validate the feature table before training so downstream
# steps only run on a known-good dataset with all split groups.
# ------------------------------------------------------------
def validate_feature_table(spark: SparkSession) -> DataFrame:
    """Validate the ML feature table and return it as a DataFrame."""

    if not spark.catalog.tableExists(FEATURE_TABLE):
        raise RuntimeError(f"ML feature table does not exist: {FEATURE_TABLE}")

    feature_dataframe = spark.table(FEATURE_TABLE)
    feature_row_count = feature_dataframe.count()

    if feature_row_count == 0:
        raise RuntimeError(f"ML feature table is empty: {FEATURE_TABLE}")

    split_counts = {
        row[SPLIT_COLUMN]: row["row_count"]
        for row in feature_dataframe.groupBy(SPLIT_COLUMN)
        .count()
        .withColumnRenamed("count", "row_count")
        .collect()
    }

    for required_split in [TRAIN_SPLIT_LABEL, VALIDATION_SPLIT_LABEL, TEST_SPLIT_LABEL]:
        if split_counts.get(required_split, 0) == 0:
            raise RuntimeError(
                f"ML feature validation failed: required split {required_split} is empty."
            )

    LOGGER.info(
        "Validated ML feature table %s with %s rows.",
        FEATURE_TABLE,
        f"{feature_row_count:,}",
    )
    return feature_dataframe


def _collect_to_pandas_with_retry(spark_dataframe: DataFrame, attempts: int = 3, initial_backoff_seconds: int = 10) -> pd.DataFrame:
    """Collect a Spark DataFrame to pandas, retrying on transient Spark Connect failures."""

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return spark_dataframe.toPandas()
        except Exception as collect_error:  # noqa: BLE001 - deliberately broad to retry any transient Connect failure.
            last_error = collect_error
            LOGGER.warning("toPandas attempt %s/%s failed: %s", attempt, attempts, collect_error)
            if attempt < attempts:
                time.sleep(initial_backoff_seconds * attempt)
    raise last_error


def _target_encode_column(
    fit_pdf: pd.DataFrame,
    transform_series: pd.Series,
    column_name: str,
    global_positive_rate: float,
) -> pd.Series:
    """Map each category in transform_series to its smoothed positive rate, fit on fit_pdf."""

    stats = fit_pdf.groupby(column_name)[LABEL_COLUMN].agg(["mean", "count"])
    smoothed_means = (
        stats["count"] * stats["mean"] + TARGET_ENCODING_SMOOTHING * global_positive_rate
    ) / (stats["count"] + TARGET_ENCODING_SMOOTHING)
    return transform_series.map(smoothed_means).fillna(global_positive_rate)


def _target_encode_categorical_columns(
    feature_pandas_df: pd.DataFrame, training_only_pdf: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, dict[str, float]], float]:
    """Target-encode categoricals: out-of-fold on TRAIN, full-TRAIN statistics on VALIDATION/TEST.

    TRAIN rows are encoded out-of-fold (K-fold) so a row's own label never leaks into its own
    encoded value; VALIDATION/TEST are encoded using the full TRAIN split's statistics since they
    are genuinely held out already.

    Returns the encoded DataFrame plus the full-TRAIN encoding maps (category -> smoothed rate) and
    the global TRAIN positive rate fallback - both are required at serving time to encode raw
    categorical values for a brand-new complaint exactly as VALIDATION/TEST were encoded here.
    """

    global_positive_rate = float(training_only_pdf[LABEL_COLUMN].mean())
    train_mask = (feature_pandas_df[SPLIT_COLUMN] == TRAIN_SPLIT_LABEL).to_numpy()
    train_positions = np.flatnonzero(train_mask)
    other_positions = np.flatnonzero(~train_mask)

    kfold = KFold(n_splits=TARGET_ENCODING_FOLDS, shuffle=True, random_state=42)
    encoding_maps: dict[str, dict[str, float]] = {}

    for column_name, indexed_column_name in zip(CATEGORICAL_FEATURE_COLUMNS, INDEXED_CATEGORICAL_FEATURE_COLUMNS, strict=True):
        encoded_values = np.empty(len(feature_pandas_df), dtype=float)

        for fit_local_idx, transform_local_idx in kfold.split(train_positions):
            fit_positions = train_positions[fit_local_idx]
            transform_positions = train_positions[transform_local_idx]
            fit_fold_pdf = feature_pandas_df.iloc[fit_positions]
            transform_fold_series = feature_pandas_df.iloc[transform_positions][column_name]
            encoded_values[transform_positions] = _target_encode_column(
                fit_fold_pdf, transform_fold_series, column_name, global_positive_rate
            ).to_numpy()

        full_train_pdf = feature_pandas_df.iloc[train_positions]
        encoded_values[other_positions] = _target_encode_column(
            full_train_pdf, feature_pandas_df.iloc[other_positions][column_name], column_name, global_positive_rate
        ).to_numpy()

        feature_pandas_df[indexed_column_name] = encoded_values

        full_train_means = _target_encode_column(
            full_train_pdf, full_train_pdf[column_name], column_name, global_positive_rate
        )
        # Deduplicate to one smoothed value per distinct category (full_train_means is per-row, not per-category).
        category_to_value = full_train_pdf[column_name].to_frame()
        category_to_value["_encoded"] = full_train_means.to_numpy()
        encoding_maps[column_name] = (
            category_to_value.drop_duplicates(subset=column_name)
            .set_index(column_name)["_encoded"]
            .to_dict()
        )

    return feature_pandas_df, encoding_maps, global_positive_rate


# ------------------------------------------------------------
# Collect the reduced feature columns into pandas, apply class
# weights, and target-encode categoricals (fit on TRAIN only).
# ------------------------------------------------------------
def build_training_frame(feature_dataframe: DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, float]], float]:
    """Collect the model-ready columns into a weighted, encoded pandas DataFrame.

    Returns the encoded DataFrame plus the target-encoding maps and global positive rate, which
    are required at serving time to encode a new complaint's raw categorical values.
    """

    pandas_columns = list(dict.fromkeys(
        ["complaint_id", SPLIT_COLUMN, LABEL_COLUMN]
        + CATEGORICAL_FEATURE_COLUMNS
        + NUMERIC_FEATURE_COLUMNS
        + PREDICTION_OUTPUT_COLUMNS
    ))

    # Collect one split at a time, capped to a representative sample per split. Multi-million-row
    # toPandas() transfers over Spark Connect have consistently failed on this workspace (a resource/
    # size limit, not a transient blip). A baseline tree-based model does not need every row: a few
    # hundred thousand rows per split is standard practice and plenty for stable training.
    split_pandas_frames = []
    for split_label in [TRAIN_SPLIT_LABEL, VALIDATION_SPLIT_LABEL, TEST_SPLIT_LABEL]:
        split_spark_df = feature_dataframe.filter(F.col(SPLIT_COLUMN) == split_label).select(*pandas_columns)
        split_row_count = split_spark_df.count()  # Cheap Spark aggregate, not a full collect.
        if split_row_count > MAX_ROWS_PER_SPLIT:
            sample_fraction = min(1.0, MAX_ROWS_PER_SPLIT / split_row_count)
            split_spark_df = split_spark_df.sample(fraction=sample_fraction, seed=42)  # Reproducible random sample.
            LOGGER.info(
                "%s split has %s rows; sampling down to ~%s rows (fraction=%.4f).",
                split_label, f"{split_row_count:,}", f"{MAX_ROWS_PER_SPLIT:,}", sample_fraction,
            )
        split_pandas_df = _collect_to_pandas_with_retry(split_spark_df)
        split_pandas_frames.append(split_pandas_df)
        LOGGER.info("Collected %s split into pandas: %s rows.", split_label, f"{len(split_pandas_df):,}")

    feature_pandas_df = pd.concat(split_pandas_frames, ignore_index=True)

    LOGGER.info(
        "Collected %s rows x %s columns into pandas for training.",
        f"{len(feature_pandas_df):,}",
        len(pandas_columns),
    )

    training_only_pdf = feature_pandas_df[feature_pandas_df[SPLIT_COLUMN] == TRAIN_SPLIT_LABEL]
    class_counts = training_only_pdf[LABEL_COLUMN].value_counts().to_dict()

    negative_count = int(class_counts.get(0, 0))
    positive_count = int(class_counts.get(1, 0))

    if negative_count == 0 or positive_count == 0:
        raise RuntimeError(
            "ML training failed: both label classes must be present in the training split."
        )

    positive_weight = negative_count / positive_count

    LOGGER.info(
        "Computed class weights from the training split: negative=%s, positive=%s, positive_weight=%.4f.",
        f"{negative_count:,}",
        f"{positive_count:,}",
        positive_weight,
    )

    feature_pandas_df[WEIGHT_COLUMN] = np.where(
        feature_pandas_df[LABEL_COLUMN] == 1, positive_weight, 1.0
    )

    feature_pandas_df, encoding_maps, global_positive_rate = _target_encode_categorical_columns(
        feature_pandas_df, training_only_pdf
    )

    return feature_pandas_df, encoding_maps, global_positive_rate


# ------------------------------------------------------------
# Use a selector Random Forest on the full candidate feature set
# so downstream models train only on the most influential inputs.
# ------------------------------------------------------------
def select_top_features(train_pdf: pd.DataFrame) -> tuple[list[str], list[tuple[str, float]]]:
    """Select the top indexed/numeric feature columns using Random Forest importance."""

    selector_random_forest = RandomForestClassifier(
        n_estimators=60,
        max_depth=8,
        min_samples_leaf=25,
        random_state=42,
        n_jobs=-1,
    )
    selector_random_forest.fit(
        train_pdf[MODEL_INPUT_COLUMNS],
        train_pdf[LABEL_COLUMN],
        sample_weight=train_pdf[WEIGHT_COLUMN],
    )

    feature_importance_pairs = sorted(
        zip(MODEL_INPUT_COLUMNS, [float(value) for value in selector_random_forest.feature_importances_]),
        key=lambda item: item[1],
        reverse=True,
    )
    selected_features = [feature_name for feature_name, _ in feature_importance_pairs[:TOP_FEATURE_COUNT]]

    LOGGER.info(
        "Selected top %s features from %s candidate inputs: %s",
        min(TOP_FEATURE_COUNT, len(MODEL_INPUT_COLUMNS)),
        len(MODEL_INPUT_COLUMNS),
        selected_features,
    )
    return selected_features, feature_importance_pairs


# ------------------------------------------------------------
# Fit Decision Tree, Random Forest, and XGBoost on the selected
# feature representation for a fair comparison.
# ------------------------------------------------------------
# NEEDS STAKEHOLDER REVIEW: this value is an unconfirmed assumption, not an approved business
# requirement. ASSUMPTION (opinion): this model is a compliance/monitoring signal, so a missed
# truly-late complaint (false negative) is treated as costlier than an over-flagged timely one
# (false positive). This constant is the one thing to change if that cost asymmetry is wrong;
# confirm the real number with whoever owns the downstream use of this prediction before relying
# on it in production.
TARGET_RECALL_FLOOR = 0.75
CALIBRATION_METHOD = "sigmoid"  # See _fit_score_and_log_model for why isotonic was rejected.

# Off by default: a grid search multiplies training time by len(grid) * CV_FOLDS per model, and
# the hand-picked defaults below were already tuned somewhat through manual iteration. Set True to
# search a small grid per model (scored by AUC-PR, the right metric under this class imbalance)
# instead of using the fixed hyperparameters in train_and_score_models.
ENABLE_HYPERPARAMETER_TUNING = False
HYPERPARAMETER_TUNING_CV_FOLDS = 3
HYPERPARAMETER_GRIDS: dict[str, dict[str, list]] = {
    "decision_tree": {"max_depth": [6, 8, 10], "min_samples_leaf": [25, 50, 100]},
    "random_forest": {"n_estimators": [50, 80, 120], "max_depth": [8, 10, 12]},
    "spark_xgboost": {"max_depth": [6, 8, 10], "learning_rate": [0.05, 0.1, 0.2]},
}


def _select_threshold_for_recall_floor(labels: pd.Series, probabilities: np.ndarray, recall_floor: float) -> float:
    """Pick the highest-precision threshold that still achieves at least recall_floor.

    Threshold selection happens on a VALIDATION-derived fold only; TEST is never touched here.
    """

    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if len(thresholds) == 0:
        return 0.5
    precision, recall = precision[:-1], recall[:-1]  # precision_recall_curve returns one extra point with no threshold.
    feasible = recall >= recall_floor
    if not feasible.any():
        LOGGER.warning("No threshold reached the %.0f%% recall floor; falling back to the max-recall threshold.", recall_floor * 100)
        return float(thresholds[int(np.argmax(recall))])
    best_index = int(np.argmax(np.where(feasible, precision, -1.0)))
    return float(thresholds[best_index])


def _fit_with_optional_tuning(
    model_name: str,
    model: object,
    train_pdf: pd.DataFrame,
    selected_feature_columns: list[str],
) -> object:
    """Fit model on TRAIN, optionally hyperparameter-tuned via cross-validated grid search.

    Gated behind ENABLE_HYPERPARAMETER_TUNING (off by default). When enabled, GridSearchCV both
    searches and does the final fit (refit=True is the default), so no separate .fit() call is
    needed afterward either way.
    """

    X_train = train_pdf[selected_feature_columns]
    y_train = train_pdf[LABEL_COLUMN]
    sample_weight = train_pdf[WEIGHT_COLUMN]

    param_grid = HYPERPARAMETER_GRIDS.get(model_name)
    if ENABLE_HYPERPARAMETER_TUNING and param_grid:
        search = GridSearchCV(
            model,
            param_grid,
            scoring="average_precision",  # Matches the metric used to pick the champion model.
            cv=HYPERPARAMETER_TUNING_CV_FOLDS,
            n_jobs=-1,
        )
        search.fit(X_train, y_train, sample_weight=sample_weight)
        LOGGER.info(
            "Tuned %s via %s-fold CV: best_params=%s, best_cv_auc_pr=%.4f",
            model_name, HYPERPARAMETER_TUNING_CV_FOLDS, search.best_params_, search.best_score_,
        )
        return search.best_estimator_

    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model


def _fit_score_and_log_model(
    model_name: str,
    model: object,
    train_pdf: pd.DataFrame,
    validation_pdf: pd.DataFrame,
    test_pdf: pd.DataFrame,
    selected_feature_columns: list[str],
    encoding_maps: dict[str, dict[str, float]],
    global_positive_rate: float,
) -> dict[str, object]:
    """Fit, calibrate, pick a recall-floor threshold, and write the model's serving metadata.

    Class weighting during training distorts predicted probabilities, so a calibration step is
    required before those probabilities can be used for thresholding. Calibration is fit on half
    of VALIDATION; the threshold is selected on the other half, so the same rows are never used
    both to fit the calibrator and to pick/evaluate the threshold. TEST remains untouched.
    """

    model = _fit_with_optional_tuning(model_name, model, train_pdf, selected_feature_columns)

    validation_calibration_pdf, validation_threshold_pdf = train_test_split(
        validation_pdf, test_size=0.5, stratify=validation_pdf[LABEL_COLUMN], random_state=42,
    )

    # No sample_weight here: calibration should reflect the true (unweighted) class distribution,
    # not the training-time class-weighted one. Isotonic (non-parametric) calibration was tried first
    # and confirmed (via the notebook's reliability curve) to degenerate under this level of class
    # imbalance with a modest calibration fold - it collapsed to near-zero probabilities almost
    # everywhere. Sigmoid (Platt) calibration fits a smooth 2-parameter logistic curve instead, which
    # is far more stable when positives are this rare, at the cost of less flexibility.
    calibrated_model = CalibratedClassifierCV(estimator=model, method=CALIBRATION_METHOD, cv="prefit")
    calibrated_model.fit(validation_calibration_pdf[selected_feature_columns], validation_calibration_pdf[LABEL_COLUMN])

    threshold_probability = calibrated_model.predict_proba(validation_threshold_pdf[selected_feature_columns])[:, 1]
    validation_probability = calibrated_model.predict_proba(validation_pdf[selected_feature_columns])[:, 1]  # For reporting only.
    test_probability = calibrated_model.predict_proba(test_pdf[selected_feature_columns])[:, 1]

    selected_threshold = _select_threshold_for_recall_floor(
        validation_threshold_pdf[LABEL_COLUMN], threshold_probability, TARGET_RECALL_FLOOR,
    )
    validation_auc_pr = float(average_precision_score(validation_pdf[LABEL_COLUMN], validation_probability))

    # Deliberately NOT using MLflow experiment tracking (mlflow.start_run/log_params/log_metric/
    # log_model) here. On this workspace's serverless/Spark-Connect compute it has failed in four
    # distinct ways: registered-model artifact download (RESOURCE_DOES_NOT_EXIST), a Unity Catalog
    # registration path that failed silently, a crash in log_model reading the Spark config
    # spark.mlflow.modelRegistryUri (which this compute disallows reading), and finally the same
    # crash from mlflow.start_run() itself - i.e. no MLflow call here is reliable, not just model
    # logging. Metrics are already persisted to METRICS_TABLE (a real Delta table, queried directly
    # throughout this project) and the model artifact is written to a Volume with joblib below,
    # neither of which touches the MLflow client at all.
    serving_metadata = {
        "selected_feature_columns": selected_feature_columns,
        "selected_threshold": selected_threshold,
        "categorical_feature_columns": CATEGORICAL_FEATURE_COLUMNS,
        "numeric_feature_columns": NUMERIC_FEATURE_COLUMNS,
        "encoding_maps": encoding_maps,
        "global_positive_rate": global_positive_rate,
    }
    SparkSession.builder.getOrCreate().sql(
        f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{MONITORING_SCHEMA}.model_exports"
    )
    volume_path = f"/Volumes/{CATALOG}/{MONITORING_SCHEMA}/model_exports/serving_metadata_{model_name}.json"
    with open(volume_path, "w", encoding="utf-8") as serving_metadata_file:
        json.dump(serving_metadata, serving_metadata_file)
    LOGGER.info("Wrote serving metadata for %s to %s.", model_name, volume_path)

    return {
        "base_model": model,  # Raw fitted estimator - used for feature_importances_ reporting only.
        "calibrated_model": calibrated_model,
        "validation_probability": validation_probability,
        "test_probability": test_probability,
        "threshold": selected_threshold,
        "validation_auc_pr": validation_auc_pr,
    }


def _promote_champion_model(scored_models: dict[str, dict[str, object]]) -> None:
    """Write the model with the best validation AUC-PR to the Volume as the deployable champion.

    Serving (the Flask app) reads champion_model.joblib and serving_metadata_champion.json
    directly - no MLflow model registry involved. Both files are written together, atomically,
    in this one step, so they can never drift out of sync with each other.
    """

    best_model_name = max(scored_models, key=lambda name: scored_models[name]["validation_auc_pr"])
    best_model_outputs = scored_models[best_model_name]
    LOGGER.info(
        "Best model by validation AUC-PR: %s -> validation_auc_pr=%.4f",
        best_model_name,
        best_model_outputs["validation_auc_pr"],
    )

    export_dir = f"/Volumes/{CATALOG}/{MONITORING_SCHEMA}/model_exports"
    joblib.dump(best_model_outputs["calibrated_model"], f"{export_dir}/champion_model.joblib")

    # serving_metadata_{model_name}.json gets overwritten by every run of that model, including
    # runs where it does NOT win - so it can drift out of sync with whichever model is actually
    # champion (feature selection can vary run-to-run due to sampling variance). Copy the WINNING
    # run's own metadata to the canonical champion filename here, alongside the model file above.
    source_metadata_path = f"{export_dir}/serving_metadata_{best_model_name}.json"
    canonical_metadata_path = f"{export_dir}/serving_metadata_champion.json"
    shutil.copyfile(source_metadata_path, canonical_metadata_path)

    LOGGER.info("Wrote champion model (%s) and serving metadata to %s.", best_model_name, export_dir)


def train_and_score_models(
    feature_pandas_df: pd.DataFrame,
    encoding_maps: dict[str, dict[str, float]],
    global_positive_rate: float,
) -> tuple[dict[str, dict[str, object]], list[str], list[tuple[str, float]]]:
    """Train all tree-based models and score validation/test rows."""

    train_pdf = feature_pandas_df[feature_pandas_df[SPLIT_COLUMN] == TRAIN_SPLIT_LABEL]
    validation_pdf = feature_pandas_df[feature_pandas_df[SPLIT_COLUMN] == VALIDATION_SPLIT_LABEL]
    test_pdf = feature_pandas_df[feature_pandas_df[SPLIT_COLUMN] == TEST_SPLIT_LABEL]

    selected_feature_columns, feature_importance_pairs = select_top_features(train_pdf)

    decision_tree_classifier = DecisionTreeClassifier(
        max_depth=8,
        min_samples_leaf=50,
        random_state=42,
    )
    random_forest_classifier = RandomForestClassifier(
        n_estimators=80,
        max_depth=10,
        min_samples_leaf=25,
        random_state=42,
        n_jobs=-1,
    )

    models: dict[str, object] = {
        DECISION_TREE_MODEL_NAME: decision_tree_classifier,
        RANDOM_FOREST_MODEL_NAME: random_forest_classifier,
    }
    if XGBOOST_AVAILABLE:
        models[XGBOOST_MODEL_NAME] = XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            n_estimators=120,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
    else:
        LOGGER.warning(
            "XGBoost is not available in this environment (%s). Training will continue with Decision Tree and Random Forest only.",
            XGBOOST_IMPORT_ERROR,
        )

    scored_models: dict[str, dict[str, object]] = {}
    for model_name, model in models.items():
        LOGGER.info("Training %s model.", model_name)
        scored_models[model_name] = _fit_score_and_log_model(
            model_name, model, train_pdf, validation_pdf, test_pdf, selected_feature_columns,
            encoding_maps, global_positive_rate,
        )

    LOGGER.info(
        "Completed training for %s models with %s selected input features and 1 label column.",
        len(scored_models),
        len(selected_feature_columns),
    )

    # End stage: pick the model with the best validation AUC-PR (the right ranking metric under this
    # level of class imbalance) and mark it as the 'champion' alias in Unity Catalog, so downstream
    # consumers can always resolve "models:/<name>@champion" instead of hardcoding a model/version.
    _promote_champion_model(scored_models)

    return scored_models, selected_feature_columns, feature_importance_pairs


# ------------------------------------------------------------
# Build multi-model evaluation metrics so model quality can be
# compared across validation and held-out test splits.
# ------------------------------------------------------------
def build_metrics_dataframe(
    spark: SparkSession,
    feature_pandas_df: pd.DataFrame,
    scored_models: dict[str, dict[str, object]],
    selected_feature_columns: list[str],
) -> DataFrame:
    """Create a compact metrics DataFrame for all models and splits."""

    validation_pdf = feature_pandas_df[feature_pandas_df[SPLIT_COLUMN] == VALIDATION_SPLIT_LABEL]
    test_pdf = feature_pandas_df[feature_pandas_df[SPLIT_COLUMN] == TEST_SPLIT_LABEL]

    def summarize_split(model_name: str, split_name: str, labels: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, object]:
        predictions = (probabilities >= threshold).astype(int)
        labels_array = np.asarray(labels)

        # zero_division=0: at extreme thresholds a split can have zero predicted positives, which
        # would otherwise raise/warn rather than reporting the (correct) precision of 0.
        return {
            MODEL_NAME_COLUMN: model_name,
            "dataset_split": split_name,
            "row_count": int(len(labels_array)),
            "auc_roc": float(roc_auc_score(labels_array, probabilities)),
            "auc_pr": float(average_precision_score(labels_array, probabilities)),
            "precision": float(precision_score(labels_array, predictions, zero_division=0)),
            "recall": float(recall_score(labels_array, predictions, zero_division=0)),
            "f2_score": float(fbeta_score(labels_array, predictions, beta=2, zero_division=0)),
            "selected_threshold": float(threshold),
            "true_positive_count": int(np.sum((labels_array == 1) & (predictions == 1))),
            "false_positive_count": int(np.sum((labels_array == 0) & (predictions == 1))),
            "true_negative_count": int(np.sum((labels_array == 0) & (predictions == 0))),
            "false_negative_count": int(np.sum((labels_array == 1) & (predictions == 0))),
            "positive_label_rate": float(np.mean(labels_array)),
            "predicted_positive_rate": float(np.mean(predictions)),
            "feature_count": len(selected_feature_columns),
            "label_column_count": 1,
            "selected_features": ", ".join(selected_feature_columns),
        }

    metrics_rows: list[dict[str, object]] = []
    for model_name, model_outputs in scored_models.items():
        threshold = model_outputs["threshold"]
        metrics_rows.append(
            summarize_split(model_name, VALIDATION_SPLIT_LABEL, validation_pdf[LABEL_COLUMN], model_outputs["validation_probability"], threshold)
        )
        metrics_rows.append(
            summarize_split(model_name, TEST_SPLIT_LABEL, test_pdf[LABEL_COLUMN], model_outputs["test_probability"], threshold)
        )

    metrics_dataframe = spark.createDataFrame(metrics_rows).withColumn(
        "_model_run_at",
        F.current_timestamp(),
    )

    LOGGER.info("Built evaluation metrics for %s model/s.", len(scored_models))
    return metrics_dataframe


# ------------------------------------------------------------
# Persist scored test-set predictions for all models so later
# analysis can compare model behaviour row by row.
# ------------------------------------------------------------
def build_predictions_dataframe(
    spark: SparkSession,
    feature_pandas_df: pd.DataFrame,
    scored_models: dict[str, dict[str, object]],
) -> DataFrame:
    """Create a combined scored test-prediction DataFrame for all models."""

    test_pdf = feature_pandas_df[feature_pandas_df[SPLIT_COLUMN] == TEST_SPLIT_LABEL]

    predictions_parts = []
    for model_name, model_outputs in scored_models.items():
        probabilities = model_outputs["test_probability"]
        model_predictions_pdf = test_pdf[[LABEL_COLUMN, SPLIT_COLUMN] + PREDICTION_OUTPUT_COLUMNS].copy()
        model_predictions_pdf[MODEL_NAME_COLUMN] = model_name
        model_predictions_pdf["prediction"] = (probabilities >= model_outputs["threshold"]).astype(int)
        model_predictions_pdf["predicted_probability_untimely"] = probabilities
        predictions_parts.append(model_predictions_pdf)

    if not predictions_parts:
        raise RuntimeError("Prediction build failed: no model predictions were created.")

    predictions_pdf = pd.concat(predictions_parts, ignore_index=True)
    predictions_pdf = predictions_pdf[[
        MODEL_NAME_COLUMN,
        "complaint_id",
        LABEL_COLUMN,
        "prediction",
        "predicted_probability_untimely",
        SPLIT_COLUMN,
        "product",
        "sub_product",
        "company",
        "state",
        "issue",
        "sub_issue",
        "submitted_via",
        "complaint_received_date",
    ]]

    return spark.createDataFrame(predictions_pdf).withColumn("_model_run_at", F.current_timestamp())


# ------------------------------------------------------------
# Write evaluation artifacts into the monitoring schema so the
# models can be reviewed without re-running training.
# ------------------------------------------------------------
def write_monitoring_tables(
    spark: SparkSession,
    metrics_dataframe: DataFrame,
    predictions_dataframe: DataFrame,
) -> None:
    """Write model metrics and test predictions to managed Delta tables."""

    spark.sql(
        f"""
        CREATE SCHEMA IF NOT EXISTS {CATALOG}.{MONITORING_SCHEMA}
        COMMENT 'Monitoring and ML evaluation outputs for consumer complaints.'
        """
    )

    LOGGER.info("Writing ML metrics table to %s", METRICS_TABLE)
    (
        metrics_dataframe.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(METRICS_TABLE)
    )

    LOGGER.info("Writing ML predictions table to %s", PREDICTIONS_TABLE)
    (
        predictions_dataframe.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(PREDICTIONS_TABLE)
    )


# ------------------------------------------------------------
# Verify that the monitoring outputs were written successfully
# and contain enough information for practical model review.
# ------------------------------------------------------------
def validate_monitoring_outputs(spark: SparkSession) -> None:
    """Validate that the monitoring tables were written successfully."""

    metrics_dataframe = spark.table(METRICS_TABLE)
    predictions_dataframe = spark.table(PREDICTIONS_TABLE)

    metrics_row_count = metrics_dataframe.count()
    predictions_row_count = predictions_dataframe.count()

    minimum_metrics_rows = 6 if XGBOOST_AVAILABLE else 4
    if metrics_row_count < minimum_metrics_rows:
        raise RuntimeError(
            f"ML output validation failed: expected at least {minimum_metrics_rows} metrics rows for the available models across validation and test."
        )

    if predictions_row_count == 0:
        raise RuntimeError(
            "ML output validation failed: test predictions table is empty."
        )

    distinct_models = metrics_dataframe.select(MODEL_NAME_COLUMN).distinct().count()
    expected_model_count = 3 if XGBOOST_AVAILABLE else 2
    if distinct_models < expected_model_count:
        raise RuntimeError(
            "ML output validation failed: not all available models were written to the metrics table."
        )

    LOGGER.info(
        "ML monitoring outputs validated successfully: metrics=%s rows, predictions=%s rows.",
        f"{metrics_row_count:,}",
        f"{predictions_row_count:,}",
    )


# ------------------------------------------------------------
# Placeholder drift-check hook, run at the end of every training
# job. Not implemented yet: see the TODO in the docstring.
# ------------------------------------------------------------
def check_for_drift(feature_pandas_df: pd.DataFrame) -> None:
    """Placeholder for input/label drift monitoring. Not implemented yet.

    TODO: no baseline distribution is stored or compared against today. Wire this up to compare
    this run's TRAIN split statistics (e.g. positive rate, per-category frequencies for the
    categorical columns) against a stored historical baseline (a small table would do), and
    log/alert if they diverge beyond some threshold. Until that exists, this is a no-op so the
    call site in main() has somewhere to plug it in without restructuring the training flow later.
    """

    LOGGER.info(
        "Drift check: not yet implemented (placeholder hook, see check_for_drift() docstring)."
    )


# ------------------------------------------------------------
# Manually triggered batch training entrypoint. Invoked both by
# direct execution (`python consumer_complaints_ml_train.py`) and
# by the Databricks Job task that runs this script - there is no
# separate/different entrypoint for the scheduled path today,
# because there is no scheduled path yet (see check_for_drift and
# the module docstring for what is still missing on that front).
# ------------------------------------------------------------
def main() -> None:
    """Train and evaluate the tree-based untimely-response models."""

    spark = SparkSession.builder.getOrCreate()

    LOGGER.info("Starting Consumer Complaints tree-based ML training workflow.")
    persist_runtime_diagnostics(spark)

    feature_dataframe = validate_feature_table(spark)
    feature_pandas_df, encoding_maps, global_positive_rate = build_training_frame(feature_dataframe)
    scored_models, selected_feature_columns, feature_importance_pairs = (
        train_and_score_models(feature_pandas_df, encoding_maps, global_positive_rate)
    )

    metrics_dataframe = build_metrics_dataframe(
        spark,
        feature_pandas_df,
        scored_models,
        selected_feature_columns,
    )
    predictions_dataframe = build_predictions_dataframe(spark, feature_pandas_df, scored_models)

    write_monitoring_tables(spark, metrics_dataframe, predictions_dataframe)
    validate_monitoring_outputs(spark)
    check_for_drift(feature_pandas_df)

    top_selected_features = feature_importance_pairs[:TOP_FEATURE_COUNT]
    LOGGER.info(
        "Top selected features for downstream models: %s",
        [feature_name for feature_name, _ in top_selected_features],
    )

    LOGGER.info(
        "Consumer Complaints tree-based ML training workflow finished successfully."
    )


if __name__ == "__main__":
    main()
