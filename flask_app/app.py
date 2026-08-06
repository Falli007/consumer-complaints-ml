"""Flask app that scores a consumer complaint for untimely-response risk.

Loads the champion model written by the Databricks training pipeline
(see databricks/src/ml/consumer_complaints_ml_train.py, _promote_champion_model)
plus its serving_metadata.json sidecar, which records everything needed to
turn a raw complaint into the encoded feature vector the model expects:
target-encoding maps for categoricals, the selected feature columns, and
the validation-tuned decision threshold. Both files are plain joblib/JSON -
no MLflow model registry involved, so there is nothing beyond scikit-learn/
xgboost required to load them here.
"""

from __future__ import annotations

import json
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "champion_model.joblib")
SERVING_METADATA_PATH = os.path.join(os.path.dirname(__file__), "model", "serving_metadata.json")

app = Flask(__name__)

# Loaded lazily and cached at module scope rather than at import time, so a missing/corrupt model
# file surfaces as a normal request error (caught by /health, or a 500 from /predict) instead of
# crashing the whole process on startup before Flask even has a chance to report anything.
_model = None
_serving_metadata = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def get_serving_metadata() -> dict:
    global _serving_metadata
    if _serving_metadata is None:
        with open(SERVING_METADATA_PATH, encoding="utf-8") as f:
            _serving_metadata = json.load(f)
    return _serving_metadata


def raw_input_fields(serving_metadata: dict) -> list[dict]:
    """Derive the raw fields the UI/API caller must supply from selected_feature_columns.

    Only the ~10 features the champion model actually uses are requested, not the full
    18-column candidate set - selected_feature_columns is the source of truth for that.
    """

    categorical_columns = set(serving_metadata["categorical_feature_columns"])
    fields = []
    for column_name in serving_metadata["selected_feature_columns"]:
        if column_name.endswith("_indexed") and column_name[: -len("_indexed")] in categorical_columns:
            raw_name = column_name[: -len("_indexed")]
            fields.append({"name": raw_name, "type": "categorical"})
        else:
            fields.append({"name": column_name, "type": "numeric"})
    return fields


def encode_request(raw_complaint: dict, serving_metadata: dict) -> pd.DataFrame:
    """Apply the same target-encoding used during training to a raw complaint payload."""

    encoding_maps = serving_metadata["encoding_maps"]
    global_positive_rate = serving_metadata["global_positive_rate"]
    categorical_columns = serving_metadata["categorical_feature_columns"]
    numeric_columns = serving_metadata["numeric_feature_columns"]
    selected_feature_columns = serving_metadata["selected_feature_columns"]

    encoded_row = {}
    for column_name in categorical_columns:
        raw_value = str(raw_complaint.get(column_name, ""))
        column_map = encoding_maps.get(column_name, {})
        # Unseen categories fall back to the global TRAIN positive rate, mirroring training-time behaviour.
        encoded_row[f"{column_name}_indexed"] = column_map.get(raw_value, global_positive_rate)
    for column_name in numeric_columns:
        # HTML form submissions arrive as strings (FormData has no concept of numeric types);
        # JSON API callers may already send real numbers. Cast explicitly either way - XGBoost
        # rejects object-dtype columns outright rather than coercing them.
        raw_value = raw_complaint.get(column_name, 0)
        encoded_row[column_name] = float(raw_value) if raw_value not in (None, "") else 0.0

    row_df = pd.DataFrame([encoded_row])
    return row_df[selected_feature_columns]


@app.route("/health", methods=["GET"])
def health():
    # Only checks that serving_metadata.json parses, not that the (larger, slower-to-load) model
    # file is valid - good enough as a "is this deployment even pointed at real artifacts" check
    # without paying the joblib.load cost on every health probe.
    try:
        get_serving_metadata()
        return jsonify({"status": "ok"})
    except Exception as error:  # noqa: BLE001 - surface whatever prevents the app from serving, not just an expected subset.
        return jsonify({"status": "error", "detail": str(error)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    # silent=True: a malformed/missing JSON body becomes {} (which encode_request then fills with
    # fallback values) rather than Flask raising its own 400 before we get a chance to respond.
    payload = request.get_json(force=True, silent=True) or {}
    serving_metadata = get_serving_metadata()
    model = get_model()

    row_df = encode_request(payload, serving_metadata)
    probability_untimely = float(model.predict_proba(row_df)[:, 1][0])
    threshold = serving_metadata["selected_threshold"]
    # >=, not >: matches the comparison used when the threshold was selected during training
    # (_select_threshold_for_recall_floor), so the reported recall/precision trade-off holds here too.
    prediction = int(probability_untimely >= threshold)

    return jsonify(
        {
            "predicted_probability_untimely": probability_untimely,
            "prediction": prediction,
            "prediction_label": "untimely" if prediction else "timely",
            "threshold_used": threshold,
        }
    )


@app.route("/", methods=["GET"])
def index():
    # raw_input_fields drives which inputs the form asks for, so the UI never falls out of sync
    # with whichever ~10 features the currently-loaded champion model actually needs.
    serving_metadata = get_serving_metadata()
    return render_template("index.html", fields=raw_input_fields(serving_metadata))


if __name__ == "__main__":
    # debug=True enables the auto-reloader and interactive debugger - convenient for local testing,
    # but this is the Werkzeug dev server: not meant to be exposed beyond localhost as-is.
    app.run(host="0.0.0.0", port=5000, debug=True)
