# Consumer Complaints Untimely-Response Risk — Flask App

Scores a single consumer complaint for the risk of an untimely CFPB response, using the
champion model trained by the Databricks pipeline in `../databricks/src/ml/consumer_complaints_ml_train.py`.

## What's in `model/`

The Databricks training pipeline (`_promote_champion_model` in `consumer_complaints_ml_train.py`)
writes both files directly to a Unity Catalog Volume the moment it picks a champion - no MLflow
model registry involved:

- `model/champion_model.joblib` — the calibrated scikit-learn/XGBoost model, loaded with `joblib.load`.
- `model/serving_metadata.json` — the serving contract: which raw fields to collect, the
  target-encoding maps for categorical fields (company, product, etc.), and the validation-tuned
  decision threshold. The model itself only understands already-encoded numbers, so this file is
  required to score a brand-new, raw complaint.

To refresh both after retraining, copy them down from the Volume:

```powershell
databricks fs cp "dbfs:/Volumes/fintech_lakehouse_dev/monitoring/model_exports/champion_model.joblib" flask_app/model/champion_model.joblib --profile consumer-complaints-dev --overwrite
databricks fs cp "dbfs:/Volumes/fintech_lakehouse_dev/monitoring/model_exports/serving_metadata_champion.json" flask_app/model/serving_metadata.json --profile consumer-complaints-dev --overwrite
```

## Setup

```powershell
cd flask_app
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
python app.py
```

Then open http://localhost:5000 in a browser, or call the API directly:

```powershell
curl -X POST http://localhost:5000/predict `
  -H "Content-Type: application/json" `
  -d '{\"product\": \"Credit reporting\", \"company\": \"EQUIFAX, INC.\", \"sub_product\": \"Credit reporting\", \"issue\": \"Incorrect information on your report\", \"sub_issue\": \"Information belongs to someone else\", \"complaint_received_year\": 2024, \"has_tags\": 0, \"has_consumer_narrative\": 1, \"zip_code_prefix3\": \"300\", \"complaint_received_day\": 15}'
```

`GET /health` reports whether the model and serving metadata loaded successfully.
