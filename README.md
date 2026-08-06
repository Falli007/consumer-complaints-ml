# Consumer Complaints Lakehouse and ML Pipeline

This repository ingests the CFPB Consumer Complaint Database, builds a full Bronze, Silver and
Gold lakehouse on Databricks with Unity Catalog, trains a tree-based machine learning model to
predict untimely complaint responses, and serves that model through a local Flask app.

It is organized as four connected stages:

1. Ingestion: download the raw CFPB dataset and upload it to Amazon S3.
2. Data engineering: Bronze, Silver and Gold pipelines on Databricks, built with PySpark and
   Delta Lake, orchestrated as a Databricks Job through a Databricks Asset Bundle.
3. Machine learning: a Gold-layer feature table, then a training pipeline that compares Decision
   Tree, Random Forest and XGBoost models and saves the best one.
4. Serving: a small Flask app that loads the trained model and scores a single complaint through
   a web form or a JSON API.

## Table of contents

- [Architecture overview](#architecture-overview)
- [1. Ingestion](#1-ingestion)
- [2. Data engineering: Bronze, Silver, Gold](#2-data-engineering-bronze-silver-gold)
- [3. Machine learning pipeline](#3-machine-learning-pipeline)
- [4. Model serving: Flask app](#4-model-serving-flask-app)
- [Databricks Asset Bundle](#databricks-asset-bundle)
- [Repository structure](#repository-structure)
- [Setup](#setup)
- [Known limitations and next steps](#known-limitations-and-next-steps)

## Architecture overview

```
CFPB dataset (S3)
      |
      v
Bronze table (fintech_lakehouse_dev.bronze)      raw data, as ingested
      |
      v
Silver table (fintech_lakehouse_dev.silver)      cleaned, validated, deduplicated
      |
      v
Gold layer (fintech_lakehouse_dev.gold)          dimensional warehouse: fact + dim tables
      |
      v
ML feature table (fintech_lakehouse_dev.gold)    complaint-level model input table
      |
      v
Training pipeline (pandas / scikit-learn / XGBoost, run on Databricks)
      |
      v
Champion model (Unity Catalog Volume: fintech_lakehouse_dev.monitoring.model_exports)
      |
      v
Flask app (local)                                 scores one complaint at a time
```

Everything upstream of the Flask app runs on Databricks, orchestrated as Databricks Jobs defined
in a Databricks Asset Bundle (`databricks.yml`). The Flask app is the only part of this project
that runs outside Databricks.

## 1. Ingestion

Location: `ingestion/consumer_complaints/download_to_s3.py`, tested in `tests/`.

A standalone script that downloads the CFPB Consumer Complaint Database ZIP file and uploads it
to an S3 bucket using multipart transfer, then verifies the uploaded object. It generates the
ingestion date at runtime from the current UTC date and uses it in both the S3 key and the S3
object metadata, so every run is timestamped and traceable.

Run it with:

```powershell
$env:S3_BUCKET_NAME = "your-bucket-name"
python ingestion\consumer_complaints\download_to_s3.py
```

Downloaded data is stored locally under `data/` before upload; that directory, along with ZIP
files, virtual environments and Python cache files, is excluded from Git.

## 2. Data engineering: Bronze, Silver, Gold

All data engineering code lives under `databricks/`, with a parallel `.py` script and `.ipynb`
notebook for every stage. The `.py` files are the production entry points wired into the
Databricks Job; the notebooks mirror the same logic for interactive exploration and review.

Catalog used throughout: `fintech_lakehouse_dev`.

### Bronze (`databricks/src/bronze/consumer_complaints_bronze.py`)

Reads the raw CFPB CSV from S3 and writes it, largely unchanged, into
`fintech_lakehouse_dev.bronze.consumer_complaints` as a managed Delta table. This is the
"as-ingested" copy of the source data: minimal transformation, so the original data is always
recoverable if a downstream bug is found later.

### Silver (`databricks/src/silver/consumer_complaints_silver.py`)

Reads from the Bronze table, applies cleaning, type casting, deduplication and data quality
checks, and writes `fintech_lakehouse_dev.silver.silver_consumer_complaints`. SQL validation
scripts for this stage live in `databricks/sql/silver/`:

- `01_bronze_profile.sql`: profiles the raw Bronze table before cleaning.
- `02_data_quality_checks.sql`: checks for nulls, duplicates and invalid values.
- `03_silver_validation.sql`: validates the cleaned Silver output.

### Gold (`databricks/src/gold/consumer_complaints_gold.py`)

Builds a dimensional warehouse on top of Silver: a fact table plus supporting dimension tables,
all under `fintech_lakehouse_dev.gold`:

- `fact_consumer_complaints`: one row per complaint, with foreign keys into the dimension tables.
- `dim_date`: calendar dimension for complaint intake dates.
- `dim_product`: complaint product/sub-product categories.
- `dim_company`: companies complaints were filed against.
- `dim_state`: US state/territory dimension.

Validation for this layer lives in `databricks/sql/gold/01_gold_validation.sql`. Screenshots of
each dimension table and the resulting output are under `docs/screenshots/gold/`.

## 3. Machine learning pipeline

The ML pipeline has two stages, both under `databricks/notebooks/` (interactive) and
`databricks/src/ml/` (the production entry points).

### Feature engineering (`consumer_complaints_ml_features`)

Reads the Gold fact and dimension tables, joins them into a single complaint-level table, derives
the binary target label (`label_untimely_response`, whether a complaint's response was late), and
writes a deterministic TRAIN/VALIDATION/TEST split into a managed feature table:
`fintech_lakehouse_dev.gold.consumer_complaints_timely_response_features`. This step is pure
Spark SQL/DataFrame work and runs entirely on Databricks Spark, since it operates over the full
complaint volume.

### Training (`consumer_complaints_ml_train`)

Reads the feature table and trains a baseline classifier to predict untimely responses. This step
runs on pandas, scikit-learn and XGBoost rather than Spark ML: this workspace's serverless
compute does not reliably support classic `pyspark.ml` (constructor whitelisting issues on some
environment versions, no Spark JVM context at all on others, and Spark Connect ML session cache
limits on the one version where it does run), and the reduced, already-aggregated feature table
is small enough to train comfortably on the driver instead.

The training flow, in order:

1. Validate the feature table exists and all three splits (TRAIN/VALIDATION/TEST) are non-empty.
2. Collect the model-input columns into pandas, capped at 500,000 rows per split (a representative
   sample is standard practice for a baseline tree model; multi-million-row `toPandas()` transfers
   were also found to be unreliable on this workspace at full volume).
3. Compute a class weight from the TRAIN split to address the rare positive class (roughly 0.6
   percent of complaints are untimely).
4. Target-encode the ten categorical columns (company, product, state, and so on) using each
   category's smoothed historical positive rate, rather than an arbitrary ordinal integer code.
   TRAIN rows are encoded out-of-fold (5-fold) so a row's own label never leaks into its own
   encoded value; VALIDATION and TEST are encoded using the full TRAIN split's statistics. This
   change alone roughly tripled AUC-PR across all three models compared to ordinal encoding,
   since ordinal codes falsely imply an order between unrelated categories like company names.
5. Use a Random Forest's feature importances to select the top 10 of 18 candidate input columns.
6. Train Decision Tree, Random Forest and XGBoost classifiers on the selected features, each with
   a sample weight column to account for class imbalance.
7. Calibrate each model's predicted probabilities. Class weighting during training distorts raw
   probabilities, so this step is required before using them for thresholding. Calibration is
   fit with Platt/sigmoid scaling on half of the VALIDATION split; isotonic calibration was tried
   first and rejected because it degenerated under this level of class imbalance with a modest
   calibration fold, collapsing to near-zero probabilities almost everywhere.
8. Select a decision threshold on the other half of VALIDATION, targeting a recall floor (75
   percent by default) rather than maximizing F1. F1 weights precision and recall equally, which
   is the wrong objective for a compliance-style signal where a missed untimely complaint is
   assumed to be costlier than an over-flagged timely one; this assumption is configurable via the
   `TARGET_RECALL_FLOOR` constant and should be confirmed against the real business cost of each
   error type before being trusted in production.
9. Evaluate every model on VALIDATION and TEST: AUC-ROC, AUC-PR, precision, recall, F2 and a full
   confusion matrix, all persisted to
   `fintech_lakehouse_dev.monitoring.consumer_complaints_timely_response_model_metrics`.
10. Persist scored TEST predictions to
    `fintech_lakehouse_dev.monitoring.consumer_complaints_timely_response_test_predictions`.
11. Pick the model with the best validation AUC-PR (AUC-PR, not AUC-ROC or accuracy, since it is
    the appropriate ranking metric under this level of class imbalance) and write it as the
    deployable champion: see [Model artifacts](#model-artifacts) below.

TEST is never touched for calibration, threshold selection or feature selection; it is only used
for final reporting, to keep the reported metrics honest.

### Model artifacts

The champion model and everything needed to serve it are written directly to a Unity Catalog
Volume, `fintech_lakehouse_dev.monitoring.model_exports`, using plain `joblib.dump()` and JSON
file writes:

- `champion_model.joblib`: the calibrated scikit-learn/XGBoost estimator.
- `serving_metadata_champion.json`: the serving contract, written atomically alongside the model
  in the same step so the two files can never drift out of sync. It records the selected feature
  columns, the target-encoding maps for each categorical column, the global fallback rate for
  unseen categories, and the selected decision threshold.
- `serving_metadata_<model_name>.json`: the same metadata for each individually trained model,
  kept for inspection even when that model is not the champion.

MLflow's experiment tracking and Unity Catalog model registry were tried first for this step and
found to be unreliable on this workspace's serverless compute: registered-model artifact download
failed, model registration failed silently under a broad exception handler, logging a model
crashed reading a Spark configuration value the compute does not allow reading, and eventually
even the plain act of starting an MLflow run crashed the same way. None of this is a property of
MLflow generally; it reflects a specific incompatibility between MLflow's Databricks integration
and this workspace's serverless/Spark Connect compute. Writing plain files to a Volume avoided
all of it, since it is a normal filesystem write rather than an MLflow artifact-store operation.
Model evaluation metrics are still tracked in the Delta table described above, which was already
the actual source of truth used throughout this project.

## 4. Model serving: Flask app

Location: `flask_app/`. This is the only part of the project that runs outside Databricks.

The app loads `champion_model.joblib` and `serving_metadata.json` (copied down from the Volume,
see below) and exposes:

- `GET /`: a simple HTML form, dynamically built from `serving_metadata.json` so it only ever
  asks for the raw fields the current champion model actually needs.
- `POST /predict`: accepts a JSON body of raw complaint fields, applies the same target encoding
  used during training, scores it with the model, and returns the predicted probability, the
  binary prediction and the threshold that was applied.
- `GET /health`: reports whether the model and serving metadata loaded successfully.

Because the model only understands already-encoded numbers, `encode_request()` in `app.py`
re-applies the exact target-encoding maps recorded in `serving_metadata.json` to any raw
complaint before scoring it; an unseen category (a company the model never saw in training, for
example) falls back to the global training-set positive rate rather than failing.

### Refreshing the model

After retraining on Databricks, pull the two artifact files down from the Volume:

```powershell
databricks fs cp "dbfs:/Volumes/fintech_lakehouse_dev/monitoring/model_exports/champion_model.joblib" flask_app/model/champion_model.joblib --profile consumer-complaints-dev --overwrite
databricks fs cp "dbfs:/Volumes/fintech_lakehouse_dev/monitoring/model_exports/serving_metadata_champion.json" flask_app/model/serving_metadata.json --profile consumer-complaints-dev --overwrite
```

### Running the app

```powershell
cd flask_app
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:5000` in a browser, or call `POST /predict` directly with a JSON
body. Full details and an example request are in `flask_app/README.md`.

## Databricks Asset Bundle

The whole Databricks side of this project (Bronze, Silver, Gold and the ML pipeline) is defined
as a single Databricks Asset Bundle, configured in `databricks.yml`, with job definitions under
`databricks/resources/`.

Two Databricks Jobs are defined:

- `consumer_complaints_bronze_job` (in `consumer_complaints_job.yml`... see the resources file for
  exact task names): runs the Bronze, Silver and Gold tasks in sequence.
- `consumer_complaints_ml_job`: builds the ML feature table, then runs the training pipeline.

Only `databricks/src/**/*.py` and `databricks/notebooks/**/*.ipynb` are synced to the Databricks
workspace; `flask_app/` and other local-only directories are explicitly excluded in
`databricks.yml`, since Flask never runs on Databricks and does not need to be uploaded there.

Common commands, using the `dev` target:

```powershell
databricks bundle validate --profile consumer-complaints-dev -t dev
databricks bundle deploy --profile consumer-complaints-dev -t dev
databricks bundle run consumer_complaints_bronze_job -t dev --profile consumer-complaints-dev
databricks bundle run consumer_complaints_ml_job -t dev --profile consumer-complaints-dev
```

## Repository structure

```
ingestion/consumer_complaints/       S3 ingestion script
tests/                               Unit tests for the ingestion script
databricks/
  notebooks/                         Interactive notebooks: bronze, silver, gold, ml_features, ml_train
  src/
    bronze/                          Bronze pipeline (production entry point)
    silver/                          Silver pipeline (production entry point)
    gold/                            Gold warehouse pipeline (production entry point)
    ml/                              ML feature engineering and training (production entry points)
  sql/
    silver/                          Silver profiling and validation SQL
    gold/                            Gold validation SQL
  resources/                         Databricks Job definitions (Asset Bundle resources)
flask_app/
  app.py                             Flask app: loads the model, serves predictions
  templates/index.html               Web form, generated from the model's serving metadata
  model/                             Local copy of the champion model and serving metadata (not in Git)
  requirements.txt                   Flask app dependencies
docs/screenshots/gold/               Screenshots of the Gold warehouse tables and app
databricks.yml                       Databricks Asset Bundle definition
```

## Setup

Three separate environments are involved, each with its own dependencies:

1. Ingestion (root `requirements.txt`): Python 3.11+, AWS credentials available to `boto3`, and
   the `S3_BUCKET_NAME` environment variable. See [Ingestion](#1-ingestion) above.
2. Databricks: no local Python environment is required to run the pipelines themselves; they run
   on Databricks compute. A local Databricks CLI profile is needed to deploy and trigger jobs (see
   [Databricks Asset Bundle](#databricks-asset-bundle) above).
3. Flask app (`flask_app/requirements.txt`): a separate virtual environment, since its dependency
   set (scikit-learn, XGBoost, joblib) is unrelated to the ingestion script's. See
   [Model serving](#4-model-serving-flask-app) above.

Run the ingestion test suite with:

```powershell
python -m unittest discover -s tests -v
```

## Known limitations and next steps

- The recall floor used for threshold selection (75 percent) is an assumption about the relative
  cost of a missed untimely complaint versus an over-flagged timely one; it has not been confirmed
  against the actual downstream use of this signal and should be revisited with a stakeholder.
- Precision at that recall level is low (roughly 4 to 16 percent depending on the model), which is
  mathematically inherent to catching most of a rare (0.6 percent) positive class, not a modeling
  bug; pushing both precision and recall up further requires a more discriminative model, not a
  different threshold.
- The complaint narrative text itself is not used as a feature, only a binary flag for whether one
  was provided. Adding real text features (for example, embeddings from a pretrained transformer)
  is a natural next step but requires new plumbing in the Silver/Gold layers to retain the raw
  text, plus new feature-engineering and training work; it was scoped out of this iteration.
- Hyperparameters for all three models are fixed by hand rather than tuned via cross-validation.
- There is no automated retraining schedule or drift monitoring; the model is a manually triggered
  batch training job today.
