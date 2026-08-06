# Progress

Handoff note. Verified against repo state on 2026-08-06, `main` (post commit `7c24825`, plus
uncommitted follow-on work below verified against a live Databricks run, job run 307098592749574).

## Complete

- Ingestion: `ingestion/consumer_complaints/download_to_s3.py`, tests in `tests/test_download_to_s3.py`.
- Bronze: `databricks/src/bronze/consumer_complaints_bronze.py` + `databricks/notebooks/consumer_complaints_bronze.ipynb`.
- Silver: `databricks/src/silver/consumer_complaints_silver.py` + notebook + validation SQL in `databricks/sql/silver/`.
- Gold warehouse: `databricks/src/gold/consumer_complaints_gold.py` + notebook + `databricks/sql/gold/01_gold_validation.sql`. Fact + 3 dim tables under `fintech_lakehouse_dev.gold`.
- ML feature table: `databricks/src/ml/consumer_complaints_ml_features.py` + notebook. Writes `fintech_lakehouse_dev.gold.consumer_complaints_timely_response_features` with TRAIN/VALIDATION/TEST split.
- ML training: `databricks/src/ml/consumer_complaints_ml_train.py` + notebook. Target-encodes categoricals (out-of-fold on TRAIN), trains Decision Tree/Random Forest/XGBoost on pandas+sklearn+xgboost (not Spark ML, see README for why), sigmoid-calibrates, picks a decision threshold, evaluates all 3, writes metrics to `fintech_lakehouse_dev.monitoring.consumer_complaints_timely_response_model_metrics`, and joblib-dumps the best model + `serving_metadata_champion.json` to the `fintech_lakehouse_dev.monitoring.model_exports` Volume.
- Flask serving app: `flask_app/app.py`, `flask_app/templates/index.html`. Loads the model from `flask_app/model/` (not in git), applies the same target-encoding to raw input, serves `/`, `/predict`, `/health`. Verified working locally end to end: health check, predict via curl (JSON) and via the HTML form (string-typed fields).
- Databricks Asset Bundle: `databricks.yml` + `databricks/resources/consumer_complaints_job.yml`. Both jobs (bronze/silver/gold pipeline, ml pipeline) deploy and run successfully as of the last verified run.
- Root `README.md`: full pipeline documentation, ingestion through serving.
- Optional hyperparameter tuning: `ENABLE_HYPERPARAMETER_TUNING` flag (default `False`) in `consumer_complaints_ml_train.py`/`.ipynb`. When on, `GridSearchCV` (3-fold, scored by AUC-PR) replaces the fixed hyperparameters per model via `_fit_with_optional_tuning`/`fit_with_optional_tuning`. Off-path verified against a live run (job 307098592749574); the on-path (`ENABLE_HYPERPARAMETER_TUNING = True`) has not been run end to end yet.
- Drift-check call site: `check_for_drift()` in `consumer_complaints_ml_train.py`, called at the end of `main()`; notebook has the equivalent placeholder print at the end of Step 10. Still a no-op, see below.
- `TARGET_RECALL_FLOOR` and the narrative-text gap now have explicit `NEEDS STAKEHOLDER REVIEW` / `TODO` comments at their definitions in both the `.py` and `.ipynb`.

## Stubbed or missing

- `flask_app/model/` is gitignored (binary artifacts). A fresh clone has no model and Flask will fail to start until the two `databricks fs cp` commands in `flask_app/README.md` are run manually. No automation for this step.
- No cross-validation for model comparison; single TRAIN/VALIDATION/TEST split only (hyperparameter tuning, when enabled, does use CV internally, but only for the grid search itself).
- Complaint narrative text is unused beyond a binary `has_consumer_narrative` flag. No text features. TODO comment in place; explicitly out of scope until Silver/Gold retain the raw text.
- `check_for_drift()` is a no-op placeholder: no baseline distribution is stored anywhere yet, nothing is actually compared. There is still no scheduler; `main()` (`.py`) / the notebook run top-to-bottom is the only entrypoint, and it is only ever triggered manually or by the Databricks Job.
- `TARGET_RECALL_FLOOR = 0.75` in `consumer_complaints_ml_train.py`/`.ipynb` is still an unconfirmed assumption about cost asymmetry, now explicitly flagged `NEEDS STAKEHOLDER REVIEW` in-line, not a stakeholder-approved number.
- No automated tests for the ML pipeline or the Flask app; only `tests/test_download_to_s3.py` exists (ingestion only). This includes no test coverage for the new tuning helper or drift-check hook.
- Flask app is the Werkzeug dev server (`debug=True`), no auth, not production-hardened.
- A leftover `champion_model/` directory sits in the `model_exports` Volume from an earlier MLflow-based export attempt that was abandoned. Unused, safe to delete, not cleaned up.
- `docs/`, `tests/`, `ingestion/` still sync to the Databricks workspace on every `bundle deploy` (only `flask_app/` was excluded). Harmless but unnecessary.

## Current failing tests or errors

None. `python -m unittest discover -s tests -v` passes 4/4:

```
test_build_s3_key_uses_runtime_date (test_download_to_s3.DownloadToS3Tests.test_build_s3_key_uses_runtime_date) ... ok
test_get_current_utc_date_returns_iso_date (test_download_to_s3.DownloadToS3Tests.test_get_current_utc_date_returns_iso_date) ... ok
test_load_config_requires_bucket_name (test_download_to_s3.DownloadToS3Tests.test_load_config_requires_bucket_name) ... ok
test_load_config_uses_dynamic_date_in_key (test_download_to_s3.DownloadToS3Tests.test_load_config_uses_dynamic_date_in_key) ... ok

Ran 4 tests in 0.005s

OK
```

No test coverage exists for `databricks/src/ml/` or `flask_app/`, so this does not mean those are verified beyond the manual end-to-end checks noted above.

## Next concrete step

Add a smoke test asserting `champion_model.feature_names_in_ == serving_metadata["selected_feature_columns"]` (see `flask_app/app.py`). This exact mismatch happened once already during development (a stale Volume file from a losing model run) and was only caught by hand; it should be caught automatically before it reaches Flask again.
