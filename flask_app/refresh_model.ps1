<#
.SYNOPSIS
Pulls the current champion model and its serving metadata down from the Unity Catalog
Volume into flask_app/model/, so the local Flask app can serve whatever was most
recently trained on Databricks.

.PARAMETER Profile
Databricks CLI profile to use. Defaults to consumer-complaints-dev.
#>

param(
    [string]$Profile = "consumer-complaints-dev"  # which Databricks CLI profile to authenticate with
)

$ErrorActionPreference = "Stop"  # any databricks fs cp failure should stop the script, not continue silently

$modelDir = Join-Path $PSScriptRoot "model"  # flask_app/model/, resolved relative to this script's own location
if (-not (Test-Path $modelDir)) {
    New-Item -ItemType Directory -Path $modelDir | Out-Null  # create it on a fresh clone, where it doesn't exist yet
}

Write-Host "Pulling champion_model.joblib..."
databricks fs cp "dbfs:/Volumes/fintech_lakehouse_dev/monitoring/model_exports/champion_model.joblib" `
    (Join-Path $modelDir "champion_model.joblib") --profile $Profile --overwrite  # the calibrated estimator

Write-Host "Pulling serving_metadata_champion.json..."
databricks fs cp "dbfs:/Volumes/fintech_lakehouse_dev/monitoring/model_exports/serving_metadata_champion.json" `
    (Join-Path $modelDir "serving_metadata.json") --profile $Profile --overwrite  # renamed to match what app.py loads

Write-Host "Model refreshed. Restart the Flask app to pick up the new files."
