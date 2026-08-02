-- Consumer Complaints Silver validation
-- Project: consumer-complaints-ml
-- Layer: Post-transformation Silver validation
-- Bronze table: fintech_lakehouse_dev.bronze.bronze_consumer_complaints
-- Silver table: fintech_lakehouse_dev.silver.silver_consumer_complaints

-- ==========================================================
-- Check: Silver table existence and schema registration
--
-- Purpose:
-- Confirm that the Silver table exists in Unity Catalog before deeper
-- validation is performed.
--
-- Why this matters:
-- A successful job run should register a queryable managed Delta table in the
-- expected catalog and schema.
--
-- Silver action:
-- If this query returns no rows, the Silver transformation did not publish to
-- the expected managed location and the job configuration should be reviewed.
-- ==========================================================
SELECT
  table_catalog,
  table_schema,
  table_name,
  table_type
FROM fintech_lakehouse_dev.information_schema.tables
WHERE table_schema = 'silver'
  AND table_name = 'silver_consumer_complaints';


-- ==========================================================
-- Check: Bronze versus Silver row comparison
--
-- Purpose:
-- Compare total Bronze and Silver row counts in one result set.
--
-- Why this matters:
-- Silver should normally be less than or equal to Bronze after invalid records
-- are removed and duplicates are collapsed.
--
-- Silver action:
-- Investigate if Silver is unexpectedly larger than Bronze or if the reduction
-- from Bronze to Silver is materially different from the duplicate/invalid-ID
-- profile observed earlier.
-- ==========================================================
WITH bronze_stats AS (
  SELECT COUNT(*) AS bronze_row_count
  FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints
),
silver_stats AS (
  SELECT COUNT(*) AS silver_row_count
  FROM fintech_lakehouse_dev.silver.silver_consumer_complaints
)
SELECT
  b.bronze_row_count,
  s.silver_row_count,
  b.bronze_row_count - s.silver_row_count AS rows_removed_in_silver
FROM bronze_stats b
CROSS JOIN silver_stats s;


-- ==========================================================
-- Check: Silver schema validation
--
-- Purpose:
-- Review the Silver schema, including the expected metadata and Silver audit
-- columns.
--
-- Why this matters:
-- This confirms that complaint_id was cast appropriately, zip_code remains a
-- string, and Silver lineage fields were added.
--
-- Silver action:
-- Validate that cleaned business fields and audit columns are present before
-- downstream reporting or ML use.
-- ==========================================================
SELECT
  ordinal_position,
  column_name,
  data_type,
  is_nullable
FROM fintech_lakehouse_dev.information_schema.columns
WHERE table_schema = 'silver'
  AND table_name = 'silver_consumer_complaints'
ORDER BY ordinal_position;


-- ==========================================================
-- Check: complaint_id uniqueness in Silver
--
-- Purpose:
-- Confirm that Silver contains exactly one row per complaint_id.
--
-- Why this matters:
-- The core Silver grain is one complaint per complaint_id after deduplication.
--
-- Silver action:
-- If duplicates remain, the deduplication window or ordering logic must be
-- corrected before the table is used downstream.
-- ==========================================================
SELECT
  COUNT(*) AS duplicate_complaint_id_groups
FROM (
  SELECT
    complaint_id
  FROM fintech_lakehouse_dev.silver.silver_consumer_complaints
  GROUP BY complaint_id
  HAVING COUNT(*) > 1
) duplicate_ids;


-- ==========================================================
-- Check: Null validation for complaint_id and core business fields
--
-- Purpose:
-- Measure remaining nulls in key Silver fields after cleaning.
--
-- Why this matters:
-- complaint_id should be fully populated in Silver, while the remaining null
-- rates on business fields give an engineer a realistic view of source quality
-- after standardisation.
--
-- Silver action:
-- Confirm that invalid complaint IDs were removed and track any still-missing
-- descriptive fields for later business remediation rather than silent drops.
-- ==========================================================
SELECT
  COUNT(*) AS silver_row_count,
  COUNT_IF(complaint_id IS NULL) AS null_complaint_id_count,
  COUNT_IF(date_received IS NULL) AS null_date_received_count,
  COUNT_IF(product IS NULL) AS null_product_count,
  COUNT_IF(issue IS NULL) AS null_issue_count,
  COUNT_IF(company IS NULL) AS null_company_count,
  COUNT_IF(state IS NULL) AS null_state_count
FROM fintech_lakehouse_dev.silver.silver_consumer_complaints;


-- ==========================================================
-- Check: Blank-string validation after Silver cleaning
--
-- Purpose:
-- Ensure that key Silver string columns no longer contain blank or
-- whitespace-only values.
--
-- Why this matters:
-- Blank strings should have been converted to null, otherwise downstream
-- analysts will still see phantom categories and misleading completeness.
--
-- Silver action:
-- If any counts are non-zero, revisit TRIM/blank-to-null handling in the
-- Silver transformation.
-- ==========================================================
SELECT 'product' AS column_name, COUNT(*) AS remaining_blank_count
FROM fintech_lakehouse_dev.silver.silver_consumer_complaints
WHERE product IS NOT NULL AND TRIM(product) = ''
UNION ALL
SELECT 'sub_product', COUNT(*)
FROM fintech_lakehouse_dev.silver.silver_consumer_complaints
WHERE sub_product IS NOT NULL AND TRIM(sub_product) = ''
UNION ALL
SELECT 'issue', COUNT(*)
FROM fintech_lakehouse_dev.silver.silver_consumer_complaints
WHERE issue IS NOT NULL AND TRIM(issue) = ''
UNION ALL
SELECT 'company', COUNT(*)
FROM fintech_lakehouse_dev.silver.silver_consumer_complaints
WHERE company IS NOT NULL AND TRIM(company) = ''
UNION ALL
SELECT 'state', COUNT(*)
FROM fintech_lakehouse_dev.silver.silver_consumer_complaints
WHERE state IS NOT NULL AND TRIM(state) = ''
UNION ALL
SELECT 'zip_code', COUNT(*)
FROM fintech_lakehouse_dev.silver.silver_consumer_complaints
WHERE zip_code IS NOT NULL AND TRIM(zip_code) = ''
UNION ALL
SELECT 'submitted_via', COUNT(*)
FROM fintech_lakehouse_dev.silver.silver_consumer_complaints
WHERE submitted_via IS NOT NULL AND TRIM(submitted_via) = ''
UNION ALL
SELECT 'company_response_to_consumer', COUNT(*)
FROM fintech_lakehouse_dev.silver.silver_consumer_complaints
WHERE company_response_to_consumer IS NOT NULL AND TRIM(company_response_to_consumer) = ''
UNION ALL
SELECT 'timely_response', COUNT(*)
FROM fintech_lakehouse_dev.silver.silver_consumer_complaints
WHERE timely_response IS NOT NULL AND TRIM(timely_response) = ''
ORDER BY remaining_blank_count DESC, column_name;


-- ==========================================================
-- Check: State standardisation validation
--
-- Purpose:
-- Verify that non-null Silver state values are standardised to uppercase.
--
-- Why this matters:
-- Geographic reporting becomes fragmented if raw state values retain mixed
-- casing or leading/trailing whitespace.
--
-- Silver action:
-- Non-zero counts here indicate the uppercase standardisation logic needs
-- adjustment.
-- ==========================================================
SELECT
  COUNT(*) AS non_uppercase_state_count
FROM fintech_lakehouse_dev.silver.silver_consumer_complaints
WHERE state IS NOT NULL
  AND state <> UPPER(TRIM(state));


-- ==========================================================
-- Check: Flag-value standardisation validation
--
-- Purpose:
-- Confirm that the Silver flag-like columns now use a controlled value set.
--
-- Why this matters:
-- Consistent categorical values are essential for analytics, dashboards, and
-- feature engineering.
--
-- Silver action:
-- Review any values outside the expected set and tighten standardisation rules
-- if required. This Silver table currently validates timely_response only,
-- because consumer_disputed is not present in the source Bronze schema.
-- ==========================================================
SELECT
  'timely_response' AS column_name,
  timely_response AS value,
  COUNT(*) AS row_count
FROM fintech_lakehouse_dev.silver.silver_consumer_complaints
WHERE timely_response IS NOT NULL
GROUP BY timely_response
ORDER BY column_name, row_count DESC, value;


-- ==========================================================
-- Check: Silver timestamp parsing validation
--
-- Purpose:
-- Validate that date_received and date_sent_to_company are present as parsed
-- timestamps at the Silver layer.
--
-- Why this matters:
-- Silver should be analytically ready for time-series reporting and SLA-style
-- calculations.
--
-- Silver action:
-- If parsed timestamps are mostly null despite valid source values, the Silver
-- date parsing patterns need to be revisited.
-- ==========================================================
SELECT
  MIN(date_received) AS min_date_received_ts,
  MAX(date_received) AS max_date_received_ts,
  MIN(date_sent_to_company) AS min_date_sent_to_company_ts,
  MAX(date_sent_to_company) AS max_date_sent_to_company_ts,
  COUNT_IF(date_received IS NOT NULL) AS populated_date_received_count,
  COUNT_IF(date_sent_to_company IS NOT NULL) AS populated_date_sent_to_company_count
FROM fintech_lakehouse_dev.silver.silver_consumer_complaints;


-- ==========================================================
-- Check: Bronze-to-Silver complaint_id preservation
--
-- Purpose:
-- Compare valid Bronze complaint IDs with the final Silver row count.
--
-- Why this matters:
-- Silver should only remove invalid or duplicate complaint IDs, not valid
-- unique complaint records.
--
-- Silver action:
-- Use this summary to explain the difference between raw Bronze volume and
-- final Silver complaint grain.
-- ==========================================================
WITH bronze_valid_ids AS (
  SELECT
    COUNT(*) AS bronze_valid_complaint_id_rows,
    COUNT(DISTINCT complaint_id) AS bronze_valid_distinct_complaint_ids
  FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints
  WHERE complaint_id IS NOT NULL
    AND TRIM(complaint_id) RLIKE '^[0-9]+$'
),
silver_stats AS (
  SELECT
    COUNT(*) AS silver_rows,
    COUNT(DISTINCT complaint_id) AS silver_distinct_complaint_ids
  FROM fintech_lakehouse_dev.silver.silver_consumer_complaints
)
SELECT
  b.bronze_valid_complaint_id_rows,
  b.bronze_valid_distinct_complaint_ids,
  s.silver_rows,
  s.silver_distinct_complaint_ids
FROM bronze_valid_ids b
CROSS JOIN silver_stats s;


-- ==========================================================
-- Check: Silver quality summary metrics
--
-- Purpose:
-- Produce a compact quality scorecard suitable for job validation, run-book
-- review, or portfolio evidence.
--
-- Why this matters:
-- A single summary table helps engineers quickly confirm whether the core
-- transformation expectations were met after each refresh.
--
-- Silver action:
-- Use this output as the first validation checkpoint after the Silver job runs.
-- ==========================================================
WITH silver_summary AS (
  SELECT
    COUNT(*) AS silver_row_count,
    COUNT(DISTINCT complaint_id) AS distinct_complaint_id_count,
    COUNT_IF(complaint_id IS NULL) AS null_complaint_id_count,
    COUNT_IF(date_received IS NULL) AS null_date_received_count,
    COUNT_IF(state IS NOT NULL AND state <> UPPER(TRIM(state))) AS non_uppercase_state_count,
    COUNT_IF(timely_response IS NOT NULL AND timely_response NOT IN ('YES', 'NO')) AS non_standard_timely_response_count,
    COUNT_IF(_silver_record_status <> 'VALID') AS non_valid_record_status_count
  FROM fintech_lakehouse_dev.silver.silver_consumer_complaints
)
SELECT 'silver_row_count' AS metric_name, CAST(silver_row_count AS STRING) AS metric_value FROM silver_summary
UNION ALL
SELECT 'distinct_complaint_id_count', CAST(distinct_complaint_id_count AS STRING) FROM silver_summary
UNION ALL
SELECT 'null_complaint_id_count', CAST(null_complaint_id_count AS STRING) FROM silver_summary
UNION ALL
SELECT 'null_date_received_count', CAST(null_date_received_count AS STRING) FROM silver_summary
UNION ALL
SELECT 'non_uppercase_state_count', CAST(non_uppercase_state_count AS STRING) FROM silver_summary
UNION ALL
SELECT 'non_standard_timely_response_count', CAST(non_standard_timely_response_count AS STRING) FROM silver_summary
UNION ALL
SELECT 'non_valid_record_status_count', CAST(non_valid_record_status_count AS STRING) FROM silver_summary;


-- ==========================================================
-- Check: Sample Silver records
--
-- Purpose:
-- Inspect representative Silver rows after cleaning, deduplication, casting,
-- and audit-column enrichment.
--
-- Why this matters:
-- Engineers should always review sample rows to verify that the table is not
-- merely structurally valid but also semantically sensible.
--
-- Silver action:
-- Confirm that complaint_id is numeric, dates are parsed, string values are
-- trimmed, and Silver metadata columns are populated.
-- ==========================================================
SELECT
  complaint_id,
  date_received,
  product,
  sub_product,
  issue,
  sub_issue,
  company,
  state,
  zip_code,
  submitted_via,
  company_response_to_consumer,
  timely_response,
  _ingestion_date,
  _ingested_at,
  _source_zip_path,
  _source_csv_name,
  _silver_processed_at,
  _silver_record_status
FROM fintech_lakehouse_dev.silver.silver_consumer_complaints
ORDER BY _silver_processed_at DESC, complaint_id
LIMIT 50;
