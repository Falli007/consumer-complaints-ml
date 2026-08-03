-- Consumer Complaints Gold validation
-- Project: consumer-complaints-ml
-- Layer: Post-transformation Gold validation
-- Silver table: fintech_lakehouse_dev.silver.silver_consumer_complaints
-- Gold dimensions:
--   fintech_lakehouse_dev.gold.dim_date
--   fintech_lakehouse_dev.gold.dim_product
--   fintech_lakehouse_dev.gold.dim_company
--   fintech_lakehouse_dev.gold.dim_state
-- Gold fact:
--   fintech_lakehouse_dev.gold.fact_consumer_complaints

-- ==========================================================
-- Check: Gold table registration in Unity Catalog
--
-- Purpose:
-- Confirm that every expected Gold dimension and fact table has been created
-- and registered in the target catalog and schema.
--
-- Why this matters:
-- A successful Gold pipeline run should publish a complete dimensional model,
-- not just partial outputs.
--
-- Gold action:
-- If any expected table is missing, review the Gold job run and downstream
-- write logic before using the warehouse layer for BI or ML features.
-- ==========================================================
SELECT
  table_catalog,
  table_schema,
  table_name,
  table_type
FROM fintech_lakehouse_dev.information_schema.tables
WHERE table_schema = 'gold'
  AND table_name IN (
    'dim_date',
    'dim_product',
    'dim_company',
    'dim_state',
    'fact_consumer_complaints'
  )
ORDER BY table_name;


-- ==========================================================
-- Check: Gold row counts by table
--
-- Purpose:
-- Produce a quick inventory of row counts across all Gold assets.
--
-- Why this matters:
-- This establishes whether each dimension and the fact table were populated
-- sensibly after the Gold run.
--
-- Gold action:
-- Empty dimensions or an empty fact table indicate a failed or incomplete
-- warehouse build.
-- ==========================================================
SELECT 'dim_date' AS table_name, COUNT(*) AS row_count
FROM fintech_lakehouse_dev.gold.dim_date
UNION ALL
SELECT 'dim_product', COUNT(*)
FROM fintech_lakehouse_dev.gold.dim_product
UNION ALL
SELECT 'dim_company', COUNT(*)
FROM fintech_lakehouse_dev.gold.dim_company
UNION ALL
SELECT 'dim_state', COUNT(*)
FROM fintech_lakehouse_dev.gold.dim_state
UNION ALL
SELECT 'fact_consumer_complaints', COUNT(*)
FROM fintech_lakehouse_dev.gold.fact_consumer_complaints;


-- ==========================================================
-- Check: Silver versus Gold fact row comparison
--
-- Purpose:
-- Verify that the Gold fact table preserves the Silver complaint grain.
--
-- Why this matters:
-- The Gold fact should contain one row per cleaned Silver complaint record.
-- Unexpected row loss or row inflation would indicate join or deduplication
-- defects in the Gold pipeline.
--
-- Gold action:
-- Investigate dimension joins and fact-building logic if the counts differ.
-- ==========================================================
WITH silver_stats AS (
  SELECT COUNT(*) AS silver_row_count
  FROM fintech_lakehouse_dev.silver.silver_consumer_complaints
),
gold_stats AS (
  SELECT COUNT(*) AS gold_fact_row_count
  FROM fintech_lakehouse_dev.gold.fact_consumer_complaints
)
SELECT
  s.silver_row_count,
  g.gold_fact_row_count,
  s.silver_row_count - g.gold_fact_row_count AS row_difference
FROM silver_stats s
CROSS JOIN gold_stats g;


-- ==========================================================
-- Check: Gold schema inspection
--
-- Purpose:
-- Review the registered schema for each Gold table.
--
-- Why this matters:
-- This confirms that surrogate keys, business attributes, and audit columns
-- were persisted with the expected data types for analytical use.
--
-- Gold action:
-- Use this output to verify the warehouse contract before dashboards, feature
-- tables, or downstream semantic models are built.
-- ==========================================================
SELECT
  table_name,
  ordinal_position,
  column_name,
  data_type,
  is_nullable
FROM fintech_lakehouse_dev.information_schema.columns
WHERE table_schema = 'gold'
  AND table_name IN (
    'dim_date',
    'dim_product',
    'dim_company',
    'dim_state',
    'fact_consumer_complaints'
  )
ORDER BY table_name, ordinal_position;


-- ==========================================================
-- Check: Fact-table complaint grain uniqueness
--
-- Purpose:
-- Confirm that Gold still contains exactly one fact row per complaint_id.
--
-- Why this matters:
-- Duplicate complaint facts would distort reporting, training data, and all
-- downstream aggregates.
--
-- Gold action:
-- Any non-zero result here should be treated as a Gold model defect.
-- ==========================================================
SELECT
  COUNT(*) AS duplicate_complaint_id_groups
FROM (
  SELECT
    complaint_id
  FROM fintech_lakehouse_dev.gold.fact_consumer_complaints
  GROUP BY complaint_id
  HAVING COUNT(*) > 1
) duplicate_ids;


-- ==========================================================
-- Check: Null business-key validation in the Gold fact
--
-- Purpose:
-- Ensure that the core fact grain and required foreign keys are populated.
--
-- Why this matters:
-- complaint_id should remain non-null, and Gold should never leave foreign
-- keys null even when values are unknown.
--
-- Gold action:
-- Null foreign keys indicate incomplete dimension joins or missing fallback
-- handling for unknown members.
-- ==========================================================
SELECT
  COUNT(*) AS fact_row_count,
  COUNT_IF(complaint_id IS NULL) AS null_complaint_id_count,
  COUNT_IF(received_date_key IS NULL) AS null_received_date_key_count,
  COUNT_IF(sent_date_key IS NULL) AS null_sent_date_key_count,
  COUNT_IF(product_key IS NULL) AS null_product_key_count,
  COUNT_IF(company_key IS NULL) AS null_company_key_count,
  COUNT_IF(state_key IS NULL) AS null_state_key_count
FROM fintech_lakehouse_dev.gold.fact_consumer_complaints;


-- ==========================================================
-- Check: Unknown-member usage in the Gold fact
--
-- Purpose:
-- Measure how often the fact table falls back to the explicit unknown members
-- defined in the dimensions.
--
-- Why this matters:
-- Unknown-key usage is expected for genuinely missing source values, but high
-- rates can indicate upstream parsing or standardisation issues.
--
-- Gold action:
-- Use this summary to decide whether additional Silver quality rules are
-- needed before expanding the model.
-- ==========================================================
SELECT
  COUNT_IF(received_date_key = 0) AS unknown_received_date_rows,
  COUNT_IF(sent_date_key = 0) AS unknown_sent_date_rows,
  COUNT_IF(product_key = 0) AS unknown_product_rows,
  COUNT_IF(company_key = 0) AS unknown_company_rows,
  COUNT_IF(state_key = 0) AS unknown_state_rows
FROM fintech_lakehouse_dev.gold.fact_consumer_complaints;


-- ==========================================================
-- Check: Referential coverage between fact and dimensions
--
-- Purpose:
-- Confirm that every foreign key used in the fact resolves to an existing
-- dimension member.
--
-- Why this matters:
-- A warehouse model is only reliable if the fact table is referentially
-- complete against its conformed dimensions.
--
-- Gold action:
-- Non-zero orphan counts indicate a serious Gold-model integrity defect.
-- ==========================================================
SELECT
  COUNT_IF(dd.date_key IS NULL) AS orphan_received_date_keys,
  COUNT_IF(ds.date_key IS NULL) AS orphan_sent_date_keys,
  COUNT_IF(dp.product_key IS NULL) AS orphan_product_keys,
  COUNT_IF(dc.company_key IS NULL) AS orphan_company_keys,
  COUNT_IF(st.state_key IS NULL) AS orphan_state_keys
FROM fintech_lakehouse_dev.gold.fact_consumer_complaints f
LEFT JOIN fintech_lakehouse_dev.gold.dim_date dd
  ON f.received_date_key = dd.date_key
LEFT JOIN fintech_lakehouse_dev.gold.dim_date ds
  ON f.sent_date_key = ds.date_key
LEFT JOIN fintech_lakehouse_dev.gold.dim_product dp
  ON f.product_key = dp.product_key
LEFT JOIN fintech_lakehouse_dev.gold.dim_company dc
  ON f.company_key = dc.company_key
LEFT JOIN fintech_lakehouse_dev.gold.dim_state st
  ON f.state_key = st.state_key;


-- ==========================================================
-- Check: Date dimension range and completeness
--
-- Purpose:
-- Profile the conformed date dimension that supports complaint reporting.
--
-- Why this matters:
-- The date dimension should cover the dates present in the fact model and
-- expose calendar attributes needed for time-based analytics.
--
-- Gold action:
-- Review this range before building daily/monthly reporting or trend charts.
-- ==========================================================
SELECT
  MIN(full_date) AS min_full_date,
  MAX(full_date) AS max_full_date,
  COUNT(*) AS total_dim_date_rows,
  COUNT_IF(date_key = 0) AS unknown_date_rows
FROM fintech_lakehouse_dev.gold.dim_date;


-- ==========================================================
-- Check: Product dimension profile
--
-- Purpose:
-- Measure the size and unknown-member usage of the product dimension.
--
-- Why this matters:
-- This confirms the product/sub-product analytical grain and shows whether the
-- raw Bronze/Silver taxonomy was preserved.
--
-- Gold action:
-- Use this as a sanity check before product-level reporting or segmentation.
-- ==========================================================
SELECT
  COUNT(*) AS total_product_members,
  COUNT_IF(product_key = 0) AS unknown_product_members,
  COUNT_IF(product IS NULL) AS null_product_labels,
  COUNT_IF(sub_product IS NULL) AS null_sub_product_labels
FROM fintech_lakehouse_dev.gold.dim_product;


-- ==========================================================
-- Check: Company dimension profile
--
-- Purpose:
-- Validate the company dimension cardinality and unknown-member handling.
--
-- Why this matters:
-- Company is a major reporting slice for complaint analysis, so the dimension
-- must be populated and clean.
--
-- Gold action:
-- Investigate unexpected company counts or null labels before BI publishing.
-- ==========================================================
SELECT
  COUNT(*) AS total_company_members,
  COUNT_IF(company_key = 0) AS unknown_company_members,
  COUNT_IF(company IS NULL) AS null_company_labels
FROM fintech_lakehouse_dev.gold.dim_company;


-- ==========================================================
-- Check: State dimension profile
--
-- Purpose:
-- Validate the geography dimension used by complaint reporting.
--
-- Why this matters:
-- State-level reporting depends on a stable, standardised state dimension.
--
-- Gold action:
-- Use this result to confirm the expected state/member footprint before
-- mapping or regional rollups are built.
-- ==========================================================
SELECT
  COUNT(*) AS total_state_members,
  COUNT_IF(state_key = 0) AS unknown_state_members,
  COUNT_IF(state IS NULL) AS null_state_labels
FROM fintech_lakehouse_dev.gold.dim_state;


-- ==========================================================
-- Check: Top complaint products in the Gold fact
--
-- Purpose:
-- Demonstrate that the Gold fact joins cleanly back to the product dimension
-- and supports straightforward analytical summaries.
--
-- Why this matters:
-- A dimensional model should immediately enable business-friendly aggregates.
--
-- Gold action:
-- Use this as a practical sanity check that joins, keys, and labels align.
-- ==========================================================
SELECT
  dp.product,
  dp.sub_product,
  COUNT(*) AS complaint_count
FROM fintech_lakehouse_dev.gold.fact_consumer_complaints f
LEFT JOIN fintech_lakehouse_dev.gold.dim_product dp
  ON f.product_key = dp.product_key
GROUP BY dp.product, dp.sub_product
ORDER BY complaint_count DESC, dp.product, dp.sub_product
LIMIT 25;


-- ==========================================================
-- Check: Complaint volume by state
--
-- Purpose:
-- Validate that state-level joins and aggregation work as expected from the
-- Gold fact and state dimension.
--
-- Why this matters:
-- Geographic reporting is one of the most common consumer-complaints use
-- cases, so this is an effective end-to-end sanity check.
--
-- Gold action:
-- Review the distribution to confirm there are no obvious keying anomalies.
-- ==========================================================
SELECT
  st.state,
  COUNT(*) AS complaint_count
FROM fintech_lakehouse_dev.gold.fact_consumer_complaints f
LEFT JOIN fintech_lakehouse_dev.gold.dim_state st
  ON f.state_key = st.state_key
GROUP BY st.state
ORDER BY complaint_count DESC, st.state
LIMIT 25;


-- ==========================================================
-- Check: Gold quality summary metrics
--
-- Purpose:
-- Produce a compact one-table quality scorecard for the final warehouse layer.
--
-- Why this matters:
-- A single summary result is useful for run-book validation, screenshots, and
-- portfolio evidence after every Gold refresh.
--
-- Gold action:
-- Treat any non-zero defect metric as a signal to investigate the Gold build
-- before downstream consumption.
-- ==========================================================
WITH fact_summary AS (
  SELECT
    COUNT(*) AS fact_row_count,
    COUNT(DISTINCT complaint_id) AS distinct_complaint_id_count,
    COUNT_IF(complaint_id IS NULL) AS null_complaint_id_count,
    COUNT_IF(received_date_key IS NULL) AS null_received_date_key_count,
    COUNT_IF(sent_date_key IS NULL) AS null_sent_date_key_count,
    COUNT_IF(product_key IS NULL) AS null_product_key_count,
    COUNT_IF(company_key IS NULL) AS null_company_key_count,
    COUNT_IF(state_key IS NULL) AS null_state_key_count,
    COUNT_IF(received_date_key = 0) AS unknown_received_date_rows,
    COUNT_IF(sent_date_key = 0) AS unknown_sent_date_rows,
    COUNT_IF(product_key = 0) AS unknown_product_rows,
    COUNT_IF(company_key = 0) AS unknown_company_rows,
    COUNT_IF(state_key = 0) AS unknown_state_rows
  FROM fintech_lakehouse_dev.gold.fact_consumer_complaints
)
SELECT 'fact_row_count' AS metric_name, CAST(fact_row_count AS STRING) AS metric_value FROM fact_summary
UNION ALL
SELECT 'distinct_complaint_id_count', CAST(distinct_complaint_id_count AS STRING) FROM fact_summary
UNION ALL
SELECT 'null_complaint_id_count', CAST(null_complaint_id_count AS STRING) FROM fact_summary
UNION ALL
SELECT 'null_received_date_key_count', CAST(null_received_date_key_count AS STRING) FROM fact_summary
UNION ALL
SELECT 'null_sent_date_key_count', CAST(null_sent_date_key_count AS STRING) FROM fact_summary
UNION ALL
SELECT 'null_product_key_count', CAST(null_product_key_count AS STRING) FROM fact_summary
UNION ALL
SELECT 'null_company_key_count', CAST(null_company_key_count AS STRING) FROM fact_summary
UNION ALL
SELECT 'null_state_key_count', CAST(null_state_key_count AS STRING) FROM fact_summary
UNION ALL
SELECT 'unknown_received_date_rows', CAST(unknown_received_date_rows AS STRING) FROM fact_summary
UNION ALL
SELECT 'unknown_sent_date_rows', CAST(unknown_sent_date_rows AS STRING) FROM fact_summary
UNION ALL
SELECT 'unknown_product_rows', CAST(unknown_product_rows AS STRING) FROM fact_summary
UNION ALL
SELECT 'unknown_company_rows', CAST(unknown_company_rows AS STRING) FROM fact_summary
UNION ALL
SELECT 'unknown_state_rows', CAST(unknown_state_rows AS STRING) FROM fact_summary;


-- ==========================================================
-- Check: Sample Gold fact rows
--
-- Purpose:
-- Inspect representative fact rows with resolved foreign keys and retained
-- business attributes.
--
-- Why this matters:
-- Structural validation is not enough on its own; engineers should also check
-- that the stored records look analytically sensible.
--
-- Gold action:
-- Use these rows to verify the final warehouse grain and lineage columns.
-- ==========================================================
SELECT
  complaint_id,
  received_date_key,
  sent_date_key,
  product_key,
  company_key,
  state_key,
  issue,
  sub_issue,
  zip_code,
  submitted_via,
  company_response_to_consumer,
  timely_response,
  has_consumer_narrative,
  has_tags,
  has_company_response_date,
  _ingestion_date,
  _ingested_at,
  _source_zip_path,
  _source_csv_name,
  _silver_processed_at,
  _silver_record_status,
  _gold_processed_at
FROM fintech_lakehouse_dev.gold.fact_consumer_complaints
ORDER BY _gold_processed_at DESC, complaint_id
LIMIT 50;


-- ==========================================================
-- Check: Sample Gold dimension members
--
-- Purpose:
-- Provide a quick human-readable sample of the dimension tables.
--
-- Why this matters:
-- This helps confirm that the conformed dimensions look business-friendly and
-- are ready for dashboard joins.
--
-- Gold action:
-- Use these samples as a final visual inspection step after the Gold run.
-- ==========================================================
SELECT
  'dim_product' AS dimension_name,
  CAST(product_key AS STRING) AS key_value,
  product AS attribute_1,
  sub_product AS attribute_2
FROM fintech_lakehouse_dev.gold.dim_product
WHERE product_key <> 0
ORDER BY product_key
LIMIT 20;

SELECT
  'dim_company' AS dimension_name,
  CAST(company_key AS STRING) AS key_value,
  company AS attribute_1,
  'NOT_APPLICABLE' AS attribute_2
FROM fintech_lakehouse_dev.gold.dim_company
WHERE company_key <> 0
ORDER BY company_key
LIMIT 20;

SELECT
  'dim_state' AS dimension_name,
  CAST(state_key AS STRING) AS key_value,
  state AS attribute_1,
  'NOT_APPLICABLE' AS attribute_2
FROM fintech_lakehouse_dev.gold.dim_state
WHERE state_key <> 0
ORDER BY state_key
LIMIT 20;

SELECT
  'dim_date' AS dimension_name,
  CAST(date_key AS STRING) AS key_value,
  COALESCE(CAST(full_date AS STRING), 'NOT_PROVIDED') AS attribute_1,
  calendar_month_name AS attribute_2
FROM fintech_lakehouse_dev.gold.dim_date
WHERE date_key <> 0
ORDER BY date_key
LIMIT 20;
