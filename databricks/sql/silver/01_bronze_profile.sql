-- Consumer Complaints Bronze profiling
-- Project: consumer-complaints-ml
-- Layer: Bronze investigation prior to Silver cleansing
-- Table: fintech_lakehouse_dev.bronze.bronze_consumer_complaints

-- ==========================================================
-- Check: Bronze table existence and row volume
--
-- Purpose:
-- Confirm that the Bronze table exists, understand the total number of rows,
-- and establish the current ingestion footprint before any cleaning logic is
-- designed.
--
-- Why this matters:
-- A Silver job should be based on the observed data volume rather than
-- assumptions. This also confirms that Bronze ingestion completed successfully.
--
-- Silver action:
-- Use the row-count baseline to validate that Silver does not create more
-- records than Bronze and to estimate the expected scale of deduplication.
-- ==========================================================
SELECT
  COUNT(*) AS bronze_row_count,
  COUNT(DISTINCT complaint_id) AS distinct_complaint_id_count,
  COUNT(DISTINCT _ingestion_date) AS ingestion_date_count,
  MIN(_ingestion_date) AS min_ingestion_date,
  MAX(_ingestion_date) AS max_ingestion_date
FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints;


-- ==========================================================
-- Check: Bronze table schema
--
-- Purpose:
-- Inspect the physical Bronze schema exactly as stored in Unity Catalog.
--
-- Why this matters:
-- Silver logic must use real Bronze column names and data types. This is
-- especially important because the ingestion pipeline normalised raw CFPB
-- headers into Delta-safe column names.
--
-- Silver action:
-- Confirm the exact fields available for trimming, parsing, deduplication,
-- and metadata preservation before writing transformation logic.
-- ==========================================================
DESCRIBE TABLE fintech_lakehouse_dev.bronze.bronze_consumer_complaints;


-- ==========================================================
-- Check: Bronze column catalogue
--
-- Purpose:
-- Review the schema from the information schema perspective, including column
-- order and data types.
--
-- Why this matters:
-- This helps an engineer confirm whether fields like zip_code remain strings
-- and whether metadata columns are available for lineage and deduplication.
--
-- Silver action:
-- Keep zip_code string-based, retain Bronze metadata columns, and ensure
-- Silver casting is applied only where intended.
-- ==========================================================
SELECT
  ordinal_position,
  column_name,
  data_type,
  is_nullable
FROM fintech_lakehouse_dev.information_schema.columns
WHERE table_schema = 'bronze'
  AND table_name = 'bronze_consumer_complaints'
ORDER BY ordinal_position;


-- ==========================================================
-- Check: Null and blank percentage profile for key Bronze fields
--
-- Purpose:
-- Measure how often important business columns are null, blank, or whitespace
-- only in the raw Bronze table.
--
-- Why this matters:
-- High missingness in complaint identifiers, product attributes, dates, or
-- response fields directly affects analytics quality and downstream feature
-- engineering.
--
-- Silver action:
-- Trim whitespace, convert blanks to null, enforce complaint_id validity, and
-- make targeted standardisation decisions based on observed completeness.
-- ==========================================================
WITH bronze_counts AS (
  SELECT COUNT(*) AS total_rows
  FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints
),
column_profile AS (
  SELECT 'complaint_id' AS column_name,
         COUNT_IF(complaint_id IS NULL) AS null_count,
         COUNT_IF(TRIM(COALESCE(complaint_id, '')) = '') AS blank_or_whitespace_count
  FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints
  UNION ALL
  SELECT 'date_received',
         COUNT_IF(date_received IS NULL),
         COUNT_IF(TRIM(COALESCE(date_received, '')) = '')
  FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints
  UNION ALL
  SELECT 'product',
         COUNT_IF(product IS NULL),
         COUNT_IF(TRIM(COALESCE(product, '')) = '')
  FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints
  UNION ALL
  SELECT 'sub_product',
         COUNT_IF(sub_product IS NULL),
         COUNT_IF(TRIM(COALESCE(sub_product, '')) = '')
  FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints
  UNION ALL
  SELECT 'issue',
         COUNT_IF(issue IS NULL),
         COUNT_IF(TRIM(COALESCE(issue, '')) = '')
  FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints
  UNION ALL
  SELECT 'sub_issue',
         COUNT_IF(sub_issue IS NULL),
         COUNT_IF(TRIM(COALESCE(sub_issue, '')) = '')
  FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints
  UNION ALL
  SELECT 'company',
         COUNT_IF(company IS NULL),
         COUNT_IF(TRIM(COALESCE(company, '')) = '')
  FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints
  UNION ALL
  SELECT 'state',
         COUNT_IF(state IS NULL),
         COUNT_IF(TRIM(COALESCE(state, '')) = '')
  FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints
  UNION ALL
  SELECT 'zip_code',
         COUNT_IF(zip_code IS NULL),
         COUNT_IF(TRIM(COALESCE(zip_code, '')) = '')
  FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints
  UNION ALL
  SELECT 'submitted_via',
         COUNT_IF(submitted_via IS NULL),
         COUNT_IF(TRIM(COALESCE(submitted_via, '')) = '')
  FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints
  UNION ALL
  SELECT 'company_response_to_consumer',
         COUNT_IF(company_response_to_consumer IS NULL),
         COUNT_IF(TRIM(COALESCE(company_response_to_consumer, '')) = '')
  FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints
  UNION ALL
  SELECT 'timely_response',
         COUNT_IF(timely_response IS NULL),
         COUNT_IF(TRIM(COALESCE(timely_response, '')) = '')
  FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints
)
SELECT
  p.column_name,
  b.total_rows,
  p.null_count,
  ROUND(100.0 * p.null_count / NULLIF(b.total_rows, 0), 2) AS null_pct,
  p.blank_or_whitespace_count,
  ROUND(100.0 * p.blank_or_whitespace_count / NULLIF(b.total_rows, 0), 2) AS blank_or_whitespace_pct
FROM column_profile p
CROSS JOIN bronze_counts b
ORDER BY null_pct DESC, blank_or_whitespace_pct DESC, column_name;


-- ==========================================================
-- Check: Duplicate complaint record profile
--
-- Purpose:
-- Identify whether Bronze contains repeated complaint IDs and quantify how
-- many duplicate rows are present.
--
-- Why this matters:
-- Duplicate complaint IDs inflate counts and bias complaint-level analytics or
-- downstream machine-learning features.
--
-- Silver action:
-- Deduplicate on complaint_id and keep the most recent version using
-- _ingested_at as the tie-breaker.
-- ==========================================================
WITH duplicate_ids AS (
  SELECT
    complaint_id,
    COUNT(*) AS row_count,
    MIN(_ingested_at) AS earliest_ingested_at,
    MAX(_ingested_at) AS latest_ingested_at
  FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints
  WHERE complaint_id IS NOT NULL
  GROUP BY complaint_id
  HAVING COUNT(*) > 1
)
SELECT
  COUNT(*) AS duplicate_complaint_id_count,
  COALESCE(SUM(row_count), 0) AS duplicate_row_count,
  COALESCE(SUM(row_count) - COUNT(*), 0) AS rows_above_unique_grain,
  MIN(earliest_ingested_at) AS first_duplicate_seen_at,
  MAX(latest_ingested_at) AS last_duplicate_seen_at
FROM duplicate_ids;


-- ==========================================================
-- Check: Sample duplicate complaint IDs
--
-- Purpose:
-- Show a targeted sample of duplicated complaint IDs for manual inspection.
--
-- Why this matters:
-- Engineers often need to verify whether duplicate IDs are exact replayed rows
-- or updated versions of the same complaint arriving at different ingestion
-- times.
--
-- Silver action:
-- Validate that complaint_id is the correct business key for deduplication and
-- confirm that keeping the latest _ingested_at record is appropriate.
-- ==========================================================
SELECT
  complaint_id,
  COUNT(*) AS row_count,
  MIN(_ingested_at) AS earliest_ingested_at,
  MAX(_ingested_at) AS latest_ingested_at
FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints
WHERE complaint_id IS NOT NULL
GROUP BY complaint_id
HAVING COUNT(*) > 1
ORDER BY row_count DESC, latest_ingested_at DESC
LIMIT 50;


-- ==========================================================
-- Check: Date range and date parseability profile
--
-- Purpose:
-- Understand the Bronze date coverage and how reliably raw date strings can be
-- interpreted as timestamps.
--
-- Why this matters:
-- Invalid or malformed date strings break trend analysis and can make time-
-- based joins or partition logic unreliable.
--
-- Silver action:
-- Parse date_received and date_sent_to_company into proper timestamps and
-- monitor any unparseable raw values for future data-quality remediation.
-- ==========================================================
WITH bronze_dates AS (
  SELECT
    COALESCE(
      TRY_TO_TIMESTAMP(date_received, 'MM/dd/yyyy'),
      TRY_TO_TIMESTAMP(date_received, 'M/d/yyyy'),
      TRY_TO_TIMESTAMP(date_received, 'yyyy-MM-dd')
    ) AS parsed_date_received,
    COALESCE(
      TRY_TO_TIMESTAMP(date_sent_to_company, 'MM/dd/yyyy'),
      TRY_TO_TIMESTAMP(date_sent_to_company, 'M/d/yyyy'),
      TRY_TO_TIMESTAMP(date_sent_to_company, 'yyyy-MM-dd')
    ) AS parsed_date_sent_to_company,
    date_received,
    date_sent_to_company
  FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints
)
SELECT
  MIN(parsed_date_received) AS min_date_received_ts,
  MAX(parsed_date_received) AS max_date_received_ts,
  MIN(parsed_date_sent_to_company) AS min_date_sent_to_company_ts,
  MAX(parsed_date_sent_to_company) AS max_date_sent_to_company_ts,
  COUNT_IF(date_received IS NOT NULL AND parsed_date_received IS NULL) AS unparseable_date_received_count,
  COUNT_IF(date_sent_to_company IS NOT NULL AND parsed_date_sent_to_company IS NULL) AS unparseable_date_sent_to_company_count
FROM bronze_dates;


-- ==========================================================
-- Check: Distinct-value profile for key business dimensions
--
-- Purpose:
-- Quantify the cardinality of the most important analytical dimensions in the
-- Bronze dataset.
--
-- Why this matters:
-- This reveals which fields are low-cardinality categories versus high-
-- cardinality descriptive dimensions and helps set expectations for grouping
-- logic, standardisation effort, and dashboard design.
--
-- Silver action:
-- Focus string cleaning on high-value analytical dimensions such as product,
-- issue, company, state, and channel fields before reporting or modelling.
-- ==========================================================
SELECT 'product' AS column_name, COUNT(DISTINCT product) AS distinct_value_count
FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints
UNION ALL
SELECT 'sub_product', COUNT(DISTINCT sub_product)
FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints
UNION ALL
SELECT 'issue', COUNT(DISTINCT issue)
FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints
UNION ALL
SELECT 'sub_issue', COUNT(DISTINCT sub_issue)
FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints
UNION ALL
SELECT 'company', COUNT(DISTINCT company)
FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints
UNION ALL
SELECT 'state', COUNT(DISTINCT state)
FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints
UNION ALL
SELECT 'zip_code', COUNT(DISTINCT zip_code)
FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints
UNION ALL
SELECT 'submitted_via', COUNT(DISTINCT submitted_via)
FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints
UNION ALL
SELECT 'company_response_to_consumer', COUNT(DISTINCT company_response_to_consumer)
FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints
UNION ALL
SELECT 'timely_response', COUNT(DISTINCT timely_response)
FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints
ORDER BY distinct_value_count DESC, column_name;


-- ==========================================================
-- Check: Product frequency distribution
--
-- Purpose:
-- Inspect the most common complaint products represented in Bronze.
--
-- Why this matters:
-- Product is one of the primary business dimensions for complaints analytics
-- and often reveals whether different string variants need standardisation.
--
-- Silver action:
-- Standardise whitespace and case so product-level reporting groups identical
-- business concepts together.
-- ==========================================================
SELECT
  product,
  COUNT(*) AS row_count
FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints
GROUP BY product
ORDER BY row_count DESC, product
LIMIT 50;


-- ==========================================================
-- Check: State frequency distribution
--
-- Purpose:
-- Profile state values and identify whether state coverage is complete or
-- inconsistent.
--
-- Why this matters:
-- State values frequently suffer from mixed casing or whitespace issues, which
-- can fragment geographic reporting.
--
-- Silver action:
-- Standardise state to uppercase in Silver while preserving nulls and raw ZIP
-- code strings.
-- ==========================================================
SELECT
  state,
  COUNT(*) AS row_count
FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints
GROUP BY state
ORDER BY row_count DESC, state
LIMIT 60;


-- ==========================================================
-- Check: Submission-channel frequency distribution
--
-- Purpose:
-- Review how complaints enter the platform via submitted_via.
--
-- Why this matters:
-- Submission channels are often used in operational reporting and can be split
-- incorrectly by stray whitespace or inconsistent text casing.
--
-- Silver action:
-- Trim submission-channel text values so like-for-like operational reporting is
-- accurate.
-- ==========================================================
SELECT
  submitted_via,
  COUNT(*) AS row_count
FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints
GROUP BY submitted_via
ORDER BY row_count DESC, submitted_via
LIMIT 30;


-- ==========================================================
-- Check: Company response frequency distribution
--
-- Purpose:
-- Inspect the CFPB company response field as normalised into the Bronze table.
--
-- Why this matters:
-- This field is business critical for complaint-outcome analysis and often
-- contains categorical variants that need careful standardisation.
--
-- Silver action:
-- Keep the business content, trim whitespace, and use the Bronze schema name
-- company_response_to_consumer when writing transformations.
-- ==========================================================
SELECT
  company_response_to_consumer,
  COUNT(*) AS row_count
FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints
GROUP BY company_response_to_consumer
ORDER BY row_count DESC, company_response_to_consumer
LIMIT 30;


-- ==========================================================
-- Check: Timely-response frequency distribution
--
-- Purpose:
-- Profile the raw timely_response values before standardisation.
--
-- Why this matters:
-- Yes/No-style fields often contain inconsistent casing, blanks, or alternate
-- encodings that can fragment metrics or model features.
--
-- Silver action:
-- Standardise timely_response into a consistent representation in Silver while
-- preserving nulls where the source is genuinely missing.
-- ==========================================================
SELECT
  timely_response,
  COUNT(*) AS row_count
FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints
GROUP BY timely_response
ORDER BY row_count DESC, timely_response;

