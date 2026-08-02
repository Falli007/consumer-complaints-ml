-- Consumer Complaints Bronze data-quality investigation
-- Project: consumer-complaints-ml
-- Layer: Bronze diagnostics to inform Silver design
-- Table: fintech_lakehouse_dev.bronze.bronze_consumer_complaints

-- ==========================================================
-- Check: Duplicate complaint records
--
-- Purpose:
-- Identify whether the Bronze layer contains duplicate complaint records by
-- complaint_id.
--
-- Why this matters:
-- Duplicate records can affect reporting accuracy, operational KPIs, and ML
-- features by overstating complaint counts.
--
-- Silver action:
-- Remove duplicates using complaint_id as the business key and retain the most
-- recent record based on _ingested_at.
-- ==========================================================
SELECT
  complaint_id,
  COUNT(*) AS duplicate_count,
  MIN(_ingested_at) AS earliest_ingested_at,
  MAX(_ingested_at) AS latest_ingested_at
FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints
WHERE complaint_id IS NOT NULL
GROUP BY complaint_id
HAVING COUNT(*) > 1
ORDER BY duplicate_count DESC, latest_ingested_at DESC
LIMIT 100;


-- ==========================================================
-- Check: Missing complaint IDs
--
-- Purpose:
-- Measure how many Bronze records are missing the primary complaint business
-- key or contain only whitespace.
--
-- Why this matters:
-- complaint_id is required to deduplicate and uniquely identify complaint
-- records downstream.
--
-- Silver action:
-- Remove rows where complaint_id is null, blank, or non-numeric before writing
-- Silver.
-- ==========================================================
SELECT
  COUNT(*) AS total_rows,
  COUNT_IF(complaint_id IS NULL) AS null_complaint_id_count,
  COUNT_IF(TRIM(COALESCE(complaint_id, '')) = '') AS blank_complaint_id_count,
  COUNT_IF(complaint_id IS NOT NULL AND NOT TRIM(complaint_id) RLIKE '^[0-9]+$') AS non_numeric_complaint_id_count
FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints;


-- ==========================================================
-- Check: Missing critical business fields
--
-- Purpose:
-- Quantify missingness in the core analytical complaint attributes.
--
-- Why this matters:
-- Critical gaps in product, issue, company, or complaint dates reduce the
-- usefulness of the dataset for analytics, governance, and model training.
--
-- Silver action:
-- Convert blanks to null consistently and track which business fields still
-- require downstream remediation or monitoring.
-- ==========================================================
SELECT
  COUNT_IF(TRIM(COALESCE(date_received, '')) = '') AS missing_date_received_count,
  COUNT_IF(TRIM(COALESCE(product, '')) = '') AS missing_product_count,
  COUNT_IF(TRIM(COALESCE(issue, '')) = '') AS missing_issue_count,
  COUNT_IF(TRIM(COALESCE(company, '')) = '') AS missing_company_count,
  COUNT_IF(TRIM(COALESCE(state, '')) = '') AS missing_state_count,
  COUNT_IF(TRIM(COALESCE(submitted_via, '')) = '') AS missing_submitted_via_count
FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints;


-- ==========================================================
-- Check: Blank-string and whitespace-only issues
--
-- Purpose:
-- Detect fields that look populated but are effectively empty once trimmed.
--
-- Why this matters:
-- Blank strings and whitespace-only values bypass simple null checks and cause
-- misleading completeness metrics in raw reporting.
--
-- Silver action:
-- Apply TRIM and convert blank strings to null across all string fields before
-- analytics.
-- ==========================================================
SELECT 'product' AS column_name, COUNT(*) AS whitespace_only_count
FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints
WHERE product IS NOT NULL AND TRIM(product) = ''
UNION ALL
SELECT 'sub_product', COUNT(*)
FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints
WHERE sub_product IS NOT NULL AND TRIM(sub_product) = ''
UNION ALL
SELECT 'issue', COUNT(*)
FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints
WHERE issue IS NOT NULL AND TRIM(issue) = ''
UNION ALL
SELECT 'sub_issue', COUNT(*)
FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints
WHERE sub_issue IS NOT NULL AND TRIM(sub_issue) = ''
UNION ALL
SELECT 'company', COUNT(*)
FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints
WHERE company IS NOT NULL AND TRIM(company) = ''
UNION ALL
SELECT 'state', COUNT(*)
FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints
WHERE state IS NOT NULL AND TRIM(state) = ''
UNION ALL
SELECT 'zip_code', COUNT(*)
FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints
WHERE zip_code IS NOT NULL AND TRIM(zip_code) = ''
UNION ALL
SELECT 'submitted_via', COUNT(*)
FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints
WHERE submitted_via IS NOT NULL AND TRIM(submitted_via) = ''
UNION ALL
SELECT 'company_response_to_consumer', COUNT(*)
FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints
WHERE company_response_to_consumer IS NOT NULL AND TRIM(company_response_to_consumer) = ''
UNION ALL
SELECT 'timely_response', COUNT(*)
FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints
WHERE timely_response IS NOT NULL AND TRIM(timely_response) = ''
ORDER BY whitespace_only_count DESC, column_name;


-- ==========================================================
-- Check: Inconsistent casing in categorical fields
--
-- Purpose:
-- Compare raw distinct counts with standardised lower/upper-cased distinct
-- counts to identify fragmentation caused by casing inconsistency.
--
-- Why this matters:
-- "Email", "EMAIL", and "email" are analytically the same category but would
-- group separately in raw reporting.
--
-- Finding:
-- Product, company, state, channel, and response fields commonly fragment when
-- case normalisation is missing.
--
-- Impact:
-- Aggregations become misleading and dimension cardinality grows artificially.
--
-- Silver transformation:
-- Apply TRIM plus targeted case standardisation, especially uppercase for
-- state and consistent yes/no values for flags.
-- ==========================================================
SELECT
  'product' AS column_name,
  COUNT(DISTINCT product) AS raw_distinct_count,
  COUNT(DISTINCT LOWER(TRIM(product))) AS normalised_distinct_count
FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints
WHERE product IS NOT NULL
UNION ALL
SELECT
  'company',
  COUNT(DISTINCT company),
  COUNT(DISTINCT LOWER(TRIM(company)))
FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints
WHERE company IS NOT NULL
UNION ALL
SELECT
  'state',
  COUNT(DISTINCT state),
  COUNT(DISTINCT UPPER(TRIM(state)))
FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints
WHERE state IS NOT NULL
UNION ALL
SELECT
  'submitted_via',
  COUNT(DISTINCT submitted_via),
  COUNT(DISTINCT LOWER(TRIM(submitted_via)))
FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints
WHERE submitted_via IS NOT NULL
UNION ALL
SELECT
  'timely_response',
  COUNT(DISTINCT timely_response),
  COUNT(DISTINCT UPPER(TRIM(timely_response)))
FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints
WHERE timely_response IS NOT NULL
ORDER BY column_name;


-- ==========================================================
-- Check: Invalid or unparseable date values
--
-- Purpose:
-- Identify non-null source date strings that cannot be converted into valid
-- timestamps.
--
-- Why this matters:
-- Date parsing errors break temporal analysis and can lead to silent data loss
-- if invalid values are not surfaced explicitly.
--
-- Finding:
-- Source values that remain unparseable after standard date patterns are
-- strong candidates for data-quality exceptions.
--
-- Impact:
-- Complaint trends by day, month, or SLA timing become unreliable.
--
-- Silver transformation:
-- Parse recognised formats, including ISO-style timestamps, into timestamps
-- and leave genuinely invalid values
-- as null for observability.
-- ==========================================================
WITH date_checks AS (
  SELECT
    date_received,
    date_sent_to_company,
    COALESCE(
      TRY_TO_TIMESTAMP(date_received, "yyyy-MM-dd'T'HH:mm:ss.SSSX"),
      TRY_TO_TIMESTAMP(date_received, "yyyy-MM-dd'T'HH:mm:ssX"),
      TRY_TO_TIMESTAMP(date_received, 'MM/dd/yyyy'),
      TRY_TO_TIMESTAMP(date_received, 'M/d/yyyy'),
      TRY_TO_TIMESTAMP(date_received, 'yyyy-MM-dd')
    ) AS parsed_date_received,
    COALESCE(
      TRY_TO_TIMESTAMP(date_sent_to_company, "yyyy-MM-dd'T'HH:mm:ss.SSSX"),
      TRY_TO_TIMESTAMP(date_sent_to_company, "yyyy-MM-dd'T'HH:mm:ssX"),
      TRY_TO_TIMESTAMP(date_sent_to_company, 'MM/dd/yyyy'),
      TRY_TO_TIMESTAMP(date_sent_to_company, 'M/d/yyyy'),
      TRY_TO_TIMESTAMP(date_sent_to_company, 'yyyy-MM-dd')
    ) AS parsed_date_sent_to_company
  FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints
)
SELECT
  COUNT_IF(date_received IS NOT NULL AND parsed_date_received IS NULL) AS invalid_date_received_count,
  COUNT_IF(date_sent_to_company IS NOT NULL AND parsed_date_sent_to_company IS NULL) AS invalid_date_sent_to_company_count
FROM date_checks;


-- ==========================================================
-- Check: Examples of invalid raw date strings
--
-- Purpose:
-- Show sample raw values that fail date parsing.
--
-- Why this matters:
-- Engineers need representative bad values to decide whether the parsing logic
-- should be expanded or whether the source data should be treated as invalid.
--
-- Silver action:
-- Review samples before expanding parsing rules; avoid overfitting Silver to a
-- small set of malformed strings unless the pattern is genuinely recurring.
-- ==========================================================
WITH date_checks AS (
  SELECT
    date_received,
    date_sent_to_company,
    COALESCE(
      TRY_TO_TIMESTAMP(date_received, "yyyy-MM-dd'T'HH:mm:ss.SSSX"),
      TRY_TO_TIMESTAMP(date_received, "yyyy-MM-dd'T'HH:mm:ssX"),
      TRY_TO_TIMESTAMP(date_received, 'MM/dd/yyyy'),
      TRY_TO_TIMESTAMP(date_received, 'M/d/yyyy'),
      TRY_TO_TIMESTAMP(date_received, 'yyyy-MM-dd')
    ) AS parsed_date_received,
    COALESCE(
      TRY_TO_TIMESTAMP(date_sent_to_company, "yyyy-MM-dd'T'HH:mm:ss.SSSX"),
      TRY_TO_TIMESTAMP(date_sent_to_company, "yyyy-MM-dd'T'HH:mm:ssX"),
      TRY_TO_TIMESTAMP(date_sent_to_company, 'MM/dd/yyyy'),
      TRY_TO_TIMESTAMP(date_sent_to_company, 'M/d/yyyy'),
      TRY_TO_TIMESTAMP(date_sent_to_company, 'yyyy-MM-dd')
    ) AS parsed_date_sent_to_company
  FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints
)
SELECT *
FROM date_checks
WHERE (date_received IS NOT NULL AND parsed_date_received IS NULL)
   OR (date_sent_to_company IS NOT NULL AND parsed_date_sent_to_company IS NULL)
LIMIT 50;


-- ==========================================================
-- Check: Invalid categorical values in flag-style fields
--
-- Purpose:
-- Identify values outside the expected yes/no style domain for flag-like
-- complaint fields.
--
-- Why this matters:
-- Flag fields are often used directly in reporting and model features, so
-- inconsistent encodings can split identical business outcomes into many
-- categories.
--
-- Finding:
-- Unexpected flag values indicate the need for controlled standardisation.
--
-- Impact:
-- Timely-response and dispute metrics can be materially wrong if values such as
-- 'yes ', 'Y', or blank strings are not handled consistently.
--
-- Silver transformation:
-- Apply TRIM and map common yes/no variants into consistent values while
-- preserving null where the source is genuinely missing. This Bronze table
-- contains timely_response but does not contain consumer_disputed.
-- ==========================================================
SELECT
  'timely_response' AS column_name,
  timely_response AS raw_value,
  COUNT(*) AS row_count
FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints
WHERE timely_response IS NOT NULL
  AND UPPER(TRIM(timely_response)) NOT IN ('YES', 'NO')
GROUP BY timely_response
ORDER BY column_name, row_count DESC, raw_value;


-- ==========================================================
-- Check: Suspicious ZIP code patterns
--
-- Purpose:
-- Identify ZIP code values that are missing, oddly formatted, or likely to
-- fail standard US ZIP expectations.
--
-- Why this matters:
-- ZIP code is a high-value geographic dimension, but it must remain a string
-- so leading zeroes are not lost during processing.
--
-- Finding:
-- Suspicious ZIP code patterns often indicate padding issues, text corruption,
-- or international/non-standard source formatting.
--
-- Impact:
-- Geographic analysis can be distorted if ZIP values are cast to numbers or
-- if malformed values are treated as valid.
--
-- Silver transformation:
-- Preserve zip_code as a string, trim whitespace, and monitor non-standard
-- patterns rather than coercing the field into an unsafe numeric type.
-- ==========================================================
SELECT
  zip_code,
  COUNT(*) AS row_count
FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints
WHERE zip_code IS NOT NULL
  AND TRIM(zip_code) <> ''
  AND NOT TRIM(zip_code) RLIKE '^[0-9]{5}(-[0-9]{4})?$'
GROUP BY zip_code
ORDER BY row_count DESC, zip_code
LIMIT 100;


-- ==========================================================
-- Check: Unexpected null patterns by source file
--
-- Purpose:
-- Look for concentrated missingness that may be associated with a particular
-- source file or ingestion slice rather than the dataset as a whole.
--
-- Why this matters:
-- Data-quality issues that cluster by source file often signal upstream source
-- corruption, truncated exports, or schema shifts.
--
-- Silver action:
-- Preserve source metadata in Silver so quality issues can be traced back to
-- the ingestion artifact that produced them.
-- ==========================================================
SELECT
  _source_csv_name,
  _ingestion_date,
  COUNT(*) AS row_count,
  COUNT_IF(TRIM(COALESCE(product, '')) = '') AS missing_product_count,
  COUNT_IF(TRIM(COALESCE(issue, '')) = '') AS missing_issue_count,
  COUNT_IF(TRIM(COALESCE(company, '')) = '') AS missing_company_count,
  COUNT_IF(TRIM(COALESCE(state, '')) = '') AS missing_state_count
FROM fintech_lakehouse_dev.bronze.bronze_consumer_complaints
GROUP BY _source_csv_name, _ingestion_date
ORDER BY _ingestion_date DESC, row_count DESC;
