"""Build the Gold Consumer Complaints dimensional model.

This job transforms the complaint-level Silver table into a small analytical
warehouse model suitable for dashboards, BI, and downstream feature engineering.
The implementation keeps one fact row per complaint and publishes supporting
dimensions for date, product, company, and state.

Gold processing flow:
1. Validate that the Silver source table exists and contains rows.
2. Create the Gold schema if needed.
3. Build reusable dimensions from the cleaned Silver table.
4. Join dimensions back to Silver to create the complaint fact table.
5. Overwrite all Gold tables idempotently.
6. Validate row counts, uniqueness, and referential completeness.
"""

from __future__ import annotations

import logging

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window


CATALOG = "fintech_lakehouse_dev"
SILVER_SCHEMA = "silver"
GOLD_SCHEMA = "gold"

SOURCE_TABLE = f"{CATALOG}.{SILVER_SCHEMA}.silver_consumer_complaints"

DIM_DATE_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.dim_date"
DIM_PRODUCT_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.dim_product"
DIM_COMPANY_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.dim_company"
DIM_STATE_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.dim_state"
FACT_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.fact_consumer_complaints"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

LOGGER = logging.getLogger(__name__)

MISSING_MEMBER_LABEL = "NOT_PROVIDED"


# ------------------------------------------------------------
# Validate the cleaned Silver table before building any Gold
# assets. Gold should only run when the complaint-level Silver
# table exists and contains rows.
# ------------------------------------------------------------
def validate_source_table(spark: SparkSession) -> tuple[DataFrame, int]:
    """Validate that the Silver source table exists and contains records."""

    if not spark.catalog.tableExists(SOURCE_TABLE):
        raise RuntimeError(f"Silver source table does not exist: {SOURCE_TABLE}")

    silver_dataframe = spark.table(SOURCE_TABLE)
    silver_row_count = silver_dataframe.count()

    if silver_row_count == 0:
        raise RuntimeError(f"Silver source table is empty: {SOURCE_TABLE}")

    LOGGER.info(
        "Validated Silver source table %s with %s rows.",
        SOURCE_TABLE,
        f"{silver_row_count:,}",
    )
    return silver_dataframe, silver_row_count


# ------------------------------------------------------------
# Ensure the Gold schema exists so dimensions and facts can be
# written as managed Unity Catalog Delta tables in a consistent
# location.
# ------------------------------------------------------------
def ensure_gold_schema_exists(spark: SparkSession) -> None:
    """Create the Gold schema if it is missing."""

    spark.sql(
        f"""
        CREATE SCHEMA IF NOT EXISTS {CATALOG}.{GOLD_SCHEMA}
        COMMENT 'Business-ready dimensional warehouse for consumer complaints analytics.'
        """
    )


# ------------------------------------------------------------
# Build a conformed date dimension from all complaint received
# and company-response dates. The dimension also includes an
# explicit fallback row for unmatched or missing dates.
# ------------------------------------------------------------
def build_dim_date(silver_dataframe: DataFrame) -> DataFrame:
    """Build a conformed date dimension from received and sent complaint dates."""

    all_dates = (
        silver_dataframe.select(F.to_date("date_received").alias("full_date"))
        .union(
            silver_dataframe.select(F.to_date("date_sent_to_company").alias("full_date"))
        )
        .filter(F.col("full_date").isNotNull())
        .distinct()
    )

    dim_date = (
        all_dates
        .withColumn("date_key", F.date_format("full_date", "yyyyMMdd").cast(T.IntegerType()))
        .withColumn("calendar_year", F.year("full_date"))
        .withColumn("calendar_quarter", F.quarter("full_date"))
        .withColumn("calendar_month", F.month("full_date"))
        .withColumn("calendar_month_name", F.date_format("full_date", "MMMM"))
        .withColumn("calendar_day", F.dayofmonth("full_date"))
        .withColumn("day_of_week", F.dayofweek("full_date"))
        .withColumn("day_name", F.date_format("full_date", "EEEE"))
        .withColumn("is_weekend", F.dayofweek("full_date").isin(1, 7))
    )

    fallback_row = spark_single_row(
        silver_dataframe.sparkSession,
        {
            "date_key": 0,
            "full_date": None,
            "calendar_year": None,
            "calendar_quarter": None,
            "calendar_month": None,
            "calendar_month_name": MISSING_MEMBER_LABEL,
            "calendar_day": None,
            "day_of_week": None,
            "day_name": MISSING_MEMBER_LABEL,
            "is_weekend": None,
        },
        schema=T.StructType(
            [
                T.StructField("date_key", T.IntegerType(), False),
                T.StructField("full_date", T.DateType(), True),
                T.StructField("calendar_year", T.IntegerType(), True),
                T.StructField("calendar_quarter", T.IntegerType(), True),
                T.StructField("calendar_month", T.IntegerType(), True),
                T.StructField("calendar_month_name", T.StringType(), True),
                T.StructField("calendar_day", T.IntegerType(), True),
                T.StructField("day_of_week", T.IntegerType(), True),
                T.StructField("day_name", T.StringType(), True),
                T.StructField("is_weekend", T.BooleanType(), True),
            ]
        ),
    )

    return fallback_row.unionByName(dim_date)


# ------------------------------------------------------------
# Build the product dimension at the product/sub-product grain.
# This gives Gold a stable surrogate key for complaint product
# analysis while preserving the business labels from Silver.
# ------------------------------------------------------------
def build_dim_product(silver_dataframe: DataFrame) -> DataFrame:
    """Build the product dimension at the product and sub-product grain."""

    base_dimension = (
        silver_dataframe
        .select("product", "sub_product")
        .distinct()
        .withColumn(
            "_sort_product",
            F.coalesce(F.col("product"), F.lit(MISSING_MEMBER_LABEL)),
        )
        .withColumn(
            "sub_product",
            F.coalesce(F.col("sub_product"), F.lit(MISSING_MEMBER_LABEL)),
        )
        .withColumn(
            "_sort_sub_product",
            F.coalesce(F.col("sub_product"), F.lit(MISSING_MEMBER_LABEL)),
        )
        .withColumn(
            "product_key",
            F.row_number().over(
                Window.orderBy("_sort_product", "_sort_sub_product")
            ),
        )
        .drop("_sort_product", "_sort_sub_product")
        .select("product_key", "product", "sub_product")
    )

    fallback_row = spark_single_row(
        silver_dataframe.sparkSession,
        {
            "product_key": 0,
            "product": MISSING_MEMBER_LABEL,
            "sub_product": MISSING_MEMBER_LABEL,
        },
        schema=T.StructType(
            [
                T.StructField("product_key", T.IntegerType(), False),
                T.StructField("product", T.StringType(), True),
                T.StructField("sub_product", T.StringType(), True),
            ]
        ),
    )

    return fallback_row.unionByName(base_dimension)


# ------------------------------------------------------------
# Build the company dimension so complaint facts can join to a
# reusable organisation lookup instead of repeating company text
# in every downstream aggregate model.
# ------------------------------------------------------------
def build_dim_company(silver_dataframe: DataFrame) -> DataFrame:
    """Build the company dimension."""

    base_dimension = (
        silver_dataframe
        .select("company")
        .distinct()
        .withColumn(
            "_sort_company",
            F.coalesce(F.col("company"), F.lit(MISSING_MEMBER_LABEL)),
        )
        .withColumn(
            "company_key",
            F.row_number().over(Window.orderBy("_sort_company")),
        )
        .drop("_sort_company")
        .select("company_key", "company")
    )

    fallback_row = spark_single_row(
        silver_dataframe.sparkSession,
        {
            "company_key": 0,
            "company": MISSING_MEMBER_LABEL,
        },
        schema=T.StructType(
            [
                T.StructField("company_key", T.IntegerType(), False),
                T.StructField("company", T.StringType(), True),
            ]
        ),
    )

    return fallback_row.unionByName(base_dimension)


# ------------------------------------------------------------
# Build the state dimension from the cleaned Silver geography
# field. A dedicated dimension supports state-level analytics
# and keeps null or missing states mapped to a single fallback member.
# ------------------------------------------------------------
def build_dim_state(silver_dataframe: DataFrame) -> DataFrame:
    """Build the state dimension from the cleaned Silver geography fields."""

    base_dimension = (
        silver_dataframe
        .select("state")
        .withColumn(
            "state",
            F.coalesce(F.col("state"), F.lit(MISSING_MEMBER_LABEL)),
        )
        .distinct()
        .withColumn(
            "_sort_state",
            F.coalesce(F.col("state"), F.lit(MISSING_MEMBER_LABEL)),
        )
        .withColumn(
            "state_key",
            F.row_number().over(Window.orderBy("_sort_state")),
        )
        .drop("_sort_state")
        .select("state_key", "state")
    )

    fallback_row = spark_single_row(
        silver_dataframe.sparkSession,
        {
            "state_key": 0,
            "state": MISSING_MEMBER_LABEL,
        },
        schema=T.StructType(
            [
                T.StructField("state_key", T.IntegerType(), False),
                T.StructField("state", T.StringType(), True),
            ]
        ),
    )

    return fallback_row.unionByName(base_dimension)


# ------------------------------------------------------------
# Build the complaint fact table by joining Silver complaints to
# each conformed dimension and preserving the operational and
# lineage fields needed for analytics and traceability.
# ------------------------------------------------------------
def build_fact_consumer_complaints(
    silver_dataframe: DataFrame,
    dim_date: DataFrame,
    dim_product: DataFrame,
    dim_company: DataFrame,
    dim_state: DataFrame,
) -> DataFrame:
    """Build the complaint fact table by joining Silver to the conformed dimensions."""

    received_date_dimension = (
        dim_date
        .select(
            F.col("date_key").alias("received_date_key"),
            F.col("full_date").alias("received_full_date"),
        )
    )

    sent_date_dimension = (
        dim_date
        .select(
            F.col("date_key").alias("sent_date_key"),
            F.col("full_date").alias("sent_full_date"),
        )
    )

    fact_dataframe = (
        silver_dataframe.alias("s")
        .join(
            dim_product.alias("p"),
            on=[
                silver_dataframe["product"].eqNullSafe(dim_product["product"]),
                silver_dataframe["sub_product"].eqNullSafe(dim_product["sub_product"]),
            ],
            how="left",
        )
        .join(
            dim_company.alias("c"),
            on=silver_dataframe["company"].eqNullSafe(dim_company["company"]),
            how="left",
        )
        .join(
            dim_state.alias("st"),
            on=silver_dataframe["state"].eqNullSafe(dim_state["state"]),
            how="left",
        )
        .join(
            received_date_dimension.alias("dr"),
            on=F.to_date(silver_dataframe["date_received"]) == received_date_dimension["received_full_date"],
            how="left",
        )
        .join(
            sent_date_dimension.alias("ds"),
            on=F.to_date(silver_dataframe["date_sent_to_company"]) == sent_date_dimension["sent_full_date"],
            how="left",
        )
        .select(
            F.col("s.complaint_id"),
            F.coalesce(F.col("dr.received_date_key"), F.lit(0)).alias("received_date_key"),
            F.coalesce(F.col("ds.sent_date_key"), F.lit(0)).alias("sent_date_key"),
            F.coalesce(F.col("p.product_key"), F.lit(0)).alias("product_key"),
            F.coalesce(F.col("c.company_key"), F.lit(0)).alias("company_key"),
            F.coalesce(F.col("st.state_key"), F.lit(0)).alias("state_key"),
            F.col("s.issue"),
            F.col("s.sub_issue"),
            F.col("s.zip_code"),
            F.col("s.tags"),
            F.col("s.submitted_via"),
            F.col("s.company_response_to_consumer"),
            F.col("s.timely_response"),
            F.when(F.col("s.consumer_complaint_narrative").isNotNull(), F.lit(True)).otherwise(F.lit(False)).alias("has_consumer_narrative"),
            F.when(F.col("s.tags").isNotNull(), F.lit(True)).otherwise(F.lit(False)).alias("has_tags"),
            F.when(F.col("s.date_sent_to_company").isNotNull(), F.lit(True)).otherwise(F.lit(False)).alias("has_company_response_date"),
            F.col("s._ingestion_date"),
            F.col("s._ingested_at"),
            F.col("s._source_zip_path"),
            F.col("s._source_csv_name"),
            F.col("s._silver_processed_at"),
            F.col("s._silver_record_status"),
            F.current_timestamp().alias("_gold_processed_at"),
        )
    )

    return fact_dataframe


# ------------------------------------------------------------
# Create a single-row Spark DataFrame with a controlled schema.
# This is used for explicit fallback members in Gold dimensions
# so foreign keys always have a safe fallback value.
# ------------------------------------------------------------
def spark_single_row(
    spark: SparkSession,
    row: dict[str, object],
    schema: T.StructType,
) -> DataFrame:
    """Create a one-row Spark DataFrame with a specific schema."""

    return spark.createDataFrame([row], schema=schema)


# ------------------------------------------------------------
# Write any Gold table using an idempotent full overwrite. Gold
# is modelled as a refreshed analytical layer rather than an
# append-only raw ingestion surface.
# ------------------------------------------------------------
def write_table(dataframe: DataFrame, target_table: str) -> None:
    """Overwrite a Gold Delta table idempotently."""

    LOGGER.info("Writing Gold table to %s", target_table)

    (
        dataframe.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(target_table)
    )


# ------------------------------------------------------------
# Validate that the dimensional model is complete after the
# write: dimensions must be populated, the fact grain must match
# Silver complaint grain, and no dimension keys may be null.
# ------------------------------------------------------------
def validate_gold_tables(
    spark: SparkSession,
    silver_row_count: int,
) -> None:
    """Run post-write validations across the Gold dimensional model."""

    dim_date_count = spark.table(DIM_DATE_TABLE).count()
    dim_product_count = spark.table(DIM_PRODUCT_TABLE).count()
    dim_company_count = spark.table(DIM_COMPANY_TABLE).count()
    dim_state_count = spark.table(DIM_STATE_TABLE).count()
    fact_row_count = spark.table(FACT_TABLE).count()

    if min(dim_date_count, dim_product_count, dim_company_count, dim_state_count) <= 0:
        raise RuntimeError("Gold validation failed: one or more dimensions are empty.")

    if fact_row_count == 0:
        raise RuntimeError("Gold validation failed: fact table row count is zero.")

    if fact_row_count != silver_row_count:
        raise RuntimeError(
            "Gold validation failed: fact row count does not match the Silver complaint grain."
        )

    duplicate_complaint_ids = (
        spark.table(FACT_TABLE)
        .groupBy("complaint_id")
        .count()
        .filter(F.col("count") > 1)
        .count()
    )
    if duplicate_complaint_ids > 0:
        raise RuntimeError(
            f"Gold validation failed: {duplicate_complaint_ids} duplicate complaint IDs found in the fact table."
        )

    null_dimension_keys = (
        spark.table(FACT_TABLE)
        .filter(
            F.col("product_key").isNull()
            | F.col("company_key").isNull()
            | F.col("state_key").isNull()
            | F.col("received_date_key").isNull()
            | F.col("sent_date_key").isNull()
        )
        .count()
    )
    if null_dimension_keys > 0:
        raise RuntimeError(
            f"Gold validation failed: {null_dimension_keys} fact rows contain null dimension keys."
        )

    LOGGER.info(
        "Gold validation successful. Dimensions: date=%s, product=%s, company=%s, state=%s. Fact rows=%s.",
        f"{dim_date_count:,}",
        f"{dim_product_count:,}",
        f"{dim_company_count:,}",
        f"{dim_state_count:,}",
        f"{fact_row_count:,}",
    )


# ------------------------------------------------------------
# Run the end-to-end Gold warehouse build in dependency order:
# validate Silver, build dimensions, build the complaint fact,
# write all Gold tables, then run final validation checks.
# ------------------------------------------------------------
def main() -> None:
    """Run the full Gold warehouse build for consumer complaints."""

    spark = SparkSession.builder.getOrCreate()

    LOGGER.info("Starting Consumer Complaints Gold processing.")

    silver_dataframe, silver_row_count = validate_source_table(spark)
    ensure_gold_schema_exists(spark)

    dim_date = build_dim_date(silver_dataframe)
    dim_product = build_dim_product(silver_dataframe)
    dim_company = build_dim_company(silver_dataframe)
    dim_state = build_dim_state(silver_dataframe)
    fact_consumer_complaints = build_fact_consumer_complaints(
        silver_dataframe=silver_dataframe,
        dim_date=dim_date,
        dim_product=dim_product,
        dim_company=dim_company,
        dim_state=dim_state,
    )

    write_table(dim_date, DIM_DATE_TABLE)
    write_table(dim_product, DIM_PRODUCT_TABLE)
    write_table(dim_company, DIM_COMPANY_TABLE)
    write_table(dim_state, DIM_STATE_TABLE)
    write_table(fact_consumer_complaints, FACT_TABLE)

    validate_gold_tables(spark, silver_row_count)

    LOGGER.info("Consumer Complaints Gold processing finished successfully.")


if __name__ == "__main__":
    main()
