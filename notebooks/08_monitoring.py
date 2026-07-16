# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Monitoramento do modelo
# MAGIC
# MAGIC Este notebook monitora:
# MAGIC
# MAGIC - qualidade dos dados transacionais;
# MAGIC - volume da população de scoring;
# MAGIC - drift das features;
# MAGIC - drift dos scores;
# MAGIC - performance sintética após a maturação da target.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Imports

# COMMAND ----------

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from pyspark.sql.functions import *

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Configurações

# COMMAND ----------

HISTORY_TABLE = (
    "workspace.synthetic_layer."
    "online_retail_transactions_full"
)

GENERATION_RUNS_TABLE = (
    "workspace.synthetic_layer."
    "generation_runs"
)

SCORING_TABLE = (
    "workspace.gold_layer."
    "customer_reactivation_scoring"
)

MODELING_TABLE = (
    "workspace.gold_layer."
    "customer_reactivation_modeling"
)

MONITORING_SCHEMA = "workspace.monitoring_layer"

SUMMARY_TABLE = (
    "workspace.monitoring_layer."
    "model_monitoring_summary"
)

FEATURE_MONITORING_TABLE = (
    "workspace.monitoring_layer."
    "feature_monitoring"
)

PERFORMANCE_TABLE = (
    "workspace.monitoring_layer."
    "performance_monitoring"
)

REFERENCE_DATE = "2011-11-08"
TARGET_WINDOW_DAYS = 30

FEATURE_COLUMNS = [
    "inactive_days",
    "number_orders",
    "total_items",
    "total_spent",
    "unique_products",
    "distinct_countries",
    "customer_tenure_days",
    "average_ticket",
    "average_amount_per_country",
    "average_amount_per_stock_code",
]

TRANSACTION_COLUMNS = [
    "invoice",
    "stock_code",
    "description",
    "quantity",
    "invoice_date",
    "price",
    "customer_id",
    "country",
]

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Parâmetro da rodada

# COMMAND ----------

try:
    dbutils.widgets.get("monitor_scoring_date")
except Exception:
    dbutils.widgets.text(
        "monitor_scoring_date",
        "",
        "Scoring date (YYYY-MM-DD)",
    )

monitor_scoring_date = (
    dbutils.widgets
    .get("monitor_scoring_date")
    .strip()
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Funções auxiliares

# COMMAND ----------

def psi_status(psi_value):

    if psi_value >= 0.25:
        return "CRITICAL"

    if psi_value >= 0.10:
        return "WARNING"

    return "PASS"


def combine_status(statuses):

    if "CRITICAL" in statuses:
        return "CRITICAL"

    if "WARNING" in statuses:
        return "WARNING"

    return "PASS"


def calculate_psi(
    reference_values,
    current_values,
    bins=10,
):

    reference_values = np.asarray(
        reference_values,
        dtype=float,
    )

    current_values = np.asarray(
        current_values,
        dtype=float,
    )

    reference_values = reference_values[
        np.isfinite(reference_values)
    ]

    current_values = current_values[
        np.isfinite(current_values)
    ]

    if (
        len(reference_values) == 0
        or len(current_values) == 0
    ):
        return np.nan

    unique_values = np.unique(
        np.concatenate(
            [
                reference_values,
                current_values,
            ]
        )
    )

    epsilon = 1e-6

    if len(unique_values) <= 10:

        reference_distribution = np.array(
            [
                np.mean(
                    reference_values == value
                )
                for value in unique_values
            ]
        )

        current_distribution = np.array(
            [
                np.mean(
                    current_values == value
                )
                for value in unique_values
            ]
        )

    else:

        quantiles = np.linspace(
            0,
            1,
            bins + 1,
        )

        edges = np.unique(
            np.quantile(
                reference_values,
                quantiles,
            )
        )

        if len(edges) < 3:
            return 0.0

        edges[0] = -np.inf
        edges[-1] = np.inf

        reference_distribution = (
            np.histogram(
                reference_values,
                bins=edges,
            )[0]
            / len(reference_values)
        )

        current_distribution = (
            np.histogram(
                current_values,
                bins=edges,
            )[0]
            / len(current_values)
        )

    reference_distribution = np.clip(
        reference_distribution,
        epsilon,
        None,
    )

    current_distribution = np.clip(
        current_distribution,
        epsilon,
        None,
    )

    psi_value = np.sum(
        (
            current_distribution
            - reference_distribution
        )
        * np.log(
            current_distribution
            / reference_distribution
        )
    )

    return float(psi_value)


def calculate_top_k_metrics(
    y_true,
    y_score,
    top_k,
):

    ranking_df = pd.DataFrame(
        {
            "target": y_true,
            "score": y_score,
        }
    ).sort_values(
        "score",
        ascending=False,
    )

    selected_rows = max(
        1,
        int(
            np.ceil(
                len(ranking_df)
                * top_k
            )
        ),
    )

    selected_df = ranking_df.head(
        selected_rows
    )

    positive_rate = ranking_df[
        "target"
    ].mean()

    precision_at_k = selected_df[
        "target"
    ].mean()

    total_positives = ranking_df[
        "target"
    ].sum()

    recall_at_k = (
        selected_df["target"].sum()
        / total_positives
        if total_positives > 0
        else 0.0
    )

    lift_at_k = (
        precision_at_k
        / positive_rate
        if positive_rate > 0
        else 0.0
    )

    return {
        "precision": float(
            precision_at_k
        ),
        "recall": float(
            recall_at_k
        ),
        "lift": float(
            lift_at_k
        ),
    }


def write_replace_where(
    dataframe,
    table_name,
    scoring_date,
):

    if spark.catalog.tableExists(
        table_name
    ):

        (
            dataframe
            .write
            .format("delta")
            .mode("overwrite")
            .option(
                "replaceWhere",
                (
                    "scoring_date = "
                    f"DATE '{scoring_date}'"
                ),
            )
            .saveAsTable(
                table_name
            )
        )

    else:

        (
            dataframe
            .write
            .format("delta")
            .mode("overwrite")
            .saveAsTable(
                table_name
            )
        )

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Validação das fontes

# COMMAND ----------

required_tables = [
    HISTORY_TABLE,
    SCORING_TABLE,
    MODELING_TABLE,
]

missing_tables = [
    table_name
    for table_name in required_tables
    if not spark.catalog.tableExists(
        table_name
    )
]

if missing_tables:
    raise ValueError(
        f"Tabelas ausentes: {missing_tables}"
    )

spark.sql(
    f"""
    CREATE SCHEMA IF NOT EXISTS
    {MONITORING_SCHEMA}
    """
)

# COMMAND ----------

history_df = spark.table(
    HISTORY_TABLE
)

scoring_df = spark.table(
    SCORING_TABLE
)

modeling_df = spark.table(
    MODELING_TABLE
)

# COMMAND ----------

if monitor_scoring_date:

    scoring_date = datetime.strptime(
        monitor_scoring_date,
        "%Y-%m-%d",
    ).date()

else:

    scoring_date = (
        scoring_df
        .agg(
            max("scoring_date").alias(
                "max_scoring_date"
            )
        )
        .first()[
            "max_scoring_date"
        ]
    )

if scoring_date is None:
    raise ValueError(
        "Não foi possível definir a scoring_date."
    )

# COMMAND ----------

current_scoring_df = (
    scoring_df
    .filter(
        col("scoring_date")
        == lit(scoring_date)
    )
)

if current_scoring_df.limit(1).count() == 0:
    raise ValueError(
        (
            "Não existem resultados de scoring "
            f"para {scoring_date}."
        )
    )

# COMMAND ----------

reference_df = (
    modeling_df
    .filter(
        col("reference_date")
        == lit(REFERENCE_DATE)
    )
)

if reference_df.limit(1).count() == 0:
    raise ValueError(
        (
            "A população de referência não existe "
            f"para {REFERENCE_DATE}."
        )
    )

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Período transacional da rodada

# COMMAND ----------

previous_scoring_date = (
    scoring_df
    .filter(
        col("scoring_date")
        < lit(scoring_date)
    )
    .agg(
        max("scoring_date").alias(
            "previous_scoring_date"
        )
    )
    .first()[
        "previous_scoring_date"
    ]
)

batch_start_date = (
    previous_scoring_date
    + timedelta(days=1)
    if previous_scoring_date is not None
    else scoring_date
    - timedelta(days=29)
)

batch_transactions_df = (
    history_df
    .withColumn(
        "purchase_date",
        to_date(
            col("invoice_date")
        ),
    )
    .filter(
        col("purchase_date")
        >= lit(batch_start_date)
    )
    .filter(
        col("purchase_date")
        <= lit(scoring_date)
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Cenário sintético da rodada

# COMMAND ----------

scenario = "unknown"

if spark.catalog.tableExists(
    GENERATION_RUNS_TABLE
):

    scenario_row = (
        spark.table(
            GENERATION_RUNS_TABLE
        )
        .filter(
            col("batch_start_date")
            <= lit(scoring_date)
        )
        .filter(
            col("batch_end_date")
            >= lit(scoring_date)
        )
        .orderBy(
            desc("generated_at")
        )
        .limit(1)
        .collect()
    )

    if scenario_row:
        scenario = scenario_row[0][
            "scenario"
        ]

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Qualidade dos dados transacionais

# COMMAND ----------

batch_rows = (
    batch_transactions_df
    .count()
)

batch_invoices = (
    batch_transactions_df
    .select("invoice")
    .distinct()
    .count()
)

batch_customers = (
    batch_transactions_df
    .select("customer_id")
    .distinct()
    .count()
)

# COMMAND ----------

null_condition = None

for column_name in TRANSACTION_COLUMNS:

    current_condition = col(
        column_name
    ).isNull()

    null_condition = (
        current_condition
        if null_condition is None
        else null_condition
        | current_condition
    )

null_rows = (
    batch_transactions_df
    .filter(null_condition)
    .count()
)

duplicate_rows = (
    batch_rows
    - batch_transactions_df
    .dropDuplicates(
        TRANSACTION_COLUMNS
    )
    .count()
)

invalid_price_rows = (
    batch_transactions_df
    .filter(
        col("price") <= 0
    )
    .count()
)

invalid_quantity_rows = (
    batch_transactions_df
    .filter(
        col("quantity") <= 0
    )
    .count()
)

cancelled_rows = (
    batch_transactions_df
    .filter(
        col("invoice").startswith("C")
    )
    .count()
)

test_rows = (
    batch_transactions_df
    .filter(
        col("stock_code").startswith(
            "TEST"
        )
    )
    .count()
)

discount_rows = (
    batch_transactions_df
    .filter(
        col("description")
        == "Discount"
    )
    .count()
)

# COMMAND ----------

dq_status = (
    "CRITICAL"
    if (
        null_rows > 0
        or duplicate_rows > 0
        or invalid_price_rows > 0
        or invalid_quantity_rows > 0
    )
    else "PASS"
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Resumo da população e dos scores

# COMMAND ----------

current_scoring_pandas = (
    current_scoring_df
    .toPandas()
)

reference_pandas = (
    reference_df
    .select(
        *FEATURE_COLUMNS
    )
    .toPandas()
)

number_customers = len(
    current_scoring_pandas
)

reference_customers = (
    reference_df
    .select("customer_id")
    .distinct()
    .count()
)

score_min = float(
    current_scoring_pandas[
        "reactivation_probability"
    ].min()
)

score_mean = float(
    current_scoring_pandas[
        "reactivation_probability"
    ].mean()
)

score_median = float(
    current_scoring_pandas[
        "reactivation_probability"
    ].median()
)

score_p90 = float(
    current_scoring_pandas[
        "reactivation_probability"
    ].quantile(0.90)
)

score_max = float(
    current_scoring_pandas[
        "reactivation_probability"
    ].max()
)

predicted_positive_rate = float(
    current_scoring_pandas[
        "prediction_0_5"
    ].mean()
)

# COMMAND ----------

priority_counts = (
    current_scoring_pandas[
        "priority_group"
    ]
    .value_counts()
    .to_dict()
)

top_10_customers = int(
    priority_counts.get(
        "TOP_10",
        0,
    )
)

top_20_customers = int(
    priority_counts.get(
        "TOP_20",
        0,
    )
)

top_30_customers = int(
    priority_counts.get(
        "TOP_30",
        0,
    )
)

remainder_customers = int(
    priority_counts.get(
        "REMAINDER",
        0,
    )
)

priority_total = (
    top_10_customers
    + top_20_customers
    + top_30_customers
    + remainder_customers
)

priority_status = (
    "PASS"
    if priority_total
    == number_customers
    else "CRITICAL"
)

# COMMAND ----------

from builtins import abs as builtin_abs

if previous_scoring_date is not None:

    previous_number_customers = (
        scoring_df
        .filter(
            col("scoring_date")
            == lit(
                previous_scoring_date
            )
        )
        .count()
    )

else:

    previous_number_customers = (
        reference_customers
    )

volume_change = (
    (
        number_customers
        - previous_number_customers
    )
    / previous_number_customers
    if previous_number_customers > 0
    else 0.0
)

absolute_volume_change = builtin_abs(
    volume_change
)

if absolute_volume_change >= 0.50:

    volume_status = "CRITICAL"

elif absolute_volume_change >= 0.30:

    volume_status = "WARNING"

else:

    volume_status = "PASS"

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Drift das features

# COMMAND ----------

feature_monitoring_rows = []

for feature_name in FEATURE_COLUMNS:

    reference_values = (
        reference_pandas[
            feature_name
        ]
        .astype(float)
    )

    current_values = (
        current_scoring_pandas[
            feature_name
        ]
        .astype(float)
    )

    psi_value = calculate_psi(
        reference_values,
        current_values,
    )

    feature_monitoring_rows.append(
        {
            "scoring_date":
                scoring_date,

            "reference_date":
                datetime.strptime(
                    REFERENCE_DATE,
                    "%Y-%m-%d",
                ).date(),

            "feature_name":
                feature_name,

            "reference_mean":
                float(
                    reference_values.mean()
                ),

            "current_mean":
                float(
                    current_values.mean()
                ),

            "reference_median":
                float(
                    reference_values.median()
                ),

            "current_median":
                float(
                    current_values.median()
                ),

            "current_std":
                float(
                    current_values.std()
                ),

            "current_min":
                float(
                    current_values.min()
                ),

            "current_p10":
                float(
                    current_values.quantile(
                        0.10
                    )
                ),

            "current_p25":
                float(
                    current_values.quantile(
                        0.25
                    )
                ),

            "current_p75":
                float(
                    current_values.quantile(
                        0.75
                    )
                ),

            "current_p90":
                float(
                    current_values.quantile(
                        0.90
                    )
                ),

            "current_max":
                float(
                    current_values.max()
                ),

            "psi":
                float(
                    psi_value
                ),

            "status":
                psi_status(
                    psi_value
                ),
        }
    )

feature_monitoring_pandas = pd.DataFrame(
    feature_monitoring_rows
)

maximum_feature_psi = float(
    feature_monitoring_pandas[
        "psi"
    ].max()
)

feature_drift_status = psi_status(
    maximum_feature_psi
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Drift dos scores

# COMMAND ----------

first_scoring_date = (
    scoring_df
    .agg(
        min("scoring_date").alias(
            "first_scoring_date"
        )
    )
    .first()[
        "first_scoring_date"
    ]
)

score_reference_values = (
    scoring_df
    .filter(
        col("scoring_date")
        == lit(first_scoring_date)
    )
    .select(
        "reactivation_probability"
    )
    .toPandas()[
        "reactivation_probability"
    ]
)

score_psi = calculate_psi(
    score_reference_values,
    current_scoring_pandas[
        "reactivation_probability"
    ],
)

score_drift_status = psi_status(
    score_psi
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Maturação da target e performance sintética

# COMMAND ----------

maximum_history_date = (
    history_df
    .agg(
        max(
            to_date(
                col("invoice_date")
            )
        ).alias(
            "maximum_history_date"
        )
    )
    .first()[
        "maximum_history_date"
    ]
)

target_end_date = (
    scoring_date
    + timedelta(
        days=TARGET_WINDOW_DAYS
    )
)

target_status = (
    "MATURED"
    if maximum_history_date
    >= target_end_date
    else "NOT_MATURED"
)

performance_row = None

# COMMAND ----------

if target_status == "MATURED":

    future_valid_transactions_df = (
        history_df
        .withColumn(
            "purchase_date",
            to_date(
                col("invoice_date")
            ),
        )
        .filter(
            col("purchase_date")
            > lit(scoring_date)
        )
        .filter(
            col("purchase_date")
            <= lit(target_end_date)
        )
        .filter(
            ~col("invoice").startswith("C")
        )
        .filter(
            ~col("stock_code").startswith(
                "TEST"
            )
        )
        .filter(
            col("description")
            != "Discount"
        )
        .filter(
            col("price") > 0
        )
        .filter(
            col("quantity") > 0
        )
        .select(
            "customer_id"
        )
        .distinct()
        .withColumn(
            "target_reactivated_30d",
            lit(1),
        )
    )

    matured_scoring_df = (
        current_scoring_df
        .join(
            future_valid_transactions_df,
            on="customer_id",
            how="left",
        )
        .fillna(
            {
                "target_reactivated_30d":
                    0
            }
        )
    )

    matured_scoring_pandas = (
        matured_scoring_df
        .select(
            "target_reactivated_30d",
            "reactivation_probability",
        )
        .toPandas()
    )

    y_true = (
        matured_scoring_pandas[
            "target_reactivated_30d"
        ]
        .astype(int)
        .to_numpy()
    )

    y_score = (
        matured_scoring_pandas[
            "reactivation_probability"
        ]
        .astype(float)
        .to_numpy()
    )

    positive_rate = float(
        y_true.mean()
    )

    pr_auc = float(
        average_precision_score(
            y_true,
            y_score,
        )
    )

    roc_auc = (
        float(
            roc_auc_score(
                y_true,
                y_score,
            )
        )
        if len(
            np.unique(
                y_true
            )
        ) == 2
        else np.nan
    )

    brier_score = float(
        brier_score_loss(
            y_true,
            y_score,
        )
    )

    metrics_10 = (
        calculate_top_k_metrics(
            y_true,
            y_score,
            0.10,
        )
    )

    metrics_20 = (
        calculate_top_k_metrics(
            y_true,
            y_score,
            0.20,
        )
    )

    metrics_30 = (
        calculate_top_k_metrics(
            y_true,
            y_score,
            0.30,
        )
    )

    performance_row = {
        "scoring_date":
            scoring_date,

        "target_end_date":
            target_end_date,

        "performance_type":
            "synthetic_performance",

        "positive_rate":
            positive_rate,

        "pr_auc":
            pr_auc,

        "roc_auc":
            roc_auc,

        "brier_score":
            brier_score,

        "precision_at_10pct":
            metrics_10[
                "precision"
            ],

        "recall_at_10pct":
            metrics_10[
                "recall"
            ],

        "lift_at_10pct":
            metrics_10[
                "lift"
            ],

        "precision_at_20pct":
            metrics_20[
                "precision"
            ],

        "recall_at_20pct":
            metrics_20[
                "recall"
            ],

        "lift_at_20pct":
            metrics_20[
                "lift"
            ],

        "precision_at_30pct":
            metrics_30[
                "precision"
            ],

        "recall_at_30pct":
            metrics_30[
                "recall"
            ],

        "lift_at_30pct":
            metrics_30[
                "lift"
            ],

        "calculated_at":
            datetime.utcnow(),
    }

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Status consolidado

# COMMAND ----------

overall_status = combine_status(
    [
        dq_status,
        volume_status,
        priority_status,
        feature_drift_status,
        score_drift_status,
    ]
)

# COMMAND ----------

summary_row = {
    "scoring_date":
        scoring_date,

    "scenario":
        scenario,

    "model_name":
        str(
            current_scoring_pandas[
                "model_name"
            ].iloc[0]
        ),

    "model_version":
        str(
            current_scoring_pandas[
                "model_version"
            ].iloc[0]
        ),

    "batch_start_date":
        batch_start_date,

    "batch_rows":
        int(
            batch_rows
        ),

    "batch_invoices":
        int(
            batch_invoices
        ),

    "batch_customers":
        int(
            batch_customers
        ),

    "null_rows":
        int(
            null_rows
        ),

    "duplicate_rows":
        int(
            duplicate_rows
        ),

    "invalid_price_rows":
        int(
            invalid_price_rows
        ),

    "invalid_quantity_rows":
        int(
            invalid_quantity_rows
        ),

    "cancelled_rows":
        int(
            cancelled_rows
        ),

    "test_rows":
        int(
            test_rows
        ),

    "discount_rows":
        int(
            discount_rows
        ),

    "number_customers":
        int(
            number_customers
        ),

    "previous_number_customers":
        int(
            previous_number_customers
        ),

    "volume_change":
        float(
            volume_change
        ),

    "score_min":
        score_min,

    "score_mean":
        score_mean,

    "score_median":
        score_median,

    "score_p90":
        score_p90,

    "score_max":
        score_max,

    "predicted_positive_rate":
        predicted_positive_rate,

    "top_10_customers":
        top_10_customers,

    "top_20_customers":
        top_20_customers,

    "top_30_customers":
        top_30_customers,

    "remainder_customers":
        remainder_customers,

    "maximum_feature_psi":
        maximum_feature_psi,

    "score_psi":
        float(
            score_psi
        ),

    "dq_status":
        dq_status,

    "volume_status":
        volume_status,

    "priority_status":
        priority_status,

    "feature_drift_status":
        feature_drift_status,

    "score_drift_status":
        score_drift_status,

    "target_status":
        target_status,

    "overall_status":
        overall_status,

    "monitored_at":
        datetime.utcnow(),
}

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Persistência

# COMMAND ----------

summary_df = spark.createDataFrame(
    pd.DataFrame(
        [
            summary_row
        ]
    )
)

feature_monitoring_df = (
    spark.createDataFrame(
        feature_monitoring_pandas
    )
)

write_replace_where(
    dataframe=summary_df,
    table_name=SUMMARY_TABLE,
    scoring_date=scoring_date,
)

write_replace_where(
    dataframe=feature_monitoring_df,
    table_name=FEATURE_MONITORING_TABLE,
    scoring_date=scoring_date,
)

# COMMAND ----------

if performance_row is not None:

    performance_df = (
        spark.createDataFrame(
            pd.DataFrame(
                [
                    performance_row
                ]
            )
        )
    )

    write_replace_where(
        dataframe=performance_df,
        table_name=PERFORMANCE_TABLE,
        scoring_date=scoring_date,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Resultado da rodada

# COMMAND ----------

display(
    summary_df
)

# COMMAND ----------

display(
    feature_monitoring_df
    .orderBy(
        desc("psi")
    )
)

# COMMAND ----------

print(
    f"""
    Monitoramento concluído.

    Scoring date:
    {scoring_date}

    Cenário:
    {scenario}

    Status geral:
    {overall_status}

    Target:
    {target_status}

    Maior PSI:
    {maximum_feature_psi:.4f}

    PSI dos scores:
    {score_psi:.4f}
    """
)

# COMMAND ----------


