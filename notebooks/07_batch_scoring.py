# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "databricks_ml_v5"
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC
# MAGIC # Batch scoring mensal
# MAGIC
# MAGIC Este notebook:
# MAGIC
# MAGIC - cria a população elegível na data de scoring;
# MAGIC - recalcula as mesmas features do treinamento;
# MAGIC - carrega o modelo `champion`;
# MAGIC - gera probabilidades e ranking;
# MAGIC - salva o resultado em uma tabela Delta.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Imports

# COMMAND ----------

from datetime import datetime

import mlflow
import numpy as np
import pandas as pd

from mlflow import MlflowClient
from pyspark.sql.functions import *

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Configurações

# COMMAND ----------

HISTORY_TABLE = "workspace.synthetic_layer." "online_retail_transactions_full"
OUTPUT_TABLE = "workspace.gold_layer." "customer_reactivation_scoring"
SNAPSHOT_FEATURES_TABLE = "workspace.gold_layer.customer_reactivation_feature_snapshot"
REGISTERED_MODEL_NAME = "workspace.default." "reactivation_customers_model"

MODEL_ALIAS = "champion"

INACTIVITY_DAYS = 60
PREDICTION_THRESHOLD = 0.50

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

REQUIRED_HISTORY_COLUMNS = [
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
# MAGIC ## Parâmetros

# COMMAND ----------

try:
    dbutils.widgets.get("scoring_date")
except Exception:
    dbutils.widgets.text(
        "scoring_date",
        "",
        "Scoring date (YYYY-MM-DD)",
    )

# COMMAND ----------

scoring_date_parameter = dbutils.widgets.get("scoring_date").strip()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Leitura e validação do histórico

# COMMAND ----------

if not spark.catalog.tableExists(HISTORY_TABLE):
    raise ValueError(
        f"Tabela de histórico não encontrada: {HISTORY_TABLE}"
    )

history_df = spark.table(HISTORY_TABLE)

# COMMAND ----------

missing_columns = sorted(
    set(REQUIRED_HISTORY_COLUMNS)
    - set(history_df.columns)
)

if missing_columns:
    raise ValueError(
        f"Colunas ausentes no histórico: {missing_columns}"
    )

if history_df.limit(1).count() == 0:
    raise ValueError(
        "A tabela de histórico está vazia."
    )

# COMMAND ----------

if scoring_date_parameter:

    scoring_date = datetime.strptime(
        scoring_date_parameter,
        "%Y-%m-%d",
    ).date()

else:

    scoring_date = (
        history_df
        .agg(
            max(
                to_date(
                    col("invoice_date")
                )
            ).alias("max_date")
        )
        .first()["max_date"]
    )

if scoring_date is None:
    raise ValueError(
        "Não foi possível definir a scoring_date."
    )

print(f"Scoring date: {scoring_date}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Compras válidas até a data de scoring

# COMMAND ----------

valid_transactions_df = (
    history_df
    .withColumn(
        "purchase_date",
        to_date(
            col("invoice_date")
        ),
    )
    .withColumn(
        "total_amount",
        col("quantity") * col("price"),
    )
    .filter(
        col("purchase_date")
        <= lit(scoring_date)
    )
    .filter(
        ~col("invoice").startswith("C")
    )
    .filter(
        ~col("stock_code").startswith("TEST")
    )
    .filter(
        col("description") != "Discount"
    )
    .filter(
        col("price") > 0
    )
    .filter(
        col("quantity") > 0
    )
)

# COMMAND ----------

if valid_transactions_df.limit(1).count() == 0:
    raise ValueError(
        "Não existem compras válidas até a scoring_date."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## População elegível e features

# COMMAND ----------

scoring_features_df = (
    valid_transactions_df.groupBy("customer_id")
    .agg(
        min("purchase_date").alias("first_purchase_date"),
        max("purchase_date").alias("last_purchase_date"),
        countDistinct("invoice").alias("number_orders"),
        sum("quantity").alias("total_items"),
        sum("total_amount").alias("total_spent"),
        countDistinct("stock_code").alias("unique_products"),
        countDistinct("country").alias("distinct_countries"),
    )
    .withColumn(
        "scoring_date",
        lit(scoring_date),
    )
    .withColumn(
        "inactive_days",
        datediff(
            col("scoring_date"),
            col("last_purchase_date"),
        ),
    )
    .withColumn(
        "customer_tenure_days",
        datediff(
            col("scoring_date"),
            col("first_purchase_date"),
        ),
    )
    .withColumn(
        "average_ticket",
        col("total_spent") / col("number_orders"),
    )
    .withColumn(
        "average_amount_per_country",
        col("total_spent") / col("distinct_countries"),
    )
    .withColumn(
        "average_amount_per_stock_code",
        col("total_spent") / col("unique_products"),
    )
    .filter(col("inactive_days") >= INACTIVITY_DAYS)
)

# COMMAND ----------

if scoring_features_df.limit(1).count() == 0:
    raise ValueError(
        "Nenhum cliente elegível foi encontrado."
    )

# COMMAND ----------

duplicate_customers = (
    scoring_features_df
    .groupBy(
        "customer_id",
        "scoring_date",
    )
    .count()
    .filter(
        col("count") > 1
    )
    .count()
)

if duplicate_customers > 0:
    raise ValueError(
        "Foram encontrados clientes duplicados na população."
    )

# COMMAND ----------

null_condition = None

for feature_name in FEATURE_COLUMNS:

    current_condition = (
        col(feature_name).isNull()
    )

    null_condition = (
        current_condition
        if null_condition is None
        else null_condition | current_condition
    )

null_rows = (
    scoring_features_df
    .filter(null_condition)
    .count()
)

if null_rows > 0:
    raise ValueError(
        f"Foram encontradas {null_rows} linhas com features nulas."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Preparação da entrada do modelo

# COMMAND ----------

scoring_input_df = (
    scoring_features_df
    .select(
        col("customer_id").cast("long"),
        "scoring_date",
        "first_purchase_date",
        "last_purchase_date",
        *[
            col(feature_name)
            .cast("double")
            .alias(feature_name)
            for feature_name in FEATURE_COLUMNS
        ],
    )
)

scoring_input_pandas = (
    scoring_input_df
    .toPandas()
)

X_scoring = (
    scoring_input_pandas[
        FEATURE_COLUMNS
    ]
    .copy()
)

# COMMAND ----------

if not np.isfinite(
    X_scoring.to_numpy(
        dtype=float
    )
).all():

    raise ValueError(
        "A entrada do modelo contém valores infinitos ou inválidos."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Salvando o snapshot das features que estão indo para o modelo

# COMMAND ----------

spark.createDataFrame(scoring_input_pandas).write.format("delta").mode(
    "overwrite"
).option("replaceWhere", f"scoring_date = '{scoring_date}'").saveAsTable(OUTPUT_TABLE)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Carregamento do modelo champion

# COMMAND ----------

mlflow.set_registry_uri(
    "databricks-uc"
)

client = MlflowClient()

model_version_details = (
    client
    .get_model_version_by_alias(
        name=REGISTERED_MODEL_NAME,
        alias=MODEL_ALIAS,
    )
)

model_version = str(
    model_version_details.version
)

source_run_id = (
    model_version_details.run_id
)

model_uri = (
    f"models:/"
    f"{REGISTERED_MODEL_NAME}"
    f"@{MODEL_ALIAS}"
)

model = (
    mlflow.xgboost
    .load_model(
        model_uri
    )
)

# COMMAND ----------

booster_feature_names = (
    model
    .get_booster()
    .feature_names
)

if (
    booster_feature_names is not None
    and booster_feature_names != FEATURE_COLUMNS
):

    raise ValueError(
        f"""
        Features incompatíveis com o modelo.

        Esperado:
        {FEATURE_COLUMNS}

        Encontrado:
        {booster_feature_names}
        """
    )

print(
    f"""
    Modelo carregado:

    {REGISTERED_MODEL_NAME}

    Alias:
    {MODEL_ALIAS}

    Versão:
    {model_version}
    """
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Scoring

# COMMAND ----------

reactivation_probability = (
    model
    .predict_proba(
        X_scoring
    )[:, 1]
)

prediction_0_5 = (
    reactivation_probability
    >= PREDICTION_THRESHOLD
).astype(int)

# COMMAND ----------

if len(
    reactivation_probability
) != len(
    scoring_input_pandas
):

    raise ValueError(
        "O volume de scores é diferente do volume de entrada."
    )

if not np.isfinite(
    reactivation_probability
).all():

    raise ValueError(
        "Foram gerados scores inválidos."
    )

if not (
    (
        reactivation_probability >= 0
    )
    &
    (
        reactivation_probability <= 1
    )
).all():

    raise ValueError(
        "Foram gerados scores fora do intervalo [0, 1]."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Ranking e grupos de prioridade

# COMMAND ----------

scoring_result_pandas = (
    scoring_input_pandas
    .copy()
)

scoring_result_pandas[
    "reactivation_probability"
] = reactivation_probability

scoring_result_pandas[
    "prediction_0_5"
] = prediction_0_5

scoring_result_pandas = (
    scoring_result_pandas
    .sort_values(
        by=[
            "reactivation_probability",
            "customer_id",
        ],
        ascending=[
            False,
            True,
        ],
    )
    .reset_index(
        drop=True
    )
)

number_customers = len(
    scoring_result_pandas
)

scoring_result_pandas[
    "score_rank"
] = np.arange(
    1,
    number_customers + 1,
)

scoring_result_pandas[
    "score_percentile"
] = (
    scoring_result_pandas[
        "score_rank"
    ]
    / number_customers
)

scoring_result_pandas[
    "priority_group"
] = np.select(
    [
        scoring_result_pandas[
            "score_percentile"
        ] <= 0.10,

        scoring_result_pandas[
            "score_percentile"
        ] <= 0.20,

        scoring_result_pandas[
            "score_percentile"
        ] <= 0.30,
    ],
    [
        "TOP_10",
        "TOP_20",
        "TOP_30",
    ],
    default="REMAINDER",
)

# COMMAND ----------

scoring_result_pandas[
    "model_name"
] = REGISTERED_MODEL_NAME

scoring_result_pandas[
    "model_version"
] = model_version

scoring_result_pandas[
    "model_alias"
] = MODEL_ALIAS

scoring_result_pandas[
    "source_run_id"
] = source_run_id

scoring_result_pandas[
    "scored_at"
] = datetime.utcnow()

# COMMAND ----------

OUTPUT_COLUMNS = [
    "customer_id",
    "scoring_date",
    "first_purchase_date",
    "last_purchase_date",
    *FEATURE_COLUMNS,
    "reactivation_probability",
    "prediction_0_5",
    "score_rank",
    "score_percentile",
    "priority_group",
    "model_name",
    "model_version",
    "model_alias",
    "source_run_id",
    "scored_at",
]

scoring_result_pandas = (
    scoring_result_pandas[
        OUTPUT_COLUMNS
    ]
)

# COMMAND ----------

if scoring_result_pandas[
    [
        "customer_id",
        "scoring_date",
    ]
].duplicated().any():

    raise ValueError(
        "O resultado contém clientes duplicados."
    )

if scoring_result_pandas[
    "priority_group"
].isna().any():

    raise ValueError(
        "Existem clientes sem grupo de prioridade."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Persistência idempotente

# COMMAND ----------

scoring_result_df = (
    spark.createDataFrame(
        scoring_result_pandas
    )
)

# COMMAND ----------

# Garante que o DataFrame contém somente a scoring_date atual

scoring_dates = scoring_result_df.select("scoring_date").distinct().collect()

if len(scoring_dates) != 1 or scoring_dates[0]["scoring_date"] != scoring_date:

    raise ValueError(
        """
        O DataFrame contém uma scoring_date diferente
        da data que será substituída na tabela.
        """
    )


# Cria a tabela na primeira execução ou substitui somente
# a scoring_date correspondente nas próximas execuções.

if spark.catalog.tableExists(OUTPUT_TABLE):

    (
        scoring_result_df.write.format("delta")
        .mode("overwrite")
        .option(
            "replaceWhere",
            f"scoring_date = DATE '{scoring_date}'",
        )
        .saveAsTable(OUTPUT_TABLE)
    )

    print(
        f"""
        Scoring substituído com sucesso.

        Scoring date:
        {scoring_date}
        """
    )

else:

    (
        scoring_result_df.write.format("delta")
        .mode("overwrite")
        .saveAsTable(OUTPUT_TABLE)
    )

    print(
        f"""
        Tabela de scoring criada com sucesso.

        Scoring date:
        {scoring_date}
        """
    )

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Resumo da execução

# COMMAND ----------

display(
    scoring_result_df
    .groupBy(
        "priority_group"
    )
    .agg(
        count("*").alias(
            "number_customers"
        ),
        min(
            "reactivation_probability"
        ).alias(
            "minimum_score"
        ),
        avg(
            "reactivation_probability"
        ).alias(
            "average_score"
        ),
        max(
            "reactivation_probability"
        ).alias(
            "maximum_score"
        ),
    )
    .orderBy(
        "priority_group"
    )
)

# COMMAND ----------

score_summary = (
    scoring_result_df
    .agg(
        count("*").alias(
            "number_customers"
        ),
        min(
            "reactivation_probability"
        ).alias(
            "minimum_score"
        ),
        avg(
            "reactivation_probability"
        ).alias(
            "average_score"
        ),
        max(
            "reactivation_probability"
        ).alias(
            "maximum_score"
        ),
        avg(
            "prediction_0_5"
        ).alias(
            "predicted_positive_rate"
        ),
    )
    .first()
)

# COMMAND ----------

print(
    f"""
    Batch scoring concluído.

    Scoring date:
    {scoring_date}

    Clientes elegíveis:
    {score_summary["number_customers"]:,}

    Score mínimo:
    {score_summary["minimum_score"]:.6f}

    Score médio:
    {score_summary["average_score"]:.6f}

    Score máximo:
    {score_summary["maximum_score"]:.6f}

    Taxa prevista positiva em 0.5:
    {score_summary["predicted_positive_rate"]:.4%}

    Modelo:
    {REGISTERED_MODEL_NAME}

    Versão:
    {model_version}

    Tabela:
    {OUTPUT_TABLE}
    """
)

# COMMAND ----------

# MAGIC %sql 
# MAGIC
# MAGIC select * from workspace.gold_layer.customer_reactivation_scoring
