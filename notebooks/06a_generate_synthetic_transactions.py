# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Geração recorrente de transações sintéticas
# MAGIC
# MAGIC Este notebook gera um novo lote de transações futuras a cada execução.
# MAGIC
# MAGIC Objetivos:
# MAGIC
# MAGIC - simular a chegada recorrente de dados em produção;
# MAGIC - preservar o schema transacional utilizado no projeto;
# MAGIC - testar batch scoring, Data Quality e monitoramento;
# MAGIC - criar cenários controlados de drift e falhas de dados.
# MAGIC
# MAGIC Os dados sintéticos devem ser utilizados para validar o processo operacional,
# MAGIC não para comprovar a performance real do modelo.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Imports

# COMMAND ----------

from datetime import datetime, timedelta
import math

import numpy as np
import pandas as pd

from pyspark.sql import Window
from pyspark.sql.functions import (
    col,
    count,
    countDistinct,
    datediff,
    desc,
    lit,
    max,
    row_number,
    to_date,
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Configurações

# COMMAND ----------

SOURCE_TABLE = "workspace.silver_layer.online_retail_transactions"

SYNTHETIC_SCHEMA = "workspace.synthetic_layer"

OUTPUT_TABLE = "workspace.synthetic_layer." "online_retail_transactions"

MANIFEST_TABLE = "workspace.synthetic_layer." "generation_runs"

FULL_HISTORY_VIEW = "workspace.synthetic_layer." "online_retail_transactions_full"

BASE_COLUMNS = [
    "invoice",
    "stock_code",
    "description",
    "quantity",
    "invoice_date",
    "price",
    "customer_id",
    "country",
]

VALID_SCENARIOS = [
    "normal",
    "reactivation_spike",
    "monetary_drift",
    "volume_spike",
    "data_quality_issue",
]

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Parâmetros do notebook
# MAGIC
# MAGIC Quando `batch_start_date` estiver vazio, o notebook começa no dia seguinte
# MAGIC à última data sintética já gerada. Na primeira execução, começa no dia seguinte
# MAGIC à última data da Silver real.

# COMMAND ----------

def create_text_widget_if_missing(
    name,
    default_value,
    label,
):
    try:
        dbutils.widgets.get(name)
    except Exception:
        dbutils.widgets.text(
            name,
            default_value,
            label,
        )


def create_dropdown_widget_if_missing(
    name,
    default_value,
    choices,
    label,
):
    try:
        dbutils.widgets.get(name)
    except Exception:
        dbutils.widgets.dropdown(
            name,
            default_value,
            choices,
            label,
        )


create_text_widget_if_missing(
    name="batch_start_date",
    default_value="",
    label="Batch start date (YYYY-MM-DD)",
)

create_text_widget_if_missing(
    name="batch_days",
    default_value="30",
    label="Number of days",
)

create_text_widget_if_missing(
    name="number_invoices",
    default_value="1500",
    label="Base number of invoices",
)

create_dropdown_widget_if_missing(
    name="scenario",
    default_value="normal",
    choices=VALID_SCENARIOS,
    label="Simulation scenario",
)

create_text_widget_if_missing(
    name="random_seed",
    default_value="",
    label="Random seed",
)

create_dropdown_widget_if_missing(
    name="replace_existing_batch",
    default_value="false",
    choices=["false", "true"],
    label="Replace existing batch",
)

# COMMAND ----------

batch_start_date_parameter = dbutils.widgets.get("batch_start_date").strip()

batch_days = int(dbutils.widgets.get("batch_days"))

base_number_invoices = int(dbutils.widgets.get("number_invoices"))

scenario = dbutils.widgets.get("scenario").strip()

random_seed_parameter = dbutils.widgets.get("random_seed").strip()

replace_existing_batch = dbutils.widgets.get("replace_existing_batch").lower() == "true"

# COMMAND ----------

if batch_days <= 0:
    raise ValueError(
        "batch_days deve ser maior que zero."
    )

if base_number_invoices <= 0:
    raise ValueError(
        "number_invoices deve ser maior que zero."
    )

if scenario not in VALID_SCENARIOS:
    raise ValueError(
        f"Cenário inválido: {scenario}"
    )

if not spark.catalog.tableExists(
    SOURCE_TABLE
):
    raise ValueError(
        f"Tabela de origem não encontrada: {SOURCE_TABLE}"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Definição automática do período

# COMMAND ----------

spark.sql(
    f"""
    CREATE SCHEMA IF NOT EXISTS
    {SYNTHETIC_SCHEMA}
    """
)

# COMMAND ----------

if batch_start_date_parameter:

    batch_start_date = datetime.strptime(
        batch_start_date_parameter,
        "%Y-%m-%d",
    ).date()

else:

    if spark.catalog.tableExists(OUTPUT_TABLE):

        latest_synthetic_date = (
            spark.table(OUTPUT_TABLE)
            .agg(max(to_date(col("invoice_date"))).alias("latest_date"))
            .first()["latest_date"]
        )

    else:

        latest_synthetic_date = None

    if latest_synthetic_date is not None:

        batch_start_date = latest_synthetic_date + timedelta(days=1)

    else:

        latest_real_date = (
            spark.table(SOURCE_TABLE)
            .agg(max(to_date(col("invoice_date"))).alias("latest_date"))
            .first()["latest_date"]
        )

        batch_start_date = latest_real_date + timedelta(days=1)

# COMMAND ----------

batch_end_date = batch_start_date + timedelta(days=batch_days - 1)

batch_id = f"{batch_start_date:%Y%m%d}_" f"{batch_end_date:%Y%m%d}_" f"{scenario}"

invoice_prefix = f"SYN-{batch_id}-"

scenario_seed_offset = sum(ord(character) for character in scenario)

if random_seed_parameter:

    random_seed = int(random_seed_parameter)

else:

    random_seed = int(batch_start_date.strftime("%Y%m%d")) + scenario_seed_offset

# COMMAND ----------

number_invoices = (
    base_number_invoices * 3 if scenario == "volume_spike" else base_number_invoices
)

print(
    f"""
    Batch ID:
    {batch_id}

    Período:
    {batch_start_date} até {batch_end_date}

    Cenário:
    {scenario}

    Seed:
    {random_seed}

    Quantidade planejada de invoices:
    {number_invoices:,}
    """
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Idempotência do lote

# COMMAND ----------

if spark.catalog.tableExists(OUTPUT_TABLE):

    existing_batch_rows = (
        spark.table(OUTPUT_TABLE)
        .filter(col("invoice").startswith(invoice_prefix))
        .count()
    )

else:

    existing_batch_rows = 0

# COMMAND ----------

if existing_batch_rows > 0:

    if not replace_existing_batch:

        raise ValueError(
            f"""
            O lote {batch_id} já possui
            {existing_batch_rows:,} linhas.

            Utilize replace_existing_batch=true
            apenas quando desejar recriá-lo.
            """
        )

    spark.sql(
        f"""
        DELETE FROM {OUTPUT_TABLE}
        WHERE invoice LIKE '{invoice_prefix}%'
        """
    )

    if spark.catalog.tableExists(MANIFEST_TABLE):

        spark.sql(
            f"""
            DELETE FROM {MANIFEST_TABLE}
            WHERE batch_id = '{batch_id}'
            """
        )

    print(f"Lote anterior removido: {batch_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Construção do histórico disponível

# COMMAND ----------

source_schema = spark.table(SOURCE_TABLE).select(*BASE_COLUMNS).schema

# COMMAND ----------

real_history_df = spark.table(SOURCE_TABLE).select(*BASE_COLUMNS)

# COMMAND ----------

if spark.catalog.tableExists(OUTPUT_TABLE):

    previous_synthetic_df = (
        spark.table(OUTPUT_TABLE)
        .select(*BASE_COLUMNS)
        .filter(to_date(col("invoice_date")) < lit(batch_start_date))
    )

    history_df = real_history_df.unionByName(previous_synthetic_df)

else:

    history_df = real_history_df

# COMMAND ----------

valid_history_df = (
    history_df.filter(~col("invoice").startswith("C"))
    .filter(~col("stock_code").startswith("TEST"))
    .filter(col("description") != "Discount")
    .filter(col("price") > 0)
    .filter(col("quantity") > 0)
)

# COMMAND ----------

if valid_history_df.limit(1).count() == 0:
    raise ValueError("Não existem transações válidas para gerar a simulação.")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Catálogo de clientes

# COMMAND ----------

latest_country_window = Window.partitionBy("customer_id").orderBy(desc("invoice_date"))

# COMMAND ----------

latest_country_df = (
    valid_history_df.withColumn(
        "row_number",
        row_number().over(latest_country_window),
    )
    .filter(col("row_number") == 1)
    .select(
        "customer_id",
        "country",
    )
)

# COMMAND ----------

customer_profiles_df = (
    valid_history_df
    .groupBy(
        "customer_id"
    )
    .agg(
        max(
            to_date(
                col("invoice_date")
            )
        ).alias(
            "last_purchase_date"
        )
    )
    .join(
        latest_country_df,
        on="customer_id",
        how="inner",
    )
    .withColumn(
        "inactive_days",
        datediff(
            lit(
                batch_start_date
            ),
            col(
                "last_purchase_date"
            ),
        ),
    )
    .select(
        col(
            "customer_id"
        ).cast(
            "long"
        ),
        "country",
        "inactive_days",
    )
)

# COMMAND ----------

customer_profiles_pandas = (
    customer_profiles_df
    .toPandas()
)

# COMMAND ----------

if customer_profiles_pandas.empty:
    raise ValueError(
        "O catálogo de clientes está vazio."
    )

inactive_customers = (
    customer_profiles_pandas
    .query(
        "inactive_days >= 60"
    )
    .reset_index(
        drop=True
    )
)

active_customers = (
    customer_profiles_pandas
    .query(
        "inactive_days < 60"
    )
    .reset_index(
        drop=True
    )
)

if inactive_customers.empty:
    inactive_customers = (
        customer_profiles_pandas
        .copy()
    )

if active_customers.empty:
    active_customers = (
        customer_profiles_pandas
        .copy()
    )

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Catálogo de produtos

# COMMAND ----------

product_catalog_df = (
    valid_history_df
    .filter(
        col("description").isNotNull()
    )
    .groupBy(
        "stock_code",
        "description",
    )
    .agg(
        {
            "price": "avg",
            "invoice": "count",
        }
    )
    .withColumnRenamed(
        "avg(price)",
        "base_price",
    )
    .withColumnRenamed(
        "count(invoice)",
        "frequency",
    )
    .filter(
        col("base_price") > 0
    )
    .select(
        "stock_code",
        "description",
        col(
            "base_price"
        ).cast(
            "double"
        ),
        col(
            "frequency"
        ).cast(
            "double"
        ),
    )
)

# COMMAND ----------

product_catalog_pandas = (
    product_catalog_df
    .toPandas()
)

# COMMAND ----------

if product_catalog_pandas.empty:
    raise ValueError(
        "O catálogo de produtos está vazio."
    )

product_probabilities = (
    product_catalog_pandas[
        "frequency"
    ]
    .to_numpy(
        dtype=float
    )
)

product_probabilities = (
    product_probabilities
    / product_probabilities.sum()
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Geração das transações

# COMMAND ----------

rng = np.random.default_rng(
    random_seed
)

# COMMAND ----------

if scenario == "reactivation_spike":

    inactive_invoice_probability = 0.70

else:

    inactive_invoice_probability = 0.20

# COMMAND ----------

if scenario == "monetary_drift":

    quantity_multiplier = 2.5
    price_multiplier = 1.4

else:

    quantity_multiplier = 1.0
    price_multiplier = 1.0

# COMMAND ----------

generated_rows = []

for invoice_number in range(
    1,
    number_invoices + 1,
):

    use_inactive_customer = (
        rng.random()
        < inactive_invoice_probability
    )

    customer_pool = (
        inactive_customers
        if use_inactive_customer
        else active_customers
    )

    customer_position = int(
        rng.integers(
            low=0,
            high=len(
                customer_pool
            ),
        )
    )

    customer = (
        customer_pool
        .iloc[
            customer_position
        ]
    )

    invoice_day_offset = int(
        rng.integers(
            low=0,
            high=batch_days,
        )
    )

    invoice_second_offset = int(
        rng.integers(
            low=0,
            high=86_400,
        )
    )

    invoice_datetime = (
        datetime.combine(
            batch_start_date,
            datetime.min.time(),
        )
        + timedelta(
            days=invoice_day_offset,
            seconds=invoice_second_offset,
        )
    )

    number_lines = int(
        rng.integers(
            low=1,
            high=6,
        )
    )

    product_positions = (
        rng.choice(
            len(
                product_catalog_pandas
            ),
            size=number_lines,
            replace=False,
            p=product_probabilities,
        )
    )

    invoice = (
        f"{invoice_prefix}"
        f"{invoice_number:08d}"
    )

    for product_position in product_positions:

        product = (
            product_catalog_pandas
            .iloc[
                int(
                    product_position
                )
            ]
        )

        quantity = int(
            np.clip(
                math.ceil(
                    (
                        rng.poisson(
                            lam=2.0
                        )
                        + 1
                    )
                    * quantity_multiplier
                ),
                1,
                100,
            )
        )

        price_noise = float(
            rng.lognormal(
                mean=0.0,
                sigma=0.10,
            )
        )

        price = float(
            np.clip(
                (
                    float(
                        product[
                            "base_price"
                        ]
                    )
                    * price_noise
                    * price_multiplier
                ),
                0.01,
                100_000.00,
            )
        )

        generated_rows.append(
            {
                "invoice":
                    invoice,

                "stock_code":
                    str(
                        product[
                            "stock_code"
                        ]
                    ),

                "description":
                    str(
                        product[
                            "description"
                        ]
                    ),

                "quantity":
                    quantity,

                "invoice_date":
                    invoice_datetime,

                "price":
                    round(
                        price,
                        6,
                    ),

                "customer_id":
                    int(
                        customer[
                            "customer_id"
                        ]
                    ),

                "country":
                    str(
                        customer[
                            "country"
                        ]
                    ),
            }
        )

# COMMAND ----------

synthetic_pandas = pd.DataFrame(
    generated_rows,
    columns=BASE_COLUMNS,
)

# COMMAND ----------

if scenario == "data_quality_issue":

    number_zero_prices = max(
        1,
        int(
            len(
                synthetic_pandas
            )
            * 0.01
        ),
    )

    number_null_descriptions = max(
        1,
        int(
            len(
                synthetic_pandas
            )
            * 0.01
        ),
    )

    number_duplicates = max(
        1,
        int(
            len(
                synthetic_pandas
            )
            * 0.005
        ),
    )

    zero_price_positions = (
        rng.choice(
            synthetic_pandas.index,
            size=number_zero_prices,
            replace=False,
        )
    )

    null_description_positions = (
        rng.choice(
            synthetic_pandas.index,
            size=number_null_descriptions,
            replace=False,
        )
    )

    synthetic_pandas.loc[
        zero_price_positions,
        "price",
    ] = 0.0

    synthetic_pandas.loc[
        null_description_positions,
        "description",
    ] = None

    duplicated_rows = (
        synthetic_pandas
        .sample(
            n=number_duplicates,
            random_state=random_seed,
        )
        .copy()
    )

    synthetic_pandas = pd.concat(
        [
            synthetic_pandas,
            duplicated_rows,
        ],
        ignore_index=True,
    )

# COMMAND ----------

synthetic_df_untyped = (
    spark.createDataFrame(
        synthetic_pandas
    )
)

# COMMAND ----------

synthetic_df = (
    synthetic_df_untyped
    .select(
        *[
            col(
                field.name
            )
            .cast(
                field.dataType
            )
            .alias(
                field.name
            )
            for field in source_schema.fields
        ]
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Validações do lote

# COMMAND ----------

generated_row_count = (
    synthetic_df
    .count()
)

generated_invoice_count = (
    synthetic_df
    .select(
        "invoice"
    )
    .distinct()
    .count()
)

generated_date_range = (
    synthetic_df
    .agg(
        {
            "invoice_date": "min",
        }
    )
    .first()[
        "min(invoice_date)"
    ],
    synthetic_df
    .agg(
        {
            "invoice_date": "max",
        }
    )
    .first()[
        "max(invoice_date)"
    ],
)

# COMMAND ----------

if generated_row_count == 0:
    raise ValueError(
        "O lote sintético ficou vazio."
    )

if (
    generated_date_range[0].date()
    < batch_start_date
    or generated_date_range[1].date()
    > batch_end_date
):
    raise ValueError(
        "Foram geradas datas fora do período do lote."
    )

# COMMAND ----------

if scenario != "data_quality_issue":

    null_count = (
        synthetic_df
        .filter(
            col("invoice").isNull()
            | col("stock_code").isNull()
            | col("description").isNull()
            | col("quantity").isNull()
            | col("invoice_date").isNull()
            | col("price").isNull()
            | col("customer_id").isNull()
            | col("country").isNull()
        )
        .count()
    )

    invalid_commercial_rows = (
        synthetic_df
        .filter(
            (col("quantity") <= 0)
            | (col("price") <= 0)
        )
        .count()
    )

    duplicate_count = (
        synthetic_df
        .groupBy(
            "invoice",
            "stock_code",
        )
        .count()
        .filter(
            col("count") > 1
        )
        .count()
    )

    if null_count > 0:
        raise ValueError(
            f"Foram encontrados {null_count} nulos."
        )

    if invalid_commercial_rows > 0:
        raise ValueError(
            f"""
            Foram encontradas
            {invalid_commercial_rows}
            transações comerciais inválidas.
            """
        )

    if duplicate_count > 0:
        raise ValueError(
            f"""
            Foram encontradas
            {duplicate_count}
            chaves invoice + stock_code duplicadas.
            """
        )

# COMMAND ----------

print(
    f"""
    Lote validado.

    Linhas:
    {generated_row_count:,}

    Invoices:
    {generated_invoice_count:,}

    Menor timestamp:
    {generated_date_range[0]}

    Maior timestamp:
    {generated_date_range[1]}
    """
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Persistência em Delta

# COMMAND ----------

if spark.catalog.tableExists(
    OUTPUT_TABLE
):

    (
        synthetic_df
        .write
        .format("delta")
        .mode("append")
        .saveAsTable(
            OUTPUT_TABLE
        )
    )

else:

    (
        synthetic_df
        .write
        .format("delta")
        .mode("overwrite")
        .saveAsTable(
            OUTPUT_TABLE
        )
    )

# COMMAND ----------

manifest_pandas = pd.DataFrame(
    [
        {
            "batch_id":
                batch_id,

            "scenario":
                scenario,

            "random_seed":
                int(
                    random_seed
                ),

            "batch_start_date":
                batch_start_date,

            "batch_end_date":
                batch_end_date,

            "planned_invoices":
                int(
                    number_invoices
                ),

            "generated_invoices":
                int(
                    generated_invoice_count
                ),

            "generated_rows":
                int(
                    generated_row_count
                ),

            "generated_at":
                datetime.utcnow(),
        }
    ]
)

# COMMAND ----------

manifest_df = (
    spark.createDataFrame(
        manifest_pandas
    )
)

# COMMAND ----------

if spark.catalog.tableExists(
    MANIFEST_TABLE
):

    (
        manifest_df
        .write
        .format("delta")
        .mode("append")
        .saveAsTable(
            MANIFEST_TABLE
        )
    )

else:

    (
        manifest_df
        .write
        .format("delta")
        .mode("overwrite")
        .saveAsTable(
            MANIFEST_TABLE
        )
    )

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## View com histórico real e sintético

# COMMAND ----------

spark.sql(
    f"""
    CREATE OR REPLACE VIEW
    {FULL_HISTORY_VIEW}
    AS

    SELECT
        invoice,
        stock_code,
        description,
        quantity,
        invoice_date,
        price,
        customer_id,
        country
    FROM
        {SOURCE_TABLE}

    UNION ALL

    SELECT
        invoice,
        stock_code,
        description,
        quantity,
        invoice_date,
        price,
        customer_id,
        country
    FROM
        {OUTPUT_TABLE}
    """
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Resumo da execução

# COMMAND ----------

display(
    synthetic_df
    .withColumn(
        "purchase_date",
        to_date(
            col("invoice_date")
        ),
    )
    .groupBy(
        "purchase_date"
    )
    .agg(
        count(
            "*"
        ).alias(
            "number_rows"
        ),
        countDistinct(
            "invoice"
        ).alias(
            "number_invoices"
        ),
        countDistinct(
            "customer_id"
        ).alias(
            "number_customers"
        ),
    )
    .orderBy(
        "purchase_date"
    )
)

# COMMAND ----------

display(
    synthetic_df
    .orderBy(
        "invoice_date"
    )
    .limit(
        20
    )
)

# COMMAND ----------

print(
    f"""
    Geração concluída com sucesso.

    Tabela sintética:
    {OUTPUT_TABLE}

    Manifesto de execuções:
    {MANIFEST_TABLE}

    Histórico completo:
    {FULL_HISTORY_VIEW}

    Próxima execução automática:
    o notebook começará no dia seguinte a {batch_end_date}.
    """
)
