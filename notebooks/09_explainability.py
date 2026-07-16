# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Explicabilidade do modelo
# MAGIC
# MAGIC Este notebook explica o modelo `champion` em dois níveis:
# MAGIC
# MAGIC - **global:** quais features mais influenciam o modelo no conjunto;
# MAGIC - **local:** por que um cliente específico recebeu determinado score.
# MAGIC
# MAGIC As contribuições são calculadas pelo próprio XGBoost com
# MAGIC `pred_contribs=True`, sem dependência adicional da biblioteca SHAP.
# MAGIC
# MAGIC > As contribuições são expressas na escala bruta do modelo
# MAGIC > (margem/log-odds), e não diretamente em pontos de probabilidade.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Imports

# COMMAND ----------

# MAGIC %pip install xgboost optuna
# MAGIC %restart_python

# COMMAND ----------

from datetime import datetime

import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

from mlflow import MlflowClient
from pyspark.sql.functions import *

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Configurações

# COMMAND ----------

SCORING_TABLE = (
    "workspace.gold_layer."
    "customer_reactivation_scoring"
)

REGISTERED_MODEL_NAME = (
    "workspace.default."
    "reactivation_customers_model"
)

MODEL_ALIAS = "champion"

EXPLAINABILITY_SCHEMA = (
    "workspace.explainability_layer"
)

GLOBAL_OUTPUT_TABLE = (
    "workspace.explainability_layer."
    "global_feature_importance"
)

LOCAL_OUTPUT_TABLE = (
    "workspace.explainability_layer."
    "local_feature_contributions"
)

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

GLOBAL_SAMPLE_SIZE = 2000
EXAMPLES_PER_GROUP = 3

EXAMPLE_GROUPS = [
    "TOP_10",
    "TOP_20",
    "TOP_30",
    "REMAINDER",
]

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Parâmetros

# COMMAND ----------

try:
    dbutils.widgets.get(
        "explanation_scoring_date"
    )
except Exception:
    dbutils.widgets.text(
        "explanation_scoring_date",
        "",
        "Scoring date (YYYY-MM-DD)",
    )

try:
    dbutils.widgets.get(
        "customer_id"
    )
except Exception:
    dbutils.widgets.text(
        "customer_id",
        "",
        "Customer ID",
    )

# COMMAND ----------

scoring_date_parameter = (
    dbutils.widgets
    .get("explanation_scoring_date")
    .strip()
)

customer_id_parameter = (
    dbutils.widgets
    .get("customer_id")
    .strip()
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Funções auxiliares

# COMMAND ----------

def sigmoid(value):

    value = np.asarray(
        value,
        dtype=float,
    )

    return 1.0 / (
        1.0 + np.exp(-value)
    )


def calculate_contributions(
    booster,
    features,
):

    dmatrix = xgb.DMatrix(
        features,
        feature_names=FEATURE_COLUMNS,
    )

    contributions = booster.predict(
        dmatrix,
        pred_contribs=True,
        validate_features=True,
    )

    if contributions.ndim == 3:

        if contributions.shape[1] != 1:
            raise ValueError(
                (
                    "O modelo retornou contribuições "
                    "para mais de uma classe."
                )
            )

        contributions = (
            contributions[:, 0, :]
        )

    expected_columns = (
        len(FEATURE_COLUMNS) + 1
    )

    if (
        contributions.ndim != 2
        or contributions.shape[1]
        != expected_columns
    ):
        raise ValueError(
            (
                "Formato inesperado das "
                "contribuições do XGBoost: "
                f"{contributions.shape}"
            )
        )

    shap_values = (
        contributions[:, :-1]
    )

    base_values = (
        contributions[:, -1]
    )

    return (
        shap_values,
        base_values,
    )


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


def build_local_explanation(
    customer_row,
    shap_row,
    base_value,
    model_name,
    model_version,
):

    customer_id = customer_row[
        "customer_id"
    ]

    scoring_date = customer_row[
        "scoring_date"
    ]

    model_probability = float(
        customer_row[
            "reactivation_probability"
        ]
    )

    raw_margin = float(
        base_value
        + np.sum(
            shap_row
        )
    )

    probability_from_margin = float(
        sigmoid(
            raw_margin
        )
    )

    rows = []

    for position, feature_name in enumerate(
        FEATURE_COLUMNS
    ):

        shap_value = float(
            shap_row[position]
        )

        if shap_value > 0:

            direction = (
                "INCREASES_SCORE"
            )

        elif shap_value < 0:

            direction = (
                "REDUCES_SCORE"
            )

        else:

            direction = "NEUTRAL"

        rows.append(
            {
                "customer_id":
                    int(customer_id),

                "scoring_date":
                    scoring_date,

                "priority_group":
                    str(
                        customer_row[
                            "priority_group"
                        ]
                    ),

                "score_rank":
                    int(
                        customer_row[
                            "score_rank"
                        ]
                    ),

                "reactivation_probability":
                    model_probability,

                "probability_from_margin":
                    probability_from_margin,

                "raw_margin":
                    raw_margin,

                "base_value":
                    float(
                        base_value
                    ),

                "feature_name":
                    feature_name,

                "feature_value":
                    float(
                        customer_row[
                            feature_name
                        ]
                    ),

                "shap_value":
                    shap_value,

                "absolute_shap_value":
                    np.abs(
                        shap_value
                    ),

                "direction":
                    direction,

                "model_name":
                    model_name,

                "model_version":
                    model_version,

                "explained_at":
                    datetime.utcnow(),
            }
        )

    return rows

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Leitura e validação da população

# COMMAND ----------

if not spark.catalog.tableExists(
    SCORING_TABLE
):
    raise ValueError(
        (
            "Tabela de scoring não "
            f"encontrada: {SCORING_TABLE}"
        )
    )

scoring_df = spark.table(
    SCORING_TABLE
)

# COMMAND ----------

required_columns = {
    "customer_id",
    "scoring_date",
    "reactivation_probability",
    "score_rank",
    "priority_group",
    *FEATURE_COLUMNS,
}

missing_columns = sorted(
    required_columns
    - set(
        scoring_df.columns
    )
)

if missing_columns:
    raise ValueError(
        (
            "Colunas ausentes na tabela "
            f"de scoring: {missing_columns}"
        )
    )

# COMMAND ----------

if scoring_date_parameter:

    scoring_date = (
        datetime.strptime(
            scoring_date_parameter,
            "%Y-%m-%d",
        )
        .date()
    )

else:

    scoring_date = (
        scoring_df
        .agg(
            max(
                "scoring_date"
            ).alias(
                "latest_scoring_date"
            )
        )
        .first()[
            "latest_scoring_date"
        ]
    )

if scoring_date is None:
    raise ValueError(
        (
            "Não foi possível definir "
            "a scoring_date."
        )
    )

# COMMAND ----------

current_scoring_df = (
    scoring_df
    .filter(
        col("scoring_date")
        == lit(scoring_date)
    )
)

if (
    current_scoring_df
    .limit(1)
    .count()
    == 0
):
    raise ValueError(
        (
            "Não existem resultados "
            f"de scoring para {scoring_date}."
        )
    )

# COMMAND ----------

current_scoring_pandas = (
    current_scoring_df
    .select(
        "customer_id",
        "scoring_date",
        "reactivation_probability",
        "score_rank",
        "priority_group",
        *FEATURE_COLUMNS,
    )
    .toPandas()
)

# COMMAND ----------

if (
    current_scoring_pandas[
        FEATURE_COLUMNS
    ]
    .isna()
    .any()
    .any()
):
    raise ValueError(
        (
            "Existem valores nulos nas "
            "features de explicabilidade."
        )
    )

if not np.isfinite(
    current_scoring_pandas[
        FEATURE_COLUMNS
    ].to_numpy(
        dtype=float
    )
).all():
    raise ValueError(
        (
            "Existem valores infinitos "
            "ou inválidos nas features."
        )
    )

# COMMAND ----------

print(
    f"""
    População carregada.

    Scoring date:
    {scoring_date}

    Clientes:
    {len(current_scoring_pandas):,}
    """
)

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

booster = model.get_booster()

# COMMAND ----------

booster_feature_names = (
    booster.feature_names
)

if (
    booster_feature_names is not None
    and booster_feature_names
    != FEATURE_COLUMNS
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
    Modelo carregado.

    Nome:
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
# MAGIC ## Explicabilidade global

# COMMAND ----------

from builtins import min as min_builtin

global_sample_size = min_builtin(
    GLOBAL_SAMPLE_SIZE,
    len(
        current_scoring_pandas
    ),
)

global_sample_pandas = (
    current_scoring_pandas
    .sample(
        n=global_sample_size,
        random_state=42,
    )
    .sort_values(
        "customer_id"
    )
    .reset_index(
        drop=True
    )
)

X_global = (
    global_sample_pandas[
        FEATURE_COLUMNS
    ]
    .astype(float)
)

# COMMAND ----------

(
    global_shap_values,
    global_base_values,
) = calculate_contributions(
    booster=booster,
    features=X_global,
)

# COMMAND ----------

global_importance_pandas = pd.DataFrame(
    {
        "feature_name":
            FEATURE_COLUMNS,

        "mean_absolute_shap":
            np.mean(
                np.abs(
                    global_shap_values
                ),
                axis=0,
            ),

        "mean_shap":
            np.mean(
                global_shap_values,
                axis=0,
            ),

        "median_shap":
            np.median(
                global_shap_values,
                axis=0,
            ),

        "positive_contribution_rate":
            np.mean(
                global_shap_values > 0,
                axis=0,
            ),

        "negative_contribution_rate":
            np.mean(
                global_shap_values < 0,
                axis=0,
            ),
    }
)

global_importance_pandas = (
    global_importance_pandas
    .sort_values(
        "mean_absolute_shap",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)

global_importance_pandas[
    "importance_rank"
] = np.arange(
    1,
    len(
        global_importance_pandas
    ) + 1,
)

global_importance_pandas[
    "scoring_date"
] = scoring_date

global_importance_pandas[
    "sample_size"
] = global_sample_size

global_importance_pandas[
    "model_name"
] = REGISTERED_MODEL_NAME

global_importance_pandas[
    "model_version"
] = model_version

global_importance_pandas[
    "calculated_at"
] = datetime.utcnow()

# COMMAND ----------

display(
    spark.createDataFrame(
        global_importance_pandas
    )
    .select(
        "importance_rank",
        "feature_name",
        "mean_absolute_shap",
        "mean_shap",
        "positive_contribution_rate",
        "negative_contribution_rate",
    )
    .orderBy(
        "importance_rank"
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Importância global

# COMMAND ----------

plot_global_importance = (
    global_importance_pandas
    .sort_values(
        "mean_absolute_shap",
        ascending=True,
    )
)

plt.figure(
    figsize=(9, 5)
)

plt.barh(
    plot_global_importance[
        "feature_name"
    ],
    plot_global_importance[
        "mean_absolute_shap"
    ],
)

plt.title(
    "Importância global das features"
)

plt.xlabel(
    "Média do valor absoluto da contribuição"
)

plt.ylabel(
    "Feature"
)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Direção média das contribuições

# COMMAND ----------

plot_signed_importance = (
    global_importance_pandas
    .sort_values(
        "mean_shap",
        ascending=True,
    )
)

plt.figure(
    figsize=(9, 5)
)

plt.barh(
    plot_signed_importance[
        "feature_name"
    ],
    plot_signed_importance[
        "mean_shap"
    ],
)

plt.axvline(
    0,
    linewidth=1,
)

plt.title(
    "Direção média das contribuições"
)

plt.xlabel(
    "Contribuição média na margem do modelo"
)

plt.ylabel(
    "Feature"
)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Distribuição das contribuições das principais features

# COMMAND ----------

top_features = (
    global_importance_pandas
    .head(8)[
        "feature_name"
    ]
    .tolist()
)

distribution_rows = []

for feature_name in top_features:

    feature_position = (
        FEATURE_COLUMNS.index(
            feature_name
        )
    )

    for shap_value in (
        global_shap_values[
            :,
            feature_position,
        ]
    ):

        distribution_rows.append(
            {
                "feature_name":
                    feature_name,

                "shap_value":
                    float(
                        shap_value
                    ),
            }
        )

distribution_pandas = pd.DataFrame(
    distribution_rows
)

# COMMAND ----------

plt.figure(
    figsize=(9, 6)
)

for feature_position, feature_name in enumerate(
    top_features
):

    feature_values = (
        distribution_pandas
        .loc[
            distribution_pandas[
                "feature_name"
            ]
            == feature_name,
            "shap_value",
        ]
        .to_numpy()
    )

    y_values = np.full(
        len(
            feature_values
        ),
        feature_position,
    )

    plt.scatter(
        feature_values,
        y_values,
        alpha=0.25,
        s=12,
    )

plt.axvline(
    0,
    linewidth=1,
)

plt.yticks(
    range(
        len(
            top_features
        )
    ),
    top_features,
)

plt.title(
    "Distribuição das contribuições SHAP"
)

plt.xlabel(
    "Contribuição na margem do modelo"
)

plt.ylabel(
    "Feature"
)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Seleção do cliente para explicação local

# COMMAND ----------

if customer_id_parameter:

    selected_customer_pandas = (
        current_scoring_pandas
        .loc[
            current_scoring_pandas[
                "customer_id"
            ]
            .astype(str)
            == customer_id_parameter
        ]
        .copy()
    )

    if selected_customer_pandas.empty:
        raise ValueError(
            (
                "O customer_id informado "
                "não existe na scoring_date "
                f"{scoring_date}."
            )
        )

else:

    selected_customer_pandas = (
        current_scoring_pandas
        .sort_values(
            "score_rank"
        )
        .head(1)
        .copy()
    )

selected_customer_pandas = (
    selected_customer_pandas
    .reset_index(
        drop=True
    )
)

selected_customer = (
    selected_customer_pandas
    .iloc[0]
)

# COMMAND ----------

X_local = (
    selected_customer_pandas[
        FEATURE_COLUMNS
    ]
    .astype(float)
)

(
    local_shap_values,
    local_base_values,
) = calculate_contributions(
    booster=booster,
    features=X_local,
)

local_explanation_rows = (
    build_local_explanation(
        customer_row=selected_customer,
        shap_row=local_shap_values[0],
        base_value=local_base_values[0],
        model_name=REGISTERED_MODEL_NAME,
        model_version=model_version,
    )
)

local_explanation_pandas = (
    pd.DataFrame(
        local_explanation_rows
    )
    .sort_values(
        "absolute_shap_value",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)

local_explanation_pandas[
    "contribution_rank"
] = np.arange(
    1,
    len(
        local_explanation_pandas
    ) + 1,
)

# COMMAND ----------

display(
    spark.createDataFrame(
        local_explanation_pandas
    )
    .select(
        "contribution_rank",
        "customer_id",
        "priority_group",
        "score_rank",
        "reactivation_probability",
        "feature_name",
        "feature_value",
        "shap_value",
        "direction",
    )
    .orderBy(
        "contribution_rank"
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Visualização da explicação local

# COMMAND ----------

local_plot_pandas = (
    local_explanation_pandas
    .sort_values(
        "shap_value"
    )
)

plt.figure(
    figsize=(9, 5)
)

plt.barh(
    local_plot_pandas[
        "feature_name"
    ],
    local_plot_pandas[
        "shap_value"
    ],
)

plt.axvline(
    0,
    linewidth=1,
)

plt.title(
    (
        "Contribuições locais — cliente "
        f"{selected_customer['customer_id']}"
    )
)

plt.xlabel(
    "Contribuição na margem do modelo"
)

plt.ylabel(
    "Feature"
)

plt.tight_layout()
plt.show()

# COMMAND ----------

print(
    f"""
    Cliente explicado:

    Customer ID:
    {selected_customer["customer_id"]}

    Priority group:
    {selected_customer["priority_group"]}

    Score rank:
    {int(selected_customer["score_rank"])}

    Probabilidade:
    {float(selected_customer["reactivation_probability"]):.6f}

    Probabilidade reconstruída pela margem:
    {float(local_explanation_pandas["probability_from_margin"].iloc[0]):.6f}
    """
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Exemplos por grupo de prioridade

# COMMAND ----------

example_customers_pandas = (
    current_scoring_pandas
    .loc[
        current_scoring_pandas[
            "priority_group"
        ]
        .isin(
            EXAMPLE_GROUPS
        )
    ]
    .sort_values(
        [
            "priority_group",
            "score_rank",
        ]
    )
    .groupby(
        "priority_group",
        as_index=False,
        group_keys=False,
    )
    .head(
        EXAMPLES_PER_GROUP
    )
    .copy()
)

selected_customer_id = int(
    selected_customer[
        "customer_id"
    ]
)

if (
    selected_customer_id
    not in example_customers_pandas[
        "customer_id"
    ].astype(int).tolist()
):

    example_customers_pandas = pd.concat(
        [
            example_customers_pandas,
            selected_customer_pandas,
        ],
        ignore_index=True,
    )

example_customers_pandas = (
    example_customers_pandas
    .drop_duplicates(
        subset=[
            "customer_id",
            "scoring_date",
        ]
    )
    .sort_values(
        "score_rank"
    )
    .reset_index(
        drop=True
    )
)

# COMMAND ----------

X_examples = (
    example_customers_pandas[
        FEATURE_COLUMNS
    ]
    .astype(float)
)

(
    examples_shap_values,
    examples_base_values,
) = calculate_contributions(
    booster=booster,
    features=X_examples,
)

# COMMAND ----------

all_local_rows = []

for row_position in range(
    len(
        example_customers_pandas
    )
):

    all_local_rows.extend(
        build_local_explanation(
            customer_row=(
                example_customers_pandas
                .iloc[
                    row_position
                ]
            ),
            shap_row=(
                examples_shap_values[
                    row_position
                ]
            ),
            base_value=(
                examples_base_values[
                    row_position
                ]
            ),
            model_name=(
                REGISTERED_MODEL_NAME
            ),
            model_version=(
                model_version
            ),
        )
    )

all_local_pandas = pd.DataFrame(
    all_local_rows
)

all_local_pandas[
    "contribution_rank"
] = (
    all_local_pandas
    .groupby(
        [
            "customer_id",
            "scoring_date",
        ]
    )[
        "absolute_shap_value"
    ]
    .rank(
        method="first",
        ascending=False,
    )
    .astype(int)
)

# COMMAND ----------

display(
    spark.createDataFrame(
        all_local_pandas
    )
    .filter(
        col("contribution_rank")
        <= 3
    )
    .select(
        "customer_id",
        "priority_group",
        "score_rank",
        "reactivation_probability",
        "contribution_rank",
        "feature_name",
        "feature_value",
        "shap_value",
        "direction",
    )
    .orderBy(
        "score_rank",
        "contribution_rank",
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Persistência

# COMMAND ----------

spark.sql(
    f"""
    CREATE SCHEMA IF NOT EXISTS
    {EXPLAINABILITY_SCHEMA}
    """
)

# COMMAND ----------

GLOBAL_OUTPUT_COLUMNS = [
    "scoring_date",
    "importance_rank",
    "feature_name",
    "mean_absolute_shap",
    "mean_shap",
    "median_shap",
    "positive_contribution_rate",
    "negative_contribution_rate",
    "sample_size",
    "model_name",
    "model_version",
    "calculated_at",
]

global_output_df = (
    spark.createDataFrame(
        global_importance_pandas[
            GLOBAL_OUTPUT_COLUMNS
        ]
    )
)

# COMMAND ----------

LOCAL_OUTPUT_COLUMNS = [
    "customer_id",
    "scoring_date",
    "priority_group",
    "score_rank",
    "reactivation_probability",
    "probability_from_margin",
    "raw_margin",
    "base_value",
    "contribution_rank",
    "feature_name",
    "feature_value",
    "shap_value",
    "absolute_shap_value",
    "direction",
    "model_name",
    "model_version",
    "explained_at",
]

local_output_df = (
    spark.createDataFrame(
        all_local_pandas[
            LOCAL_OUTPUT_COLUMNS
        ]
    )
)

# COMMAND ----------

write_replace_where(
    dataframe=global_output_df,
    table_name=GLOBAL_OUTPUT_TABLE,
    scoring_date=scoring_date,
)

write_replace_where(
    dataframe=local_output_df,
    table_name=LOCAL_OUTPUT_TABLE,
    scoring_date=scoring_date,
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Validações finais

# COMMAND ----------

global_output_count = (
    global_output_df
    .count()
)

local_output_count = (
    local_output_df
    .count()
)

if global_output_count != len(
    FEATURE_COLUMNS
):
    raise ValueError(
        (
            "A saída global não contém "
            "uma linha por feature."
        )
    )

expected_local_rows = (
    example_customers_pandas[
        [
            "customer_id",
            "scoring_date",
        ]
    ]
    .drop_duplicates()
    .shape[0]
    * len(
        FEATURE_COLUMNS
    )
)

if (
    local_output_count
    != expected_local_rows
):
    raise ValueError(
        (
            "A saída local possui volume "
            "diferente do esperado."
        )
    )

# COMMAND ----------

print(
    f"""
    Explicabilidade concluída.

    Scoring date:
    {scoring_date}

    Modelo:
    {REGISTERED_MODEL_NAME}

    Versão:
    {model_version}

    Amostra global:
    {global_sample_size:,}

    Features globais:
    {global_output_count:,}

    Clientes com explicação local:
    {len(example_customers_pandas):,}

    Contribuições locais:
    {local_output_count:,}

    Tabela global:
    {GLOBAL_OUTPUT_TABLE}

    Tabela local:
    {LOCAL_OUTPUT_TABLE}
    """
)
