"""Unit tests for the project configuration helpers."""

from __future__ import annotations

import pytest

from reactivation_model.config import (
    get_environment,
    get_full_schema_name,
    get_model_alias,
    get_modeling_param,
    get_registered_model_name,
    get_table_name,
    load_config,
    summarize_config,
)


def test_load_dev_config_exposes_the_modeling_contract() -> None:
    """The development configuration contains the expected project contract."""
    config = load_config("dev")

    assert config["project"]["environment"] == "dev"
    assert config["modeling"]["inactive_days_threshold"] == 60
    assert config["modeling"]["observation_window_days"] == 180
    assert config["modeling"]["prediction_window_days"] == 30


def test_table_name_is_resolved_from_catalog_schema_and_logical_key() -> None:
    """Logical table names resolve to a fully qualified Delta table name."""
    config = load_config("dev")

    assert get_full_schema_name(config) == "workspace.gold_layer"
    assert (
        get_table_name(config, "gold", "modeling")
        == "workspace.gold_layer.customer_reactivation_modeling"
    )
    assert (
        get_table_name(config, "synthetic", "full_history")
        == "workspace.synthetic_layer.online_retail_transactions_full"
    )


def test_unknown_table_and_modeling_parameter_raise_diagnostic_errors() -> None:
    """Invalid configuration lookups identify the missing logical key."""
    config = load_config("dev")

    with pytest.raises(KeyError, match="unknown_table"):
        get_table_name(config, "gold", "unknown_table")

    with pytest.raises(KeyError, match="unknown_parameter"):
        get_modeling_param(config, "unknown_parameter")


def test_project_environment_can_be_overridden(monkeypatch: pytest.MonkeyPatch) -> None:
    """PROJECT_ENV selects an environment without changing source code."""
    monkeypatch.setenv("PROJECT_ENV", "test")

    assert get_environment() == "test"

    monkeypatch.delenv("PROJECT_ENV")
    assert get_environment() == "dev"


def test_summary_uses_the_same_values_as_the_configuration() -> None:
    """The summary preserves the key values emitted by setup code and logs."""
    summary = summarize_config(load_config("dev"))

    assert summary == {
        "project_name": "reactivation_customers_model",
        "environment": "dev",
        "catalog": "workspace",
        "schema": "gold_layer",
        "full_schema_name": "workspace.gold_layer",
        "operational_schemas": {
            "bronze": "bronze_layer",
            "silver": "silver_layer",
            "gold": "gold_layer",
            "synthetic": "synthetic_layer",
            "monitoring": "monitoring_layer",
            "explainability": "explainability_layer",
        },
        "inactive_days_threshold": 60,
        "observation_window_days": 180,
        "prediction_window_days": 30,
        "mlflow_experiment_name": "/Shared/bettor_crm_ml/dev/reactivation_30d",
        "registered_model_name": "workspace.default.reactivation_customers_model",
        "model_alias": "champion",
    }

    assert get_registered_model_name(load_config("dev")) == "workspace.default.reactivation_customers_model"
    assert get_model_alias(load_config("dev")) == "champion"
