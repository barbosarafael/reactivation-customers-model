"""Project configuration helpers.

This module centralizes environment configuration, table naming and project
paths. It is intentionally simple so it can run both inside Databricks and in
local tests.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


DEFAULT_ENV = "dev"


def find_project_root(start_path: str | Path | None = None) -> Path:
    """Find the project root by looking for the `conf` directory.

    Parameters
    ----------
    start_path:
        Optional starting path. If not provided, the search starts from this
        file location.

    Returns
    -------
    Path
        Project root path.

    Raises
    ------
    FileNotFoundError
        If the project root cannot be found.
    """
    if start_path is None:
        start = Path(__file__).resolve()
    else:
        start = Path(start_path).resolve()

    candidates = [start] if start.is_dir() else [start.parent]
    candidates.extend(candidates[0].parents)

    for candidate in candidates:
        if (candidate / "conf").exists():
            return candidate

    raise FileNotFoundError(
        "Could not find project root. Expected to find a 'conf/' directory "
        "in the current path or one of its parents."
    )


def get_environment(default: str = DEFAULT_ENV) -> str:
    """Return the active environment name.

    The environment can be controlled with the PROJECT_ENV environment variable.
    """
    return os.getenv("PROJECT_ENV", default)


def load_config(env: str | None = None, config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML config for the selected environment.

    Parameters
    ----------
    env:
        Environment name, for example `dev`.
    config_path:
        Optional explicit path to a YAML config file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    selected_env = env or get_environment()

    if config_path is None:
        project_root = find_project_root()
        config_file = project_root / "conf" / f"{selected_env}.yml"
    else:
        config_file = Path(config_path).resolve()

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with config_file.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid config file. Expected a dictionary: {config_file}")

    return config


def get_catalog(config: dict[str, Any]) -> str:
    """Return the Databricks catalog configured for the project."""
    return config["databricks"]["catalog"]


def get_schema(config: dict[str, Any]) -> str:
    """Return the Databricks schema configured for the project."""
    return config["databricks"]["schema"]


def get_full_schema_name(config: dict[str, Any]) -> str:
    """Return the fully qualified schema name: catalog.schema."""
    return f"{get_catalog(config)}.{get_schema(config)}"


def get_table_name(config: dict[str, Any], layer: str, table_key: str) -> str:
    """Return a fully qualified table name.

    Parameters
    ----------
    config:
        Project config dictionary.
    layer:
        Table group, for example `bronze`, `silver`, `gold`, `scoring` or
        `monitoring`.
    table_key:
        Logical table key inside the selected group.

    Examples
    --------
    >>> config = load_config("dev")
    >>> get_table_name(config, "bronze", "online_retail_raw")
    'workspace.bettor_crm_ml_dev.bronze_online_retail_raw'
    """
    try:
        raw_table_name = config["tables"][layer][table_key]
    except KeyError as exc:
        available_layers = list(config.get("tables", {}).keys())
        available_tables = list(config.get("tables", {}).get(layer, {}).keys())

        raise KeyError(
            f"Table not found for layer='{layer}' and table_key='{table_key}'. "
            f"Available layers: {available_layers}. "
            f"Available tables in this layer: {available_tables}."
        ) from exc

    return f"{get_full_schema_name(config)}.{raw_table_name}"


def get_modeling_param(config: dict[str, Any], key: str) -> Any:
    """Return a modeling parameter from the config file."""
    try:
        return config["modeling"][key]
    except KeyError as exc:
        available_keys = list(config.get("modeling", {}).keys())
        raise KeyError(
            f"Modeling parameter not found: {key}. "
            f"Available modeling parameters: {available_keys}."
        ) from exc


def get_mlflow_experiment_name(config: dict[str, Any]) -> str:
    """Return the MLflow experiment name for the current environment."""
    return config["mlflow"]["experiment_name"]


def get_registered_model_name(config: dict[str, Any]) -> str:
    """Return the registered model name for the current environment."""
    return config["mlflow"]["registered_model_name"]


def summarize_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a compact summary useful for setup notebooks and logs."""
    return {
        "project_name": config["project"]["name"],
        "environment": config["project"]["environment"],
        "catalog": get_catalog(config),
        "schema": get_schema(config),
        "full_schema_name": get_full_schema_name(config),
        "inactive_days_threshold": config["modeling"]["inactive_days_threshold"],
        "observation_window_days": config["modeling"]["observation_window_days"],
        "prediction_window_days": config["modeling"]["prediction_window_days"],
        "mlflow_experiment_name": get_mlflow_experiment_name(config),
    }
