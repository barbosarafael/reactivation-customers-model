"""Unit tests for synthetic-data helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from reactivation_model.synthetic_data import consolidate_product_catalog


def test_consolidate_product_catalog_preserves_one_row_per_stock_code() -> None:
    """Different descriptions for one code cannot create duplicate invoice lines."""
    catalog = pd.DataFrame(
        {
            "stock_code": ["A", "A", "B"],
            "description": ["Zulu", "Alpha", "Beta"],
            "base_price": [10.0, 20.0, 5.0],
            "frequency": [2.0, 3.0, 4.0],
        }
    )

    result = consolidate_product_catalog(catalog)

    assert result["stock_code"].is_unique
    assert result.to_dict("records") == [
        {
            "stock_code": "A",
            "description": "Alpha",
            "base_price": 16.0,
            "frequency": 5.0,
        },
        {
            "stock_code": "B",
            "description": "Beta",
            "base_price": 5.0,
            "frequency": 4.0,
        },
    ]


def test_consolidate_product_catalog_rejects_an_incomplete_input() -> None:
    """The helper fails clearly if the generator contract is incomplete."""
    catalog = pd.DataFrame(
        {
            "stock_code": ["A"],
            "description": ["Alpha"],
            "base_price": [10.0],
        }
    )

    with pytest.raises(ValueError, match="frequency"):
        consolidate_product_catalog(catalog)
