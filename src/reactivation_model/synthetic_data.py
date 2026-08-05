"""Helpers for deterministic synthetic-transaction generation."""

from __future__ import annotations

import pandas as pd


PRODUCT_CATALOG_COLUMNS = {
    "stock_code",
    "description",
    "base_price",
    "frequency",
}


def consolidate_product_catalog(product_catalog: pd.DataFrame) -> pd.DataFrame:
    """Return one deterministic product candidate per ``stock_code``.

    The source history can associate a product code with more than one textual
    description. The synthetic generator samples catalog rows without
    replacement within an invoice, so duplicate product codes must be
    consolidated first to preserve the ``invoice + stock_code`` key.

    Prices are combined as a frequency-weighted mean and descriptions use the
    lexicographically smallest available value to keep the result deterministic.
    """
    missing_columns = PRODUCT_CATALOG_COLUMNS.difference(product_catalog.columns)
    if missing_columns:
        raise ValueError(
            "Product catalog is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if product_catalog.empty:
        return product_catalog.copy()

    catalog = product_catalog.copy()
    catalog["_weighted_price"] = catalog["base_price"] * catalog["frequency"]

    consolidated = (
        catalog.groupby("stock_code", as_index=False, sort=True)
        .agg(
            description=("description", "min"),
            frequency=("frequency", "sum"),
            _weighted_price=("_weighted_price", "sum"),
        )
    )
    consolidated["base_price"] = (
        consolidated["_weighted_price"] / consolidated["frequency"]
    )

    return consolidated.loc[
        :,
        ["stock_code", "description", "base_price", "frequency"],
    ]
