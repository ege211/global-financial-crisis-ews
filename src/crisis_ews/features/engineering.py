"""Feature transformations applied separately within countries and using past data only."""

from __future__ import annotations

import pandas as pd


def add_configured_transformations(
    panel: pd.DataFrame, variables: dict[str, dict[str, object]]
) -> pd.DataFrame:
    """Add configured year-over-year percentage changes without imputation.

    The change at t uses only value(t) and value(t-1). Zero denominators generate a
    missing value rather than an arbitrary number.
    """
    transformed = panel.sort_values(["country_code", "year"]).copy()
    for name, spec in variables.items():
        if spec.get("transform") != "yoy_pct_change":
            continue
        previous = transformed.groupby("country_code")[name].shift(1)
        transformed[f"{name}_yoy_pct_change"] = (transformed[name] / previous - 1.0) * 100.0
        transformed.loc[previous.eq(0), f"{name}_yoy_pct_change"] = float("nan")
    return transformed
