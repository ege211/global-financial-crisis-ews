"""Non-destructive conversion of raw long data into an annual research panel."""

from __future__ import annotations

import pandas as pd

from crisis_ews.data.validation import validate_long_data


def build_panel(long_data: pd.DataFrame, variables: dict[str, dict[str, object]]) -> pd.DataFrame:
    """Pivot unique country-year indicator data; no imputation is performed here."""
    validate_long_data(long_data, variables)
    panel = long_data.pivot(index=["country_code", "year"], columns="variable", values="value")
    return panel.reset_index().sort_values(["country_code", "year"]).reset_index(drop=True)
