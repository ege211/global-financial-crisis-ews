"""Transparent model summaries; coefficients describe association, not causation."""

from __future__ import annotations

import pandas as pd


def logistic_coefficient_table(fitted_pipeline: object, features: list[str]) -> pd.DataFrame:
    """Return standardized regularized-logit coefficients and odds ratios.

    Penalized sklearn estimates do not have classical confidence intervals. This table
    intentionally does not manufacture p-values; uncertainty analysis belongs in a
    separately documented bootstrap or unpenalized sensitivity specification.
    """
    estimator = fitted_pipeline.named_steps["model"]
    coefficients = estimator.coef_.ravel()
    table = pd.DataFrame({"feature": features, "coefficient": coefficients})
    table["odds_ratio_per_standard_deviation"] = table["coefficient"].apply(__import__("math").exp)
    table["association_direction"] = table["coefficient"].map(
        lambda value: "higher predicted risk" if value > 0 else "lower predicted risk"
    )
    return table.sort_values("coefficient", key=lambda series: series.abs(), ascending=False)
