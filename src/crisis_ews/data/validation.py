"""Fail-loud data-quality checks and coverage reporting."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

REQUIRED_COLUMNS = {"country_code", "year", "variable", "value"}


@dataclass(frozen=True)
class ValidationResult:
    observations: int
    duplicate_rows: int
    invalid_country_codes: int
    invalid_years: int
    impossible_values: int


def validate_long_data(frame: pd.DataFrame, variables: dict[str, dict[str, object]]) -> ValidationResult:
    """Validate long-form macro data, raising on violations that invalidate modeling."""
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Data missing required columns: {sorted(missing)}")
    duplicate_rows = int(frame.duplicated(["country_code", "year", "variable"]).sum())
    country_codes = frame["country_code"].astype("string").fillna("")
    invalid_codes = int((~country_codes.str.fullmatch(r"[A-Z]{3}")).sum())
    invalid_years = int((~frame["year"].between(1900, 2100)).sum())
    impossible_values = 0
    for name, spec in variables.items():
        subset = frame.loc[frame["variable"].eq(name), "value"]
        if "min" in spec:
            impossible_values += int((subset < float(spec["min"])).sum())
        if "max" in spec:
            impossible_values += int((subset > float(spec["max"])).sum())
    result = ValidationResult(len(frame), duplicate_rows, invalid_codes, invalid_years, impossible_values)
    if any((duplicate_rows, invalid_codes, invalid_years, impossible_values)):
        raise ValueError(f"Critical data quality failure: {result}")
    return result


def coverage_report(frame: pd.DataFrame, countries: list[str], variables: list[str]) -> pd.DataFrame:
    """Summarize each country-variable series without filling missing observations."""
    index = pd.MultiIndex.from_product([countries, variables], names=["country_code", "variable"])
    grouped = frame.groupby(["country_code", "variable"])["year"].agg(
        first_available_year="min", last_available_year="max", observations="count"
    )
    report = grouped.reindex(index).reset_index()
    span = report["last_available_year"] - report["first_available_year"] + 1
    report["missing_years_within_span"] = (span - report["observations"]).where(span.notna())
    return report


def country_coverage_report(
    long_data: pd.DataFrame,
    panel: pd.DataFrame,
    countries: list[str],
    variables: list[str],
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Report full-window country coverage, including structural missingness.

    Missingness is over all requested country-year-variable cells, rather than only
    between a series' first and last observed values.
    """
    expected_cells = (end_year - start_year + 1) * len(variables)
    names = long_data.groupby("country_code")["country_name"].first().to_dict()
    rows: list[dict[str, object]] = []
    for country in countries:
        observed = long_data.loc[long_data["country_code"].eq(country)]
        country_panel = panel.loc[panel["country_code"].eq(country)]
        rows.append(
            {
                "country_code": country,
                "country": names.get(country),
                "first_available_year": observed["year"].min() if not observed.empty else None,
                "last_available_year": observed["year"].max() if not observed.empty else None,
                "country_year_observations": int(country_panel.shape[0]),
                "complete_case_country_years": int(country_panel[variables].notna().all(axis=1).sum()),
                "crisis_observations": int(country_panel.get("crisis_within_horizon", pd.Series(dtype=int)).sum()),
                "observed_variable_year_cells": int(observed.shape[0]),
                "expected_variable_year_cells": expected_cells,
                "missingness_percent": round(100 * (1 - len(observed) / expected_cells), 2),
            }
        )
    return pd.DataFrame(rows)
