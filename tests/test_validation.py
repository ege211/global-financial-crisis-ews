import pandas as pd
import pytest

from crisis_ews.data.validation import country_coverage_report, validate_long_data


def test_validation_rejects_duplicate_country_year_variable() -> None:
    data = pd.DataFrame(
        {
            "country_code": ["AAA", "AAA"],
            "year": [2000, 2000],
            "variable": ["x", "x"],
            "value": [1.0, 1.0],
        }
    )
    with pytest.raises(ValueError, match="Critical data quality"):
        validate_long_data(data, {"x": {"min": 0}})


def test_country_coverage_counts_requested_window_structural_missingness() -> None:
    long_data = pd.DataFrame(
        {
            "country_code": ["AAA", "AAA"],
            "country_name": ["A", "A"],
            "year": [2000, 2001],
            "variable": ["x", "x"],
            "value": [1.0, 2.0],
        }
    )
    panel = pd.DataFrame(
        {"country_code": ["AAA"], "year": [2000], "x": [1.0], "crisis_within_horizon": [0]}
    )
    report = country_coverage_report(long_data, panel, ["AAA", "BBB"], ["x", "y"], 2000, 2001)
    assert report.loc[report.country_code.eq("AAA"), "missingness_percent"].item() == 50.0
    assert report.loc[report.country_code.eq("BBB"), "missingness_percent"].item() == 100.0
