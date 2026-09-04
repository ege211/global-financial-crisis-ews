"""Unit tests for Phase 5D predictor selection, coverage audit, and eligibility logic."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crisis_ews.evaluation.predictor_audit import (
    CANDIDATE_SPECS,
    audit_candidate_coverage,
    compute_candidate_panel,
    compute_correlation_matrices,
    load_candidate_raw_data,
    run_phase_5d_predictor_audit,
)


def test_candidate_specs_completeness() -> None:
    """Verify that all candidate concepts are represented in CANDIDATE_SPECS."""
    names = [spec.candidate for spec in CANDIDATE_SPECS]
    assert "private_credit_pct_gdp" in names
    assert "private_credit_growth" in names
    assert "current_account_pct_gdp" in names
    assert "exchange_rate_depreciation" in names
    assert "unemployment_rate_ilo" in names
    assert "unemployment_rate_national" in names
    assert len(names) == 6


def test_no_future_year_usage_in_transformations() -> None:
    """Verify that growth and depreciation transformations strictly use past data (t-1 to t)."""
    # Create synthetic two-year, two-country panel
    panel = pd.DataFrame(
        {
            "country_code": ["AAA", "AAA", "BBB", "BBB"],
            "year": [2000, 2001, 2000, 2001],
            "crisis_within_horizon": [0, 0, 0, 0],
        }
    )
    raw_data = {
        "private_credit_pct_gdp": pd.DataFrame(
            {
                "country_code": ["AAA", "AAA", "BBB", "BBB"],
                "year": [2000, 2001, 2000, 2001],
                "value": [100.0, 120.0, 50.0, 40.0],
            }
        ),
        "current_account_pct_gdp": pd.DataFrame(
            {
                "country_code": ["AAA", "AAA", "BBB", "BBB"],
                "year": [2000, 2001, 2000, 2001],
                "value": [-2.0, -3.0, 1.0, 2.0],
            }
        ),
        "official_exchange_rate": pd.DataFrame(
            {
                "country_code": ["AAA", "AAA", "BBB", "BBB"],
                "year": [2000, 2001, 2000, 2001],
                "value": [10.0, 11.0, 1.0, 0.9],
            }
        ),
        "unemployment_ilo": pd.DataFrame(
            {
                "country_code": ["AAA", "AAA", "BBB", "BBB"],
                "year": [2000, 2001, 2000, 2001],
                "value": [5.0, 6.0, 4.0, 4.5],
            }
        ),
        "unemployment_nat": pd.DataFrame(
            {
                "country_code": ["AAA", "AAA", "BBB", "BBB"],
                "year": [2000, 2001, 2000, 2001],
                "value": [5.2, 6.1, 4.1, 4.4],
            }
        ),
    }

    result = compute_candidate_panel(panel, raw_data)

    # In year 2000 (first available year), YoY growth must be NaN (no t-1)
    aaa_2000 = result[(result["country_code"] == "AAA") & (result["year"] == 2000)].iloc[0]
    assert np.isnan(aaa_2000["private_credit_growth"])
    assert np.isnan(aaa_2000["exchange_rate_depreciation"])

    # In year 2001, growth must strictly equal (v_2001 - v_2000) / v_2000 * 100
    aaa_2001 = result[(result["country_code"] == "AAA") & (result["year"] == 2001)].iloc[0]
    assert aaa_2001["private_credit_growth"] == pytest.approx(20.0)
    assert aaa_2001["exchange_rate_depreciation"] == pytest.approx(10.0)

    # Country BBB values must not leak into AAA
    bbb_2001 = result[(result["country_code"] == "BBB") & (result["year"] == 2001)].iloc[0]
    assert bbb_2001["private_credit_growth"] == pytest.approx(-20.0)
    assert bbb_2001["exchange_rate_depreciation"] == pytest.approx(-10.0)


def test_country_year_uniqueness_in_candidate_panel() -> None:
    """Verify that candidate panel has zero duplicate (country_code, year) pairs."""
    root = Path(__file__).parents[1]
    panel = pd.read_csv(root / "data/processed/expanded_research/modeling_panel.csv")
    countries = sorted(panel["country_code"].unique())
    raw_data = load_candidate_raw_data(root / "data/raw/expanded_research", countries)
    panel_with_candidates = compute_candidate_panel(panel, raw_data)

    assert len(panel_with_candidates) == len(panel)
    dupes = panel_with_candidates.duplicated(subset=["country_code", "year"]).sum()
    assert dupes == 0


def test_coverage_calculations_and_thresholds() -> None:
    """Verify exact coverage percentages and validation period metrics."""
    root = Path(__file__).parents[1]
    cov_df = pd.read_csv(root / "results/tables/phase5d_candidate_coverage.csv")

    assert len(cov_df) == 6
    pc_row = cov_df[cov_df["candidate"] == "private_credit_pct_gdp"].iloc[0]
    ca_row = cov_df[cov_df["candidate"] == "current_account_pct_gdp"].iloc[0]
    uem_row = cov_df[cov_df["candidate"] == "unemployment_rate_ilo"].iloc[0]

    # Current account must have > 95% coverage in validation period
    assert ca_row["validation_period_coverage_pct"] >= 95.0
    assert "19/19" in ca_row["positive_events_retained"]

    # Unemployment ILO must have > 97% coverage in validation period
    assert uem_row["validation_period_coverage_pct"] >= 97.0
    assert "19/19" in uem_row["positive_events_retained"]

    # Private credit level must have >= 85% validation coverage
    assert pc_row["validation_period_coverage_pct"] >= 85.0
    assert "18/19" in pc_row["positive_events_retained"]


def test_direct_audit_and_correlation_calculation() -> None:
    """Verify that audit_candidate_coverage and compute_correlation_matrices function correctly on panel."""
    root = Path(__file__).parents[1]
    panel = pd.read_csv(root / "data/processed/expanded_research/modeling_panel.csv")
    countries = sorted(panel["country_code"].unique())
    raw_data = load_candidate_raw_data(root / "data/raw/expanded_research", countries)
    panel_cand = compute_candidate_panel(panel, raw_data)

    cov_df = audit_candidate_coverage(panel_cand)
    assert len(cov_df) == 6
    assert set(cov_df["candidate"].tolist()) == {spec.candidate for spec in CANDIDATE_SPECS}

    pearson, spearman = compute_correlation_matrices(panel_cand)
    assert pearson.shape == (9, 9)
    assert spearman.shape == (9, 9)
    assert (pearson.loc["inflation", "exchange_rate_depreciation"]) > 0.90


def test_eligibility_logic_consistency() -> None:
    """Verify that eligible set matches the pre-registered 3 indicators."""
    root = Path(__file__).parents[1]
    cov_df = pd.read_csv(root / "results/tables/phase5d_candidate_coverage.csv")

    eligible = set(cov_df[cov_df["eligible"]]["candidate"].tolist())
    expected_eligible = {
        "private_credit_pct_gdp",
        "current_account_pct_gdp",
        "unemployment_rate_ilo",
    }
    assert eligible == expected_eligible

    # Excluded indicators must have non-empty exclusion reasons
    excluded = cov_df[~cov_df["eligible"]]
    assert len(excluded) == 3
    for _, row in excluded.iterrows():
        assert len(str(row["exclusion_reason"]).strip()) > 10


def test_reproducibility_of_phase_5d_audit() -> None:
    """Verify that run_phase_5d_predictor_audit executes deterministically and returns expected summary."""
    root = Path(__file__).parents[1]
    summary = run_phase_5d_predictor_audit(root)

    assert summary["candidate_count"] == 6
    assert len(summary["eligible_candidates"]) == 3
    assert set(summary["eligible_candidates"]) == {
        "private_credit_pct_gdp",
        "current_account_pct_gdp",
        "unemployment_rate_ilo",
    }
    assert (root / "results/tables/phase5d_candidate_coverage.csv").exists()
    assert (root / "results/tables/expanded_phase5d_candidate_coverage.csv").exists()
    assert (root / "results/tables/phase5d_correlations_pearson.csv").exists()
    assert (root / "results/tables/phase5d_correlations_spearman.csv").exists()
