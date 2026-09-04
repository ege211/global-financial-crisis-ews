"""Unit tests for Phase 5A modeling infrastructure and pre-fit audit."""

from pathlib import Path

import pandas as pd

from crisis_ews.evaluation.pre_fit_audit import (
    audit_feature_missingness,
    audit_panel_integrity,
    audit_temporal_folds,
    compute_pre_fit_correlations,
    compute_pre_fit_feature_statistics,
    run_pre_fit_audit,
    verify_preprocessing_isolation,
)
from crisis_ews.evaluation.walk_forward import expanding_window_folds


def test_expanding_window_folds_bounded_by_test_end_year() -> None:
    years = pd.Series(range(1990, 2026))
    folds = list(
        expanding_window_folds(
            years,
            minimum_train_years=10,
            test_start_year=2000,
            test_end_year=2008,
        )
    )
    assert len(folds) == 9
    assert [fold.test_year for fold in folds] == list(range(2000, 2009))
    assert [fold.train_end_year for fold in folds] == list(range(1999, 2008))


def test_expanding_window_folds_temporal_ordering_and_isolation() -> None:
    years = pd.Series(range(1990, 2026))
    folds = list(
        expanding_window_folds(
            years,
            minimum_train_years=10,
            test_start_year=2000,
            test_end_year=2008,
        )
    )
    for fold in folds:
        assert fold.train_end_year < fold.test_year
        assert fold.test_year - fold.train_end_year == 1


def test_pre_fit_audit_panel_integrity() -> None:
    root = Path(__file__).parents[1]
    panel_path = root / "data/processed/expanded_research/modeling_panel.csv"
    assert panel_path.exists()
    panel = pd.read_csv(panel_path)
    info = audit_panel_integrity(panel)
    assert info["total_rows"] == 2736
    assert info["unique_countries"] == 76
    assert info["min_year"] == 1990
    assert info["max_year"] == 2025
    assert info["duplicates"] == 0
    assert info["positive_labels"] == 44
    assert info["negative_labels"] == 2692


def test_pre_fit_missingness_confined_to_1990() -> None:
    root = Path(__file__).parents[1]
    panel = pd.read_csv(root / "data/processed/expanded_research/modeling_panel.csv")
    missing_df = audit_feature_missingness(panel)
    assert len(missing_df) == 4
    for _, row in missing_df.iterrows():
        assert row["status"] == "passed"
    reserves_yoy = missing_df.loc[missing_df["variable"] == "reserves_usd_yoy_pct_change"].iloc[0]
    assert reserves_yoy["missing_count"] == 76
    assert reserves_yoy["years_with_missing"] == "[1990]"


def test_pre_fit_temporal_folds_have_dual_classes_and_zero_leakage() -> None:
    root = Path(__file__).parents[1]
    panel = pd.read_csv(root / "data/processed/expanded_research/modeling_panel.csv")
    folds_df = audit_temporal_folds(panel, minimum_train_years=10, test_start_year=2000, test_end_year=2008)
    assert len(folds_df) == 9
    assert folds_df["leakage_check"].eq("PASSED").all()
    assert folds_df["train_has_both_classes"].all()
    assert (folds_df["train_positive_labels"] >= 24).all()
    assert folds_df["test_observations"].sum() == 684
    assert folds_df["test_positive_labels"].sum() == 19


def test_pre_fit_feature_statistics() -> None:
    root = Path(__file__).parents[1]
    panel = pd.read_csv(root / "data/processed/expanded_research/modeling_panel.csv")
    stats_df = compute_pre_fit_feature_statistics(panel)
    assert len(stats_df) == 4
    assert set(stats_df["variable"]) == {"gdp_growth", "inflation", "reserves_usd", "reserves_usd_yoy_pct_change"}
    gdp_row = stats_df.loc[stats_df["variable"] == "gdp_growth"].iloc[0]
    assert gdp_row["observations"] == 2736
    assert -60 < gdp_row["mean"] < 10


def test_pre_fit_feature_correlations_low_multicollinearity() -> None:
    root = Path(__file__).parents[1]
    panel = pd.read_csv(root / "data/processed/expanded_research/modeling_panel.csv")
    corr_df = compute_pre_fit_correlations(panel)
    off_diagonal = corr_df.loc[corr_df["variable_1"] != corr_df["variable_2"]]
    assert (off_diagonal["pearson_correlation"].abs() < 0.20).all()
    assert (off_diagonal["spearman_correlation"].abs() < 0.20).all()


def test_preprocessing_isolation_verification() -> None:
    root = Path(__file__).parents[1]
    panel = pd.read_csv(root / "data/processed/expanded_research/modeling_panel.csv")
    result = verify_preprocessing_isolation(panel)
    assert result["isolation_verified"] is True


def test_run_pre_fit_audit_end_to_end() -> None:
    root = Path(__file__).parents[1]
    summary = run_pre_fit_audit(root, root / "config/expanded_research.yaml")
    assert summary["folds_count"] == 9
    assert summary["cumulative_test_rows"] == 684
    assert summary["cumulative_test_positives"] == 19
    assert summary["post_2008_rows"] == 1292
    assert summary["post_2008_positives"] == 1
    assert summary["preprocessing_isolation_passed"] is True
    assert summary["models_fitted"] == 0
    assert summary["performance_metrics_calculated"] is False
