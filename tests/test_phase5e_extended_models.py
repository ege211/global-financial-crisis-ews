"""Unit tests for Phase 5E 6-variable Extended Model evaluation."""

from pathlib import Path

import pandas as pd
import pytest

from crisis_ews.evaluation.extended_models import (
    LOCKED_EXTENDED_FEATURES,
    build_extended_modeling_panel,
)


def test_six_feature_schema() -> None:
    """Verify that the locked extended set contains exactly the approved 6 features."""
    assert len(LOCKED_EXTENDED_FEATURES) == 6
    assert LOCKED_EXTENDED_FEATURES == [
        "gdp_growth",
        "inflation",
        "reserves_usd",
        "reserves_usd_yoy_pct_change",
        "private_credit_pct_gdp",
        "current_account_pct_gdp",
    ]


def test_prediction_integrity() -> None:
    """Verify prediction integrity: valid probabilities [0, 1], binary classes {0, 1}, and uniqueness."""
    root = Path(__file__).parents[1]
    outputs_dir = root / "results/model_outputs"
    lr_preds = pd.read_csv(outputs_dir / "extended_logistic_regression_predictions.csv")
    rf_preds = pd.read_csv(outputs_dir / "extended_random_forest_predictions.csv")

    for df in [lr_preds, rf_preds]:
        assert df.duplicated(subset=["country_code", "year"]).sum() == 0
        assert (df["predicted_probability"] >= 0.0).all()
        assert (df["predicted_probability"] <= 1.0).all()
        assert set(df["predicted_class"].unique()).issubset({0, 1})
        assert not df["predicted_probability"].isna().any()
        assert not df["predicted_class"].isna().any()


def test_no_leakage() -> None:
    """Verify that training end years are strictly before test evaluation years."""
    root = Path(__file__).parents[1]
    outputs_dir = root / "results/model_outputs"
    lr_preds = pd.read_csv(outputs_dir / "extended_logistic_regression_predictions.csv")
    rf_preds = pd.read_csv(outputs_dir / "extended_random_forest_predictions.csv")

    for df in [lr_preds, rf_preds]:
        assert len(df) == 684
        assert (df["training_end_year"] < df["test_year"]).all()
        assert (df["training_end_year"] < df["year"]).all()
        assert (df["test_year"] == df["year"]).all()
        assert (df["training_start_year"] == 1990).all()
        assert (df["training_end_year"] >= 1999).all()
        assert (df["training_end_year"] - df["training_start_year"] + 1 >= 10).all()


def test_fold_integrity() -> None:
    """Verify that extended predictions contain 9 folds with 76 observations each and 19 total events."""
    root = Path(__file__).parents[1]
    outputs_dir = root / "results/model_outputs"
    lr_preds = pd.read_csv(outputs_dir / "extended_logistic_regression_predictions.csv")
    rf_preds = pd.read_csv(outputs_dir / "extended_random_forest_predictions.csv")

    for df in [lr_preds, rf_preds]:
        counts = df.groupby("test_year").size()
        assert len(counts) == 9
        assert (counts == 76).all()
        assert sorted(counts.index.tolist()) == list(range(2000, 2009))
        assert df["crisis_within_horizon"].sum() == 19


def test_baseline_artifact_preservation() -> None:
    """Verify that Phase 5B baseline predictions exist and match locked values."""
    root = Path(__file__).parents[1]
    outputs_dir = root / "results/model_outputs"
    tables_dir = root / "results/tables"

    # Baseline prediction files must exist and remain untouched
    base_lr = pd.read_csv(outputs_dir / "logistic_regression_predictions.csv")
    base_rf = pd.read_csv(outputs_dir / "random_forest_predictions.csv")
    assert len(base_lr) == 684
    assert len(base_rf) == 684

    # Comparison table must preserve baseline metrics exactly
    comp_df = pd.read_csv(tables_dir / "expanded_extended_model_comparison.csv")
    assert len(comp_df) == 4

    b_lr_row = comp_df[comp_df["model"] == "logistic_regression"].iloc[0]
    assert b_lr_row["pr_auc"] == pytest.approx(0.0278, abs=1e-4)
    assert b_lr_row["roc_auc"] == pytest.approx(0.5106, abs=1e-4)
    assert b_lr_row["true_positives"] == 4
    assert b_lr_row["false_positives"] == 208
    assert b_lr_row["false_negatives"] == 15
    assert b_lr_row["true_negatives"] == 457

    b_rf_row = comp_df[comp_df["model"] == "random_forest"].iloc[0]
    assert b_rf_row["pr_auc"] == pytest.approx(0.0256, abs=1e-4)
    assert b_rf_row["roc_auc"] == pytest.approx(0.4712, abs=1e-4)
    assert b_rf_row["true_positives"] == 4
    assert b_rf_row["false_positives"] == 161
    assert b_rf_row["false_negatives"] == 15
    assert b_rf_row["true_negatives"] == 504


def test_expected_output_sizes() -> None:
    """Verify expected output sizes for panel, predictions, fold metrics, and summary tables."""
    root = Path(__file__).parents[1]
    tables_dir = root / "results/tables"
    outputs_dir = root / "results/model_outputs"

    panel = build_extended_modeling_panel(root)
    assert len(panel) == 2736
    assert panel["country_code"].nunique() == 76

    lr_preds = pd.read_csv(outputs_dir / "extended_logistic_regression_predictions.csv")
    rf_preds = pd.read_csv(outputs_dir / "extended_random_forest_predictions.csv")
    assert len(lr_preds) == 684
    assert len(rf_preds) == 684

    lr_folds = pd.read_csv(tables_dir / "expanded_extended_logistic_regression_fold_metrics.csv")
    rf_folds = pd.read_csv(tables_dir / "expanded_extended_random_forest_fold_metrics.csv")
    assert len(lr_folds) == 9
    assert len(rf_folds) == 9

    comp_df = pd.read_csv(tables_dir / "expanded_extended_model_comparison.csv")
    assert len(comp_df) == 4


def test_extended_feature_interpretations_schema() -> None:
    """Verify that feature interpretations table has 54 rows (9 folds * 6 features)."""
    root = Path(__file__).parents[1]
    interp_path = root / "results/tables/expanded_extended_feature_interpretations.csv"
    assert interp_path.exists()
    df = pd.read_csv(interp_path)

    assert len(df) == 54  # 9 folds * 6 features
    assert set(df["feature"].unique()) == set(LOCKED_EXTENDED_FEATURES)
    assert (df["logistic_odds_ratio"] > 0).all()
    assert (df["random_forest_importance"] >= 0).all()

    # In each fold, tree importances should sum to approximately 1.0
    for _, grp in df.groupby("test_year"):
        assert grp["random_forest_importance"].sum() == pytest.approx(1.0, abs=1e-2)
