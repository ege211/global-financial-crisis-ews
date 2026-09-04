"""Unit tests for Phase 5C robustness and sensitivity analysis."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crisis_ews.evaluation.robustness_analysis import (
    TrainingFoldWinsorizer,
    calculate_metrics_dict,
)


def test_training_fold_winsorizer_strict_training_bounds() -> None:
    """Verify that winsorizer learns bounds strictly from training data and clips test outliers."""
    # Synthetic training data with values from 0 to 100
    train_data = pd.DataFrame({"inflation": np.linspace(0, 100, 101)})
    winsorizer = TrainingFoldWinsorizer(column_name="inflation", lower_quantile=0.01, upper_quantile=0.99)
    winsorizer.fit(train_data)

    assert winsorizer.lower_bound_ == pytest.approx(1.0)
    assert winsorizer.upper_bound_ == pytest.approx(99.0)

    # Test data contains an extreme hyperinflation outlier of 5000%
    test_data = pd.DataFrame({"inflation": [-50.0, 10.0, 50.0, 5000.0]})
    transformed = winsorizer.transform(test_data)

    # Outliers should be strictly clamped to training bounds [1.0, 99.0]
    assert transformed["inflation"].iloc[0] == pytest.approx(1.0)
    assert transformed["inflation"].iloc[1] == pytest.approx(10.0)
    assert transformed["inflation"].iloc[2] == pytest.approx(50.0)
    assert transformed["inflation"].iloc[3] == pytest.approx(99.0)

    # Ensure bounds were not shifted by the test data
    assert winsorizer.lower_bound_ == pytest.approx(1.0)
    assert winsorizer.upper_bound_ == pytest.approx(99.0)


def test_calculate_metrics_dict_single_class_handling() -> None:
    """Verify that single-class test folds produce None for ROC-AUC and PR-AUC."""
    y_zero = np.zeros(76, dtype=int)
    prob = np.full(76, 0.3)
    metrics = calculate_metrics_dict(y_zero, prob, threshold=0.50)

    assert metrics["roc_auc"] is None
    assert metrics["pr_auc"] is None
    assert metrics["positive_events"] == 0
    assert metrics["observations"] == 76
    assert metrics["true_positives"] == 0
    assert metrics["false_positives"] == 0


def test_phase5c_artifact_existence() -> None:
    """Verify that all four required Phase 5C research tables exist."""
    root = Path(__file__).parents[1]
    tables_dir = root / "results/tables"

    assert (tables_dir / "expanded_robustness_inflation.csv").exists()
    assert (tables_dir / "expanded_robustness_class_weight.csv").exists()
    assert (tables_dir / "expanded_threshold_sensitivity.csv").exists()
    assert (tables_dir / "expanded_fold_stability.csv").exists()


def test_phase5b_baseline_predictions_not_overwritten() -> None:
    """Verify that locked Phase 5B predictions exist and were not overwritten."""
    root = Path(__file__).parents[1]
    lr_base = pd.read_csv(root / "results/model_outputs/logistic_regression_predictions.csv")
    rf_base = pd.read_csv(root / "results/model_outputs/random_forest_predictions.csv")

    assert len(lr_base) == 684
    assert len(rf_base) == 684
    assert lr_base["crisis_within_horizon"].sum() == 19
    assert rf_base["crisis_within_horizon"].sum() == 19

    # Separate sensitivity files must exist
    outputs_dir = root / "results/model_outputs"
    assert (outputs_dir / "logistic_regression_inflation_winsorized_predictions.csv").exists()
    assert (outputs_dir / "random_forest_inflation_winsorized_predictions.csv").exists()
    assert (outputs_dir / "logistic_regression_unweighted_predictions.csv").exists()
    assert (outputs_dir / "random_forest_unweighted_predictions.csv").exists()


def test_threshold_sensitivity_monotonicity() -> None:
    """Verify that false positive rates decrease monotonically as threshold increases."""
    root = Path(__file__).parents[1]
    thresh_df = pd.read_csv(root / "results/tables/expanded_threshold_sensitivity.csv")

    for model_name in ["logistic_regression", "random_forest"]:
        sub = thresh_df[thresh_df["model"] == model_name].sort_values("threshold")
        fpr_values = sub["false_positive_rate"].tolist()
        assert fpr_values[0] >= fpr_values[1] >= fpr_values[2]

        recalls = sub["recall"].tolist()
        assert recalls[0] >= recalls[1] >= recalls[2]


def test_fold_stability_table_structure() -> None:
    """Verify that fold stability contains 18 rows, explicit 'Not estimable' labels, and 15 events in 2007."""
    root = Path(__file__).parents[1]
    stab_df = pd.read_csv(root / "results/tables/expanded_fold_stability.csv")

    assert len(stab_df) == 18
    assert set(stab_df["test_year"].unique()) == set(range(2000, 2009))

    for model_name in ["logistic_regression", "random_forest"]:
        m_df = stab_df[stab_df["model"] == model_name]
        assert len(m_df) == 9
        assert m_df["positive_events"].sum() == 19

        # Single class years
        for yr in [2000, 2003, 2004, 2005]:
            row = m_df[m_df["test_year"] == yr].iloc[0]
            assert row["positive_events"] == 0
            assert row["roc_auc"] == "Not estimable"
            assert row["pr_auc"] == "Not estimable"

        # Concentration year 2007
        row_2007 = m_df[m_df["test_year"] == 2007].iloc[0]
        assert row_2007["positive_events"] == 15


def test_sensitivity_prediction_integrity() -> None:
    """Verify that sensitivity predictions have no duplicates and probabilities are bounded [0, 1]."""
    root = Path(__file__).parents[1]
    outputs_dir = root / "results/model_outputs"
    sensitivity_files = [
        "logistic_regression_inflation_winsorized_predictions.csv",
        "random_forest_inflation_winsorized_predictions.csv",
        "logistic_regression_unweighted_predictions.csv",
        "random_forest_unweighted_predictions.csv",
    ]

    for fname in sensitivity_files:
        df = pd.read_csv(outputs_dir / fname)
        assert len(df) == 684
        assert df.duplicated(subset=["country_code", "year"]).sum() == 0
        assert (df["predicted_probability"] >= 0.0).all()
        assert (df["predicted_probability"] <= 1.0).all()
        assert (df["training_end_year"] < df["test_year"]).all()
