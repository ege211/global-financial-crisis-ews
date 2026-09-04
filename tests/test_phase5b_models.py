"""Unit tests for Phase 5B initial model estimation, predictions, and metrics."""

from pathlib import Path

import numpy as np
import pandas as pd

from crisis_ews.evaluation.initial_models import (
    audit_prediction_integrity,
    compute_calibration_assessment,
    run_phase_5b_initial_models,
)
from crisis_ews.evaluation.metrics import classification_metrics
from crisis_ews.evaluation.walk_forward import compute_per_fold_metrics


def test_prediction_test_year_alignment() -> None:
    root = Path(__file__).parents[1]
    lr_preds = pd.read_csv(root / "results/model_outputs/logistic_regression_predictions.csv")
    rf_preds = pd.read_csv(root / "results/model_outputs/random_forest_predictions.csv")

    for preds in [lr_preds, rf_preds]:
        assert len(preds) == 684
        assert (preds["test_year"] == preds["year"]).all()
        assert set(preds["test_year"].unique()) == set(range(2000, 2009))


def test_no_training_observations_in_prediction_outputs() -> None:
    root = Path(__file__).parents[1]
    lr_preds = pd.read_csv(root / "results/model_outputs/logistic_regression_predictions.csv")
    rf_preds = pd.read_csv(root / "results/model_outputs/random_forest_predictions.csv")

    for preds in [lr_preds, rf_preds]:
        assert (preds["training_end_year"] < preds["year"]).all()
        assert (preds["training_end_year"] < preds["test_year"]).all()


def test_probability_bounds() -> None:
    root = Path(__file__).parents[1]
    lr_preds = pd.read_csv(root / "results/model_outputs/logistic_regression_predictions.csv")
    rf_preds = pd.read_csv(root / "results/model_outputs/random_forest_predictions.csv")

    for preds in [lr_preds, rf_preds]:
        assert (preds["predicted_probability"] >= 0.0).all()
        assert (preds["predicted_probability"] <= 1.0).all()


def test_no_duplicate_prediction_keys() -> None:
    root = Path(__file__).parents[1]
    lr_preds = pd.read_csv(root / "results/model_outputs/logistic_regression_predictions.csv")
    rf_preds = pd.read_csv(root / "results/model_outputs/random_forest_predictions.csv")

    for preds in [lr_preds, rf_preds]:
        dupes = preds.duplicated(subset=["country_code", "year"]).sum()
        assert dupes == 0


def test_fold_integrity_and_positive_counts() -> None:
    root = Path(__file__).parents[1]
    lr_preds = pd.read_csv(root / "results/model_outputs/logistic_regression_predictions.csv")
    fold_counts = lr_preds.groupby("test_year").size()
    assert len(fold_counts) == 9
    assert (fold_counts == 76).all()
    assert lr_preds["crisis_within_horizon"].sum() == 19


def test_metric_handling_when_fold_contains_one_class() -> None:
    # Test fold with zero crisis events (e.g. year 2000)
    y_true = np.zeros(76, dtype=int)
    prob = np.full(76, 0.45)
    metrics = classification_metrics(y_true, prob, threshold=0.50)
    assert metrics["roc_auc"] is None
    assert metrics["pr_auc"] is None
    assert metrics["crises"] == 0
    assert metrics["observations"] == 76


def test_audit_prediction_integrity_utility() -> None:
    root = Path(__file__).parents[1]
    lr_preds = pd.read_csv(root / "results/model_outputs/logistic_regression_predictions.csv")
    panel = pd.read_csv(root / "data/processed/expanded_research/modeling_panel.csv")
    audit = audit_prediction_integrity(lr_preds, panel, expected_rows=684)
    assert audit["integrity_verified"] is True
    assert audit["duplicates"] == 0


def test_compute_per_fold_metrics_returns_9_rows() -> None:
    root = Path(__file__).parents[1]
    lr_preds = pd.read_csv(root / "results/model_outputs/logistic_regression_predictions.csv")
    fold_metrics = compute_per_fold_metrics(lr_preds, threshold=0.50)
    assert len(fold_metrics) == 9
    assert list(fold_metrics["test_year"]) == list(range(2000, 2009))


def test_compute_calibration_assessment() -> None:
    root = Path(__file__).parents[1]
    lr_preds = pd.read_csv(root / "results/model_outputs/logistic_regression_predictions.csv")
    summary, binned = compute_calibration_assessment(lr_preds, "logistic_regression")
    assert summary["observations"] == 684
    assert summary["crises"] == 19
    assert summary["brier_score"] > 0
    assert len(binned) == 5


def test_feature_interpretations_table() -> None:
    root = Path(__file__).parents[1]
    interp_path = root / "results/tables/expanded_feature_interpretations.csv"
    assert interp_path.exists()
    interp_df = pd.read_csv(interp_path)
    # 9 folds * 4 features = 36 rows
    assert len(interp_df) == 36
    assert set(interp_df["feature"]) == {"gdp_growth", "inflation", "reserves_usd", "reserves_usd_yoy_pct_change"}
    assert (interp_df["logistic_odds_ratio"] > 0).all()
    assert (interp_df["random_forest_importance"] >= 0).all()


def test_run_phase_5b_initial_models_end_to_end() -> None:
    root = Path(__file__).parents[1]
    res = run_phase_5b_initial_models(root, root / "config/expanded_research.yaml")
    assert res["total_test_observations"] == 684
    assert res["test_positive_events"] == 19
    assert len(res["comparison"]) == 2
