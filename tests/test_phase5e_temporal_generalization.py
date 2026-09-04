"""Unit tests for Phase 5E Temporal Generalization and GFC Dependence Analysis."""

from pathlib import Path

import pandas as pd
import pytest

from crisis_ews.evaluation.temporal_generalization import (
    SUBSET_DEFINITIONS,
    evaluate_prediction_subsets,
    run_phase_5e_temporal_generalization,
)


def test_subset_definitions_and_filtering() -> None:
    """Verify that subset definitions filter folds correctly, especially excluding 2007."""
    assert 2007 not in SUBSET_DEFINITIONS["excluding_2007"]["years"]
    assert len(SUBSET_DEFINITIONS["excluding_2007"]["years"]) == 8
    assert SUBSET_DEFINITIONS["excluding_2007"]["folds_count"] == 8

    assert SUBSET_DEFINITIONS["pre_gfc_2000_2006"]["years"] == list(range(2000, 2007))
    assert 2007 not in SUBSET_DEFINITIONS["pre_gfc_2000_2006"]["years"]
    assert 2008 not in SUBSET_DEFINITIONS["pre_gfc_2000_2006"]["years"]

    assert SUBSET_DEFINITIONS["gfc_2007"]["years"] == [2007]
    assert SUBSET_DEFINITIONS["post_gfc_2008"]["years"] == [2008]
    assert SUBSET_DEFINITIONS["pooled_2000_2008"]["years"] == list(range(2000, 2009))


def test_correct_observation_and_positive_event_counts() -> None:
    """Verify that each subset produces exact expected observation and crisis event counts."""
    root = Path(__file__).parents[1]
    ext_preds = pd.read_csv(root / "results/model_outputs/extended_logistic_regression_predictions.csv")

    for cfg in SUBSET_DEFINITIONS.values():
        subset = ext_preds[ext_preds["test_year"].isin(cfg["years"])]
        assert len(subset) == cfg["expected_obs"]
        assert subset["crisis_within_horizon"].sum() == cfg["expected_crises"]


def test_no_prediction_modification() -> None:
    """Verify that temporal generalization analysis does not alter prediction values."""
    root = Path(__file__).parents[1]
    ext_path = root / "results/model_outputs/extended_logistic_regression_predictions.csv"
    base_path = root / "results/model_outputs/logistic_regression_predictions.csv"

    ext_before = pd.read_csv(ext_path)
    base_before = pd.read_csv(base_path)

    df_gen = run_phase_5e_temporal_generalization(root)
    assert len(df_gen) == 10  # 5 subsets * 2 models

    ext_after = pd.read_csv(ext_path)
    base_after = pd.read_csv(base_path)

    pd.testing.assert_frame_equal(ext_before, ext_after)
    pd.testing.assert_frame_equal(base_before, base_after)


def test_metric_handling_and_values() -> None:
    """Verify metric calculations, bounds, and edge-case behavior across subsets."""
    root = Path(__file__).parents[1]
    ext_preds = pd.read_csv(root / "results/model_outputs/extended_logistic_regression_predictions.csv")

    records = evaluate_prediction_subsets(ext_preds, "extended_logistic_regression")
    rec_dict = {r["subset_key"]: r for r in records}

    # Pooled metrics match Phase 5D exactly
    pooled = rec_dict["pooled_2000_2008"]
    assert pooled["observations"] == 684
    assert pooled["crises"] == 19
    assert pooled["pr_auc"] == pytest.approx(0.0344, abs=1e-4)
    assert pooled["roc_auc"] == pytest.approx(0.5734, abs=1e-4)
    assert pooled["true_positives"] == 9
    assert pooled["false_positives"] == 187

    # Excluding 2007 has exactly 0 TP and 4 FN
    excl = rec_dict["excluding_2007"]
    assert excl["observations"] == 608
    assert excl["crises"] == 4
    assert excl["true_positives"] == 0
    assert excl["false_positives"] == 167
    assert excl["false_negatives"] == 4
    assert excl["recall"] == 0.0
    assert excl["precision"] == 0.0
    assert excl["roc_auc"] < 0.50  # inverted ranking outside GFC


def test_baseline_artifact_preservation() -> None:
    """Verify that Phase 5B baseline files and comparisons remain untouched."""
    root = Path(__file__).parents[1]
    table_path = root / "results/tables/phase5e_temporal_generalization.csv"
    assert table_path.exists()

    df = pd.read_csv(table_path)
    base_pooled = df[
        (df["model"] == "baseline_logistic_regression")
        & (df["subset_key"] == "pooled_2000_2008")
    ].iloc[0]

    assert base_pooled["pr_auc"] == pytest.approx(0.0278, abs=1e-4)
    assert base_pooled["roc_auc"] == pytest.approx(0.5106, abs=1e-4)
    assert base_pooled["true_positives"] == 4
    assert base_pooled["false_positives"] == 208
    assert base_pooled["false_negatives"] == 15
    assert base_pooled["true_negatives"] == 457
