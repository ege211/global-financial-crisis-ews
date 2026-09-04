"""Unit tests for Phase 7 Final Model Selection & Research Synthesis."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crisis_ews.evaluation.final_synthesis import (
    build_final_model_coefficients_table,
    build_final_model_comparison_table,
    build_final_robustness_summary_table,
    run_phase_7_final_synthesis,
)


def test_final_model_selection_logic() -> None:
    """Verify that Extended Logistic Regression is designated as Primary Research Model with regime qualification."""
    root = Path(__file__).parents[1]
    df = build_final_model_comparison_table(root)

    assert len(df) == 4
    ext_lr = df[df["model"] == "Extended Logistic Regression"].iloc[0]
    assert ext_lr["primary_selection"] == "Yes (Regime-Qualified)"
    assert ext_lr["selection_role"] == "Primary Research Model"

    # Verify other models are not selected as primary
    others = df[df["model"] != "Extended Logistic Regression"]
    assert (others["primary_selection"] == "No").all()


def test_preservation_of_historical_metrics() -> None:
    """Verify that all consolidated metrics in comparison table match locked Phase 5 results."""
    root = Path(__file__).parents[1]
    df = build_final_model_comparison_table(root)

    # Baseline LR
    b_lr = df[df["model"] == "Baseline Logistic Regression"].iloc[0]
    assert b_lr["pr_auc_pooled"] == pytest.approx(0.0278, abs=1e-4)
    assert b_lr["roc_auc_pooled"] == pytest.approx(0.5106, abs=1e-4)
    assert b_lr["recall_pooled"] == pytest.approx(0.2105, abs=1e-4)
    assert b_lr["precision_pooled"] == pytest.approx(0.0189, abs=1e-4)
    assert b_lr["brier_score_pooled"] == pytest.approx(0.2512, abs=1e-4)
    assert b_lr["true_positives_pooled"] == 4
    assert b_lr["false_positives_pooled"] == 208

    # Extended LR
    e_lr = df[df["model"] == "Extended Logistic Regression"].iloc[0]
    assert e_lr["pr_auc_pooled"] == pytest.approx(0.0344, abs=1e-4)
    assert e_lr["roc_auc_pooled"] == pytest.approx(0.5734, abs=1e-4)
    assert e_lr["recall_pooled"] == pytest.approx(0.4737, abs=1e-4)
    assert e_lr["precision_pooled"] == pytest.approx(0.0459, abs=1e-4)
    assert e_lr["brier_score_pooled"] == pytest.approx(0.2482, abs=1e-4)
    assert e_lr["true_positives_pooled"] == 9
    assert e_lr["false_positives_pooled"] == 187
    assert e_lr["recall_2007_gfc"] == pytest.approx(0.6000, abs=1e-4)
    assert e_lr["recall_non_2007"] == pytest.approx(0.0000, abs=1e-4)
    assert e_lr["roc_auc_non_2007"] == pytest.approx(0.2301, abs=1e-4)

    # Baseline RF
    b_rf = df[df["model"] == "Baseline Random Forest"].iloc[0]
    assert b_rf["pr_auc_pooled"] == pytest.approx(0.0256, abs=1e-4)
    assert b_rf["roc_auc_pooled"] == pytest.approx(0.4712, abs=1e-4)
    assert b_rf["brier_score_pooled"] == pytest.approx(0.1524, abs=1e-4)
    assert b_rf["recall_pooled"] == pytest.approx(0.2105, abs=1e-4)
    assert b_rf["precision_pooled"] == pytest.approx(0.0242, abs=1e-4)
    assert b_rf["true_positives_pooled"] == 4
    assert b_rf["false_positives_pooled"] == 161
    assert b_rf["recall_2007_gfc"] == pytest.approx(0.2000, abs=1e-4)
    assert b_rf["recall_non_2007"] == pytest.approx(0.2500, abs=1e-4)
    assert b_rf["roc_auc_2007_gfc"] == pytest.approx(0.3454, abs=1e-4)
    assert b_rf["roc_auc_non_2007"] == pytest.approx(0.5555, abs=1e-4)

    # Extended RF
    e_rf = df[df["model"] == "Extended Random Forest"].iloc[0]
    assert e_rf["pr_auc_pooled"] == pytest.approx(0.0274, abs=1e-4)
    assert e_rf["roc_auc_pooled"] == pytest.approx(0.4715, abs=1e-4)
    assert e_rf["brier_score_pooled"] == pytest.approx(0.1347, abs=1e-4)
    assert e_rf["recall_pooled"] == pytest.approx(0.1053, abs=1e-4)
    assert e_rf["precision_pooled"] == pytest.approx(0.0161, abs=1e-4)
    assert e_rf["true_positives_pooled"] == 2
    assert e_rf["false_positives_pooled"] == 122
    assert e_rf["recall_2007_gfc"] == pytest.approx(0.1333, abs=1e-4)
    assert e_rf["recall_non_2007"] == pytest.approx(0.0000, abs=1e-4)
    assert e_rf["roc_auc_2007_gfc"] == pytest.approx(0.3552, abs=1e-4)
    assert e_rf["roc_auc_non_2007"] == pytest.approx(0.5401, abs=1e-4)


def test_coefficient_table_integrity() -> None:
    """Verify coefficient values, odds ratios, and non-causal interpretations."""
    root = Path(__file__).parents[1]
    coef_df = build_final_model_coefficients_table(root)

    assert len(coef_df) == 6
    expected_features = {
        "private_credit_pct_gdp",
        "reserves_usd",
        "gdp_growth",
        "inflation",
        "current_account_pct_gdp",
        "reserves_usd_yoy_pct_change",
    }
    assert set(coef_df["feature"]) == expected_features

    for _, row in coef_df.iterrows():
        # OR must equal exp(beta)
        expected_or = float(np.exp(row["logistic_coefficient"]))
        assert row["odds_ratio"] == pytest.approx(expected_or, rel=1e-3)
        # Verify associative phrasing
        assert "associates" in row["economic_interpretation"].lower()
        assert "caused" not in row["economic_interpretation"].lower()


def test_robustness_summary_coverage() -> None:
    """Verify that all 6 required robustness dimensions are represented in the summary."""
    root = Path(__file__).parents[1]
    rob_df = build_final_robustness_summary_table(root)

    assert len(rob_df) == 6
    expected_dims = {
        "Winsorization Sensitivity",
        "Class Weight Sensitivity",
        "Threshold Sensitivity",
        "Expanding-Window Fold Stability",
        "Predictor-Set Expansion",
        "Out-of-Sample Regime Generalization",
    }
    assert set(rob_df["dimension"]) == expected_dims


def test_deterministic_output_generation() -> None:
    """Verify deterministic generation of Phase 7 tables and figures."""
    root = Path(__file__).parents[1]
    summary = run_phase_7_final_synthesis(root)

    assert summary["models_compared"] == 4
    assert summary["robustness_dimensions_summarized"] == 6
    assert summary["coefficients_documented"] == 6

    tables_dir = root / "results/tables"
    assert (tables_dir / "final_model_comparison.csv").exists()
    assert (tables_dir / "final_robustness_summary.csv").exists()
    assert (tables_dir / "final_model_coefficients.csv").exists()

    figs_dir = root / "results/figures"
    assert (figs_dir / "phase7_performance_synthesis.svg").exists()
    assert (figs_dir / "phase7_performance_synthesis.png").exists()


def test_preservation_of_phase5_and_phase6_artifacts() -> None:
    """Verify that running Phase 7 does not touch or overwrite previous phase artifacts."""
    root = Path(__file__).parents[1]
    outputs_dir = root / "results/model_outputs"
    tables_dir = root / "results/tables"

    # Prediction files
    base_preds = pd.read_csv(outputs_dir / "logistic_regression_predictions.csv")
    ext_preds = pd.read_csv(outputs_dir / "extended_logistic_regression_predictions.csv")
    assert len(base_preds) == 684
    assert len(ext_preds) == 684

    # Phase 6 tables
    p6_mech = pd.read_csv(tables_dir / "phase6_mechanism_categorization.csv")
    assert len(p6_mech) == 19
    assert (p6_mech["mechanism_category"] == "Unknown / insufficient evidence").all()
