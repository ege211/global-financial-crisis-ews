"""Unit tests for Phase 6 Crisis Mechanism and Temporal Concentration Analysis."""

from pathlib import Path

import pandas as pd
import pytest

from crisis_ews.evaluation.crisis_mechanisms import (
    load_phase6_evaluation_data,
    run_phase_6_crisis_mechanisms,
)


def test_event_profile_temporal_alignment() -> None:
    """Verify that predictor values are measured at t strictly before crisis onset at t+1."""
    root = Path(__file__).parents[1]
    _eval_df, crises_df = load_phase6_evaluation_data(root)

    assert len(crises_df) == 19
    for _, row in crises_df.iterrows():
        # Predictors measured at t (year), evaluation fold is t (test_year), onset is t+1
        assert row["year"] == row["test_year"]
        assert row["crisis_onset_year"] == row["year"] + 1
        assert row["training_end_year"] < row["test_year"]
        assert row["crisis_within_horizon"] == 1


def test_detected_vs_missed_classification() -> None:
    """Verify that detected and missed events are partitioned exactly by tau = 0.50."""
    root = Path(__file__).parents[1]
    _, crises_df = load_phase6_evaluation_data(root)

    detected = crises_df[crises_df["predicted_class"] == 1]
    missed = crises_df[crises_df["predicted_class"] == 0]

    assert len(detected) == 9
    assert len(missed) == 10
    assert len(detected) + len(missed) == 19

    # Check that detected probabilities are >= 0.50, missed are < 0.50
    assert (detected["predicted_probability"] >= 0.50).all()
    assert (missed["predicted_probability"] < 0.50).all()

    # Exact country lists
    expected_detected = {"DNK", "ESP", "FRA", "GRC", "HUN", "IRL", "ISL", "ITA", "PRT"}
    expected_missed = {"URY", "DOM", "GBR", "AUT", "BEL", "CHE", "DEU", "NLD", "SWE", "NGA"}
    assert set(detected["country_code"]) == expected_detected
    assert set(missed["country_code"]) == expected_missed


def test_correct_separation_of_2007_vs_non_2007() -> None:
    """Verify exact count and country separation between 2007 GFC fold and other folds."""
    root = Path(__file__).parents[1]
    _, crises_df = load_phase6_evaluation_data(root)

    c2007 = crises_df[crises_df["is_2007_fold"]]
    cnon = crises_df[~crises_df["is_2007_fold"]]

    assert len(c2007) == 15
    assert len(cnon) == 4

    assert set(cnon["country_code"]) == {"URY", "DOM", "GBR", "NGA"}
    assert sorted(cnon["test_year"].tolist()) == [2001, 2002, 2006, 2008]

    # In 2007, 9 detected and 6 missed
    assert (c2007["predicted_class"] == 1).sum() == 9
    assert (c2007["predicted_class"] == 0).sum() == 6

    # Outside 2007, 0 detected and 4 missed
    assert (cnon["predicted_class"] == 1).sum() == 0
    assert (cnon["predicted_class"] == 0).sum() == 4


def test_no_future_year_predictor_leakage() -> None:
    """Verify that no t+1 values or test-fold outcomes enter the predictor matrices."""
    root = Path(__file__).parents[1]
    _eval_df, crises_df = load_phase6_evaluation_data(root)
    panel = pd.read_csv(root / "data/processed/expanded_research/modeling_panel.csv")

    for _, row in crises_df.iterrows():
        iso = row["country_code"]
        t = row["year"]
        # Raw panel at year t must match merged feature values
        raw_row = panel[(panel["country_code"] == iso) & (panel["year"] == t)].iloc[0]
        assert raw_row["gdp_growth"] == pytest.approx(row["gdp_growth"], abs=1e-5)
        assert raw_row["inflation"] == pytest.approx(row["inflation"], abs=1e-5)
        assert raw_row["reserves_usd"] == pytest.approx(row["reserves_usd"], abs=1e-5)


def test_preservation_of_existing_phase5e_prediction_files() -> None:
    """Verify that existing out-of-sample prediction files and Phase 5E tables remain untouched."""
    root = Path(__file__).parents[1]
    ext_path = root / "results/model_outputs/extended_logistic_regression_predictions.csv"
    gen_path = root / "results/tables/phase5e_temporal_generalization.csv"

    assert ext_path.exists()
    assert gen_path.exists()

    ext_df = pd.read_csv(ext_path)
    assert len(ext_df) == 684
    assert ext_df["crisis_within_horizon"].sum() == 19

    gen_df = pd.read_csv(gen_path)
    assert len(gen_df) == 10
    pooled_row = gen_df[
        (gen_df["model"] == "extended_logistic_regression") & (gen_df["subset_key"] == "pooled_2000_2008")
    ].iloc[0]
    assert pooled_row["pr_auc"] == pytest.approx(0.0344, abs=1e-4)
    assert pooled_row["roc_auc"] == pytest.approx(0.5734, abs=1e-4)


def test_deterministic_output_generation() -> None:
    """Verify that running Phase 6 generates deterministic tables and figures with exact row counts."""
    root = Path(__file__).parents[1]
    summary = run_phase_6_crisis_mechanisms(root)

    assert summary["total_eval_observations"] == 684
    assert summary["total_crisis_events"] == 19
    assert summary["crisis_2007_count"] == 15
    assert summary["crisis_non_2007_count"] == 4
    assert summary["detected_count"] == 9
    assert summary["missed_count"] == 10

    # Verify generated table schemas
    tables_dir = root / "results/tables"
    profiles_df = pd.read_csv(tables_dir / "phase6_crisis_event_profiles.csv")
    assert len(profiles_df) == 19

    det_mis_df = pd.read_csv(tables_dir / "phase6_detected_vs_missed_comparison.csv")
    assert len(det_mis_df) == 21  # 3 subsets * 7 variables

    stats_df = pd.read_csv(tables_dir / "phase6_group_descriptive_statistics.csv")
    assert len(stats_df) == 21  # 7 variables * 3 groups

    prob_df = pd.read_csv(tables_dir / "phase6_probability_distributions.csv")
    assert len(prob_df) == 5  # 5 groups

    mech_df = pd.read_csv(tables_dir / "phase6_mechanism_categorization.csv")
    assert len(mech_df) == 19

    # Verify figures exist
    figs_dir = root / "results/figures"
    assert (figs_dir / "phase6_predictor_profile_comparison.svg").exists()
    assert (figs_dir / "phase6_predictor_profile_comparison.png").exists()
    assert (figs_dir / "phase6_probability_distribution.svg").exists()
    assert (figs_dir / "phase6_probability_distribution.png").exists()
    assert (figs_dir / "phase6_standardized_profiles.svg").exists()
    assert (figs_dir / "phase6_standardized_profiles.png").exists()
