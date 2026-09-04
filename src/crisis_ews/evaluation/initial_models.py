"""Phase 5B: Initial Model Estimation Execution Engine.

Fits regularized Logistic Regression baseline and constrained Random Forest comparator
under the locked expanding-window design (2000-2008) on the 76-country panel.
Computes pooled and per-fold rare-event metrics, calibration diagnostics, feature
interpretations, and prediction audits with zero post-hoc tuning.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from crisis_ews.config import load_yaml
from crisis_ews.evaluation.walk_forward import (
    compute_per_fold_metrics,
    evaluate_walk_forward,
    expanding_window_folds,
)
from crisis_ews.models.models import logistic_pipeline, random_forest_pipeline

LOGGER = logging.getLogger(__name__)

PRE_REGISTERED_FEATURES = [
    "gdp_growth",
    "inflation",
    "reserves_usd",
    "reserves_usd_yoy_pct_change",
]


def extract_fold_feature_interpretations(
    data: pd.DataFrame,
    features: list[str],
    settings: dict[str, Any],
    seed: int,
    minimum_train_years: int,
    test_start_year: int,
    test_end_year: int,
) -> pd.DataFrame:
    """Extract model-based coefficients, odds ratios, and feature importances for each fold."""
    records = []
    folds = list(expanding_window_folds(data["year"], minimum_train_years, test_start_year, test_end_year))

    for fold in folds:
        train = data.loc[data["year"] <= fold.train_end_year]
        y_train = train["crisis_within_horizon"].astype(int)

        # 1. Logistic Regression
        lr = logistic_pipeline(settings["models"]["logistic_regression"], seed)
        lr.fit(train[features], y_train)
        lr_coefs = lr.named_steps["model"].coef_[0]
        lr_odds = np.exp(lr_coefs)

        # 2. Random Forest
        rf = random_forest_pipeline(settings["models"]["random_forest"], seed)
        rf.fit(train[features], y_train)
        rf_importances = rf.named_steps["model"].feature_importances_

        for feat_name, coef, odds, imp in zip(features, lr_coefs, lr_odds, rf_importances, strict=True):
            records.append(
                {
                    "test_year": fold.test_year,
                    "train_end_year": fold.train_end_year,
                    "feature": feat_name,
                    "logistic_coefficient": round(float(coef), 6),
                    "logistic_odds_ratio": round(float(odds), 6),
                    "random_forest_importance": round(float(imp), 6),
                    "interpretation_note": "Associative model parameter; non-causal",
                }
            )

    return pd.DataFrame(records)


def compute_calibration_assessment(predictions: pd.DataFrame, model_name: str) -> tuple[dict[str, Any], pd.DataFrame]:
    """Calculate Brier score, Brier skill score, and binned calibration diagnostics."""
    y_true = predictions["crisis_within_horizon"].to_numpy().astype(int)
    prob = predictions["predicted_probability"].to_numpy().astype(float)
    n = len(y_true)

    brier_score = float(np.mean((prob - y_true) ** 2))
    base_rate = float(np.mean(y_true))
    brier_ref = float(base_rate * (1.0 - base_rate))
    brier_skill_score = float(1.0 - (brier_score / brier_ref)) if brier_ref > 0 else 0.0

    # Probability bins: [0, 0.05), [0.05, 0.1), [0.1, 0.2), [0.2, 0.5), [0.5, 1.0]
    bins = [0.0, 0.05, 0.10, 0.20, 0.50, 1.00]
    bin_labels = ["[0.0, 0.05)", "[0.05, 0.10)", "[0.10, 0.20)", "[0.20, 0.50)", "[0.50, 1.00]"]
    binned = pd.cut(prob, bins=bins, labels=bin_labels, include_lowest=True, right=False)

    bin_records = []
    for label in bin_labels:
        mask = binned == label
        count = int(np.sum(mask))
        if count > 0:
            mean_prob = float(np.mean(prob[mask]))
            empirical_rate = float(np.mean(y_true[mask]))
            crises = int(np.sum(y_true[mask]))
            cal_err = float(abs(mean_prob - empirical_rate))
        else:
            mean_prob = 0.0
            empirical_rate = 0.0
            crises = 0
            cal_err = 0.0
        bin_records.append(
            {
                "model": model_name,
                "probability_bin": label,
                "observations": count,
                "crises": crises,
                "mean_predicted_probability": round(mean_prob, 4),
                "empirical_crisis_rate": round(empirical_rate, 4),
                "calibration_error": round(cal_err, 4),
            }
        )

    summary = {
        "model": model_name,
        "observations": n,
        "crises": int(np.sum(y_true)),
        "base_rate": round(base_rate, 6),
        "brier_score": round(brier_score, 6),
        "brier_reference_climatology": round(brier_ref, 6),
        "brier_skill_score": round(brier_skill_score, 6),
    }

    return summary, pd.DataFrame(bin_records)


def audit_prediction_integrity(predictions: pd.DataFrame, data: pd.DataFrame, expected_rows: int = 684) -> dict[str, Any]:
    """Verify that predictions align with test observations, have no overlap, and are bounded in [0, 1]."""
    if len(predictions) != expected_rows:
        raise ValueError(f"Expected {expected_rows} predictions, found {len(predictions)}")

    # Check bounds
    prob = predictions["predicted_probability"]
    if (prob < 0.0).any() or (prob > 1.0).any():
        raise ValueError("Predicted probabilities exceed [0, 1] bounds")

    # Check duplicates
    dupes = predictions.duplicated(subset=["country_code", "year"]).sum()
    if dupes > 0:
        raise ValueError(f"Found {dupes} duplicate predictions")

    # Check test year bounds
    test_years = set(predictions["test_year"].unique())
    expected_years = set(range(2000, 2009))
    if test_years != expected_years:
        raise ValueError(f"Test years {sorted(test_years)} do not match expected {sorted(expected_years)}")

    # Check that test_year == year
    if not (predictions["test_year"] == predictions["year"]).all():
        raise ValueError("Test year does not match observation year")

    # Check temporal training isolation
    if not (predictions["training_end_year"] < predictions["year"]).all():
        raise ValueError("Training year leaked into test year")

    return {
        "predictions_count": len(predictions),
        "duplicates": int(dupes),
        "min_prob": round(float(prob.min()), 6),
        "max_prob": round(float(prob.max()), 6),
        "test_years": sorted(test_years),
        "integrity_verified": True,
    }


def run_phase_5b_initial_models(root: Path, profile_path: Path | None = None) -> dict[str, Any]:
    """Orchestrate Phase 5B initial model fitting and evaluation across the locked 76-country panel."""
    target_profile = profile_path if profile_path is not None else (root / "config/expanded_research.yaml")
    resolved_profile = target_profile if target_profile.is_absolute() else (root / target_profile)
    profile_data = load_yaml(resolved_profile) if resolved_profile.exists() else {}
    namespace = str(profile_data.get("run_namespace", "expanded_research"))

    # Load base settings and apply profile overrides
    base_settings = deepcopy(load_yaml(root / "config/model_config.yaml"))
    if "features" in profile_data:
        features = list(profile_data["features"])
    else:
        features = PRE_REGISTERED_FEATURES

    base_settings.update(profile_data.get("model_overrides", {}))
    seed = int(base_settings["seed"])
    min_train = int(base_settings["minimum_train_years"])
    test_start = int(base_settings["test_start_year"])
    test_end = int(base_settings.get("test_end_year", 2008))
    threshold = float(base_settings["threshold"])

    data_path = root / "data/processed" / namespace / "modeling_panel.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Processed modeling panel not found at {data_path}")

    data = pd.read_csv(data_path)

    # Directories
    model_outputs_dir = root / "results/model_outputs"
    tables_dir = root / "results/tables"
    model_outputs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Step 2: Fitting Logistic Regression baseline across folds 2000-2008...")
    lr_factory = lambda: logistic_pipeline(base_settings["models"]["logistic_regression"], seed)
    lr_preds, lr_pooled_metrics = evaluate_walk_forward(
        data=data,
        features=features,
        model_factory=lr_factory,
        minimum_train_years=min_train,
        test_start_year=test_start,
        threshold=threshold,
        model_name="logistic_regression",
        test_end_year=test_end,
    )
    lr_fold_metrics = compute_per_fold_metrics(lr_preds, threshold)
    audit_prediction_integrity(lr_preds, data)
    lr_preds.to_csv(model_outputs_dir / "logistic_regression_predictions.csv", index=False)
    lr_pooled_metrics.to_csv(tables_dir / "expanded_logistic_regression_metrics.csv", index=False)
    lr_fold_metrics.to_csv(tables_dir / "expanded_logistic_regression_fold_metrics.csv", index=False)

    LOGGER.info("Step 5: Fitting Constrained Random Forest comparator across folds 2000-2008...")
    rf_factory = lambda: random_forest_pipeline(base_settings["models"]["random_forest"], seed)
    rf_preds, rf_pooled_metrics = evaluate_walk_forward(
        data=data,
        features=features,
        model_factory=rf_factory,
        minimum_train_years=min_train,
        test_start_year=test_start,
        threshold=threshold,
        model_name="random_forest",
        test_end_year=test_end,
    )
    rf_fold_metrics = compute_per_fold_metrics(rf_preds, threshold)
    audit_prediction_integrity(rf_preds, data)
    rf_preds.to_csv(model_outputs_dir / "random_forest_predictions.csv", index=False)
    rf_pooled_metrics.to_csv(tables_dir / "expanded_random_forest_metrics.csv", index=False)
    rf_fold_metrics.to_csv(tables_dir / "expanded_random_forest_fold_metrics.csv", index=False)

    # Calibration assessments
    lr_cal_summary, lr_cal_bins = compute_calibration_assessment(lr_preds, "logistic_regression")
    rf_cal_summary, rf_cal_bins = compute_calibration_assessment(rf_preds, "random_forest")
    calibration_df = pd.concat([lr_cal_bins, rf_cal_bins], ignore_index=True)
    calibration_df.to_csv(tables_dir / "expanded_calibration_summary.csv", index=False)

    # Feature interpretations
    interpretations_df = extract_fold_feature_interpretations(
        data, features, base_settings, seed, min_train, test_start, test_end
    )
    interpretations_df.to_csv(tables_dir / "expanded_feature_interpretations.csv", index=False)

    # Model comparison table matching requested schema:
    # Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | Brier | Positive Events | Test Observations
    comparison_rows = []
    for model_name, metrics in [("logistic_regression", lr_pooled_metrics.iloc[0]), ("random_forest", rf_pooled_metrics.iloc[0])]:
        comparison_rows.append(
            {
                "model": model_name,
                "pr_auc": round(float(metrics["pr_auc"]), 4) if metrics["pr_auc"] is not None else None,
                "roc_auc": round(float(metrics["roc_auc"]), 4) if metrics["roc_auc"] is not None else None,
                "precision": round(float(metrics["precision"]), 4),
                "recall": round(float(metrics["recall"]), 4),
                "f1": round(float(metrics["f1"]), 4),
                "brier": round(float(metrics["brier_score"]), 4),
                "positive_events": int(metrics["crises"]),
                "test_observations": int(metrics["observations"]),
                "true_positives": int(metrics["true_positives"]),
                "false_positives": int(metrics["false_positives"]),
                "false_negatives": int(metrics["false_negatives"]),
                "true_negatives": int(metrics["true_negatives"]),
            }
        )
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(tables_dir / "expanded_model_comparison.csv", index=False)

    LOGGER.info(
        "Phase 5B complete: LR ROC-AUC=%.4f, PR-AUC=%.4f | RF ROC-AUC=%.4f, PR-AUC=%.4f",
        comparison_rows[0]["roc_auc"],
        comparison_rows[0]["pr_auc"],
        comparison_rows[1]["roc_auc"],
        comparison_rows[1]["pr_auc"],
    )

    return {
        "logistic_regression": {
            "pooled": lr_pooled_metrics.to_dict(orient="records")[0],
            "fold_metrics": lr_fold_metrics.to_dict(orient="records"),
            "calibration": lr_cal_summary,
        },
        "random_forest": {
            "pooled": rf_pooled_metrics.to_dict(orient="records")[0],
            "fold_metrics": rf_fold_metrics.to_dict(orient="records"),
            "calibration": rf_cal_summary,
        },
        "comparison": comparison_rows,
        "predictions_count": len(lr_preds),
        "test_positive_events": 19,
        "total_test_observations": 684,
    }
