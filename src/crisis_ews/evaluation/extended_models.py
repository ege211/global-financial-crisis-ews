"""Phase 5E: 6-Variable Extended Model Estimation Execution Engine.

Evaluates the locked 6-variable Extended Predictor Set:
  1. gdp_growth
  2. inflation
  3. reserves_usd
  4. reserves_usd_yoy_pct_change
  5. private_credit_pct_gdp
  6. current_account_pct_gdp

Uses the identical expanding-window evaluation protocol (T=2000..2008) on the
76-country panel, identical hyperparameters, and fold-isolated median imputation.
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from crisis_ews.config import load_yaml
from crisis_ews.data.world_bank import WorldBankCollector
from crisis_ews.evaluation.walk_forward import (
    compute_per_fold_metrics,
    evaluate_walk_forward,
    expanding_window_folds,
)
from crisis_ews.models.models import logistic_pipeline, random_forest_pipeline

LOGGER = logging.getLogger(__name__)

LOCKED_EXTENDED_FEATURES = [
    "gdp_growth",
    "inflation",
    "reserves_usd",
    "reserves_usd_yoy_pct_change",
    "private_credit_pct_gdp",
    "current_account_pct_gdp",
]


def build_extended_modeling_panel(root: Path) -> pd.DataFrame:
    """Merge locked extended additions (private credit & current account) onto 76-country panel."""
    panel_path = root / "data/processed/expanded_research/modeling_panel.csv"
    if not panel_path.exists():
        raise FileNotFoundError(f"Modeling panel not found at {panel_path}")
    panel = pd.read_csv(panel_path)
    countries = sorted(panel["country_code"].unique())
    raw_dir = root / "data/raw/expanded_research"

    collector = WorldBankCollector()

    # 1. Private credit / GDP
    pc_path = raw_dir / "world_bank_private_credit_pct_gdp.json"
    with pc_path.open(encoding="utf-8") as f:
        pc_payload = json.load(f)
    pc_df = collector.parse_indicator_payload(pc_payload, "FS.AST.PRVT.GD.ZS")
    pc_df = pc_df[pc_df["country_code"].isin(countries)][["country_code", "year", "value"]].rename(
        columns={"value": "private_credit_pct_gdp"}
    )

    # 2. Current account / GDP
    ca_path = raw_dir / "world_bank_current_account_pct_gdp.json"
    with ca_path.open(encoding="utf-8") as f:
        ca_payload = json.load(f)
    ca_df = collector.parse_indicator_payload(ca_payload, "BN.CAB.XOKA.GD.ZS")
    ca_df = ca_df[ca_df["country_code"].isin(countries)][["country_code", "year", "value"]].rename(
        columns={"value": "current_account_pct_gdp"}
    )

    merged = panel.merge(pc_df, on=["country_code", "year"], how="left")
    merged = merged.merge(ca_df, on=["country_code", "year"], how="left")
    return merged.sort_values(["country_code", "year"]).reset_index(drop=True)


def extract_extended_feature_interpretations(
    data: pd.DataFrame,
    features: list[str],
    settings: dict[str, Any],
    seed: int,
    minimum_train_years: int,
    test_start_year: int,
    test_end_year: int,
) -> pd.DataFrame:
    """Extract coefficients, odds ratios, and tree importances for all 6 features across folds."""
    records = []
    folds = list(
        expanding_window_folds(
            data["year"], minimum_train_years, test_start_year, test_end_year
        )
    )

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

        for feat_name, coef, odds, imp in zip(
            features, lr_coefs, lr_odds, rf_importances, strict=True
        ):
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


def compute_extended_calibration(
    predictions: pd.DataFrame, model_name: str
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Calculate Brier score, BSS, and binned calibration diagnostics for extended model."""
    y_true = predictions["crisis_within_horizon"].to_numpy().astype(int)
    prob = predictions["predicted_probability"].to_numpy().astype(float)

    brier_score = float(np.mean((prob - y_true) ** 2))
    base_rate = float(np.mean(y_true))
    brier_ref = float(base_rate * (1.0 - base_rate))
    brier_skill_score = (
        float(1.0 - (brier_score / brier_ref)) if brier_ref > 0 else 0.0
    )

    bins = [0.0, 0.05, 0.10, 0.20, 0.50, 1.00]
    bin_labels = [
        "[0.0, 0.05)",
        "[0.05, 0.10)",
        "[0.10, 0.20)",
        "[0.20, 0.50)",
        "[0.50, 1.00]",
    ]
    binned = pd.cut(
        prob, bins=bins, labels=bin_labels, include_lowest=True, right=False
    )

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
        "observations": len(y_true),
        "crises": int(np.sum(y_true)),
        "unconditional_rate": round(base_rate, 4),
        "brier_score": round(brier_score, 4),
        "brier_skill_score": round(brier_skill_score, 4),
    }
    return summary, pd.DataFrame(bin_records)


def run_phase_5e_extended_models(
    root: Path, profile_path: Path | None = None
) -> dict[str, Any]:
    """Run expanding-window evaluation on the 6-variable Extended Predictor Set."""
    target_profile = (
        profile_path
        if profile_path is not None
        else (root / "config/expanded_research.yaml")
    )
    resolved_profile = (
        target_profile
        if target_profile.is_absolute()
        else (root / target_profile)
    )
    profile_data = (
        load_yaml(resolved_profile) if resolved_profile.exists() else {}
    )

    base_settings = deepcopy(load_yaml(root / "config/model_config.yaml"))
    base_settings.update(profile_data.get("model_overrides", {}))

    seed = int(base_settings["seed"])
    threshold = float(base_settings.get("threshold", 0.50))
    min_train = int(base_settings["minimum_train_years"])
    test_start = int(base_settings["test_start_year"])
    test_end = int(base_settings.get("test_end_year", 2008))

    features = LOCKED_EXTENDED_FEATURES

    LOGGER.info("Constructing 6-variable extended modeling panel...")
    data = build_extended_modeling_panel(root)

    # 1. Extended Logistic Regression
    LOGGER.info("Evaluating Extended Logistic Regression across folds 2000-2008...")
    lr_factory = lambda: logistic_pipeline(base_settings["models"]["logistic_regression"], seed)
    lr_preds, lr_metrics = evaluate_walk_forward(
        data=data,
        features=features,
        model_factory=lr_factory,
        minimum_train_years=min_train,
        test_start_year=test_start,
        threshold=threshold,
        model_name="extended_logistic_regression",
        feature_version="extended-6var-v1",
        test_end_year=test_end,
    )
    lr_fold_metrics = compute_per_fold_metrics(lr_preds, threshold=threshold)

    # 2. Extended Random Forest
    LOGGER.info("Evaluating Extended Constrained Random Forest across folds 2000-2008...")
    rf_factory = lambda: random_forest_pipeline(base_settings["models"]["random_forest"], seed)
    rf_preds, rf_metrics = evaluate_walk_forward(
        data=data,
        features=features,
        model_factory=rf_factory,
        minimum_train_years=min_train,
        test_start_year=test_start,
        threshold=threshold,
        model_name="extended_random_forest",
        feature_version="extended-6var-v1",
        test_end_year=test_end,
    )
    rf_fold_metrics = compute_per_fold_metrics(rf_preds, threshold=threshold)

    # Save distinct predictions in results/model_outputs/ (without touching baseline predictions)
    outputs_dir = root / "results/model_outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    lr_preds.to_csv(
        outputs_dir / "extended_logistic_regression_predictions.csv", index=False
    )
    rf_preds.to_csv(
        outputs_dir / "extended_random_forest_predictions.csv", index=False
    )

    # 3. Extract Feature Interpretations
    interpretations_df = extract_extended_feature_interpretations(
        data, features, base_settings, seed, min_train, test_start, test_end
    )

    # 4. Calibration Assessment
    lr_cal_sum, _lr_binned = compute_extended_calibration(
        lr_preds, "extended_logistic_regression"
    )
    rf_cal_sum, _rf_binned = compute_extended_calibration(
        rf_preds, "extended_random_forest"
    )
    cal_summary_df = pd.DataFrame([lr_cal_sum, rf_cal_sum])

    # 5. Build Comprehensive Model Comparison (Baseline LR, Baseline RF, Extended LR, Extended RF)
    baseline_lr_preds = pd.read_csv(
        outputs_dir / "logistic_regression_predictions.csv"
    )
    baseline_rf_preds = pd.read_csv(outputs_dir / "random_forest_predictions.csv")

    def calc_row(name: str, spec: str, p_df: pd.DataFrame) -> dict[str, Any]:
        y = p_df["crisis_within_horizon"].to_numpy().astype(int)
        p = p_df["predicted_probability"].to_numpy().astype(float)
        c = (p >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, c, labels=[0, 1]).ravel()
        return {
            "model": name,
            "specification": spec,
            "pr_auc": round(float(average_precision_score(y, p)), 4),
            "roc_auc": round(float(roc_auc_score(y, p)), 4),
            "precision": round(float(precision_score(y, c, zero_division=0)), 4),
            "recall": round(float(recall_score(y, c, zero_division=0)), 4),
            "f1": round(float(f1_score(y, c, zero_division=0)), 4),
            "brier_score": round(float(brier_score_loss(y, p)), 4),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
            "observations": len(y),
            "positive_events": int(y.sum()),
        }

    comparison_rows = [
        calc_row("logistic_regression", "4-var flow baseline", baseline_lr_preds),
        calc_row("random_forest", "4-var flow baseline", baseline_rf_preds),
        calc_row(
            "extended_logistic_regression", "6-var (+credit, +ca)", lr_preds
        ),
        calc_row("extended_random_forest", "6-var (+credit, +ca)", rf_preds),
    ]
    comparison_df = pd.DataFrame(comparison_rows)

    # Save Tables
    tables_dir = root / "results/tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(
        tables_dir / "expanded_extended_model_comparison.csv", index=False
    )
    lr_metrics.to_csv(
        tables_dir / "expanded_extended_logistic_regression_metrics.csv",
        index=False,
    )
    rf_metrics.to_csv(
        tables_dir / "expanded_extended_random_forest_metrics.csv", index=False
    )
    lr_fold_metrics.to_csv(
        tables_dir / "expanded_extended_logistic_regression_fold_metrics.csv",
        index=False,
    )
    rf_fold_metrics.to_csv(
        tables_dir / "expanded_extended_random_forest_fold_metrics.csv",
        index=False,
    )
    interpretations_df.to_csv(
        tables_dir / "expanded_extended_feature_interpretations.csv", index=False
    )
    cal_summary_df.to_csv(
        tables_dir / "expanded_extended_calibration_summary.csv", index=False
    )

    LOGGER.info("Phase 5E Extended Models completed successfully.")
    return {
        "comparison": comparison_df.to_dict(orient="records"),
        "total_test_observations": len(lr_preds),
        "test_positive_events": int(lr_preds["crisis_within_horizon"].sum()),
    }
