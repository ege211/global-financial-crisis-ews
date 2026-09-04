"""Phase 5E: Out-of-Sample GFC Dependence and Temporal Generalization Analysis.

Performs a diagnostic temporal-generalization assessment of the Phase 5D Extended
Logistic Regression predictions relative to the Phase 5B Baseline Logistic Regression.

Evaluates:
  1. Pooled 2000-2008 (all 9 folds, 684 obs, 19 crises)
  2. Excluding 2007 (folds 2000-2006 and 2008, 608 obs, 4 crises)
  3. Pre-GFC period (folds 2000-2006, 532 obs, 3 crises)
  4. 2007 GFC fold (76 obs, 15 crises)
  5. 2008 post-GFC fold (76 obs, 1 crisis)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from crisis_ews.evaluation.metrics import classification_metrics

LOGGER = logging.getLogger(__name__)

SUBSET_DEFINITIONS: dict[str, dict[str, Any]] = {
    "pooled_2000_2008": {
        "years": list(range(2000, 2009)),
        "description": "Pooled Out-of-Sample (2000-2008)",
        "folds_count": 9,
        "expected_obs": 684,
        "expected_crises": 19,
    },
    "excluding_2007": {
        "years": [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2008],
        "description": "Excluding 2007 Test Fold (2000-2006, 2008)",
        "folds_count": 8,
        "expected_obs": 608,
        "expected_crises": 4,
    },
    "pre_gfc_2000_2006": {
        "years": list(range(2000, 2007)),
        "description": "Pre-GFC Historical Period (2000-2006)",
        "folds_count": 7,
        "expected_obs": 532,
        "expected_crises": 3,
    },
    "gfc_2007": {
        "years": [2007],
        "description": "2007 GFC Test Fold (Forecasting 2008)",
        "folds_count": 1,
        "expected_obs": 76,
        "expected_crises": 15,
    },
    "post_gfc_2008": {
        "years": [2008],
        "description": "2008 Test Fold (Forecasting 2009)",
        "folds_count": 1,
        "expected_obs": 76,
        "expected_crises": 1,
    },
}


def evaluate_prediction_subsets(
    predictions_df: pd.DataFrame,
    model_label: str,
    threshold: float = 0.50,
) -> list[dict[str, Any]]:
    """Compute metrics across defined temporal slices without altering predictions."""
    records = []
    for sub_key, config in SUBSET_DEFINITIONS.items():
        subset_df = predictions_df[predictions_df["test_year"].isin(config["years"])]
        y_true = subset_df["crisis_within_horizon"].to_numpy()
        probs = subset_df["predicted_probability"].to_numpy()

        metrics = classification_metrics(y_true, probs, threshold=threshold)
        metrics["subset_key"] = sub_key
        metrics["subset_description"] = config["description"]
        metrics["model"] = model_label
        metrics["folds_count"] = config["folds_count"]
        records.append(metrics)
    return records


def run_phase_5e_temporal_generalization(
    root: Path,
    threshold: float = 0.50,
) -> pd.DataFrame:
    """Run temporal generalization diagnostics on existing out-of-sample predictions."""
    outputs_dir = root / "results/model_outputs"
    tables_dir = root / "results/tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    base_lr_path = outputs_dir / "logistic_regression_predictions.csv"
    ext_lr_path = outputs_dir / "extended_logistic_regression_predictions.csv"

    if not base_lr_path.exists():
        raise FileNotFoundError(f"Baseline predictions not found: {base_lr_path}")
    if not ext_lr_path.exists():
        raise FileNotFoundError(f"Extended predictions not found: {ext_lr_path}")

    base_lr_preds = pd.read_csv(base_lr_path)
    ext_lr_preds = pd.read_csv(ext_lr_path)

    # Verification: check row counts
    if len(base_lr_preds) != 684 or len(ext_lr_preds) != 684:
        raise ValueError("Prediction tables must have exactly 684 rows.")

    # Evaluate Extended Logistic Regression
    ext_records = evaluate_prediction_subsets(
        ext_lr_preds,
        model_label="extended_logistic_regression",
        threshold=threshold,
    )

    # Evaluate Baseline Logistic Regression
    base_records = evaluate_prediction_subsets(
        base_lr_preds,
        model_label="baseline_logistic_regression",
        threshold=threshold,
    )

    combined_df = pd.DataFrame(ext_records + base_records)

    # Reorder columns for presentation
    column_order = [
        "subset_key",
        "subset_description",
        "model",
        "folds_count",
        "observations",
        "crises",
        "pr_auc",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "brier_score",
        "true_positives",
        "false_positives",
        "false_negatives",
        "true_negatives",
        "false_negative_rate",
    ]
    combined_df = combined_df[column_order]

    output_csv = tables_dir / "phase5e_temporal_generalization.csv"
    combined_df.to_csv(output_csv, index=False)
    LOGGER.info("Saved Phase 5E temporal generalization table to %s", output_csv)

    return combined_df
