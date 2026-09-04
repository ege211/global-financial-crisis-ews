"""Phase 5C: Robustness and Sensitivity Analysis Layer.

Implements four pre-specified, non-interfering sensitivity analyses:
A. Inflation Outlier Sensitivity (training-fold 1%/99% winsorization)
B. Class-Weight Sensitivity (None vs balanced)
C. Threshold Sensitivity (tau = 0.25, 0.50, 0.75 on locked predictions)
D. Fold-Level Stability across expanding test years 2000-2008
"""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from crisis_ews.config import load_yaml
from crisis_ews.evaluation.walk_forward import expanding_window_folds

LOGGER = logging.getLogger(__name__)

CORE_FEATURES = [
    "gdp_growth",
    "inflation",
    "reserves_usd",
    "reserves_usd_yoy_pct_change",
]


class TrainingFoldWinsorizer(BaseEstimator, TransformerMixin):
    """Winsorize specified column using quantiles learned strictly from training data."""

    def __init__(
        self,
        column_name: str = "inflation",
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
    ) -> None:
        self.column_name = column_name
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.lower_bound_: float | None = None
        self.upper_bound_: float | None = None

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any = None) -> TrainingFoldWinsorizer:
        if isinstance(X, pd.DataFrame):
            col_data = X[self.column_name].dropna().to_numpy()
        else:
            # Assume inflation is column index 1 matching CORE_FEATURES
            col_data = X[:, 1]
            col_data = col_data[~np.isnan(col_data)]
        self.lower_bound_ = float(np.percentile(col_data, self.lower_quantile * 100.0))
        self.upper_bound_ = float(np.percentile(col_data, self.upper_quantile * 100.0))
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        if self.lower_bound_ is None or self.upper_bound_ is None:
            raise RuntimeError("Transformer must be fitted before transform")
        if isinstance(X, pd.DataFrame):
            X_out = X.copy()
            X_out[self.column_name] = np.clip(
                X_out[self.column_name], self.lower_bound_, self.upper_bound_
            )
            return X_out
        X_out = np.copy(X)
        X_out[:, 1] = np.clip(X_out[:, 1], self.lower_bound_, self.upper_bound_)
        return X_out


def evaluate_custom_walk_forward(
    data: pd.DataFrame,
    features: list[str],
    pipeline_builder: Any,
    minimum_train_years: int = 10,
    test_start_year: int = 2000,
    test_end_year: int = 2008,
    threshold: float = 0.50,
    model_name: str = "custom_model",
) -> pd.DataFrame:
    """Run walk forward with a custom pipeline builder callable."""
    predictions: list[pd.DataFrame] = []
    folds = list(
        expanding_window_folds(
            data["year"], minimum_train_years, test_start_year, test_end_year
        )
    )

    for fold in folds:
        train = data.loc[data["year"] <= fold.train_end_year]
        test = data.loc[data["year"] == fold.test_year]
        y_train = train["crisis_within_horizon"].astype(int)

        if y_train.nunique() < 2:
            continue

        model = pipeline_builder()
        model.fit(train[features], y_train)
        probability = model.predict_proba(test[features])[:, 1]

        output = test[["country_code", "year", "crisis_within_horizon"]].copy()
        output["test_year"] = fold.test_year
        output["y_true"] = output["crisis_within_horizon"]
        output["predicted_probability"] = probability
        output["predicted_class"] = (probability >= threshold).astype(int)
        output["model_name"] = model_name
        output["training_start_year"] = int(train["year"].min())
        output["training_end_year"] = fold.train_end_year
        predictions.append(output)

    return pd.concat(predictions, ignore_index=True)


def calculate_metrics_dict(
    y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.50
) -> dict[str, Any]:
    """Calculate rare-event metrics, marking undefined metrics as None."""
    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    one_class = np.unique(y_true).size < 2

    roc_auc = None if one_class else float(roc_auc_score(y_true, prob))
    pr_auc = None if not y_true.any() else float(average_precision_score(y_true, prob))

    prec = float(precision_score(y_true, pred, zero_division=0))
    rec = float(recall_score(y_true, pred, zero_division=0))
    f1 = float(f1_score(y_true, pred, zero_division=0))
    brier = float(brier_score_loss(y_true, prob))

    return {
        "observations": len(y_true),
        "positive_events": int(y_true.sum()),
        "negative_observations": int(len(y_true) - y_true.sum()),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "brier_score": brier,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "false_positive_rate": (
            round(float(fp / (fp + tn)), 4) if (fp + tn) > 0 else 0.0
        ),
        "false_negative_rate": (
            round(float(fn / (tp + fn)), 4) if (tp + fn) > 0 else 0.0
        ),
    }


def run_inflation_sensitivity(
    data: pd.DataFrame,
    features: list[str],
    settings: dict[str, Any],
    seed: int,
    output_dir: Path,
) -> pd.DataFrame:
    """Analysis A: Compare baseline vs training-fold 1%/99% winsorized inflation."""
    min_train = int(settings["minimum_train_years"])
    test_start = int(settings["test_start_year"])
    test_end = int(settings.get("test_end_year", 2008))
    threshold = float(settings["threshold"])

    # Builders with fold-isolated winsorization
    def lr_winsorized_builder() -> Pipeline:
        return Pipeline(
            [
                ("winsorizer", TrainingFoldWinsorizer(column_name="inflation")),
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=float(settings["models"]["logistic_regression"].get("C", 1.0)),
                        class_weight=settings["models"]["logistic_regression"].get(
                            "class_weight", "balanced"
                        ),
                        max_iter=int(
                            settings["models"]["logistic_regression"].get("max_iter", 2000)
                        ),
                        random_state=seed,
                        solver="lbfgs",
                    ),
                ),
            ]
        )

    def rf_winsorized_builder() -> Pipeline:
        return Pipeline(
            [
                ("winsorizer", TrainingFoldWinsorizer(column_name="inflation")),
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=int(
                            settings["models"]["random_forest"].get("n_estimators", 400)
                        ),
                        max_depth=int(
                            settings["models"]["random_forest"].get("max_depth", 4)
                        ),
                        min_samples_leaf=int(
                            settings["models"]["random_forest"].get("min_samples_leaf", 5)
                        ),
                        class_weight=settings["models"]["random_forest"].get(
                            "class_weight", "balanced"
                        ),
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    lr_win_preds = evaluate_custom_walk_forward(
        data,
        features,
        lr_winsorized_builder,
        min_train,
        test_start,
        test_end,
        threshold,
        "logistic_regression_inflation_winsorized",
    )
    rf_win_preds = evaluate_custom_walk_forward(
        data,
        features,
        rf_winsorized_builder,
        min_train,
        test_start,
        test_end,
        threshold,
        "random_forest_inflation_winsorized",
    )

    # Save sensitivity predictions to avoid overwriting baseline
    (output_dir / "model_outputs").mkdir(parents=True, exist_ok=True)
    lr_win_preds.to_csv(
        output_dir
        / "model_outputs"
        / "logistic_regression_inflation_winsorized_predictions.csv",
        index=False,
    )
    rf_win_preds.to_csv(
        output_dir
        / "model_outputs"
        / "random_forest_inflation_winsorized_predictions.csv",
        index=False,
    )

    # Load baseline predictions to ensure exact comparison
    lr_base_preds = pd.read_csv(
        output_dir / "model_outputs/logistic_regression_predictions.csv"
    )
    rf_base_preds = pd.read_csv(
        output_dir / "model_outputs/random_forest_predictions.csv"
    )

    models_specs = [
        ("logistic_regression", "baseline", lr_base_preds),
        ("logistic_regression", "inflation_winsorized_1_99", lr_win_preds),
        ("random_forest", "baseline", rf_base_preds),
        ("random_forest", "inflation_winsorized_1_99", rf_win_preds),
    ]

    records = []
    for model_name, spec, preds in models_specs:
        m = calculate_metrics_dict(
            preds["y_true"].to_numpy(),
            preds["predicted_probability"].to_numpy(),
            threshold,
        )
        records.append(
            {
                "model": model_name,
                "specification": spec,
                "pr_auc": round(m["pr_auc"], 4) if m["pr_auc"] is not None else None,
                "roc_auc": round(m["roc_auc"], 4) if m["roc_auc"] is not None else None,
                "precision": round(m["precision"], 4),
                "recall": round(m["recall"], 4),
                "f1": round(m["f1"], 4),
                "brier_score": round(m["brier_score"], 4),
                "true_positives": m["true_positives"],
                "false_positives": m["false_positives"],
                "false_negatives": m["false_negatives"],
                "true_negatives": m["true_negatives"],
            }
        )

    df_out = pd.DataFrame(records)
    df_out.to_csv(
        output_dir / "tables/expanded_robustness_inflation.csv", index=False
    )
    return df_out


def run_class_weight_sensitivity(
    data: pd.DataFrame,
    features: list[str],
    settings: dict[str, Any],
    seed: int,
    output_dir: Path,
) -> pd.DataFrame:
    """Analysis B: Compare class_weight='balanced' vs class_weight=None."""
    min_train = int(settings["minimum_train_years"])
    test_start = int(settings["test_start_year"])
    test_end = int(settings.get("test_end_year", 2008))
    threshold = float(settings["threshold"])

    # Builders with unweighted (class_weight=None)
    def lr_unweighted_builder() -> Pipeline:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=float(settings["models"]["logistic_regression"].get("C", 1.0)),
                        class_weight=None,
                        max_iter=int(
                            settings["models"]["logistic_regression"].get("max_iter", 2000)
                        ),
                        random_state=seed,
                        solver="lbfgs",
                    ),
                ),
            ]
        )

    def rf_unweighted_builder() -> Pipeline:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=int(
                            settings["models"]["random_forest"].get("n_estimators", 400)
                        ),
                        max_depth=int(
                            settings["models"]["random_forest"].get("max_depth", 4)
                        ),
                        min_samples_leaf=int(
                            settings["models"]["random_forest"].get("min_samples_leaf", 5)
                        ),
                        class_weight=None,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    lr_unw_preds = evaluate_custom_walk_forward(
        data,
        features,
        lr_unweighted_builder,
        min_train,
        test_start,
        test_end,
        threshold,
        "logistic_regression_unweighted",
    )
    rf_unw_preds = evaluate_custom_walk_forward(
        data,
        features,
        rf_unweighted_builder,
        min_train,
        test_start,
        test_end,
        threshold,
        "random_forest_unweighted",
    )

    lr_unw_preds.to_csv(
        output_dir
        / "model_outputs"
        / "logistic_regression_unweighted_predictions.csv",
        index=False,
    )
    rf_unw_preds.to_csv(
        output_dir
        / "model_outputs"
        / "random_forest_unweighted_predictions.csv",
        index=False,
    )

    # Baseline predictions
    lr_base_preds = pd.read_csv(
        output_dir / "model_outputs/logistic_regression_predictions.csv"
    )
    rf_base_preds = pd.read_csv(
        output_dir / "model_outputs/random_forest_predictions.csv"
    )

    specs = [
        ("logistic_regression", "balanced", lr_base_preds),
        ("logistic_regression", "none (unweighted)", lr_unw_preds),
        ("random_forest", "balanced", rf_base_preds),
        ("random_forest", "none (unweighted)", rf_unw_preds),
    ]

    records = []
    for model_name, class_weight, preds in specs:
        m = calculate_metrics_dict(
            preds["y_true"].to_numpy(),
            preds["predicted_probability"].to_numpy(),
            threshold,
        )
        records.append(
            {
                "model": model_name,
                "class_weight": class_weight,
                "pr_auc": round(m["pr_auc"], 4) if m["pr_auc"] is not None else None,
                "roc_auc": round(m["roc_auc"], 4) if m["roc_auc"] is not None else None,
                "precision": round(m["precision"], 4),
                "recall": round(m["recall"], 4),
                "f1": round(m["f1"], 4),
                "brier_score": round(m["brier_score"], 4),
                "true_positives": m["true_positives"],
                "false_positives": m["false_positives"],
                "false_negatives": m["false_negatives"],
                "true_negatives": m["true_negatives"],
            }
        )

    df_out = pd.DataFrame(records)
    df_out.to_csv(
        output_dir / "tables/expanded_robustness_class_weight.csv", index=False
    )
    return df_out


def run_threshold_sensitivity(output_dir: Path) -> pd.DataFrame:
    """Analysis C: Evaluate tau in [0.25, 0.50, 0.75] on existing Phase 5B predictions."""
    lr_preds = pd.read_csv(
        output_dir / "model_outputs/logistic_regression_predictions.csv"
    )
    rf_preds = pd.read_csv(
        output_dir / "model_outputs/random_forest_predictions.csv"
    )

    thresholds = [0.25, 0.50, 0.75]
    records = []

    for model_name, preds in [
        ("logistic_regression", lr_preds),
        ("random_forest", rf_preds),
    ]:
        for tau in thresholds:
            m = calculate_metrics_dict(
                preds["y_true"].to_numpy(),
                preds["predicted_probability"].to_numpy(),
                threshold=tau,
            )
            records.append(
                {
                    "model": model_name,
                    "threshold": tau,
                    "precision": round(m["precision"], 4),
                    "recall": round(m["recall"], 4),
                    "f1": round(m["f1"], 4),
                    "brier_score": round(m["brier_score"], 4),
                    "true_positives": m["true_positives"],
                    "false_positives": m["false_positives"],
                    "false_negatives": m["false_negatives"],
                    "true_negatives": m["true_negatives"],
                    "false_positive_rate": m["false_positive_rate"],
                    "false_negative_rate": m["false_negative_rate"],
                }
            )

    df_out = pd.DataFrame(records)
    df_out.to_csv(
        output_dir / "tables/expanded_threshold_sensitivity.csv", index=False
    )
    return df_out


def run_fold_stability_analysis(output_dir: Path) -> pd.DataFrame:
    """Analysis D: Report per-fold performance across all test years 2000-2008."""
    lr_preds = pd.read_csv(
        output_dir / "model_outputs/logistic_regression_predictions.csv"
    )
    rf_preds = pd.read_csv(
        output_dir / "model_outputs/random_forest_predictions.csv"
    )

    records = []
    for model_name, preds in [
        ("logistic_regression", lr_preds),
        ("random_forest", rf_preds),
    ]:
        for test_year, group in preds.groupby("test_year"):
            m = calculate_metrics_dict(
                group["y_true"].to_numpy(),
                group["predicted_probability"].to_numpy(),
                threshold=0.50,
            )
            roc_str = (
                f"{m['roc_auc']:.4f}"
                if m["roc_auc"] is not None
                else "Not estimable"
            )
            pr_str = (
                f"{m['pr_auc']:.4f}"
                if m["pr_auc"] is not None
                else "Not estimable"
            )

            records.append(
                {
                    "model": model_name,
                    "test_year": int(test_year),
                    "test_observations": m["observations"],
                    "positive_events": m["positive_events"],
                    "roc_auc": roc_str,
                    "pr_auc": pr_str,
                    "precision": round(m["precision"], 4),
                    "recall": round(m["recall"], 4),
                    "f1": round(m["f1"], 4),
                    "brier_score": round(m["brier_score"], 4),
                    "true_positives": m["true_positives"],
                    "false_positives": m["false_positives"],
                    "false_negatives": m["false_negatives"],
                    "true_negatives": m["true_negatives"],
                    "note": (
                        "High crisis concentration (15 of 19 events)"
                        if test_year == 2007
                        else (
                            "Single-class test fold"
                            if m["positive_events"] == 0
                            else "Normal event incidence"
                        )
                    ),
                }
            )

    df_out = pd.DataFrame(records)
    df_out.to_csv(
        output_dir / "tables/expanded_fold_stability.csv", index=False
    )
    return df_out


def run_phase_5c_robustness(
    root: Path, profile_path: Path | None = None
) -> dict[str, Any]:
    """Execute all Phase 5C robustness and sensitivity analyses."""
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
    namespace = str(profile_data.get("run_namespace", "expanded_research"))

    base_settings = deepcopy(load_yaml(root / "config/model_config.yaml"))
    base_settings.update(profile_data.get("model_overrides", {}))
    seed = int(base_settings["seed"])
    features = list(profile_data.get("features", CORE_FEATURES))

    data_path = root / "data/processed" / namespace / "modeling_panel.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Modeling panel not found at {data_path}")
    data = pd.read_csv(data_path)

    output_dir = root / "results"

    LOGGER.info("Executing Phase 5C Analysis A: Inflation Outlier Sensitivity...")
    inflation_df = run_inflation_sensitivity(
        data, features, base_settings, seed, output_dir
    )

    LOGGER.info("Executing Phase 5C Analysis B: Class-Weight Sensitivity...")
    class_weight_df = run_class_weight_sensitivity(
        data, features, base_settings, seed, output_dir
    )

    LOGGER.info("Executing Phase 5C Analysis C: Threshold Sensitivity...")
    threshold_df = run_threshold_sensitivity(output_dir)

    LOGGER.info("Executing Phase 5C Analysis D: Fold-Level Stability...")
    stability_df = run_fold_stability_analysis(output_dir)

    return {
        "inflation_sensitivity": inflation_df.to_dict(orient="records"),
        "class_weight_sensitivity": class_weight_df.to_dict(orient="records"),
        "threshold_sensitivity": threshold_df.to_dict(orient="records"),
        "fold_stability": stability_df.to_dict(orient="records"),
    }
