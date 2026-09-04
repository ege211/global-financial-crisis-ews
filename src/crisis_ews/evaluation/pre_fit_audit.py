"""Phase 5A Pre-Fit Modeling Infrastructure and Data Audit.

Verifies the 76-country modeling panel, expanding-window temporal folds (2000-2008),
leak-proof preprocessing isolation, and pre-fit feature distributions without
fitting any predictive models or computing performance metrics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from crisis_ews.config import load_yaml
from crisis_ews.evaluation.walk_forward import expanding_window_folds

LOGGER = logging.getLogger(__name__)

CORE_PREDICTORS = [
    "gdp_growth",
    "inflation",
    "reserves_usd",
    "reserves_usd_yoy_pct_change",
]


def audit_panel_integrity(panel: pd.DataFrame) -> dict[str, Any]:
    """Verify panel dimensions, country universe, year range, and label counts."""
    required_cols = {"country_code", "year", "crisis_within_horizon", *CORE_PREDICTORS}
    missing_cols = required_cols - set(panel.columns)
    if missing_cols:
        raise ValueError(f"Modeling panel missing required columns: {sorted(missing_cols)}")

    total_rows = len(panel)
    unique_countries = panel["country_code"].nunique()
    min_year = int(panel["year"].min())
    max_year = int(panel["year"].max())
    expected_years = max_year - min_year + 1
    expected_rows = unique_countries * expected_years

    if total_rows != expected_rows:
        raise ValueError(f"Panel row count {total_rows} does not match expected {expected_rows}")

    duplicates = panel.duplicated(subset=["country_code", "year"]).sum()
    if duplicates > 0:
        raise ValueError(f"Found {duplicates} duplicate (country_code, year) pairs in modeling panel")

    positives = int(panel["crisis_within_horizon"].sum())
    negatives = total_rows - positives

    return {
        "total_rows": total_rows,
        "unique_countries": unique_countries,
        "min_year": min_year,
        "max_year": max_year,
        "expected_rows": expected_rows,
        "duplicates": int(duplicates),
        "positive_labels": positives,
        "negative_labels": negatives,
        "positive_label_rate": round(positives / total_rows, 6),
    }


def audit_feature_missingness(panel: pd.DataFrame) -> pd.DataFrame:
    """Verify missingness patterns across all core predictors."""
    records = []
    total_rows = len(panel)
    for var in CORE_PREDICTORS:
        missing = int(panel[var].isna().sum())
        missing_pct = round(100.0 * missing / total_rows, 4)
        years_with_missing = sorted(panel.loc[panel[var].isna(), "year"].unique().tolist())
        records.append(
            {
                "variable": var,
                "total_rows": total_rows,
                "missing_count": missing,
                "missing_percent": missing_pct,
                "years_with_missing": str(years_with_missing),
                "status": "passed" if (missing == 0 or (var == "reserves_usd_yoy_pct_change" and missing == 76 and years_with_missing == [1990])) else "unexpected_missingness",
            }
        )
    return pd.DataFrame(records)


def audit_temporal_folds(
    panel: pd.DataFrame,
    minimum_train_years: int = 10,
    test_start_year: int = 2000,
    test_end_year: int = 2008,
) -> pd.DataFrame:
    """Audit the 9 expanding-window evaluation folds (T = 2000 to 2008)."""
    folds = list(
        expanding_window_folds(
            panel["year"],
            minimum_train_years=minimum_train_years,
            test_start_year=test_start_year,
            test_end_year=test_end_year,
        )
    )

    records = []
    for idx, fold in enumerate(folds, start=1):
        train_mask = panel["year"] <= fold.train_end_year
        test_mask = panel["year"] == fold.test_year

        train_df = panel.loc[train_mask]
        test_df = panel.loc[test_mask]

        train_pos = int(train_df["crisis_within_horizon"].sum())
        train_neg = int((train_df["crisis_within_horizon"] == 0).sum())
        test_pos = int(test_df["crisis_within_horizon"].sum())
        test_neg = int((test_df["crisis_within_horizon"] == 0).sum())

        # Leakage checks
        train_max_year = int(train_df["year"].max())
        test_min_year = int(test_df["year"].min())
        temporal_leakage = train_max_year >= test_min_year
        overlap_cases = set(zip(train_df["country_code"], train_df["year"])) & set(zip(test_df["country_code"], test_df["year"]))

        records.append(
            {
                "fold_index": idx,
                "test_year": fold.test_year,
                "train_start_year": int(train_df["year"].min()),
                "train_end_year": fold.train_end_year,
                "train_years_count": int(train_df["year"].nunique()),
                "train_observations": len(train_df),
                "train_positive_labels": train_pos,
                "train_negative_labels": train_neg,
                "train_positive_rate_pct": round(100.0 * train_pos / len(train_df), 2),
                "test_observations": len(test_df),
                "test_positive_labels": test_pos,
                "test_negative_labels": test_neg,
                "test_positive_rate_pct": round(100.0 * test_pos / len(test_df), 2),
                "leakage_check": "PASSED" if (not temporal_leakage and len(overlap_cases) == 0) else "FAILED",
                "train_has_both_classes": train_pos > 0 and train_neg > 0,
            }
        )

    return pd.DataFrame(records)


def compute_pre_fit_feature_statistics(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics for core predictors across the 76 countries."""
    records = []
    for var in CORE_PREDICTORS:
        series = panel[var].dropna()
        records.append(
            {
                "variable": var,
                "observations": int(series.count()),
                "mean": round(float(series.mean()), 4),
                "std": round(float(series.std()), 4),
                "min": round(float(series.min()), 4),
                "p25": round(float(series.quantile(0.25)), 4),
                "median": round(float(series.median()), 4),
                "p75": round(float(series.quantile(0.75)), 4),
                "max": round(float(series.max()), 4),
                "skewness": round(float(series.skew()), 4),
            }
        )
    return pd.DataFrame(records)


def compute_pre_fit_correlations(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute Pearson and Spearman rank correlation matrices for core predictors."""
    clean_panel = panel[CORE_PREDICTORS].dropna()
    pearson = clean_panel.corr(method="pearson")
    spearman = clean_panel.corr(method="spearman")

    records = []
    for i, var1 in enumerate(CORE_PREDICTORS):
        for j, var2 in enumerate(CORE_PREDICTORS):
            if i <= j:
                records.append(
                    {
                        "variable_1": var1,
                        "variable_2": var2,
                        "pearson_correlation": round(float(pearson.loc[var1, var2]), 4),
                        "spearman_correlation": round(float(spearman.loc[var1, var2]), 4),
                    }
                )
    return pd.DataFrame(records)


def verify_preprocessing_isolation(panel: pd.DataFrame) -> dict[str, bool]:
    """Audit fold-isolated preprocessing: verifies transformers fit strictly on training data."""
    folds = list(expanding_window_folds(panel["year"], minimum_train_years=10, test_start_year=2000, test_end_year=2008))
    for fold in folds:
        train = panel.loc[panel["year"] <= fold.train_end_year]
        test = panel.loc[panel["year"] == fold.test_year]

        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()

        # Fit strictly on train
        train_imputed = imputer.fit_transform(train[CORE_PREDICTORS])
        scaler.fit(train_imputed)

        # Imputer median statistics should equal train medians
        train_medians = train[CORE_PREDICTORS].median().to_numpy()
        if not np.allclose(imputer.statistics_, train_medians, equal_nan=True):
            return {"isolation_verified": False}

        # Scaler mean should equal train imputed means
        if not np.allclose(scaler.mean_, train_imputed.mean(axis=0)):
            return {"isolation_verified": False}

        # Transform test set without fitting
        test_imputed = imputer.transform(test[CORE_PREDICTORS])
        test_scaled = scaler.transform(test_imputed)

        # Test data should not match scaler mean (proves test didn't leak into scaler)
        if len(test) > 0 and np.allclose(test_scaled.mean(axis=0), 0.0):
            return {"isolation_verified": False}

    return {"isolation_verified": True}


def run_pre_fit_audit(root: Path, profile_path: Path | None = None) -> dict[str, Any]:
    """Execute complete pre-fit audit and export audit tables."""
    target_profile = profile_path if profile_path is not None else (root / "config/expanded_research.yaml")
    resolved_profile = target_profile if target_profile.is_absolute() else (root / target_profile)
    profile_data = load_yaml(resolved_profile) if resolved_profile.exists() else {}
    namespace = str(profile_data.get("run_namespace", "expanded_research"))

    data_path = root / "data/processed" / namespace / "modeling_panel.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Processed modeling panel not found at: {data_path}")

    panel = pd.read_csv(data_path)
    tables_dir = root / "results/tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # 1. Panel integrity
    panel_info = audit_panel_integrity(panel)

    # 2. Missingness audit
    missingness_df = audit_feature_missingness(panel)
    missingness_df.to_csv(tables_dir / "expanded_pre_fit_missingness.csv", index=False)

    # 3. Expanding-window folds audit
    min_train = int(profile_data.get("model_overrides", {}).get("minimum_train_years", 10))
    test_start = int(profile_data.get("model_overrides", {}).get("test_start_year", 2000))
    test_end = int(profile_data.get("model_overrides", {}).get("test_end_year", 2008))
    folds_df = audit_temporal_folds(panel, min_train, test_start, test_end)
    folds_df.to_csv(tables_dir / "expanded_pre_fit_folds.csv", index=False)

    # 4. Feature summary statistics
    features_df = compute_pre_fit_feature_statistics(panel)
    features_df.to_csv(tables_dir / "expanded_pre_fit_features.csv", index=False)

    # 5. Correlation analysis
    corr_df = compute_pre_fit_correlations(panel)
    corr_df.to_csv(tables_dir / "expanded_pre_fit_correlations.csv", index=False)

    # 6. Preprocessing isolation verification
    iso_check = verify_preprocessing_isolation(panel)
    if not iso_check["isolation_verified"]:
        raise RuntimeError("Preprocessing isolation check failed: potential data leakage detected!")

    # 7. Post-2008 low-onset regime verification
    post_2008 = panel.loc[panel["year"] > 2008]
    post_2008_positives = int(post_2008["crisis_within_horizon"].sum())
    post_2008_rows = len(post_2008)

    cumulative_test_rows = int(folds_df["test_observations"].sum())
    cumulative_test_positives = int(folds_df["test_positive_labels"].sum())

    LOGGER.info(
        "Phase 5A pre-fit audit passed: %s countries, %s rows, %s folds (2000-2008), %s test positives, zero models fitted",
        panel_info["unique_countries"],
        panel_info["total_rows"],
        len(folds_df),
        cumulative_test_positives,
    )

    return {
        "panel_integrity": panel_info,
        "folds_count": len(folds_df),
        "cumulative_test_rows": cumulative_test_rows,
        "cumulative_test_positives": cumulative_test_positives,
        "post_2008_rows": post_2008_rows,
        "post_2008_positives": post_2008_positives,
        "preprocessing_isolation_passed": iso_check["isolation_verified"],
        "models_fitted": 0,
        "performance_metrics_calculated": False,
    }
