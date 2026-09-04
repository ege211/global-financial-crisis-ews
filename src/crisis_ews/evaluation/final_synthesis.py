"""Phase 7: Final Model Selection & Research Synthesis Engine.

Synthesizes the complete empirical results from Phases 4B, 5A, 5B, 5C, 5D, 5E, and 6:
- Consolidates model performance metrics across baseline and extended specifications.
- Synthesizes all robustness and sensitivity dimensions into a standardized table.
- Documents the parameter estimates and associative interpretations of the Primary Research Specification.
- Generates a publication-quality final performance synthesis figure.

Methodology:
- Strictly post-estimation and observational: ZERO refitting, ZERO post-hoc tuning.
- Preserves all historical Phase 5 and Phase 6 artifacts intact.
- Enforces non-causal associative formulations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw

LOGGER = logging.getLogger(__name__)


def build_final_model_comparison_table(root: Path) -> pd.DataFrame:
    """Construct consolidated model comparison table across all 4 evaluated architectures.

    Directly traces and extracts metrics from committed Phase 5 and Phase 6 artifacts:
    - Baseline LR metrics: results/tables/expanded_logistic_regression_metrics.csv
    - Extended LR metrics: results/tables/expanded_extended_logistic_regression_metrics.csv
    - Baseline RF metrics: results/tables/expanded_random_forest_metrics.csv
    - Extended RF metrics: results/tables/expanded_extended_random_forest_metrics.csv
    - Temporal slices: results/tables/phase5e_temporal_generalization.csv and prediction files
    """
    from sklearn.metrics import roc_auc_score

    tables_dir = root / "results/tables"
    outputs_dir = root / "results/model_outputs"

    blr_m = pd.read_csv(tables_dir / "expanded_logistic_regression_metrics.csv").iloc[0]
    elr_m = pd.read_csv(tables_dir / "expanded_extended_logistic_regression_metrics.csv").iloc[0]
    brf_m = pd.read_csv(tables_dir / "expanded_random_forest_metrics.csv").iloc[0]
    erf_m = pd.read_csv(tables_dir / "expanded_extended_random_forest_metrics.csv").iloc[0]

    tg = pd.read_csv(tables_dir / "phase5e_temporal_generalization.csv")
    elr_07 = tg[(tg["model"] == "extended_logistic_regression") & (tg["subset_key"] == "gfc_2007")].iloc[0]
    elr_non07 = tg[(tg["model"] == "extended_logistic_regression") & (tg["subset_key"] == "excluding_2007")].iloc[0]
    blr_07 = tg[(tg["model"] == "baseline_logistic_regression") & (tg["subset_key"] == "gfc_2007")].iloc[0]
    blr_non07 = tg[(tg["model"] == "baseline_logistic_regression") & (tg["subset_key"] == "excluding_2007")].iloc[0]

    brf_preds = pd.read_csv(outputs_dir / "random_forest_predictions.csv")
    erf_preds = pd.read_csv(outputs_dir / "extended_random_forest_predictions.csv")

    brf_07_preds = brf_preds[brf_preds["test_year"] == 2007]
    brf_non07_preds = brf_preds[brf_preds["test_year"] != 2007]
    brf_07_rec = float(((brf_07_preds["predicted_class"] == 1) & (brf_07_preds["y_true"] == 1)).sum() / brf_07_preds["y_true"].sum())
    brf_non07_rec = float(((brf_non07_preds["predicted_class"] == 1) & (brf_non07_preds["y_true"] == 1)).sum() / brf_non07_preds["y_true"].sum())
    brf_07_roc = float(roc_auc_score(brf_07_preds["y_true"], brf_07_preds["predicted_probability"]))
    brf_non07_roc = float(roc_auc_score(brf_non07_preds["y_true"], brf_non07_preds["predicted_probability"]))

    erf_07_preds = erf_preds[erf_preds["test_year"] == 2007]
    erf_non07_preds = erf_preds[erf_preds["test_year"] != 2007]
    erf_07_rec = float(((erf_07_preds["predicted_class"] == 1) & (erf_07_preds["y_true"] == 1)).sum() / erf_07_preds["y_true"].sum())
    erf_non07_rec = float(((erf_non07_preds["predicted_class"] == 1) & (erf_non07_preds["y_true"] == 1)).sum() / erf_non07_preds["y_true"].sum())
    erf_07_roc = float(roc_auc_score(erf_07_preds["y_true"], erf_07_preds["predicted_probability"]))
    erf_non07_roc = float(roc_auc_score(erf_non07_preds["y_true"], erf_non07_preds["predicted_probability"]))

    records = [
        {
            "model": "Baseline Logistic Regression",
            "specification": "4-variable flow baseline (GDP, inflation, reserves, reserve growth)",
            "predictor_count": 4,
            "pr_auc_pooled": round(float(blr_m["pr_auc"]), 4),
            "roc_auc_pooled": round(float(blr_m["roc_auc"]), 4),
            "brier_score_pooled": round(float(blr_m["brier_score"]), 4),
            "recall_pooled": round(float(blr_m["recall"]), 4),
            "precision_pooled": round(float(blr_m["precision"]), 4),
            "f1_pooled": round(float(blr_m["f1"]), 4),
            "true_positives_pooled": int(blr_m["true_positives"]),
            "false_positives_pooled": int(blr_m["false_positives"]),
            "recall_2007_gfc": round(float(blr_07["recall"]), 4),
            "recall_non_2007": round(float(blr_non07["recall"]), 4),
            "roc_auc_2007_gfc": round(float(blr_07["roc_auc"]), 4),
            "roc_auc_non_2007": round(float(blr_non07["roc_auc"]), 4),
            "interpretability": "High (linear log-odds)",
            "parsimony": "4 parameters",
            "selection_role": "Baseline Benchmark",
            "primary_selection": "No",
            "selection_rationale": "Simple linear baseline; lower pooled discrimination and recall than extended specification.",
        },
        {
            "model": "Extended Logistic Regression",
            "specification": "6-variable macro-financial (+private credit/GDP, +current account/GDP)",
            "predictor_count": 6,
            "pr_auc_pooled": round(float(elr_m["pr_auc"]), 4),
            "roc_auc_pooled": round(float(elr_m["roc_auc"]), 4),
            "brier_score_pooled": round(float(elr_m["brier_score"]), 4),
            "recall_pooled": round(float(elr_m["recall"]), 4),
            "precision_pooled": round(float(elr_m["precision"]), 4),
            "f1_pooled": round(float(elr_m["f1"]), 4),
            "true_positives_pooled": int(elr_m["true_positives"]),
            "false_positives_pooled": int(elr_m["false_positives"]),
            "recall_2007_gfc": round(float(elr_07["recall"]), 4),
            "recall_non_2007": round(float(elr_non07["recall"]), 4),
            "roc_auc_2007_gfc": round(float(elr_07["roc_auc"]), 4),
            "roc_auc_non_2007": round(float(elr_non07["roc_auc"]), 4),
            "interpretability": "High (linear log-odds)",
            "parsimony": "6 parameters",
            "selection_role": "Primary Research Model",
            "primary_selection": "Yes (Regime-Qualified)",
            "selection_rationale": "Retained as primary interpretable research specification; superior pooled discrimination and GFC detection, but explicitly qualified by lack of non-2007 generalization.",
        },
        {
            "model": "Baseline Random Forest",
            "specification": "4-variable flow baseline (100 trees, max_depth=4, balanced)",
            "predictor_count": 4,
            "pr_auc_pooled": round(float(brf_m["pr_auc"]), 4),
            "roc_auc_pooled": round(float(brf_m["roc_auc"]), 4),
            "brier_score_pooled": round(float(brf_m["brier_score"]), 4),
            "recall_pooled": round(float(brf_m["recall"]), 4),
            "precision_pooled": round(float(brf_m["precision"]), 4),
            "f1_pooled": round(float(brf_m["f1"]), 4),
            "true_positives_pooled": int(brf_m["true_positives"]),
            "false_positives_pooled": int(brf_m["false_positives"]),
            "recall_2007_gfc": round(brf_07_rec, 4),
            "recall_non_2007": round(brf_non07_rec, 4),
            "roc_auc_2007_gfc": round(brf_07_roc, 4),
            "roc_auc_non_2007": round(brf_non07_roc, 4),
            "interpretability": "Moderate (Gini importance)",
            "parsimony": "Non-parametric ensemble",
            "selection_role": "Nonlinear Benchmark",
            "primary_selection": "No",
            "selection_rationale": "Lower Brier score due to conservative probability calibration, but lower discrimination and higher complexity than logistic models.",
        },
        {
            "model": "Extended Random Forest",
            "specification": "6-variable macro-financial (100 trees, max_depth=4, balanced)",
            "predictor_count": 6,
            "pr_auc_pooled": round(float(erf_m["pr_auc"]), 4),
            "roc_auc_pooled": round(float(erf_m["roc_auc"]), 4),
            "brier_score_pooled": round(float(erf_m["brier_score"]), 4),
            "recall_pooled": round(float(erf_m["recall"]), 4),
            "precision_pooled": round(float(erf_m["precision"]), 4),
            "f1_pooled": round(float(erf_m["f1"]), 4),
            "true_positives_pooled": int(erf_m["true_positives"]),
            "false_positives_pooled": int(erf_m["false_positives"]),
            "recall_2007_gfc": round(erf_07_rec, 4),
            "recall_non_2007": round(erf_non07_rec, 4),
            "roc_auc_2007_gfc": round(erf_07_roc, 4),
            "roc_auc_non_2007": round(erf_non07_roc, 4),
            "interpretability": "Moderate (Gini importance)",
            "parsimony": "Non-parametric ensemble",
            "selection_role": "Exploratory ML Specification",
            "primary_selection": "No",
            "selection_rationale": "Strong probability compression toward zero; misses 17 of 19 crises at default threshold (recall 10.5%); not suitable as primary early warning model.",
        },
    ]
    return pd.DataFrame(records)


def build_final_robustness_summary_table(root: Path) -> pd.DataFrame:
    """Construct comprehensive synthesis table of all robustness dimensions evaluated in Phase 5."""
    records = [
        {
            "dimension": "Winsorization Sensitivity",
            "evaluation_focus": "Sensitivity of model performance to extreme inflation tails",
            "baseline_condition": "Raw CPI inflation (untrimmed)",
            "robustness_condition": "CPI inflation winsorized at [1st, 99th] percentiles across training folds",
            "empirical_outcome": "LR PR-AUC drops from 0.0278 to 0.0219 (-21.2%), ROC-AUC drops from 0.5106 to 0.3893, Recall collapses from 21.1% to 5.3% (1 detection); RF metrics remain identical (PR-AUC 0.0256, Recall 21.1%).",
            "methodological_implication": "Linear logistic model relies heavily on extreme inflation tails for crisis separation; truncating outliers destroys predictive signal. Tree model is inherently robust to monotonic outlier shifts.",
        },
        {
            "dimension": "Class Weight Sensitivity",
            "evaluation_focus": "Impact of rare-event cost-sensitive loss weighting",
            "baseline_condition": "Balanced class weighting (loss weighted inversely to class frequency)",
            "robustness_condition": "Unweighted standard cross-entropy / Gini loss (class_weight=None)",
            "empirical_outcome": "Both LR and RF collapse to 0.0% Recall (0 true positives, 0 false alarms) at standard 0.50 cutoff, as models predict calm state for 100% of observations.",
            "methodological_implication": "In severely imbalanced panels (2.7% positive rate), cost-sensitive reweighting is strictly mandatory to prevent default majority-class convergence.",
        },
        {
            "dimension": "Threshold Sensitivity",
            "evaluation_focus": "Trade-off between early crisis detection and operational false alarms",
            "baseline_condition": "Standard classification cutoff (tau = 0.50)",
            "robustness_condition": "Conservative (tau = 0.75) and Aggressive (tau = 0.25) cutoffs",
            "empirical_outcome": "At tau=0.25, LR achieves 100% Recall (19/19) but produces 664 false positives (FPR 99.8%); at tau=0.75, LR achieves 0% Recall with 15 false alarms.",
            "methodological_implication": "Demonstrates severe policy trade-off; early warning systems cannot eliminate false alarms without missing almost all crises. Threshold tuning must be policy-loss-driven.",
        },
        {
            "dimension": "Expanding-Window Fold Stability",
            "evaluation_focus": "Temporal consistency of performance across 9 sequential test years (2000-2008)",
            "baseline_condition": "Pooled out-of-sample aggregation (684 observations, 19 crises)",
            "robustness_condition": "Fold-by-fold metric evaluation across expanding windows",
            "empirical_outcome": "In 4 calm single-class folds (2000, 2003, 2004, 2005), AUC metrics are Not estimable; 15 of 19 positive events cluster in fold 2007 (forecasting 2008).",
            "methodological_implication": "Cross-country crisis warning evaluation is structurally dominated by multi-country systemic waves. Out-of-sample metrics are episodic rather than steady-state.",
        },
        {
            "dimension": "Predictor-Set Expansion",
            "evaluation_focus": "Value of adding financial balance-sheet indicators to macroeconomic flows",
            "baseline_condition": "4-variable flow baseline (GDP, inflation, reserves, reserve growth)",
            "robustness_condition": "6-variable extended set (+private credit/GDP, +current account/GDP)",
            "empirical_outcome": "Pooled PR-AUC increases from 0.0278 to 0.0344 (+23.7%), ROC-AUC increases from 0.5106 to 0.5734, Recall surges from 21.1% to 47.4% (+125%), false positives decrease from 208 to 187.",
            "methodological_implication": "Inclusion of private credit leverage and external current account deficits provides substantially stronger empirical discrimination than flow variables alone.",
        },
        {
            "dimension": "Out-of-Sample Regime Generalization",
            "evaluation_focus": "Generalizability of predictive gains outside the 2007 pre-GFC episode",
            "baseline_condition": "Pooled 2000-2008 evaluation window (19 crises)",
            "robustness_condition": "Subsets excluding 2007 fold (N=4 crises), pre-GFC 2000-2006 (N=3 crises), and 2007 fold alone (N=15 crises)",
            "empirical_outcome": "Excluding 2007, Extended LR Recall drops to 0.0% (0/4) and ROC-AUC inverts to 0.2301; in 2007, PR-AUC is 0.2647 and Recall is 60.0% (9/15).",
            "methodological_implication": "The extended specification is highly regime-dependent. It functions as an effective indicator of private credit overhang with external deficits, but does not generalize to idiosyncratic emerging-market crises.",
        },
    ]
    return pd.DataFrame(records)


def build_final_model_coefficients_table(root: Path) -> pd.DataFrame:
    """Construct final parameter estimates and associative interpretations for Primary Model."""
    # Sourced strictly from final expanding training fold T=2008 (trained on 1990-2007)
    records = [
        {
            "feature": "private_credit_pct_gdp",
            "feature_label": "Domestic Private Credit (% of GDP)",
            "logistic_coefficient": 0.514200,
            "odds_ratio": 1.672301,
            "direction": "Positive",
            "random_forest_importance": 0.224627,
            "economic_interpretation": (
                "Holding the other included predictors constant, the fitted model associates a one standard-deviation "
                "increase in domestic private credit-to-GDP with a 67.2% increase in the estimated odds of a systemic banking "
                "crisis onset in t+1 (OR = 1.6723). This reflects the historical empirical alignment between rapid banking "
                "leverage accumulation and subsequent systemic banking distress."
            ),
        },
        {
            "feature": "reserves_usd",
            "feature_label": "Gross International Reserves (USD Level)",
            "logistic_coefficient": -0.170843,
            "odds_ratio": 0.842954,
            "direction": "Negative",
            "random_forest_importance": 0.133434,
            "economic_interpretation": (
                "Holding the other included predictors constant, the fitted model associates larger gross international "
                "reserve stocks with a 15.7% decrease in the estimated odds of crisis onset (OR = 0.8430). Substantial liquid "
                "foreign-exchange buffers historically correlate with lower systemic vulnerability."
            ),
        },
        {
            "feature": "gdp_growth",
            "feature_label": "Real GDP Growth (Annual %)",
            "logistic_coefficient": -0.127668,
            "odds_ratio": 0.880146,
            "direction": "Negative",
            "random_forest_importance": 0.090146,
            "economic_interpretation": (
                "Holding the other included predictors constant, the fitted model associates higher real GDP growth with a "
                "12.0% decrease in the estimated odds of crisis onset (OR = 0.8801). Pre-crisis macroeconomic deceleration is "
                "statistically associated with heightened banking system stress."
            ),
        },
        {
            "feature": "inflation",
            "feature_label": "CPI Inflation Rate (Annual %)",
            "logistic_coefficient": 0.096273,
            "odds_ratio": 1.101059,
            "direction": "Positive",
            "random_forest_importance": 0.183310,
            "economic_interpretation": (
                "Holding the other included predictors constant, the fitted model associates higher consumer price inflation "
                "with a 10.1% increase in the estimated odds of crisis onset (OR = 1.1011), capturing price overheating and "
                "underlying monetary instability."
            ),
        },
        {
            "feature": "current_account_pct_gdp",
            "feature_label": "Current Account Balance (% of GDP)",
            "logistic_coefficient": -0.046319,
            "odds_ratio": 0.954737,
            "direction": "Negative",
            "random_forest_importance": 0.110577,
            "economic_interpretation": (
                "Holding the other included predictors constant, the fitted model associates a one standard-deviation "
                "increase in current account balance (greater surplus) with a 4.5% decrease in the estimated odds of crisis onset "
                "(OR = 0.9547). External deficits (negative values) correspondingly elevate estimated model risk probabilities."
            ),
        },
        {
            "feature": "reserves_usd_yoy_pct_change",
            "feature_label": "Reserves YoY Growth Rate (%)",
            "logistic_coefficient": -0.044913,
            "odds_ratio": 0.956080,
            "direction": "Negative",
            "random_forest_importance": 0.257906,
            "economic_interpretation": (
                "Holding the other included predictors constant, the fitted model associates positive annual reserve accumulation "
                "with a 4.4% decrease in the estimated odds of crisis onset (OR = 0.9561). Severe reserve drawdowns reflect "
                "balance-of-payments pressures that correlate with systemic banking vulnerability."
            ),
        },
    ]
    return pd.DataFrame(records)


def generate_phase7_visualizations(root: Path) -> dict[str, Path]:
    """Generate publication-quality SVG and PNG synthesis visualizations."""
    figs_dir = root / "results/figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    svg_path = figs_dir / "phase7_performance_synthesis.svg"
    png_path = figs_dir / "phase7_performance_synthesis.png"

    # Multi-panel SVG showing:
    # Panel 1: Out-of-Sample Recall across Models (Pooled vs 2007 vs Non-2007)
    # Panel 2: PR-AUC and Brier Score Comparison
    svg = """<svg width="860" height="540" viewBox="0 0 860 540" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif">
  <rect width="860" height="540" fill="#ffffff"/>
  <!-- Title & Subtitle -->
  <text x="40" y="42" font-size="20" font-weight="bold" fill="#1e293b">Figure 4: Final Model Performance &amp; Temporal Generalization Synthesis</text>
  <text x="40" y="66" font-size="13" fill="#64748b">Comparing Out-of-Sample Crisis Detection (Recall) and Precision-Recall Dynamics across Evaluated Architectures</text>

  <!-- Legend -->
  <g transform="translate(480, 42)">
    <rect x="0" y="0" width="14" height="14" fill="#0284c7" rx="2"/>
    <text x="20" y="12" font-size="11" fill="#334155">Baseline LR (4 vars)</text>
    <rect x="150" y="0" width="14" height="14" fill="#dc2626" rx="2"/>
    <text x="170" y="12" font-size="11" fill="#334155">Extended LR (6 vars)</text>
    <rect x="300" y="0" width="14" height="14" fill="#16a34a" rx="2"/>
    <text x="320" y="12" font-size="11" fill="#334155">Random Forest (6 vars)</text>
  </g>

  <!-- Panel 1: Recall across Evaluation Subsets -->
  <g transform="translate(60, 110)">
    <text x="180" y="0" font-size="15" font-weight="600" fill="#334155" text-anchor="middle">Out-of-Sample Recall (%) by Temporal Subset</text>
    <!-- Axes -->
    <line x1="50" y1="20" x2="50" y2="280" stroke="#cbd5e1" stroke-width="1.5"/>
    <line x1="50" y1="280" x2="350" y2="280" stroke="#cbd5e1" stroke-width="1.5"/>

    <!-- Y-axis ticks (0 to 100%, scale: 2.6px per 1%) -->
    <text x="40" y="284" font-size="11" fill="#64748b" text-anchor="end">0%</text>
    <line x1="45" y1="280" x2="50" y2="280" stroke="#94a3b8"/>

    <text x="40" y="219" font-size="11" fill="#64748b" text-anchor="end">25%</text>
    <line x1="45" y1="215" x2="50" y2="215" stroke="#94a3b8"/>
    <line x1="50" y1="215" x2="350" y2="215" stroke="#f1f5f9" stroke-dasharray="4"/>

    <text x="40" y="154" font-size="11" fill="#64748b" text-anchor="end">50%</text>
    <line x1="45" y1="150" x2="50" y2="150" stroke="#94a3b8"/>
    <line x1="50" y1="150" x2="350" y2="150" stroke="#f1f5f9" stroke-dasharray="4"/>

    <text x="40" y="89" font-size="11" fill="#64748b" text-anchor="end">75%</text>
    <line x1="45" y1="85" x2="50" y2="85" stroke="#94a3b8"/>
    <line x1="50" y1="85" x2="350" y2="85" stroke="#f1f5f9" stroke-dasharray="4"/>

    <!-- Subset 1: Pooled (N=19) -->
    <!-- Base LR: 21.1% -> h=54.9, y=225.1 -->
    <rect x="75" y="225" width="22" height="55" fill="#0284c7" rx="2"/>
    <text x="86" y="218" font-size="10" font-weight="bold" fill="#0284c7" text-anchor="middle">21.1%</text>
    <!-- Ext LR: 47.4% -> h=123.2, y=156.8 -->
    <rect x="100" y="157" width="22" height="123" fill="#dc2626" rx="2"/>
    <text x="111" y="150" font-size="10" font-weight="bold" fill="#dc2626" text-anchor="middle">47.4%</text>
    <!-- Ext RF: 10.5% -> h=27.3, y=252.7 -->
    <rect x="125" y="253" width="22" height="27" fill="#16a34a" rx="2"/>
    <text x="136" y="246" font-size="10" font-weight="bold" fill="#16a34a" text-anchor="middle">10.5%</text>
    <text x="111" y="300" font-size="11" font-weight="600" fill="#334155" text-anchor="middle">Pooled</text>
    <text x="111" y="315" font-size="10" fill="#64748b" text-anchor="middle">(N=19)</text>

    <!-- Subset 2: 2007 GFC Fold (N=15) -->
    <!-- Base LR: 20.0% -> h=52, y=228 -->
    <rect x="175" y="228" width="22" height="52" fill="#0284c7" rx="2"/>
    <text x="186" y="221" font-size="10" font-weight="bold" fill="#0284c7" text-anchor="middle">20.0%</text>
    <!-- Ext LR: 60.0% -> h=156, y=124 -->
    <rect x="200" y="124" width="22" height="156" fill="#dc2626" rx="2"/>
    <text x="211" y="117" font-size="10" font-weight="bold" fill="#dc2626" text-anchor="middle">60.0%</text>
    <!-- Ext RF: 13.3% -> h=35, y=245 -->
    <rect x="225" y="245" width="22" height="35" fill="#16a34a" rx="2"/>
    <text x="236" y="238" font-size="10" font-weight="bold" fill="#16a34a" text-anchor="middle">13.3%</text>
    <text x="211" y="300" font-size="11" font-weight="600" fill="#334155" text-anchor="middle">2007 Fold</text>
    <text x="211" y="315" font-size="10" fill="#64748b" text-anchor="middle">(N=15)</text>

    <!-- Subset 3: Non-2007 Folds (N=4) -->
    <!-- Base LR: 25.0% -> h=65, y=215 -->
    <rect x="275" y="215" width="22" height="65" fill="#0284c7" rx="2"/>
    <text x="286" y="208" font-size="10" font-weight="bold" fill="#0284c7" text-anchor="middle">25.0%</text>
    <!-- Ext LR: 0.0% -> h=2, y=278 -->
    <rect x="300" y="278" width="22" height="2" fill="#dc2626" rx="1"/>
    <text x="311" y="270" font-size="10" font-weight="bold" fill="#dc2626" text-anchor="middle">0.0%</text>
    <!-- Ext RF: 0.0% -> h=2, y=278 -->
    <rect x="325" y="278" width="22" height="2" fill="#16a34a" rx="1"/>
    <text x="336" y="270" font-size="10" font-weight="bold" fill="#16a34a" text-anchor="middle">0.0%</text>
    <text x="311" y="300" font-size="11" font-weight="600" fill="#334155" text-anchor="middle">Non-2007</text>
    <text x="311" y="315" font-size="10" fill="#64748b" text-anchor="middle">(N=4)</text>
  </g>

  <!-- Panel 2: PR-AUC vs Unconditional Prior -->
  <g transform="translate(480, 110)">
    <text x="160" y="0" font-size="15" font-weight="600" fill="#334155" text-anchor="middle">Pooled Precision-Recall AUC vs Prior</text>
    <!-- Axes -->
    <line x1="50" y1="20" x2="50" y2="280" stroke="#cbd5e1" stroke-width="1.5"/>
    <line x1="50" y1="280" x2="330" y2="280" stroke="#cbd5e1" stroke-width="1.5"/>

    <!-- Y-axis ticks (0 to 0.040, scale: 6500px per 1.0 -> 260px for 0.040) -->
    <text x="40" y="284" font-size="11" fill="#64748b" text-anchor="end">0.000</text>
    <line x1="45" y1="280" x2="50" y2="280" stroke="#94a3b8"/>

    <text x="40" y="219" font-size="11" fill="#64748b" text-anchor="end">0.010</text>
    <line x1="45" y1="215" x2="50" y2="215" stroke="#94a3b8"/>
    <line x1="50" y1="215" x2="330" y2="215" stroke="#f1f5f9" stroke-dasharray="4"/>

    <text x="40" y="154" font-size="11" fill="#64748b" text-anchor="end">0.020</text>
    <line x1="45" y1="150" x2="50" y2="150" stroke="#94a3b8"/>
    <line x1="50" y1="150" x2="330" y2="150" stroke="#f1f5f9" stroke-dasharray="4"/>

    <text x="40" y="89" font-size="11" fill="#64748b" text-anchor="end">0.030</text>
    <line x1="45" y1="85" x2="50" y2="85" stroke="#94a3b8"/>
    <line x1="50" y1="85" x2="330" y2="85" stroke="#f1f5f9" stroke-dasharray="4"/>

    <!-- Unconditional sample prior line (pi = 19/684 = 0.02778 -> y = 280 - (0.02778 * 6500) = 99.4) -->
    <line x1="50" y1="99" x2="330" y2="99" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="5"/>
    <text x="335" y="103" font-size="10" fill="#64748b">Prior &pi; = 0.0278</text>

    <!-- Bars for PR-AUC -->
    <!-- Base LR: 0.0278 -> y=99.4, h=180.6 -->
    <rect x="80" y="99" width="35" height="181" fill="#0284c7" rx="2"/>
    <text x="97" y="90" font-size="11" font-weight="bold" fill="#0284c7" text-anchor="middle">0.0278</text>
    <text x="97" y="300" font-size="11" font-weight="600" fill="#334155" text-anchor="middle">Base LR</text>

    <!-- Ext LR: 0.0344 -> y = 280 - (0.0344 * 6500) = 56.4, h=223.6 -->
    <rect x="145" y="56" width="35" height="224" fill="#dc2626" rx="2"/>
    <text x="162" y="47" font-size="11" font-weight="bold" fill="#dc2626" text-anchor="middle">0.0344</text>
    <text x="162" y="300" font-size="11" font-weight="600" fill="#334155" text-anchor="middle">Ext LR</text>

    <!-- Base RF: 0.0256 -> y = 280 - (0.0256 * 6500) = 113.6, h=166.4 -->
    <rect x="210" y="114" width="35" height="166" fill="#64748b" rx="2"/>
    <text x="227" y="105" font-size="11" font-weight="bold" fill="#64748b" text-anchor="middle">0.0256</text>
    <text x="227" y="300" font-size="11" font-weight="600" fill="#334155" text-anchor="middle">Base RF</text>

    <!-- Ext RF: 0.0274 -> y = 280 - (0.0274 * 6500) = 101.9, h=178.1 -->
    <rect x="275" y="102" width="35" height="178" fill="#16a34a" rx="2"/>
    <text x="292" y="93" font-size="11" font-weight="bold" fill="#16a34a" text-anchor="middle">0.0274</text>
    <text x="292" y="300" font-size="11" font-weight="600" fill="#334155" text-anchor="middle">Ext RF</text>
  </g>

  <!-- Footnote -->
  <text x="40" y="500" font-size="11" fill="#94a3b8">Note: Illustrates the central empirical finding: Extended LR improves pooled PR-AUC and GFC detection (60%), but non-2007 recall collapses to 0.0% (N=4).</text>
</svg>"""

    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg)

    # Render companion PNG raster via Pillow
    width, height = 1200, 750
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    draw.rectangle([0, 0, width, 100], fill=(248, 250, 252))
    draw.line([0, 100, width, 100], fill=(226, 232, 240), width=2)
    draw.text((50, 25), "Figure 4: Final Model Performance & Temporal Generalization Synthesis", fill=(30, 41, 59))
    draw.text((50, 60), "Out-of-Sample Recall and Precision-Recall Dynamics across Evaluated Architectures", fill=(100, 116, 139))
    draw.rectangle([50, 130, width - 50, height - 70], outline=(203, 213, 225), width=2)

    draw.text(
        (70, 160),
        f"Artifact: {svg_path.name}\nVector graphic format available in repository.",
        fill=(51, 65, 85),
    )
    draw.text(
        (70, 220),
        "Key Quantitative Findings:\n"
        "- Extended Logistic Regression achieves highest pooled PR-AUC (0.0344) and Recall (47.4%).\n"
        "- In the 2007 GFC fold, Extended LR achieves 60.0% Recall (9 of 15 detected).\n"
        "- Outside 2007, Extended LR achieves 0.0% Recall across the four observed crisis events (N=4).\n"
        "- Brier scores: Random Forest models achieve lower Brier scores (~0.13-0.15) via probability compression,\n"
        "  but miss almost all crises (Recall 10.5%-21.1%).",
        fill=(100, 116, 139),
    )
    draw.text(
        (50, height - 40),
        "Global Financial Crisis Early Warning System (Phase 7 Final Synthesis)",
        fill=(148, 163, 184),
    )
    img.save(png_path, "PNG")

    LOGGER.info("Generated Phase 7 final performance synthesis figures in %s", figs_dir)
    return {"svg": svg_path, "png": png_path}


def run_phase_7_final_synthesis(root: Path) -> dict[str, Any]:
    """Execute complete Phase 7 Final Model & Research Synthesis pipeline."""
    tables_dir = root / "results/tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Executing Phase 7 Final Synthesis pipeline...")

    # 1. Final Model Comparison Table
    LOGGER.info("Building final model comparison table...")
    comparison_df = build_final_model_comparison_table(root)
    comparison_path = tables_dir / "final_model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    # 2. Final Robustness Summary Table
    LOGGER.info("Building final robustness summary table...")
    robustness_df = build_final_robustness_summary_table(root)
    robustness_path = tables_dir / "final_robustness_summary.csv"
    robustness_df.to_csv(robustness_path, index=False)

    # 3. Final Model Coefficients Table
    LOGGER.info("Building final model coefficients table...")
    coefficients_df = build_final_model_coefficients_table(root)
    coefficients_path = tables_dir / "final_model_coefficients.csv"
    coefficients_df.to_csv(coefficients_path, index=False)

    # 4. Final Visualizations
    LOGGER.info("Generating final synthesis visualizations...")
    figs = generate_phase7_visualizations(root)

    summary = {
        "models_compared": len(comparison_df),
        "robustness_dimensions_summarized": len(robustness_df),
        "coefficients_documented": len(coefficients_df),
        "tables_generated": [
            str(comparison_path),
            str(robustness_path),
            str(coefficients_path),
        ],
        "figures_generated": [str(p) for p in figs.values()],
    }
    LOGGER.info("Phase 7 Final Synthesis completed successfully.")
    return summary
