# Phase 5C: Robustness & Sensitivity Analysis Results

**Project:** Global Financial Crisis Early Warning System  
**Repository:** `global-financial-crisis-ews`  
**Evaluation Scope:** 76 economies, 1990–2025 panel, expanding-window walk-forward validation ($T \in [2000, 2008]$)  
**Total Test Observations:** 684 country-years across 9 expanding folds  
**Total Test Positive Events:** 19 systemic banking crisis onsets ($t \to t+1$)  
**Unconditional Positive Rate:** 2.78% (19 / 684)  
**Baseline Status:** LOCKED (Phase 5B results preserved without alteration)  

---

## 1. Executive Summary & Research Principles

Phase 5C implements a strict, pre-specified robustness and sensitivity analysis layer designed to evaluate the stability, calibration, and sensitivity of the Phase 5B initial models under four methodological perturbations:

1. **Analysis A — Inflation Outlier Sensitivity:** Evaluates whether the extreme hyperinflation outlier (1,058.4% inflation observed in Angola 1996) distorts model estimation by testing fold-isolated 1st and 99th percentile winsorization.
2. **Analysis B — Class-Weight Sensitivity:** Evaluates the trade-off between unweighted loss optimization (`class_weight=None`) and balanced loss optimization (`class_weight='balanced'`).
3. **Analysis C — Decision Threshold Sensitivity:** Evaluates the operating characteristic curve across policy-relevant decision thresholds ($\tau \in \{0.25, 0.50, 0.75\}$) using the locked Phase 5B baseline predictions.
4. **Analysis D — Fold-Level Stability:** Analyzes fold-by-fold performance across all 9 expanding test years ($2000 \dots 2008$), explicitly identifying single-class evaluation folds where rank-order discrimination metrics are mathematically undefined.

### Strict Integrity Commitments
- **No Baseline Modification:** The locked Phase 5B baseline metrics and prediction files were preserved intact and unedited.
- **Zero Data Leakage:** All data transformations (including winsorization percentiles, imputation medians, and scaling parameters) were fitted **strictly on training fold observations** ($\text{year} \le \text{test\_year} - 1$) and applied forward out-of-sample.
- **No Hyperparameter Tuning or P-Hacking:** No model parameters were tuned against test metrics, no features were added or selected, and no thresholds were retroactively optimized.
- **Transparent Econometric Reporting:** Undefined metrics on single-class folds are explicitly reported as `"Not estimable"`, rather than imputed or omitted.

---

## 2. Analysis A: Inflation Outlier Sensitivity

### Methodology
In the Phase 5A pre-fit audit, `inflation` exhibited extreme right-skewness (kurtosis = 452.92, maximum = 1,058.38%), driven primarily by sub-Saharan and transition-economy hyperinflationary episodes in the 1990s. 

To determine whether baseline model behavior was driven by these tail observations, we implemented a custom scikit-learn transformer, `TrainingFoldWinsorizer`. For each expanding fold $T \in [2000, 2008]$:
- The 1st percentile ($q_{0.01}$) and 99th percentile ($q_{0.99}$) of `inflation` were estimated **exclusively on the training observations** ($\text{year} \le T - 1$).
- Training and out-of-sample test features were clipped to $[q_{0.01}, q_{0.99}]$.
- Out-of-sample predictions were evaluated across all 684 test observations.

### Empirical Results

| Model | Specification | PR-AUC | ROC-AUC | Precision | Recall | F1 | Brier Score | TP | FP | FN | TN |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Baseline (Raw) | **0.0278** | **0.5106** | 0.0189 | **0.2105** | **0.0346** | 0.2512 | 4 | 208 | 15 | 457 |
| **Logistic Regression** | Winsorized (1%/99%) | 0.0219 | 0.3893 | 0.0078 | 0.0526 | 0.0135 | **0.2223** | 1 | 128 | 18 | 537 |
| **Random Forest** | Baseline (Raw) | **0.0256** | **0.4712** | **0.0242** | **0.2105** | **0.0435** | **0.1524** | 4 | 161 | 15 | 504 |
| **Random Forest** | Winsorized (1%/99%) | **0.0256** | **0.4712** | **0.0242** | **0.2105** | **0.0435** | **0.1524** | 4 | 161 | 15 | 504 |

*Table: Comparison of baseline vs training-fold 1%/99% winsorized inflation across 684 pooled out-of-sample test observations ($\tau=0.50$). Output source: `results/tables/expanded_robustness_inflation.csv`.*

### Econometric Interpretation
1. **Random Forest Invariance:** Random Forest performance is **numerically identical** between the baseline and winsorized models. This is a fundamental mathematical property of decision trees: split decisions depend strictly on ordinal rank (order statistics) rather than metric distance. Clipping extreme values at the 1st and 99th percentiles preserves the ranking of observations outside the interior distribution, leaving partition points and tree structures unchanged.
2. **Logistic Regression Sensitivity:** Unlike tree ensembles, Logistic Regression relies on linear combinations of standardized features. In the baseline, extreme inflation values expanded the empirical standard deviation ($\sigma_{\text{inf}}$), compressing the relative scale of normal inflation observations. When winsorized, the training variance decreased, altering the coefficient scaling. This resulted in fewer false alarms (FP dropped from 208 to 128, improving the Brier score from 0.2512 to 0.2223), but severely degraded crisis detection: True Positives dropped from 4 to 1 (Recall = 5.26%), and ROC-AUC collapsed from 0.5106 to 0.3893.
3. **Conclusion:** Outlier winsorization does not rescue predictive discrimination for linear models; rather, it suppresses warning issuance without improving positive identification.

---

## 3. Analysis B: Class-Weight Sensitivity

### Methodology
In the Phase 5B baseline, models were estimated with `class_weight='balanced'` to prevent estimators from degenerating into trivial majority-class classifiers in the presence of severe class imbalance (unconditional event rate of 2.78%). 

Analysis B tests the sensitivity of this choice by estimating identical pipeline architectures with `class_weight=None` (standard unweighted empirical risk minimization).

### Empirical Results

| Model | Class Weight | PR-AUC | ROC-AUC | Precision | Recall | F1 | Brier Score | TP | FP | FN | TN |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | `balanced` (Baseline) | **0.0278** | **0.5106** | **0.0189** | **0.2105** | **0.0346** | 0.2512 | 4 | 208 | 15 | 457 |
| **Logistic Regression** | `None` (Unweighted) | 0.0169 | 0.1796 | 0.0000 | 0.0000 | 0.0000 | **0.0278** | 0 | 0 | 19 | 665 |
| **Random Forest** | `balanced` (Baseline) | **0.0256** | **0.4712** | **0.0242** | **0.2105** | **0.0435** | 0.1524 | 4 | 161 | 15 | 504 |
| **Random Forest** | `None` (Unweighted) | 0.0243 | 0.4313 | 0.0000 | 0.0000 | 0.0000 | **0.0281** | 0 | 0 | 19 | 665 |

*Table: Class-weight sensitivity comparison across 684 pooled out-of-sample test observations ($\tau=0.50$). Output source: `results/tables/expanded_robustness_class_weight.csv`.*

### Econometric Interpretation
1. **The Base-Rate Dilemma:** When `class_weight=None`, both Logistic Regression and Random Forest predict probabilities strictly below 0.50 for every observation. As a result, both models produce **zero true positives and zero false positives** ($\text{TP}=0, \text{FP}=0, \text{FN}=19, \text{TN}=665$).
2. **The "Accuracy/Brier Paradox":** Unweighted models achieve an ostensibly superior Brier score (0.0278 vs 0.2512 for LR; 0.0281 vs 0.1524 for RF). However, this improvement is purely an artifact of base-rate climatology: predicting zero everywhere in a panel where 97.22% of observations are non-crises yields high numerical accuracy while failing completely as an Early Warning System.
3. **Rank Discrimination Degradation:** For Logistic Regression, unweighted estimation causes ROC-AUC to collapse to 0.1796. This occurs because the small number of positive training cases are overwhelmed by majority-class observations during $L_2$-penalized log-loss minimization.
4. **Conclusion:** Balanced weighting is an operational necessity to force models to differentiate risk in rare-event settings, though it shifts probability calibration toward the center, inflating Brier scores.

---

## 4. Analysis C: Decision Threshold Sensitivity

### Methodology
In operational macroprudential surveillance, decision-makers face an asymmetric loss function: the cost of missing a systemic financial crisis ($\text{Cost}_{\text{FN}}$) is typically far larger than the cost of conducting preemptive regulatory scrutiny on a false alarm ($\text{Cost}_{\text{FP}}$).

Analysis C evaluates the locked Phase 5B out-of-sample probability predictions across three pre-specified policy thresholds without model retraining:
- $\tau = 0.25$ (High-sensitivity / supervisory vigilance)
- $\tau = 0.50$ (Standard classification baseline)
- $\tau = 0.75$ (High-specificity / conservative alarm)

### Empirical Results

| Model | Threshold ($\tau$) | Precision | Recall | F1 | Brier Score | TP | FP | FN | TN | False Positive Rate | False Negative Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.25 | 0.0278 | **1.0000** | **0.0541** | 0.2512 | 19 | 664 | 0 | 1 | 99.85% | **0.00%** |
| **Logistic Regression** | 0.50 | 0.0189 | 0.2105 | 0.0346 | 0.2512 | 4 | 208 | 15 | 457 | 31.28% | 78.95% |
| **Logistic Regression** | 0.75 | 0.0000 | 0.0000 | 0.0000 | 0.2512 | 0 | 15 | 19 | 650 | **2.26%** | 100.00% |
| **Random Forest** | 0.25 | 0.0182 | **0.3158** | 0.0344 | 0.1524 | 6 | 324 | 13 | 341 | 48.72% | 68.42% |
| **Random Forest** | 0.50 | **0.0242** | 0.2105 | **0.0435** | 0.1524 | 4 | 161 | 15 | 504 | 24.21% | 78.95% |
| **Random Forest** | 0.75 | 0.0000 | 0.0000 | 0.0000 | 0.1524 | 0 | 3 | 19 | 662 | **0.45%** | 100.00% |

*Table: Threshold sensitivity evaluation on locked Phase 5B predictions across 684 pooled out-of-sample test observations. Output source: `results/tables/expanded_threshold_sensitivity.csv`.*

### Policy & Operational Insights
1. **The Extreme Vigilance Regime ($\tau = 0.25$):**
   - For Logistic Regression, lowering $\tau$ to 0.25 achieves **100% Recall** ($\text{TP}=19, \text{FN}=0$), missing zero banking panics. However, it issues warnings for 683 out of 684 country-years ($\text{FPR} = 99.85\%$). Such a signal provides zero discriminatory value to policymakers and causes total alarm fatigue.
   - For Random Forest, $\tau = 0.25$ captures 6 of 19 crises (Recall = 31.58%) with a false positive rate of 48.72% (324 false alarms out of 665 tranquil years).
2. **The Conservative Regime ($\tau = 0.75$):**
   - At $\tau = 0.75$, both models suffer from **100% False Negative Rates** ($\text{Recall} = 0.0\%$). Although false positive rates drop to 2.26% (LR) and 0.45% (RF), the system fails to issue a warning for every single crisis.
3. **Implication:** The trade-off between sensitivity and specificity in 4-variable macroeconomic flow models is steep. There exists no fixed probability threshold that provides high crisis recall ($> 70\%$) while maintaining a tolerable false alarm rate ($< 15\%$).

---

## 5. Analysis D: Fold-Level Stability

### Methodology
Pooled out-of-sample metrics can conceal severe time-varying performance instability. Analysis D evaluates performance fold-by-fold across all 9 individual test years ($2000 \dots 2008$). 

Under standard evaluation conventions, when a test fold contains observations of only a single class (i.e., zero positive events), ROC-AUC and PR-AUC are mathematically undefined ($0/0$ division in denominator). Rather than fabricating values or substituting global averages, these folds are explicitly designated as `"Not estimable"`.

### Empirical Results

| Model | Test Year | Observations | Positive Events | ROC-AUC | PR-AUC | Precision | Recall | F1 | Brier Score | TP | FP | FN | TN | Evaluation Note |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression** | 2000 | 76 | 0 | Not estimable | Not estimable | 0.0000 | 0.0000 | 0.0000 | 0.2502 | 0 | 21 | 0 | 55 | Single-class test fold |
| **Logistic Regression** | 2001 | 76 | 1 | 0.0800 | 0.0143 | 0.0000 | 0.0000 | 0.0000 | 0.2449 | 0 | 16 | 1 | 59 | Normal event incidence |
| **Logistic Regression** | 2002 | 76 | 1 | 0.3200 | 0.0192 | 0.0000 | 0.0000 | 0.0000 | 0.2524 | 0 | 19 | 1 | 56 | Normal event incidence |
| **Logistic Regression** | 2003 | 76 | 0 | Not estimable | Not estimable | 0.0000 | 0.0000 | 0.0000 | 0.2525 | 0 | 25 | 0 | 51 | Single-class test fold |
| **Logistic Regression** | 2004 | 76 | 0 | Not estimable | Not estimable | 0.0000 | 0.0000 | 0.0000 | 0.2529 | 0 | 27 | 0 | 49 | Single-class test fold |
| **Logistic Regression** | 2005 | 76 | 0 | Not estimable | Not estimable | 0.0000 | 0.0000 | 0.0000 | 0.2532 | 0 | 32 | 0 | 44 | Single-class test fold |
| **Logistic Regression** | 2006 | 76 | 1 | 0.8667 | 0.0909 | 0.0667 | 1.0000 | 0.1250 | 0.2416 | 1 | 14 | 0 | 61 | Normal event incidence |
| **Logistic Regression** | 2007 | 76 | 15 | 0.6590 | 0.2743 | 0.2308 | 0.2000 | 0.2143 | 0.2405 | 3 | 10 | 12 | 51 | High crisis concentration (78.9%) |
| **Logistic Regression** | 2008 | 76 | 1 | 0.2267 | 0.0169 | 0.0000 | 0.0000 | 0.0000 | 0.2723 | 0 | 44 | 1 | 31 | Normal event incidence |
| **Random Forest** | 2000 | 76 | 0 | Not estimable | Not estimable | 0.0000 | 0.0000 | 0.0000 | 0.0928 | 0 | 10 | 0 | 66 | Single-class test fold |
| **Random Forest** | 2001 | 76 | 1 | 0.8267 | 0.0714 | 0.0000 | 0.0000 | 0.0000 | 0.1007 | 0 | 12 | 1 | 63 | Normal event incidence |
| **Random Forest** | 2002 | 76 | 1 | 0.5867 | 0.0312 | 0.0000 | 0.0000 | 0.0000 | 0.1318 | 0 | 18 | 1 | 57 | Normal event incidence |
| **Random Forest** | 2003 | 76 | 0 | Not estimable | Not estimable | 0.0000 | 0.0000 | 0.0000 | 0.1196 | 0 | 17 | 0 | 59 | Single-class test fold |
| **Random Forest** | 2004 | 76 | 0 | Not estimable | Not estimable | 0.0000 | 0.0000 | 0.0000 | 0.1391 | 0 | 19 | 0 | 57 | Single-class test fold |
| **Random Forest** | 2005 | 76 | 0 | Not estimable | Not estimable | 0.0000 | 0.0000 | 0.0000 | 0.1319 | 0 | 18 | 0 | 58 | Single-class test fold |
| **Random Forest** | 2006 | 76 | 1 | 0.1733 | 0.0159 | 0.0000 | 0.0000 | 0.0000 | 0.1851 | 0 | 24 | 1 | 51 | Normal event incidence |
| **Random Forest** | 2007 | 76 | 15 | 0.3454 | 0.1554 | 0.1304 | 0.2000 | 0.1579 | 0.2713 | 3 | 20 | 12 | 41 | High crisis concentration (78.9%) |
| **Random Forest** | 2008 | 76 | 1 | 0.7067 | 0.0435 | 0.0417 | 1.0000 | 0.0800 | 0.1992 | 1 | 23 | 0 | 52 | Normal event incidence |

*Table: Fold-level out-of-sample metrics across all 9 expanding window evaluation folds ($T \in [2000, 2008]$). Output source: `results/tables/expanded_fold_stability.csv`.*

### Methodological Insights on Temporal Clustering
1. **Severe Temporal Concentration:**
   - Out of 19 total out-of-sample crises across the 9-year evaluation window, **15 events (78.9%) occur in a single test year: 2007** (which predicts 2008 systemic banking crisis onsets).
   - In 4 out of the 9 test years (2000, 2003, 2004, 2005), there are **zero crisis onsets** across the entire 76-country universe.
   - In 3 out of 9 test years (2001, 2002, 2006 for LR / 2008 for RF), there is **exactly one crisis onset**.
2. **Small-Sample Metric Volatility:**
   - In folds with only 1 event, metric stability breaks down. For example, in 2006 ($t=2006$ predicting 2007 onsets), Logistic Regression correctly flagged the single crisis country, driving its intra-fold Recall to 1.0000 and ROC-AUC to 0.8667. In 2001, missing the single event caused intra-fold ROC-AUC to collapse to 0.0800.
   - Conversely, Random Forest missed the single crisis in 2006 (ROC-AUC = 0.1733), but caught the single crisis in 2008 ($t=2008$ predicting 2009 onsets), achieving an intra-fold ROC-AUC of 0.7067 and Recall of 1.0000.
3. **The GFC Epicenter Fold (Test Year 2007):**
   - Test year 2007 represents the true multi-event test of early-warning capacity.
   - Logistic Regression achieved an intra-fold ROC-AUC of **0.6590** and PR-AUC of **0.2743** (against an intra-fold base rate of $15/76 = 19.74\%$), capturing 3 crisis countries (Recall = 20.0%) with 10 false alarms (Precision = 23.08%).
   - Random Forest achieved an intra-fold ROC-AUC of **0.3454** and PR-AUC of **0.1554**, capturing 3 crisis countries (Recall = 20.0%) with 20 false alarms (Precision = 13.04%).

---

## 6. Synthesis & Methodological Conclusions

The four Phase 5C sensitivity analyses converge on three fundamental econometric realities for financial crisis early-warning systems:

1. **The Structural Limits of Flow Macroeconomic Indicators:**
   Across all perturbations (raw vs winsorized, balanced vs unweighted, varying decision thresholds), the 4 flow macroeconomic predictors (`gdp_growth`, `inflation`, `reserves_usd`, `reserves_usd_yoy_pct_change`) fail to achieve robust discrimination. Systemic banking crises are balance-sheet, leverage, and liquidity phenomenona. In the absence of private credit-to-GDP gaps, foreign-currency liability mismatches, asset price bubbles, or interbank contagion metrics, flow variables alone cannot reliably separate pre-crisis buildup from ordinary business cycle fluctuations.
2. **The Reality of Temporal Clustering:**
   Systemic financial crises do not occur as independent, identically distributed (i.i.d.) annual Bernoulli trials. They cluster in time around global credit cycles (the 2008 Global Financial Crisis). Evaluating models across tranquil intervals (e.g., 2000–2005) yields single-class folds where traditional discrimination metrics cannot be estimated, while the pooled score is overwhelmingly dominated by a single cluster year (2007).
3. **Evaluation Honesty as a Scientific Principle:**
   By transparently reporting undefined metrics as `"Not estimable"`, preserving the baseline intact, and demonstrating the complete breakdown of unweighted models, this research avoids the widespread publication bias and over-fitting prevalent in retrospective early-warning literature.

---

## 7. Artifact & Reproducibility Checklist

- [x] Baseline preserved: `logistic_regression_predictions.csv` and `random_forest_predictions.csv` untouched.
- [x] Analysis A table generated: `results/tables/expanded_robustness_inflation.csv`
- [x] Analysis B table generated: `results/tables/expanded_robustness_class_weight.csv`
- [x] Analysis C table generated: `results/tables/expanded_threshold_sensitivity.csv`
- [x] Analysis D table generated: `results/tables/expanded_fold_stability.csv`
- [x] Sensitivity predictions saved separately: `results/model_outputs/*_winsorized_predictions.csv`, `*_unweighted_predictions.csv`
- [x] CLI subcommand integrated: `run-phase-5c` (aliases `robustness`, `robustness-analysis`)
- [x] Dedicated unit tests implemented: `tests/test_phase5c_robustness.py` (7/7 passed)
- [x] Full regression test suite verified: 49/49 passed
- [x] Code style verified: `ruff check` passed without errors on all modified files
