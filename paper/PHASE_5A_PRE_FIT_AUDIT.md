# Phase 5A: Modeling Infrastructure & Pre-Fit Audit

**Date:** 2026-09-04  
**Project:** Global Financial Crisis Early Warning System (`global-financial-crisis-ews`)  
**Status:** Completed & Pre-Fit Locked. Zero predictive models fitted.

---

## 1. Executive Summary & Research-Integrity Commitments

Phase 5A establishes the formal modeling infrastructure, fold-generation architecture, leakage-prevention guarantees, and pre-fit diagnostic audits for the objectively selected 76-country expanded universe across the 1990–2025 observation window.

In accordance with strict quantitative research-integrity standards:
1. **Zero Predictive Modeling in Phase 5A:** No predictive models (Logistic Regression, Random Forest, XGBoost, or any other estimator) were trained or fitted.
2. **Zero Predictive Metrics:** No out-of-sample performance scores (ROC-AUC, PR-AUC, Brier score, sensitivity, specificity, precision, or recall) were calculated or estimated.
3. **No Cherry-Picking:** The country universe (76 economies), feature set (4 core predictors), target contract ($t \to t+1$), and temporal evaluation window ($T \in [2000, 2008]$) remain strictly identical to the pre-registered Phase 4B lock.
4. **Pre-Fit Goal:** Ensure that the modeling pipeline is mathematically sound, free of data leakage, and fully audited before a single model parameter is estimated in Phase 5B.

---

## 2. Modeled Panel Contract & Pre-Fit Data Verification

The modeling dataset is loaded from `data/processed/expanded_research/modeling_panel.csv`.

* **Candidate Universe:** 217 non-aggregate World Bank economies.
* **Objective Inclusion Rule:** 100% complete annual raw data coverage across 1990–2025 for GDP growth, inflation, and total reserves.
* **Qualifying Sample:** Exactly 76 economies; 141 excluded solely on coverage with documented reasons in [`results/tables/expanded_country_coverage.csv`](../results/tables/expanded_country_coverage.csv).
* **Panel Dimensions:** Exactly 76 countries × 36 annual observations (1990–2025) = **2,736 country-year rows**.
* **Duplicates:** 0 duplicate `(country_code, year)` observations.
* **Supervised Target:**
  $$y(c, t) = 1 \quad \text{if a qualifying systemic banking crisis onset starts in calendar year } t+1, \quad 0 \text{ otherwise.}$$
* **Crisis Source:** Laeven & Valencia (2026), IMF Working Paper WP/26/94 (*Systemic Banking Crises Database: 1970–2025*), excluding author-identified borderline episodes (Nicaragua 2018, Sri Lanka 2023, Vietnam 2022).
* **Positive Labels:** Exactly 44 positive forward labels across 1990–2025 (positive rate: 1.608%).
* **Lookahead Prohibition:** Contemporaneous crisis year $t$ features are strictly excluded from predictive targets.

---

## 3. Core Predictors & Pre-Fit Feature Diagnostics

The Phase 5A core feature set consists of 4 economically motivated predictors:

$$\mathbf{x}_{c, t} = \begin{bmatrix} \text{GDP Growth}_{c, t} \\ \text{Inflation}_{c, t} \\ \text{Reserves USD}_{c, t} \\ \Delta\% \text{Reserves}_{c, t} \end{bmatrix}$$

### Economic Hypotheses & Directional Expectations
* **Real GDP Growth (`gdp_growth`):** Counter-cyclical crisis risk ($\partial \text{risk} / \partial \text{growth} < 0$). Sharp economic deceleration or contraction precedes banking sector distress.
* **Inflation (`inflation`):** Pro-cyclical crisis risk ($\partial \text{risk} / \partial \text{inflation} > 0$). High or accelerating inflation erodes purchasing power, tightens financial conditions, and precipitates currency/banking twin vulnerabilities.
* **Reserves in USD (`reserves_usd`):** Macro scale indicator of foreign exchange liquidity.
* **Reserves YoY Percentage Change (`reserves_usd_yoy_pct_change`):** Flow indicator of external buffer depletion ($\partial \text{risk} / \partial \Delta\% \text{reserves} < 0$). Rapid reserve drawdowns signal balance-of-payments stress and imminent crisis vulnerability.

### Pre-Fit Distributional Summary (`results/tables/expanded_pre_fit_features.csv`)

| Variable | Count | Mean | Std | Min | 25% | Median | 75% | Max | Skewness |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `gdp_growth` | 2,736 | 3.40% | 5.09% | -54.40% | 1.44% | 3.41% | 5.27% | 86.83% | 1.57 |
| `inflation` | 2,736 | 11.68% | 164.39% | -9.80% | 1.63% | 3.29% | 6.47% | 7,481.66% | 37.80 |
| `reserves_usd` | 2,736 | \$80.77B | \$301.66B | \$8.32M | \$1.64B | \$13.09B | \$50.61B | \$3.90T | 8.76 |
| `reserves_usd_yoy_pct_change` | 2,660 | 11.07% | 31.78% | -90.17% | -2.80% | 7.02% | 19.21% | 554.05% | 6.01 |

### Missingness Audit
* `gdp_growth`: 0 missing (0.00%).
* `inflation`: 0 missing (0.00%).
* `reserves_usd`: 0 missing (0.00%).
* `reserves_usd_yoy_pct_change`: Exactly 76 missing (2.78%), strictly and exclusively located in year $1990$ across all 76 countries due to the 1-year backward lag operator ($t-1 = 1989$, which precedes the panel start).
* **Missingness Handling:** Within each temporal fold, missing values are imputed strictly using the **median of the training fold**. Test fold values are never used during imputation fitting.

### Correlation Analysis (`results/tables/expanded_pre_fit_correlations.csv`)

| Predictor Pair | Pearson Correlation ($r$) | Spearman Rank Correlation ($\rho$) | Multicollinearity Assessment |
|---|---:|---:|---|
| `gdp_growth` & `inflation` | +0.0064 | +0.1383 | Negligible collinearity |
| `gdp_growth` & `reserves_usd` | +0.0299 | -0.0666 | Negligible collinearity |
| `gdp_growth` & `reserves_usd_yoy_pct_change` | +0.0272 | +0.0891 | Negligible collinearity |
| `inflation` & `reserves_usd` | -0.0186 | -0.1424 | Low negative correlation |
| `inflation` & `reserves_usd_yoy_pct_change` | +0.0645 | +0.0427 | Negligible collinearity |
| `reserves_usd` & `reserves_usd_yoy_pct_change` | -0.0237 | +0.0010 | Uncorrelated |

**Conclusion:** All pairwise correlations are strictly below $|0.15|$. The 4 core predictors provide orthogonal economic signals with zero threat of variance inflation or matrix rank deficiency.

---

## 4. Expanding-Window Evaluation Protocol ($T \in [2000, 2008]$)

### Mathematical Definition of Folds
For each evaluation year $T \in [2000, 2008]$:
1. **Training Set:** All country-year observations with $\text{year} \in [1990, T-1]$.
2. **Test Set:** All 76 country observations with $\text{year} = T$.
3. **Temporal Monotonicity:** $\max(\text{train\_year}) = T - 1 < T = \min(\text{test\_year})$.

### Audit of Temporal Folds (`results/tables/expanded_pre_fit_folds.csv`)

| Fold | Test Year ($T$) | Training Years | Train Obs | Train Crises | Train Crisis % | Test Obs | Test Crises | Test Crisis % | Leakage Check |
|---:|---:|---|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 2000 | 1990–1999 (10 yrs) | 760 | 24 | 3.16% | 76 | 0 | 0.00% | **PASSED** |
| 2 | 2001 | 1990–2000 (11 yrs) | 836 | 24 | 2.87% | 76 | 1 | 1.32% | **PASSED** |
| 3 | 2002 | 1990–2001 (12 yrs) | 912 | 25 | 2.74% | 76 | 1 | 1.32% | **PASSED** |
| 4 | 2003 | 1990–2002 (13 yrs) | 988 | 26 | 2.63% | 76 | 0 | 0.00% | **PASSED** |
| 5 | 2004 | 1990–2003 (14 yrs) | 1,064 | 26 | 2.44% | 76 | 0 | 0.00% | **PASSED** |
| 6 | 2005 | 1990–2004 (15 yrs) | 1,140 | 26 | 2.28% | 76 | 0 | 0.00% | **PASSED** |
| 7 | 2006 | 1990–2005 (16 yrs) | 1,216 | 26 | 2.14% | 76 | 1 | 1.32% | **PASSED** |
| 8 | 2007 | 1990–2006 (17 yrs) | 1,292 | 27 | 2.09% | 76 | 15 | 19.74% | **PASSED** |
| 9 | 2008 | 1990–2007 (18 yrs) | 1,368 | 42 | 3.07% | 76 | 1 | 1.32% | **PASSED** |
| **Total** | **2000–2008** | **Expanding** | — | — | — | **684** | **19** | **2.78%** | **PASSED** |

### Key Pre-Fit Structural Properties
1. **Dual-Class Feasibility:** Every single training fold has at least 24 positive crisis events (and over 730 non-crisis observations). Degenerate single-class training is mathematically impossible.
2. **Cumulative Test Events:** Across the 9 test folds (684 country-year observations), there are **19 positive forward crisis events** (2.78% positive rate). This provides sufficient rare-event signal to evaluate discrimination (ROC-AUC, PR-AUC), calibration (Brier score), and operational trade-offs (sensitivity vs. false positive rate).
3. **GFC Anticipation:** Test year 2007 contains 15 positive forward labels (evaluating crisis onset in 2008, the peak of the Global Financial Crisis), providing a crucial prospective stress test.

---

## 5. Preprocessing Isolation & Leak-Proof Architecture

To prevent data leakage across temporal folds:
1. **Pipeline Encapsulation:** Preprocessing estimators (`SimpleImputer`, `StandardScaler`) are encapsulated within an `sklearn.pipeline.Pipeline` instance.
2. **Fold-Level Fitting:** In each expanding fold $T$, the pipeline is fitted exclusively on $\text{train}_T = \{ (c, t) \mid t < T \}$.
3. **Strict Out-of-Sample Transformation:** The fitted pipeline is applied directly to $\text{test}_T = \{ (c, t) \mid t = T \}$ via `.transform()` and `.predict_proba()`.
4. **Automated Verification:** The automated pre-fit test confirms that:
   $$\text{imputer.statistics\_} = \operatorname{median}(\text{train}_T)$$
   $$\text{scaler.mean\_} = \operatorname{mean}(\text{imputed\_train}_T)$$
   and verifies that test observations do not influence training statistics.

---

## 6. Post-2008 Low-Onset Regime Audit ($2009 \dots 2025$)

* **Observations:** 1,292 country-years ($17 \text{ years} \times 76 \text{ countries}$).
* **Total Positive Forward Events:** Exactly 1 (Cyprus in $t=2010$ forecasting 2011 onset; Nigeria 2009 onset corresponds to $t=2008$).
* **Years 2011–2025:** Exactly 0 positive forward events.
* **Methodological Lock:** 
  - Prospective ROC-AUC and PR-AUC cannot be computed on single-class datasets (mathematically undefined denominator).
  - Rather than fabricating an artificial post-2008 holdout through data dredging or relaxing the IMF crisis criteria, this period is treated as a **documented structural limitation** (a post-GFC low-onset regime in countries with complete World Bank reporting).
  - Out-of-sample predictions for 2009–2025 will be reported descriptively in Phase 5B focusing on false-positive rates and specificity.

---

## 7. Pre-Registered Model Specifications for Phase 5B

The models pre-registered for estimation in Phase 5B are:

1. **Baseline: Regularized Logistic Regression**
   * Preprocessing: Fold median imputation + standard scaling ($z$-score).
   * Penalty: $L_2$ (Ridge regularization), $C = 1.0$.
   * Class weighting: `balanced` (inversely proportional to class frequencies).
   * Solver: `lbfgs`, `max_iter = 2000`, `random_state = 2026`.
2. **Comparator: Constrained Random Forest**
   * Preprocessing: Fold median imputation (scaling omitted as trees are invariant to monotonic transformations).
   * Hyperparameters: `n_estimators = 400`, `max_depth = 4`, `min_samples_leaf = 5`.
   * Class weighting: `balanced`.
   * Random state: `random_state = 2026`.
3. **Decision Threshold:** Fixed at $\tau = 0.50$ (no post-hoc threshold tuning on test folds).

---

## 8. Verification & Sign-Off

* [x] **76-country panel schema verified** (2,736 rows, 44 positives).
* [x] **4 core predictors audited** (low correlation, expected missingness confined to 1990).
* [x] **9 expanding-window folds audited** (2000–2008; 684 test rows, 19 test positives; zero leakage).
* [x] **Preprocessing isolation verified** (training-only fitting confirmed).
* [x] **CLI subcommand operational** (`ews pre-fit-audit`).
* [x] **Zero models fitted, zero performance metrics calculated.**
