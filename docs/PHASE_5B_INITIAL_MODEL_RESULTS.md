# Phase 5B: Initial Model Results & Honest Out-of-Sample Baseline

**Date:** 2026-09-04  
**Project:** Global Financial Crisis Early Warning System (`global-financial-crisis-ews`)  
**Status:** Initial Model Estimation Complete. Pre-registered protocol evaluated out-of-sample.

---

## 1. Research Question & Working Hypothesis

**Research Question:** Can a parsimonious, economically motivated set of annual macro-financial indicators provide useful early-warning discrimination for non-borderline systemic banking crisis onsets occurring in the subsequent calendar year ($t \to t+1$)?

**Working Hypothesis (Associative, Non-Causal):**
Systemic banking crises are preceded by macroeconomic deterioration and external balance pressure:
* Real GDP deceleration or contraction ($\partial \text{risk} / \partial \text{growth} < 0$).
* Elevated consumer price inflation eroding asset quality and purchasing power ($\partial \text{risk} / \partial \text{inflation} > 0$).
* Depletion of foreign exchange reserves signalling currency vulnerability and external illiquidity ($\partial \text{risk} / \partial \Delta\% \text{reserves} < 0$).

---

## 2. Data & Country Universe

* **Country Universe:** Exactly 76 economies selected objectively by the pre-specified complete-core data coverage rule (1990–2025) across all 217 candidate World Bank non-aggregate economies. No country was added or removed based on crisis incidence or model performance.
* **Observation Period:** 1990–2025 (36 annual periods per economy).
* **Modeling Panel:** Exactly 76 countries × 36 years = **2,736 country-year rows** in `data/processed/expanded_research/modeling_panel.csv`.
* **Missingness:** Zero missing values for `gdp_growth`, `inflation`, and `reserves_usd`. Missingness in `reserves_usd_yoy_pct_change` is strictly confined to the initial year 1990 across all 76 countries (2.78%) due to the backward lag operator ($t-1 = 1989$), and is handled via fold-fitted median imputation.

---

## 3. Pre-Registered Feature Set

$$\mathbf{x}_{c, t} = \begin{bmatrix} \text{GDP Growth}_{c, t} \\ \text{Inflation}_{c, t} \\ \text{Reserves USD}_{c, t} \\ \Delta\% \text{Reserves}_{c, t} \end{bmatrix}$$

1. `gdp_growth`: Real GDP growth (annual %) from World Bank WDI (`NY.GDP.MKTP.KD.ZG`).
2. `inflation`: Inflation, consumer prices (annual %) from World Bank WDI (`FP.CPI.TOTL.ZG`).
3. `reserves_usd`: Total foreign exchange reserves in current USD (`FI.RES.TOTL.CD`).
4. `reserves_usd_yoy_pct_change`: Percentage change in total reserves from $t-1$ to $t$ ($\frac{\text{reserves}_t}{\text{reserves}_{t-1}} - 1$).

---

## 4. Supervised Target & Timing Contract

* **Target Definition:** $y(c, t) = 1$ if a qualifying systemic banking crisis onset starts in calendar year $t+1$, and $0$ otherwise.
* **Chronology:** Laeven & Valencia (2026), IMF Working Paper WP/26/94 (*Systemic Banking Crises Database: 1970–2025*), excluding author-identified borderline episodes (Nicaragua 2018, Sri Lanka 2023, Vietnam 2022).
* **Lookahead Prohibition:** Features observed in crisis onset year $t+1$ are strictly excluded from predicting year $t+1$. Features at year $t$ precede the crisis start.
* **Sample Labels:** Exactly 44 positive forward crisis events and 2,692 negative observations across the entire 1990–2025 panel (unconditional positive rate: 1.608%).

---

## 5. Locked Temporal Validation Design

* **Protocol:** Historical expanding-window annual evaluation over test years $T \in [2000, 2008]$ (9 evaluation folds).
* **Minimum Training Period:** 10 years ($1990 \le \text{year} \le 1999$).
* **Training Monotonicity:** In each fold $T$, the training set contains exclusively observations with $\text{year} \le T - 1$. Preprocessing (median imputation and standardization) is fitted solely on training rows.
* **Out-of-Sample Test Pool:** Exactly 9 test folds × 76 countries = **684 country-year test observations**.
* **Test Event Count:** Exactly **19 positive forward crisis events** across the test pool (test positive rate: 2.778%).
* **Post-2008 Structural Regime:** Documented as a low-onset regime (only 1 positive forward event across 2009–2025: Cyprus in $t=2010$). No artificial holdout was manufactured.

---

## 6. Model Specifications

### Baseline: Regularized Logistic Regression
* **Pipeline:** Fold-level `SimpleImputer(strategy='median')` $\to$ Fold-level `StandardScaler()` $\to$ `LogisticRegression()`.
* **Hyperparameters:**
  * Penalty: $L_2$ (Ridge regularization), $C = 1.0$.
  * Class Weight: `balanced` (inversely proportional to class frequencies).
  * Solver: `lbfgs`, `max_iter = 2000`, `random_state = 2026`.
* **Decision Threshold:** Fixed at $\tau = 0.50$ (unweighted probability cutoff).

### Comparator: Constrained Random Forest
* **Pipeline:** Fold-level `SimpleImputer(strategy='median')` $\to$ `RandomForestClassifier()`.
* **Hyperparameters:**
  * Estimators: `n_estimators = 400`.
  * Tree Depth: `max_depth = 4` (constrained to limit overfitting on rare events).
  * Leaf Size: `min_samples_leaf = 5`.
  * Class Weight: `balanced`.
  * Random State: `random_state = 2026`.
* **Decision Threshold:** Fixed at $\tau = 0.50$.

---

## 7. Out-of-Sample Performance Results

### A. Pooled Out-of-Sample Comparison (2000–2008, $N=684$)

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 Score | Brier Score | Positive Events | Test Observations | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Logistic Regression** | **0.0278** | **0.5106** | 0.0189 | 0.2105 | 0.0346 | 0.2512 | 19 | 684 | 4 | 208 | 15 | 457 |
| **Random Forest** | **0.0256** | **0.4712** | 0.0242 | 0.2105 | 0.0435 | 0.1524 | 19 | 684 | 4 | 161 | 15 | 504 |
| *Baseline / Climatology* | *0.0278* | *0.5000* | *0.0278* | *—* | *—* | *0.0270* | *19* | *684* | *—* | *—* | *—* | *—* |

### B. Per-Fold Out-of-Sample Breakdown: Logistic Regression

| Test Year ($T$) | Train Window | Test Obs | Crises ($y=1$) | ROC-AUC | PR-AUC | Precision | Recall | F1 | Brier Score | TP | FP | FN | TN |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2000 | 1990–1999 | 76 | 0 | *Not estimable* | *Not estimable* | 0.0000 | 0.0000 | 0.0000 | 0.2502 | 0 | 21 | 0 | 55 |
| 2001 | 1990–2000 | 76 | 1 | 0.0800 | 0.0143 | 0.0000 | 0.0000 | 0.0000 | 0.2449 | 0 | 16 | 1 | 59 |
| 2002 | 1990–2001 | 76 | 1 | 0.3200 | 0.0192 | 0.0000 | 0.0000 | 0.0000 | 0.2524 | 0 | 19 | 1 | 56 |
| 2003 | 1990–2002 | 76 | 0 | *Not estimable* | *Not estimable* | 0.0000 | 0.0000 | 0.0000 | 0.2525 | 0 | 25 | 0 | 51 |
| 2004 | 1990–2003 | 76 | 0 | *Not estimable* | *Not estimable* | 0.0000 | 0.0000 | 0.0000 | 0.2529 | 0 | 27 | 0 | 49 |
| 2005 | 1990–2004 | 76 | 0 | *Not estimable* | *Not estimable* | 0.0000 | 0.0000 | 0.0000 | 0.2532 | 0 | 32 | 0 | 44 |
| 2006 | 1990–2005 | 76 | 1 | 0.8667 | 0.0909 | 0.0667 | 1.0000 | 0.1250 | 0.2416 | 1 | 14 | 0 | 61 |
| 2007 | 1990–2006 | 76 | 15 | 0.6590 | 0.2743 | 0.2308 | 0.2000 | 0.2143 | 0.2405 | 3 | 10 | 12 | 51 |
| 2008 | 1990–2007 | 76 | 1 | 0.2267 | 0.0169 | 0.0000 | 0.0000 | 0.0000 | 0.2723 | 0 | 44 | 1 | 31 |

### C. Per-Fold Out-of-Sample Breakdown: Constrained Random Forest

| Test Year ($T$) | Train Window | Test Obs | Crises ($y=1$) | ROC-AUC | PR-AUC | Precision | Recall | F1 | Brier Score | TP | FP | FN | TN |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2000 | 1990–1999 | 76 | 0 | *Not estimable* | *Not estimable* | 0.0000 | 0.0000 | 0.0000 | 0.0928 | 0 | 10 | 0 | 66 |
| 2001 | 1990–2000 | 76 | 1 | 0.8267 | 0.0714 | 0.0000 | 0.0000 | 0.0000 | 0.1007 | 0 | 12 | 1 | 63 |
| 2002 | 1990–2001 | 76 | 1 | 0.5867 | 0.0312 | 0.0000 | 0.0000 | 0.0000 | 0.1318 | 0 | 18 | 1 | 57 |
| 2003 | 1990–2002 | 76 | 0 | *Not estimable* | *Not estimable* | 0.0000 | 0.0000 | 0.0000 | 0.1196 | 0 | 17 | 0 | 59 |
| 2004 | 1990–2003 | 76 | 0 | *Not estimable* | *Not estimable* | 0.0000 | 0.0000 | 0.0000 | 0.1391 | 0 | 19 | 0 | 57 |
| 2005 | 1990–2004 | 76 | 0 | *Not estimable* | *Not estimable* | 0.0000 | 0.0000 | 0.0000 | 0.1319 | 0 | 18 | 0 | 58 |
| 2006 | 1990–2005 | 76 | 1 | 0.1733 | 0.0159 | 0.0000 | 0.0000 | 0.0000 | 0.1851 | 0 | 24 | 1 | 51 |
| 2007 | 1990–2006 | 76 | 15 | 0.3454 | 0.1554 | 0.1304 | 0.2000 | 0.1579 | 0.2713 | 3 | 20 | 12 | 41 |
| 2008 | 1990–2007 | 76 | 1 | 0.7067 | 0.0435 | 0.0417 | 1.0000 | 0.0800 | 0.1992 | 1 | 23 | 0 | 52 |

---

## 8. Out-of-Sample Calibration Assessment

`results/tables/expanded_calibration_summary.csv`:

| Model | Probability Bin | Count | Crises | Mean Pred Prob | Empirical Crisis Rate | Calibration Error |
|---|---|---:|---:|---:|---:|---:|
| **Logistic Regression** | $[0.00, 0.20)$ | 0 | 0 | — | — | — |
| | $[0.20, 0.50)$ | 472 | 15 | 0.4701 | 0.0318 | 0.4383 |
| | $[0.50, 1.00]$ | 212 | 4 | 0.5534 | 0.0189 | 0.5346 |
| **Random Forest** | $[0.00, 0.05)$ | 128 | 1 | 0.0254 | 0.0078 | 0.0176 |
| | $[0.05, 0.10)$ | 116 | 6 | 0.0699 | 0.0517 | 0.0182 |
| | $[0.10, 0.20)$ | 91 | 5 | 0.1462 | 0.0549 | 0.0913 |
| | $[0.20, 0.50)$ | 184 | 3 | 0.3660 | 0.0163 | 0.3497 |
| | $[0.50, 1.00]$ | 165 | 4 | 0.6269 | 0.0242 | 0.6027 |

### Calibration Findings & Mathematical Rationale
1. **Balanced Class Weighting Distortion:** Because both models were estimated with `class_weight="balanced"` to counteract extreme class imbalance ($~2.8\%$ event rate), the raw probability outputs are shifted upward toward $\approx 0.50$.
2. **Probability Clustering:** In Logistic Regression, all predictions fall in $[0.20, 1.00]$, with a mean predicted probability of $0.495$ compared to an empirical base rate of $0.0278$.
3. **Brier Skill Score:**
   * Reference Climatology Brier: $\text{Brier}_{\text{ref}} = \bar{y}(1 - \bar{y}) = 0.027778 \times 0.972222 = 0.027007$.
   * Logistic Regression Brier: $0.2512$ ($\text{BSS} = -8.30$).
   * Random Forest Brier: $0.1524$ ($\text{BSS} = -4.64$).
   * The negative Brier skill scores are the direct mathematical consequence of evaluating probability scores shifted by balanced inverse weights against an unweighted, rare-event outcome.
4. **Small-Sample Warning:** With only 19 cumulative positive test events across 9 years, empirical binning in the upper probability deciles is subject to high small-sample variance.

---

## 9. Feature Interpretation (Model-Based, Non-Causal)

`results/tables/expanded_feature_interpretations.csv`:

### Logistic Regression Coefficients across Expanding Training Folds
(Standardized coefficients $\beta_j$, representing change in log-odds per 1 standard deviation increase in the fold-standardized predictor):

| Test Year ($T$) | Train End Year | GDP Growth $\beta$ | Inflation $\beta$ | Reserves (USD) $\beta$ | Reserves YoY $\Delta\% \beta$ |
|---:|---:|---:|---:|---:|---:|
| 2000 | 1999 | +0.0528 | +0.2304 | +0.2164 | -0.0529 |
| 2001 | 2000 | +0.0336 | +0.2407 | +0.2099 | -0.0280 |
| 2002 | 2001 | -0.0120 | +0.2392 | +0.1870 | -0.0079 |
| 2003 | 2002 | +0.0151 | +0.2522 | +0.1592 | -0.1280 |
| 2004 | 2003 | +0.0134 | +0.2617 | +0.1398 | -0.1364 |
| 2005 | 2004 | -0.0021 | +0.2693 | +0.1094 | -0.1488 |
| 2006 | 2005 | -0.0055 | +0.2741 | +0.0720 | -0.1489 |
| 2007 | 2006 | -0.0349 | +0.2717 | +0.0619 | -0.1682 |
| **2008** | **2007** | **-0.1125** | **+0.1778** | **+0.0990** | **-0.0704** |

### Final Fold ($T=2008$, trained 1990–2007) Odds Ratios & Economic Interpretation:
* **GDP Growth ($\beta = -0.1125, \text{OR} = 0.8936$):** Each 1-SD increase in GDP growth is associated with an approximate $10.6\%$ decrease in subsequent crisis odds ($\text{OR} < 1.0$), matching the counter-cyclical vulnerability hypothesis ($\partial \text{risk} / \partial \text{growth} < 0$).
* **Inflation ($\beta = +0.1778, \text{OR} = 1.1946$):** Each 1-SD increase in inflation is associated with an approximate $19.5\%$ increase in subsequent crisis odds ($\text{OR} > 1.0$), matching the macroeconomic overheating hypothesis ($\partial \text{risk} / \partial \text{inflation} > 0$).
* **Reserves YoY % Change ($\beta = -0.0704, \text{OR} = 0.9320$):** Reserve accumulation decreases crisis odds, while reserve drawdown increases crisis odds ($\text{OR} < 1.0$), matching the external buffer depletion hypothesis ($\partial \text{risk} / \partial \Delta\% \text{reserves} < 0$).
* **Reserves USD ($\beta = +0.0990, \text{OR} = 1.1041$):** Positive association reflecting scale effects (larger economies with larger nominal reserve stocks experiencing banking distress during the GFC).

### Random Forest Feature Importance
In the final training fold ($T=2008$):
1. **Reserves YoY % Change:** $41.95\%$
2. **Inflation:** $23.01\%$
3. **Reserves USD:** $22.33\%$
4. **GDP Growth:** $12.71\%$

*Methodological Caution:* These coefficients and importances reflect model-based correlational patterns within the sample. They are **not causal estimates**, and no claim of formal statistical significance is made in the absence of asymptotic or clustered standard error computations.

---

## 10. Methodological Discussion & Conservative Scientific Interpretation

1. **Neither Model Outperforms Baseline Discrimination:**
   * Logistic Regression achieves a pooled ROC-AUC of **0.5106** and a PR-AUC of **0.0278** (matching the uninformative baseline event rate of $19/684 = 0.0278$).
   * Constrained Random Forest achieves a pooled ROC-AUC of **0.4712** and a PR-AUC of **0.0256** (slightly below the unconditional base rate).
   * Neither model demonstrates robust discrimination over the 2000–2008 backtest as a whole.
2. **Crisis Anticipation in GFC (Test Year 2007):**
   * Both models detected only 3 of the 15 crisis countries in test year 2007 at the fixed $\tau=0.50$ threshold (Recall = $20.0\%$, False Negative Rate = $80.0\%$).
   * Logistic Regression achieved an intra-fold ROC-AUC of $0.6590$ and PR-AUC of $0.2743$ (above the fold base rate of $0.1974$) in 2007, indicating modest directional ranking ability during the systemic shock, but at the expense of a high false alarm rate ($10$ false positives out of $61$ non-crisis countries).
3. **High False Positive Burden:**
   * At threshold $\tau=0.50$, Logistic Regression produced **208 false alarms** to capture 4 true crises (Precision = $1.89\%$).
   * Constrained Random Forest produced **161 false alarms** to capture 4 true crises (Precision = $2.42\%$).
   * In a policy early-warning context, such high noise-to-signal ratios would trigger systemic policy fatigue.
4. **Conclusion on Out-of-Sample Performance:**
   * These initial results honestly document that four basic, annual macroeconomic indicators alone are insufficient to reliably anticipate systemic banking crises out-of-sample under a strict expanding-window design.
   * This finding aligns with leading academic literature (e.g. Reinhart & Rogoff, 2009; Schularick & Taylor, 2012), which stresses that credit growth, leverage, asset prices, and banking-system maturity mismatches—rather than standard flow macroeconomic variables alone—are the primary drivers of systemic financial fragility.

---

## 11. Study Limitations

1. **Rare Event Rarity:** 19 crisis events across 684 test observations limits statistical power. Four evaluation folds (2000, 2003, 2004, 2005) contained zero positive events, rendering individual fold ROC-AUC mathematically undefined.
2. **Post-2008 Low-Event Regime:** The post-2008 era exhibits near-zero crisis incidence in countries with complete World Bank data, preventing prospective holdout evaluation on recent years without relaxing definitions.
3. **Annual Aggregation:** Annual data obscures sub-annual bank liquidity runs and interbank contagion.
4. **Data Revisions:** World Bank historical values reflect post-hoc revisions rather than real-time vintage data available to policymakers contemporaneously.

---

## 12. Research-Integrity Sign-Off

- [x] **Pre-model snapshot recorded** (commit `0c4d118`, 76 countries, 2,736 rows, 44 labels).
- [x] **Zero hyperparameter search performed** (fixed pre-registered values used throughout).
- [x] **Zero threshold optimization performed** ($\tau = 0.50$ fixed across all folds).
- [x] **Zero data leakage** ($\max(\text{train\_year}) < \min(\text{test\_year})$, train-only preprocessing verified).
- [x] **Full 42-test test suite passing** (`.venv/bin/pytest -v`).
- [x] **Zero code lint errors** (`.venv/bin/ruff check src tests`).
- [x] **Results reported conservatively and honestly** with zero cherry-picking.
