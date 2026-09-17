# Final Research Synthesis: Systemic Banking Crisis Early Warning System

**Project:** Global Financial Crisis Early Warning System (`global-financial-crisis-ews`)  
**Repository:** `https://github.com/ege211/global-financial-crisis-ews`  
**Evaluation Scope:** 684 Pooled Out-of-Sample Country-Years across 76 Economies (Test Folds $T \in [2000, 2008]$, 19 Crisis Onset Events)  
**Primary Research Specification:** 6-Variable Extended Logistic Regression (Regime-Qualified)  
**Status:** Complete, Audited, Verified, Fully Reproducible  

---

## 1. Executive Summary

This study investigates whether macro-financial indicators can provide reliable early warning signals for systemic banking crises. Using an objective 76-country panel spanning 1990–2025 and a strict expanding-window out-of-sample validation design ($T=2000,\dots,2008$), we evaluate linear logistic regression models and non-linear random forest ensembles across a 4-variable baseline flow specification and a pre-registered 6-variable extended balance-sheet specification.

Across the 684 pooled out-of-sample country-year observations (containing 19 forward crisis onset events), the 6-variable Extended Logistic Regression improves pooled discrimination over the 4-variable baseline (PR-AUC increases from 0.0278 to 0.0344; ROC-AUC increases from 0.5106 to 0.5734; Recall at threshold $\tau = 0.50$ increases from 21.1% to 47.4%). However, dedicated temporal-generalization and mechanism diagnostics reveal that this performance advantage is concentrated almost exclusively in the 2007 pre-GFC evaluation fold (which anticipates the 2008 trans-Atlantic crisis wave). In that single fold, the extended model detected 9 of 15 crisis economies (60.0% Recall). In contrast, outside the 2007 fold, within the four observed non-2007 evaluation events, all four events were missed at the fixed 0.50 threshold (0.0% Recall), and the model exhibited an observed ranking inversion (ROC-AUC = 0.2301), assigning lower predicted probabilities to crisis countries than to the median calm observation.

Consequently, the empirical evidence provides limited support for a broadly generalizable crisis early warning system. The extended model is retained as the Primary Research Specification solely as an interpretable diagnostic tool for a specific macro-financial archetype—namely, domestic private banking leverage accumulation accompanied by external current account deficits—rather than as a general-purpose forecasting system.

---

## 2. Research Question

> *"Can macro-financial indicators provide useful early warning signals of systemic banking crises?"*

Specifically, does augmenting standard macroeconomic flow indicators (real GDP growth, inflation, international reserves, and reserve growth) with domestic banking credit depth and external current account balances improve out-of-sample crisis detection across diverse historical episodes, or does predictive performance reflect regime-specific sensitivity to the 2008 Global Financial Crisis (GFC)?

---

## 3. Hypothesis & Economic Motivation

Systemic banking crises impose catastrophic real economic costs, often leading to protracted output losses, sovereign debt crises, and long-term stagnation. The classical economic literature (e.g., Kaminsky & Reinhart, 1999; Gourinchas & Obstfeld, 2012; Schularick & Taylor, 2012) identifies two primary macro-financial vectors of banking fragility:
1. **Domestic Credit Booms:** Rapid accumulation of private credit relative to GDP strains bank balance sheets, inflates collateral values, and lowers underwriting standards, creating vulnerability to sudden asset-price corrections.
2. **External Financing Imbalances:** Persistent current account deficits financed by short-term foreign borrowing expose domestic banking systems to sudden stops in international capital flows and currency depreciations.

We hypothesized that incorporating measures of domestic credit depth (`private_credit_pct_gdp`) and external vulnerability (`current_account_pct_gdp`) into an empirical early warning system would yield superior out-of-sample crisis discrimination relative to flow variables alone.

---

## 4. Data & Multilateral Sources

All data analyzed in this project derive strictly from authoritative, multilateral datasets:
1. **Crisis Labels:** Official IMF Systemic Banking Crises Database (Laeven & Valencia, 2026, IMF Working Paper WP/26/94). Provides binary onset chronologies for non-borderline systemic banking crises across all 76 sample economies.
2. **Macro-Financial Predictors:** World Bank World Development Indicators (WDI), collected and harmonized in Phase 4B and Phase 5D:
   - `gdp_growth`: Real GDP Growth (`NY.GDP.MKTP.KD.ZG`, annual %)
   - `inflation`: CPI Inflation Rate (`FP.CPI.TOTL.ZG`, annual %)
   - `reserves_usd`: Gross International Reserves in current USD (`FI.RES.TOTL.CD`)
   - `reserves_usd_yoy_pct_change`: Annual percentage change in gross reserves (derived from `FI.RES.TOTL.CD`)
   - `private_credit_pct_gdp`: Domestic Credit to Private Sector as % of GDP (`FS.AST.PRVT.GD.ZS`)
   - `current_account_pct_gdp`: Current Account Balance as % of GDP (`BN.CAB.XOKA.GD.ZS`)

---

## 5. Crisis Definition & Temporal Horizon

- **Systemic Banking Crisis Event:** Defined strictly by the IMF WP/26/94 chronology as the initial onset year of a non-borderline systemic banking crisis. Subsequent crisis continuation years are not treated as new onsets.
- **Prediction Target ($y_{c,t}$):** Binary indicator defined at calendar year $t$:
  $$y_{c,t} = 1 \quad \iff \quad \text{Systemic banking crisis onset occurs in country } c \text{ at calendar year } t+1$$
- **Temporal Alignment ($t \to t+1$):** Predictor vector $\mathbf{x}_{c,t}$ is measured strictly in year $t$ prior to crisis onset in year $t+1$. No contemporaneous or forward-looking information from $t+1$ enters feature construction, guaranteeing zero future leakage.

---

## 6. Research Universe

The objective sample universe consists of **76 World Bank economies** selected on data availability and macroeconomic relevance:
- **Time Horizon:** 1990–2025 (36 calendar years).
- **Total Panel Size:** 2,736 country-year observations.
- **Panel-Wide Positives:** Exactly 44 qualifying systemic banking crisis onset events across the 36-year panel (1.61% unconditional base rate).
- **Out-of-Sample Evaluation Slice ($T=2000,\dots,2008$):** Exactly 684 country-years across 76 economies, containing exactly 19 forward crisis onset events (2.78% base rate).

---

## 7. Predictor Design & Selection Audit

The predictor space was audited in Phase 5D to prevent specification search and post-hoc data mining:

| Predictor | WDI Code | Economic Mechanism | Transformation | 2000–2008 Coverage | Eligible? | Selection Decision |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| **gdp_growth** | `NY.GDP.MKTP.KD.ZG` | Macroeconomic slowdown | Raw annual % | 99.1% | YES | Locked Baseline (1) |
| **inflation** | `FP.CPI.TOTL.ZG` | Price overheating / monetary stress | Raw annual % | 98.4% | YES | Locked Baseline (2) |
| **reserves_usd** | `FI.RES.TOTL.CD` | Foreign currency liquidity buffer | Log USD level | 96.5% | YES | Locked Baseline (3) |
| **reserves_usd_yoy_pct_change** | Derived | Reserve depletion / balance of payments | Annual % change | 95.9% | YES | Locked Baseline (4) |
| **private_credit_pct_gdp** | `FS.AST.PRVT.GD.ZS` | Banking leverage accumulation | % of GDP | 87.6% | YES | Locked Extended Addition (5) |
| **current_account_pct_gdp** | `BN.CAB.XOKA.GD.ZS` | External financing vulnerability | % of GDP | 92.8% | YES | Locked Extended Addition (6) |
| **exchange_rate_depreciation** | `PA.NUS.FCRF` | Currency pressure | % change | 95.8% | NO | Excluded (Degrees of freedom) |
| **unemployment_rate_ilo** | `SL.UEM.TOTL.ZS` | Labor market slack | % of labor force | 96.8% | NO | Excluded (Degrees of freedom) |

---

## 8. Validation Strategy

To simulate realistic policymaker decision-making and prevent temporal leakage, all models were evaluated using **expanding-window out-of-sample validation**:
- **Evaluation Folds:** 9 expanding annual folds evaluated at test years $T \in \{2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008\}$.
- **Minimum Training Horizon:** 10 years ($1990 \le t \le T-1$).
- **Strict Train-Only Preprocessing:** Imputation of missing values (training fold median) and robust standardizations are fitted strictly on the training partition $[1990, T-1]$ and applied forward to test year $T$.
- **Fixed Decision Threshold:** Evaluated at $\tau = 0.50$ without threshold tuning or post-hoc threshold optimization.

---

## 9. Evaluated Model Architectures

Four distinct model specifications were implemented and evaluated:
1. **Baseline Logistic Regression (4 variables):** Linear log-odds model fitted with balanced class weights.
2. **Baseline Random Forest (4 variables):** 100 trees, maximum depth 4, balanced class weights.
3. **Extended Logistic Regression (6 variables):** Linear log-odds model fitted with balanced class weights on 6 predictors.
4. **Extended Random Forest (6 variables):** 100 trees, maximum depth 4, balanced class weights on 6 predictors.

---

## 10. Baseline Model Results

*Summary from [final_model_comparison.csv](file:///Users/macbookair/Documents/ChatGPT/finance/results/tables/final_model_comparison.csv).*

The 4-variable baseline models established the initial empirical benchmark across the 684 pooled out-of-sample observations (19 crises):
- **Baseline Logistic Regression:** PR-AUC = 0.0278 (relative to prior $\pi = 0.0278$), ROC-AUC = 0.5106, Precision = 0.0189, Recall = 21.05% (4 of 19 crises detected: DEU, ESP, FRA in 2007; GBR in 2006), F1 = 0.0346, Brier score = 0.2512, False Positives = 208.
- **Baseline Random Forest:** PR-AUC = 0.0256, ROC-AUC = 0.4712, Precision = 0.0242, Recall = 21.05% (4 of 19 detected), F1 = 0.0435, Brier score = 0.1524, False Positives = 161.

---

## 11. Extended Model Results

Augmenting the feature vector with private credit depth and current account balances produced significant improvements in pooled metrics:
- **Extended Logistic Regression:**
  - **PR-AUC:** **0.0344** (+23.7% relative improvement over baseline 0.0278).
  - **ROC-AUC:** **0.5734** (+12.3% improvement over baseline 0.5106).
  - **Recall ($\tau=0.50$):** **47.37%** (9 of 19 crises detected, more than doubling baseline recall).
  - **Precision:** **0.0459** (more than double baseline 0.0189).
  - **F1 Score:** **0.0837** (vs baseline 0.0346).
  - **Brier Score:** **0.2482** (slight improvement over baseline 0.2512).
  - **False Positives:** **187** (reduced by 21 relative to baseline 208).
- **Extended Random Forest:** PR-AUC = 0.0274, ROC-AUC = 0.4715, Recall = 10.53% (2 of 19 detected: DOM in 2002, NGA in 2008), Precision = 0.0161, Brier score = 0.1347, False Positives = 122.

---

## 12. Robustness Synthesis

*Synthesized in [final_robustness_summary.csv](file:///Users/macbookair/Documents/ChatGPT/finance/results/tables/final_robustness_summary.csv).*

Six distinct robustness dimensions were formally evaluated:

| Robustness Dimension | Baseline Specification | Robustness Alteration | Empirical Outcome | Methodological Implication |
| :--- | :--- | :--- | :--- | :--- |
| **Inflation Winsorization** | Raw CPI inflation | Winsorized at [1st, 99th] percentiles | LR PR-AUC drops from 0.0278 to 0.0219 (-21.2%); Recall drops to 5.3% (1 detection); RF unchanged (0.0256). | Linear logistic model relies on extreme inflation tails for rare-event separation; truncating tails degrades signal. |
| **Class Weighting** | Balanced class weighting | Unweighted standard loss (`None`) | Both LR and RF collapse to 0.0% Recall (0 true positives) at standard $\tau = 0.50$ cutoff. | In rare-event panels (2.7% positives), cost-sensitive loss reweighting is mandatory to prevent majority-class collapse. |
| **Threshold Sensitivity** | Standard $\tau = 0.50$ | $\tau = 0.25$ and $\tau = 0.75$ | At $\tau = 0.25$, LR Recall reaches 100% (19/19) but FP explodes to 664 (FPR 99.8%); at $\tau=0.75$, Recall drops to 0.0%. | Early warning systems face a severe trade-off; false alarms cannot be reduced without missing crises. |
| **Fold Stability** | Pooled 2000–2008 | Fold-by-fold expanding metrics | In 4 calm folds (2000, 2003, 2004, 2005), AUC is Not estimable; 15 of 19 crises cluster in fold 2007. | Out-of-sample evaluation is episodic and driven by systemic global waves rather than steady-state incidence. |
| **Predictor Expansion** | 4 flow variables | 6 variables (+credit, +CA) | Pooled PR-AUC rises from 0.0278 to 0.0344 (+23.7%); Recall surges from 21.1% to 47.4% (+125%). | Inclusion of leverage and external imbalances improves empirical discrimination over flow indicators alone. |
| **Regime Generalization** | Full panel (19 crises) | Excluding 2007 fold (4 crises) | Excluding 2007, Extended LR Recall drops to 0.0% (0/4) and ROC-AUC inverts to 0.2301; 2007 Recall is 60.0%. | Extended model is regime-dependent; it detects domestic credit booms with external deficits, but fails outside that setting. |

---

## 13. Temporal Generalization & GFC Dependence

*Results from [phase5e_temporal_generalization.csv](file:///Users/macbookair/Documents/ChatGPT/finance/results/tables/phase5e_temporal_generalization.csv).*

Phase 5E partitioned the out-of-sample evaluation into five temporal slices:

```
                                    TEMPORAL DISCRIMINATION DICHOTOMY
                     +-------------------------------------------------------------+
                     | 2007 GFC Fold (N=15 crises)                                 |
                     | Extended LR: Recall = 60.0% (9/15), PR-AUC = 0.2647         |
                     | Baseline LR: Recall = 20.0% (3/15), PR-AUC = 0.2743         |
                     +-------------------------------------------------------------+
                                                    |
                                                    v
                     +-------------------------------------------------------------+
                     | Excluding 2007 Fold (N=4 crises: URY, DOM, GBR, NGA)        |
                     | Extended LR: Recall = 0.0% (0/4),   ROC-AUC = 0.2301        |
                     | Baseline LR: Recall = 25.0% (1/4),  ROC-AUC = 0.3845        |
                     +-------------------------------------------------------------+
```

1. **The 2007 Pre-GFC Epicenter Fold ($t=2007 \to 2008$ crises, 15 events):** Extended Logistic Regression achieves PR-AUC = 0.2647, ROC-AUC = 0.6470, Precision = 31.0%, and Recall = **60.0%** (9 of 15 detected: DNK, ESP, FRA, GRC, HUN, IRL, ISL, ITA, PRT).
2. **Excluding 2007 Fold ($N=4$ crises: URY 2001, DOM 2002, GBR 2006, NGA 2008):** Extended Logistic Regression achieves PR-AUC = 0.0051, Recall = **0.0%** (0 of 4 detected), and ROC-AUC = **0.2301**.
3. **Statistical Inversion Mechanism:** In the non-crisis panel ($N=665$), median predicted probability is **0.4733** (mean = 0.4874). Within the four observed non-2007 crisis events, median predicted probability is **0.4411** (mean = 0.4109; URY 0.4264, DOM 0.4635, GBR 0.4559, NGA 0.2978). Because the model assigned systematically lower probabilities to non-2007 crisis events than to calm panel observations, an observed ranking inversion occurred (ROC-AUC = 0.2301).

---

## 14. Crisis Mechanism Analysis Summary

*Findings from [phase6_mechanism_categorization.csv](file:///Users/macbookair/Documents/ChatGPT/finance/results/tables/phase6_mechanism_categorization.csv) and [phase6_crisis_event_profiles.csv](file:///Users/macbookair/Documents/ChatGPT/finance/results/tables/phase6_crisis_event_profiles.csv).*

Phase 6 established a strict 3-layer architecture:
- **Layer A (Authoritative Ground Truth):** IMF WP/26/94 confirms all 19 episodes as non-borderline systemic banking crises, but provides no official sub-mechanism taxonomy. Sub-mechanisms are authoritatively recorded as **"Unknown / insufficient evidence"**.
- **Layer B (Data-Driven Descriptive Typology at $t$):**
  - **Descriptive Profile A (High credit / external deficit, $N=9$ detected in 2007):** Median private credit = 142.2% of GDP, median current account = -7.29% of GDP. All predicted probabilities $\ge 0.50$ (median 0.5222).
  - **Descriptive Profile B (High current-account surplus, $N=6$ missed in 2007: AUT, BEL, CHE, DEU, NLD, SWE):** Median credit = 103.4% of GDP, median current account = **+6.19% of GDP**. Every missed 2007 country ran a current account surplus, keeping risk scores in the narrow range $[0.4445, 0.4795]$.
  - **Descriptive Profile C (Lower credit / higher inflation / weaker reserves, $N=4$ non-2007):** Median private credit = 53.9% of GDP, median inflation = 4.79%, median reserve growth = +5.59%. Predicted probabilities in $[0.2978, 0.4635]$ (median 0.4411).
- **Layer C (Non-Causal Economic Hypotheses):** The model's large positive private credit weight and negative current account weight explain why it responded strongly to Profile A while assigning lower risk probabilities to Profiles B and C.

---

## 15. Final Model Selection

*Comparison synthesis from [final_model_comparison.csv](file:///Users/macbookair/Documents/ChatGPT/finance/results/tables/final_model_comparison.csv).*

We evaluated the four architectures across 10 multi-criteria dimensions:

| Evaluation Dimension | Baseline LR | Extended LR | Baseline RF | Extended RF |
| :--- | :--- | :--- | :--- | :--- |
| **1. Out-of-Sample Discrimination** | Weak (PR-AUC 0.0278, ROC-AUC 0.5106) | **Superior** (PR-AUC 0.0344, ROC-AUC 0.5734) | Weak (PR-AUC 0.0256, ROC-AUC 0.4712) | Weak (PR-AUC 0.0274, ROC-AUC 0.4715) |
| **2. Rare-Event Detection (Recall)** | 21.05% (4/19) | **Highest** (47.37%, 9/19) | 21.05% (4/19) | Poor (10.53%, 2/19; 13.3% 2007, 0.0% non-2007) |
| **3. False Alarm Burden** | High (208 FPs, Prec 1.89%) | **Moderate** (187 FPs, Prec 4.59%) | Moderate (161 FPs, Prec 2.42%) | Lowest (122 FPs, Prec 1.61%) |
| **4. Temporal Generalization** | Weak (25% non-2007 recall) | **Regime-Dependent** (0% non-2007 recall) | Weak (25% non-2007 recall) | Poor (0% non-2007 recall, 0/4) |
| **5. Calibration & Probability Behavior** | Linear log-odds (Brier 0.2512) | Linear log-odds (Brier 0.2482) | Conservative (Brier 0.1524) | Compressed (Brier 0.1347) |
| **6. Interpretability** | High (linear coefficients) | **High** (linear log-odds) | Moderate (tree importances) | Moderate (tree importances) |
| **7. Parsimony** | **High** (4 parameters) | High (6 parameters) | Low (ensemble of 100 trees) | Low (ensemble of 100 trees) |
| **8. Stability Across Folds** | Moderate | Moderate | Moderate | Moderate |
| **9. Economic Alignment** | Flow indicators only | **Balance-sheet + flows** | Complex non-linear | Complex non-linear |
| **10. Final Selection Status** | Baseline Benchmark | **Primary Research Model (Qualified)** | Nonlinear Benchmark | Exploratory ML Specification |

### Final Selection Determination:
**Extended Logistic Regression is retained as the Primary Research Specification (with explicit regime qualifications).**
- **Rationale:** Extended Logistic Regression achieves the strongest pooled discrimination (PR-AUC 0.0344, Recall 47.4%) and highest precision (4.59%) while maintaining full parametric transparency. Random Forest models achieved lower Brier scores solely by compressing predicted probabilities toward zero, missing 17 of 19 crises.
- **Explicit Boundary Qualification:** Extended Logistic Regression is **not** selected as a general-purpose crisis forecasting model. Its selection is strictly conditioned on the finding that its predictive power is concentrated in the 2007 pre-GFC episode.

---

## 16. Coefficient Interpretation for Primary Specification

*Estimates from [final_model_coefficients.csv](file:///Users/macbookair/Documents/ChatGPT/finance/results/tables/final_model_coefficients.csv).*

Parameter estimates from the final expanding training fold ($T=2008$, trained on 1990–2007):

| Feature | Feature Label | Coeff ($\beta$) | Odds Ratio | Direction | RF Importance | Cautious Associative Interpretation |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `private_credit_pct_gdp` | Private Credit (% GDP) | **+0.5142** | **1.6723** | Positive | 22.5% | Holding other included predictors constant, the fitted model associates a 1 SD increase in private credit/GDP with a 67.2% increase in crisis onset odds. |
| `reserves_usd` | Gross Reserves (USD Level) | **-0.1708** | **0.8430** | Negative | 13.3% | Holding other included predictors constant, the fitted model associates larger reserve stocks with a 15.7% decrease in crisis onset odds. |
| `gdp_growth` | Real GDP Growth (%) | **-0.1277** | **0.8801** | Negative | 9.0% | Holding other included predictors constant, the fitted model associates higher GDP growth with a 12.0% decrease in crisis onset odds. |
| `inflation` | CPI Inflation (%) | **+0.0963** | **1.1011** | Positive | 18.3% | Holding other included predictors constant, the fitted model associates higher inflation with a 10.1% increase in crisis onset odds. |
| `current_account_pct_gdp` | Current Account (% GDP) | **-0.0463** | **0.9547** | Negative | 11.1% | Holding other included predictors constant, the fitted model associates larger current account surpluses with a 4.5% decrease in crisis onset odds (deficits elevate odds). |
| `reserves_usd_yoy_pct_change`| Reserves Growth (%) | **-0.0449** | **0.9561** | Negative | 25.8% | Holding other included predictors constant, the fitted model associates positive reserve growth with a 4.4% decrease in crisis onset odds. |

*Note: All statements describe statistical associations in the fitted observational model; zero causal effects are implied.*

---

## 17. Main Empirical Findings

1. **Value of Balance-Sheet Indicators:** Adding domestic private credit and current account balances to macroeconomic flow variables improves pooled out-of-sample PR-AUC by +23.7% and doubles Recall from 21.1% to 47.4%.
2. **Dominance of Private Credit in Model Weights:** Private credit-to-GDP emerged as the largest positive coefficient ($\beta = +0.5142, \text{OR} = 1.6723$), consistent with the historical credit-boom literature.
3. **Temporal Concentration in the 2007 Pre-GFC Fold:** 15 of 19 out-of-sample crisis events (78.9%) occurred in the 2007 fold. In that fold, the extended model detected 9 of 15 crisis economies (60.0% Recall).
4. **Surplus Damping Effect:** In the 2007 fold, all 6 missed economies ran current account surpluses (+1.50% to +8.63%), keeping their risk scores below 0.50 despite high banking sector exposure to US securitized assets.
5. **Observed Ranking Inversion Outside 2007:** Within the four observed non-2007 evaluation events, all four events were missed at $\tau = 0.50$, and predicted probabilities fell below the median of the calm non-crisis panel (ROC-AUC = 0.2301).

---

## 18. Failure Analysis

A rigorous early warning evaluation must document where and why the model fails:
1. **Severe Class Imbalance:** Systemic banking crises are rare events (2.78% in evaluation sample). Unweighted training collapses completely, and balanced models face severe precision penalties.
2. **High False-Positive Burden:** At $\tau = 0.50$, Extended Logistic Regression generates 187 false alarms across 684 evaluations, yielding a precision of only 4.59% (approx. 21 false alarms per true detection).
3. **Regime Fragility Outside 2007:** The model fails to detect non-credit-boom crises (such as terms-of-trade collapses, bank embezzlement, or currency contagion).
4. **Probability Compression in Tree Ensembles:** Random Forests compress probabilities toward the empirical base rate, rendering standard decision cutoffs ineffective.
5. **Small Evaluation Sample Size ($N=4$ outside 2007):** The empirical evidence outside 2007 is limited to four observations, preventing broad statistical generalizations across non-GFC crisis types.

---

## 19. Methodological Limitations

1. **No Causal Identification:** Observational panel associations do not establish that private credit expansions caused systemic crises.
2. **Severe Small-Sample Limitation ($N=4$ non-2007):** Findings outside 2007 apply strictly to the four observed episodes in the evaluation panel: Uruguay (evaluated $t=2001 \to \text{onset } 2002$), Dominican Republic (evaluated $t=2002 \to \text{onset } 2003$), United Kingdom (evaluated $t=2006 \to \text{onset } 2007$), and Nigeria (evaluated $t=2008 \to \text{onset } 2009$).
3. **Aggregate Annual Reporting Lags:** Annual World Bank WDI indicators cannot capture rapid intra-year liquidity freezes (e.g., Northern Rock in September 2007).
4. **Omission of Interconnectedness & Off-Balance-Sheet Exposures:** Macroeconomic balance sheets do not measure cross-border interbank exposures or special investment vehicles (SIVs) that triggered banking distress in surplus economies like Germany and Switzerland.
5. **No 2026 Real-Time Forecasting Claim:** This study evaluates historical performance from 2000–2008 and does not claim real-time operational validity for modern banking environments.

---

## 20. Research Contribution

Despite finding limited generalizability, this research provides several contributions:
1. **Reproducible End-to-End Pipeline:** Fully scripted, deterministic pipeline from raw WDI/IMF sources to final evaluation.
2. **Strict Out-of-Sample Protocol:** Expanding-window evaluation with train-only preprocessing, preventing temporal leakage.
3. **Methodological Transparency on Rare Events:** Demonstrates why standard metrics (ROC-AUC, Brier score) can be misleading in rare-event settings without Precision-Recall analysis.
4. **Rigorous Regime-Dependence Diagnosis:** Unmasks how pooled performance gains can be driven almost entirely by a single historical crisis cluster.
5. **Non-Causal Separation Framework:** Establishes a template separating authoritative data, descriptive typologies, and economic hypotheses.

---

## 21. Reproducibility Checklist

- [x] **Data Provenance:** Raw WDI and IMF WP/26/94 datasets documented and archived.
- [x] **Panel Construction:** Deterministic country-year panel construction (`data/processed/expanded_research/modeling_panel.csv`).
- [x] **Target Alignment:** Binary crisis onset strictly aligned at $t+1$ with zero future leakage.
- [x] **Train-Only Preprocessing:** Imputations and standardizations fitted strictly on training partition $[1990, T-1]$.
- [x] **CLI Automation:** Entire pipeline reproducible via command line:
  ```bash
  .venv/bin/python -m crisis_ews.cli run-final-synthesis
  .venv/bin/pytest -v
  .venv/bin/ruff check .
  ```
- [x] **Artifact Immutability:** Baseline predictions (`logistic_regression_predictions.csv`), extended predictions (`extended_logistic_regression_predictions.csv`), and Phase 5/6 tables remain byte-identical.

---

## 22. Final Research Conclusion

> *"The observed predictive advantage of the extended specification is concentrated in the 2007 pre-GFC evaluation fold and appears to reflect a particular macro-financial configuration rather than a broadly generalizable crisis signal."*

The empirical findings provide limited support for a broadly generalizable crisis early warning signal. While the 6-variable Extended Logistic Regression improves pooled discrimination over the 4-variable baseline (PR-AUC 0.0344 vs. 0.0278, Recall 47.4% vs. 21.1%), this gain is concentrated in the 2007 pre-GFC evaluation fold (60.0% Recall) and does not generalize across the four observed non-2007 crisis episodes (0.0% Recall, ROC-AUC = 0.2301). The model functions as an informative diagnostic tool for detecting credit-expansion and external-deficit vulnerabilities characteristic of the 2008 trans-Atlantic crisis wave, but does not provide a robust general-purpose early warning system for disparate crisis mechanisms.
