# Phase 5E: Out-of-Sample GFC Dependence and Temporal Generalization Diagnostic Analysis

**Project:** Global Financial Crisis Early Warning System (`global-financial-crisis-ews`)  
**Scope:** Post-Estimation Diagnostic Assessment (No Retraining, No Threshold Tuning)  
**Evaluation Period:** 2000–2008 (684 Pooled Out-of-Sample Country-Years across 76 Economies)  
**Evaluated Specification:** 6-Variable Extended Logistic Regression vs 4-Variable Baseline Logistic Regression  

---

## 1. Research Question

> *"Is the apparent out-of-sample predictive improvement of the 6-variable Extended Logistic Regression over the 4-variable baseline a temporally robust, general early warning signal, or is it an artifact of performance gains concentrated exclusively in the 2007 pre-GFC episode?"*

In Phase 5D, augmenting the 4-variable macroeconomic baseline (`gdp_growth`, `inflation`, `reserves_usd`, `reserves_usd_yoy_pct_change`) with `private_credit_pct_gdp` and `current_account_pct_gdp` produced an apparent surge in pooled performance:
- PR-AUC increased from $0.0278$ to **$0.0344$** ($+23.7\%$).
- ROC-AUC increased from $0.5106$ to **$0.5734$** ($+0.0628$).
- Crisis detection recall more than doubled from $21.1\%$ ($4/19$) to **$47.4\%$ ($9/19$)** at the locked $\tau = 0.50$ threshold.

However, an audit of the temporal distribution of labels reveals that **15 of the 19 pooled out-of-sample crisis events** occur in a single test fold: **Test Year 2007** (forecasting systemic banking crises in 2008). 

To prevent misleading claims of general forecasting capability, this diagnostic isolates the performance of the Extended and Baseline models across multiple temporal sub-periods without altering predictions, modifying features, or tuning decision thresholds.

---

## 2. Methodology

This analysis performs a strict, non-destructive leave-fold-slice diagnostic on the existing out-of-sample predictions:
1. **Zero Retraining or Parameter Optimization:** Models are not refitted. The exact out-of-sample predictions generated under the locked expanding-window protocol ($T=2000,\dots,2008$) are preserved verbatim.
2. **Fixed Decision Threshold:** The pre-registered classification threshold $\tau = 0.50$ is held constant across all subsets.
3. **Temporal Slices Evaluated:**
   - **Subset 1: Pooled Out-of-Sample Reference (2000–2008):** All 9 test folds ($N = 684$, 19 crises, unconditional prevalence $= 2.78\%$).
   - **Subset 2: Excluding 2007 Test Fold (2000–2006, 2008):** 8 test folds ($N = 608$, 4 crises, unconditional prevalence $= 0.66\%$).
   - **Subset 3: Pre-GFC Historical Period (2000–2006):** 7 test folds ($N = 532$, 3 crises, unconditional prevalence $= 0.56\%$).
   - **Subset 4: 2007 GFC Fold (Forecasting 2008):** 1 test fold ($N = 76$, 15 crises, unconditional prevalence $= 19.74\%$).
   - **Subset 5: 2008 Test Fold (Forecasting 2009):** 1 test fold ($N = 76$, 1 crisis, unconditional prevalence $= 1.32\%$).
4. **Primary Comparison Models:**
   - Extended Logistic Regression (6 variables: flow baseline + credit-to-GDP + current account).
   - Baseline Logistic Regression (4 flow variables).
5. **Metric Convention:** Where a test slice contains zero positive events ($N_{\text{crisis}} = 0$), ranking metrics dependent on class separation are reported strictly as `"Not estimable."`

---

## 3. Analysis 1: Pooled Out-of-Sample Reference (2000–2008)

Evaluating across all 9 expanding folds ($684$ observations, 19 crisis events):

| Metric | Extended Logistic Regression | Baseline Logistic Regression | Absolute Difference ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Observations** | 684 | 684 | — |
| **Crisis Events** | 19 | 19 | — |
| **PR-AUC** | **0.0344** | 0.0278 | $+0.0066$ ($+23.7\%$) |
| **ROC-AUC** | **0.5734** | 0.5106 | $+0.0628$ ($+12.3\%$) |
| **Precision** | **0.0459** | 0.0189 | $+0.0270$ ($+142.9\%$) |
| **Recall (TPR)** | **0.4737** ($9/19$) | 0.2105 ($4/19$) | $+0.2632$ ($+125.0\%$) |
| **F1-Score** | **0.0837** | 0.0346 | $+0.0491$ ($+141.9\%$) |
| **Brier Score** | **0.2482** | 0.2512 | $-0.0030$ (improved) |
| **True Positives (TP)** | **9** | 4 | $+5$ |
| **False Positives (FP)**| **187** | 208 | $-21$ (fewer false alarms) |
| **False Negatives (FN)**| **10** | 15 | $-5$ |
| **True Negatives (TN)** | **478** | 457 | $+21$ |

In the aggregate pooled sample, the Extended Model exhibits superior discrimination, higher recall, fewer false alarms, and improved precision.

---

## 4. Analysis 2: Excluding the 2007 GFC Fold (2000–2006, 2008)

When the 2007 test fold is removed from the out-of-sample predictions, 8 test folds remain ($608$ observations, 4 true crisis onsets):

| Metric | Extended Logistic Regression | Baseline Logistic Regression | Evaluation / Contrast |
| :--- | :---: | :---: | :--- |
| **Observations** | 608 | 608 | Exact match |
| **Crisis Events** | 4 | 4 | Turkey 2001, Argentina 2002, Nigeria 2006, Mongolia 2008 |
| **PR-AUC** | **0.0051** | **0.0065** | Below unconditional baseline ($4/608 = 0.0066$) |
| **ROC-AUC** | **0.2301** | **0.3845** | **Severe ranking inversion (< 0.50)** |
| **Precision** | **0.0000** | **0.0050** | Baseline caught 1 true alarm |
| **Recall (TPR)** | **0.0000 (0/4)** | **0.2500 (1/4)** | Extended LR misses 100% of non-GFC crises |
| **F1-Score** | **0.0000** | **0.0099** | Complete collapse of extended F1 |
| **Brier Score** | **0.2496** | **0.2525** | Extended slightly lower squared error |
| **True Positives (TP)** | **0** | **1** | Extended catches zero events |
| **False Positives (FP)**| **167** | **198** | Extended issues 167 false alarms |
| **False Negatives (FN)**| **4** | **3** | All 4 crises missed by Extended LR |
| **True Negatives (TN)** | **437** | **406** | Extended has 31 more true negatives |

### Critical Finding:
When the 2007 test fold is excluded, the Extended Logistic Regression suffers a complete breakdown in predictive discrimination:
- **Recall collapses to 0.0% ($0/4$).**
- **ROC-AUC drops to $0.2301$,** meaning the model systematically assigned *lower* predicted crisis probabilities to actual crisis countries than to calm countries.
- Baseline LR, while weak, detected 1 of the 4 crises (Turkey 2001) and achieved a higher ROC-AUC ($0.3845$) and higher PR-AUC ($0.0065$).

---

## 5. Analysis 3: Pre-GFC Historical Period (2000–2006)

Evaluating strictly across the pre-crisis expansion folds ($532$ observations, 3 crisis events: Turkey 2001, Argentina 2002, Nigeria 2006):

| Metric | Extended Logistic Regression | Baseline Logistic Regression | Evaluation / Contrast |
| :--- | :---: | :---: | :--- |
| **Observations** | 532 | 532 | 7 expanding folds |
| **Crisis Events** | 3 | 3 | Turkey 2001, Argentina 2002, Nigeria 2006 |
| **PR-AUC** | **0.0051** | **0.0063** | Extended below unconditional prior ($3/532 = 0.0056$) |
| **ROC-AUC** | **0.2936** | **0.3655** | Both models inverted; Extended is worse |
| **Precision** | **0.0000** | **0.0065** | Extended precision is zero |
| **Recall (TPR)** | **0.0000 (0/3)** | **0.3333 (1/3)** | Baseline catches Turkey; Extended catches zero |
| **F1-Score** | **0.0000** | **0.0127** | Extended F1 is zero |
| **Brier Score** | **0.2427** | **0.2497** | Similar probability dispersion |
| **True Positives (TP)** | **0** | **1** | Turkey (TUR) caught by Baseline only |
| **False Positives (FP)**| **132** | **154** | Both models generate high false alarm rates |
| **False Negatives (FN)**| **3** | **2** | Extended misses Turkey, Argentina, Nigeria |
| **True Negatives (TN)** | **397** | **375** | Extended avoids 22 false alarms |

Throughout the 2000–2006 pre-GFC period, adding private credit and current account indicators provided **zero additional crisis detection capability**. It failed to alert on the sovereign/banking crises in Turkey (2001) and Argentina (2002), where crisis dynamics were driven by fiscal dominance, exchange-rate pegs, and sovereign debt default rather than private credit booms.

---

## 6. Analysis 4: The 2007 GFC Test Fold (Forecasting 2008 Crises)

Evaluating exclusively the 2007 test fold ($76$ observations, 15 true crisis onsets):

| Metric | Extended Logistic Regression | Baseline Logistic Regression | Absolute Difference ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Observations** | 76 | 76 | All 76 panel economies in 2007 |
| **Crisis Events** | 15 | 15 | 2008 GFC systemic banking crises |
| **PR-AUC** | **0.2647** | **0.2743** | Baseline slightly higher peak ranking |
| **ROC-AUC** | **0.6470** | **0.6590** | Both models demonstrate clear separation (> 0.50) |
| **Precision** | **0.3103** | **0.2308** | $+0.0795$ (higher purity for Extended) |
| **Recall (TPR)** | **0.6000 (9/15)** | **0.2000 (3/15)** | **$+0.4000$ ($+6$ crises caught)** |
| **F1-Score** | **0.4091** | **0.2143** | $+0.1948$ ($+90.9\%$) |
| **Brier Score** | **0.2371** | **0.2405** | $-0.0034$ (improved) |
| **True Positives (TP)** | **9** | **3** | Extended triples detection count |
| **False Positives (FP)**| **20** | **10** | Extended takes on 10 additional alarms |
| **False Negatives (FN)**| **6** | **12** | Extended halves missed crises |
| **True Negatives (TN)** | **41** | **51** | Baseline is more conservative |

### Crises Detected in 2007 Fold:
- **Baseline LR (3/15 caught):** Germany (`DEU`), Spain (`ESP`), France (`FRA`).
- **Extended LR (9/15 caught):** Denmark (`DNK`), Spain (`ESP`), France (`FRA`), Greece (`GRC`), Hungary (`HUN`), Ireland (`IRL`), Iceland (`ISL`), Italy (`ITA`), Portugal (`PRT`).

In 2007, the macro-financial indicators correctly flagged the massive buildup of private banking credit and cross-border deficits across Europe, tripling crisis recall from $20.0\%$ to **$60.0\%$**.

---

## 7. Analysis 5: The 2008 Test Fold (Forecasting 2009 Crises)

Evaluating the single post-shock fold ($76$ observations, 1 true crisis onset: Mongolia 2009):

| Metric | Extended Logistic Regression | Baseline Logistic Regression | Evaluation / Contrast |
| :--- | :---: | :---: | :--- |
| **Observations** | 76 | 76 | Post-shock macro environment |
| **Crisis Events** | 1 | 1 | Mongolia (`MNG`) |
| **PR-AUC** | **0.0139** | **0.0169** | Roughly equal to unconditional rate ($1/76 = 0.0132$) |
| **ROC-AUC** | **0.0533** | **0.2267** | Severe ranking inversion in both models |
| **Precision** | **0.0000** | **0.0000** | Both models miss Mongolia |
| **Recall (TPR)** | **0.0000 (0/1)** | **0.0000 (0/1)** | Recall is zero |
| **F1-Score** | **0.0000** | **0.0000** | F1 is zero |
| **Brier Score** | **0.2979** | **0.2723** | Baseline has lower squared error |
| **True Positives (TP)** | **0** | **0** | Neither model detected Mongolia |
| **False Positives (FP)**| **35** | **44** | Extended produces 9 fewer false alarms |
| **False Negatives (FN)**| **1** | **1** | Both missed Mongolia |
| **True Negatives (TN)** | **40** | **31** | Extended has higher specificity |

Both models failed to detect Mongolia's 2009 crisis onset. With 2008 data reflecting extreme global stress across nearly all economies, both models experienced a surge in false alarms (35 for Extended, 44 for Baseline).

---

## 8. Summary Comparison: Baseline LR vs Extended LR Across Subsets

| Evaluation Subset | Observations | Crises | Baseline PR-AUC | Extended PR-AUC | Baseline ROC-AUC | Extended ROC-AUC | Baseline Recall | Extended Recall | Baseline FP | Extended FP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A. Pooled 2000–2008** | 684 | 19 | 0.0278 | **0.0344** | 0.5106 | **0.5734** | 0.2105 (4/19) | **0.4737 (9/19)** | 208 | **187** |
| **B. Excluding 2007** | 608 | 4 | **0.0065** | 0.0051 | **0.3845** | 0.2301 | **0.2500 (1/4)** | 0.0000 (0/4) | 198 | **167** |
| **C. Pre-GFC (2000–2006)** | 532 | 3 | **0.0063** | 0.0051 | **0.3655** | 0.2936 | **0.3333 (1/3)** | 0.0000 (0/3) | 154 | **132** |
| **D. 2007 GFC Fold** | 76 | 15 | **0.2743** | 0.2647 | **0.6590** | 0.6470 | 0.2000 (3/15) | **0.6000 (9/15)** | **10** | 20 |
| **E. 2008 Fold** | 76 | 1 | **0.0169** | 0.0139 | **0.2267** | 0.0533 | 0.0000 (0/1) | 0.0000 (0/1) | 44 | **35** |

---

## 9. Substantive Research Interpretation

Based strictly on the empirical calculations across temporal subsets, the evidence supports an unambiguous conclusion:

### Verdict: **GFC-Specific Signal**

The apparent predictive superiority of the 6-variable Extended Logistic Regression is **not a temporally general early warning signal**; it is an **overwhelmingly GFC-specific signal**.

### Detailed Analytical Justification:

1. **Complete Absence of Detection Outside 2007:**
   Outside of the 2007 test fold, the Extended Logistic Regression achieves a **recall of exactly 0.0% ($0/4$)**. Every single positive event detected by the Extended Model ($9$ out of $9$ true positives) occurred in the 2007 fold forecasting the 2008 crisis wave.

2. **Inverted Ranking on Non-GFC Episodes:**
   When 2007 is excluded, the Extended Model's ROC-AUC collapses from $0.5734$ to **$0.2301$**. In a panel containing Turkey (2001), Argentina (2002), Nigeria (2006), and Mongolia (2008), the model assigned *lower* risk scores to the crisis countries than to calm countries. This occurs because these emerging-market crises were not preceded by prolonged multi-year domestic private-credit-to-GDP booms, but rather by currency crashes, fiscal insolvencies, or sudden commodity price shocks.

3. **Nature of the GFC Predictive Surge:**
   In 2007, the financial world experienced a synchronized credit-overhang crisis across advanced economies. Because `private_credit_pct_gdp` carries a large positive coefficient ($\beta = +0.514$, $\text{OR} = 1.67$), the Extended Model aggressively elevated the risk scores of highly leveraged European banking systems (`IRL`, `ISL`, `ESP`, `PRT`, `GRC`, `ITA`, `DNK`, `FRA`). Because 15 crises occurred simultaneously in these specific economies, the pooled statistics show a dramatic leap in performance.

4. **False Alarm Specificity:**
   The one modest benefit of the Extended Model outside 2007 is that it reduces false alarms (167 vs 198 across the remaining 8 folds). By conditioning on credit depth, it was less prone to triggering false alarms on developing economies experiencing volatile inflation or reserve swings without large domestic banking sectors.

---

## 10. Methodological Limitations and Implications

1. **Extreme Historical Concentration:**
   Systemic banking crises do not arrive in smooth, independent Poisson processes. They occur in clusters during global financial regime shifts. Any panel-wide quantitative model estimated across the 2000–2008 period is inevitably dominated by the 2008 Global Financial Crisis.
2. **Structural Heterogeneity Across Crisis Types:**
   Private-credit-to-GDP is the canonical vulnerability indicator for advanced-economy banking crises (Schularick & Taylor, 2012). However, emerging-market banking crises frequently erupt from exchange-rate pass-through, reserve depletion, and external debt rollover difficulties without high domestic credit-to-GDP ratios. A single linear model pooling both groups will fit the dominant crisis archetype (GFC) while failing on idiosyncratic emerging-market episodes.
3. **Implications for Research Integrity:**
   Presenting the Phase 5D Extended Model as an "improved overall early warning system with 47.4% recall" would be fundamentally misleading without this diagnostic. Academic integrity requires stating clearly that **the model is an effective detector of trans-Atlantic credit-boom banking crises (catching 60% of 2008 onsets), but exhibits zero predictive utility for isolated emerging-market crises.**
