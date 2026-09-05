# Project Overview: An Empirical Journey into Macro-Financial Early Warning

**Author:** Global Financial Crisis Early Warning System Research Project
**Scope:** 76 Economies, 1990–2025 Panel, Expanding-Window Temporal Validation
**Primary Research Paper:** [`docs/RESEARCH_PAPER.md`](RESEARCH_PAPER.md)

---

### 1. What question did I ask?
I investigated: **"Can macro-financial indicators provide useful early warning signals of systemic banking crises?"**

Specifically, I sought to evaluate whether adding banking-sector balance-sheet leverage (private credit to GDP) and external financing imbalances (current account to GDP) to traditional macroeconomic flow indicators (real GDP growth, inflation, international reserves) materially improves out-of-sample crisis detection, and whether any observed predictive advantage is temporally stable across distinct historical regimes.

### 2. Why did I choose it?
Systemic banking crises are among the most catastrophic macroeconomic events, imposing long-lasting output contractions, soaring public debt, and widespread social costs. While post-2008 economic consensus emphasized that debt accumulation and balance-sheet vulnerabilities precede crises, much of the applied machine learning literature in finance suffers from critical methodological flaws—most notably temporal look-ahead leakage and overly optimistic performance claims derived from randomized cross-validation. I wanted to build an end-to-end, leak-free research pipeline to see what linear and machine-learning models could—and genuinely could not—detect under realistic historical conditions.

### 3. What data did I collect?
I constructed an objective, balanced annual panel of **76 World Bank member economies** (28 advanced, 48 emerging and developing) spanning **1990 through 2025** ($N = 2,736$ country-years). Macroeconomic indicators were ingested directly from the World Bank World Development Indicators (WDI) API.

Ground-truth crisis labels derive strictly from the International Monetary Fund (IMF) Systemic Banking Crises Database (Laeven & Valencia, 2026, WP/26/94), which identifies systemic episodes based on significant banking sector distress and substantial public policy intervention measures.

### 4. How did I design the prediction problem?
I formulated a binary classification task predicting whether a systemic banking crisis onset will occur in country $c$ in calendar year $t+1$ ($y_{c,t} = 1$) conditioning strictly on information observed at calendar year $t$.

To replicate true prospective surveillance and eliminate temporal look-ahead leakage:
- I implemented an **expanding-window temporal validation design** across nine sequential test folds ($T = 2000, \dots, 2008$), yielding 684 pooled out-of-sample evaluations containing 19 positive crisis onsets (an unconditional prevalence of 2.78%).
- All data transformations, median imputations, and feature standardizations were fitted **strictly on the historical training window $[1990, T-1]$** before being applied forward to test year $T$.
- The classification threshold was pre-registered and locked at $\tau = 0.50$.

### 5. What models did I test?
I evaluated a formal $2 \times 2$ comparative matrix:
1. **Baseline Logistic Regression:** 4 macroeconomic flow variables (`gdp_growth`, `inflation`, `reserves_usd`, `reserves_usd_yoy_pct_change`).
2. **Extended Logistic Regression:** Augments the baseline with `private_credit_pct_gdp` and `current_account_pct_gdp`.
3. **Baseline Random Forest:** 100 balanced bootstrap trees ($d=4$) on 4 flow variables.
4. **Extended Random Forest:** 100 balanced bootstrap trees ($d=4$) on 6 variables.

To handle severe rare-event class imbalance (2.78% positive base rate), models were trained using cost-sensitive balanced loss weighting inversely proportional to class frequencies.

### 6. What happened?
In the pooled out-of-sample evaluation ($N=684$, 19 crises):
- Augmenting the feature set with balance-sheet variables improved pooled discrimination: the Extended Logistic Regression increased PR-AUC from 0.0278 to 0.0344 (+23.7%), ROC-AUC from 0.5106 to 0.5734, and out-of-sample Recall from 21.05% ($4/19$) to 47.37% ($9/19$), while reducing false alarms from 208 to 187.
- Domestic private credit emerged as the dominant risk indicator ($\beta = +0.5142, \text{OR} = 1.6723$), confirming that elevated banking leverage is strongly associated with heightened crisis probability.

### 7. What surprised me?
Parametric linear Logistic Regression decisively outperformed non-linear Random Forest ensembles in detecting crises.

Although the Extended Random Forest achieved an attractive Brier calibration score (0.1347 vs. 0.2482), it suffered from severe **base-rate probability compression**. In a sample where 97.2% of observations are tranquil, the decision trees minimized squared loss by shrinking predicted probabilities toward the 2.78% mean. Consequently, probabilities rarely crossed the operational 0.50 threshold, causing the Random Forest to miss 17 of 19 crises (Recall = 10.53%). Linear logit models, by enforcing monotonic log-odds scaling and cost-sensitive reweighting, allowed probabilities to expand during credit booms, making timely early warnings possible.

### 8. What failed?
When I decomposed the pooled results into sequential temporal slices, the apparent predictive power disintegrated outside of the 2008 Global Financial Crisis:
- **Concentration in 2007:** 15 of the 19 evaluation events (78.9%) clustered in the single 2007 test fold. In that fold, the extended model performed well, detecting 9 of 15 crises (60.0% Recall, PR-AUC = 0.2647) across economies experiencing credit-fueled property and deficit booms (Spain, Ireland, Iceland, Greece, Portugal).
- **Collapse Outside 2007:** Across the remaining eight evaluation folds containing four crisis events (Uruguay 2002, Dominican Republic 2003, United Kingdom 2007, Nigeria 2009), the extended model detected **zero crises** (0.0% Recall) and exhibited an observed ranking inversion (ROC-AUC = 0.2301).
- **Why it failed:** The model was tuned to a specific macro-financial profile: high domestic credit paired with current account deficits. When confronted with banking crises caused by cross-border interbank wholesale contagion in surplus economies (Germany, Switzerland) or idiosyncratic shocks in emerging markets (depositor runs from Argentina in Uruguay, commercial bank fraud in the Dominican Republic), the model’s macro indicators failed to trigger alarms.

### 9. What did I learn?
1. **Validation Design Trumps Algorithmic Complexity:** The choice between random cross-validation and leak-free temporal validation completely alters reported results. Standard $k$-fold splits produce misleadingly high performance by leaking future crisis information.
2. **Pooled Metrics Can Conceal Severe Regime Brittleness:** Aggregate out-of-sample metrics can look impressive when performance is driven by a single dominant historical wave. Slicing performance across epochs is mandatory in macro-finance.
3. **Intellectual Honesty in Negative Results:** Discovering that the model does not generalize universally is not a project failure—it is the primary scientific finding. It demonstrates that macro-financial early warning systems function as specialized regime diagnostics rather than universal crisis prediction engines.

### 10. What would I do next?
1. **Incorporate Cross-Border Wholesale Banking Data:** National accounts data miss off-balance-sheet vehicles and interbank networks. Integrating Bank for International Settlements (BIS) consolidated banking statistics and bilateral interbank exposure matrices would help capture wholesale contagion in surplus nations.
2. **Real-Time Data Vintages:** Re-evaluating the pipeline on historical publication vintages rather than revised World Bank series to measure the impact of reporting lags (often 6 to 18 months).
3. **Regime-Switching Specifications:** Moving from a single pooled parameter vector to structural regime-switching models or hierarchical models that treat advanced and emerging market financial architectures differently.
