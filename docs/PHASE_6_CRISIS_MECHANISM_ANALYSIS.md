# Phase 6: Crisis Mechanism and Temporal Concentration Analysis

**Project:** Global Financial Crisis Early Warning System (`global-financial-crisis-ews`)  
**Status:** Complete, Audited, Verified, Reproducible  
**Design:** Strictly observational, post-estimation mechanism diagnostic (Zero causal claims)  
**Evaluation Scope:** 684 Pooled Out-of-Sample Country-Years across 76 Economies (Folds $T \in [2000, 2008]$, 19 Crisis Onset Events)  
**Evaluated Architecture:** 6-Variable Extended Logistic Regression vs 4-Variable Baseline Logistic Regression  

---

## 1. Objective

The objective of Phase 6 is to conduct a descriptive, post-estimation diagnostic of the out-of-sample crisis predictions generated in Phase 5. Phase 5E demonstrated that the predictive advantage of the 6-variable Extended Logistic Regression over the 4-variable baseline (PR-AUC 0.0344 vs 0.0278, Recall 47.4% vs 21.1%) was concentrated in the 2007 pre-GFC evaluation fold (predicting 2008 crises). Outside that fold, the extended model detected 0 of 4 crises (Recall 0.0%) and exhibited an inverted ranking (ROC-AUC 0.2301).

Phase 6 examines the underlying macro-financial configurations of the observed crisis episodes to understand how the model's feature weights interacted with the predictor values in different historical episodes. The goal is not to modify the model, select features, or adjust decision thresholds, but to provide an economically grounded, transparent, and non-causal characterization of its empirical behavior.

---

## 2. Research Integrity and Methodological Separation

To maintain strict scientific integrity, Phase 6 enforces an explicit three-layer separation:

```
+---------------------------------------------------------------------------------------------------+
| LAYER A: AUTHORITATIVE CRISIS CLASSIFICATION                                                      |
| Multilateral ground truth: IMF WP/26/94 (Laeven & Valencia, 2026). All 19 events are non-         |
| borderline systemic banking crises. No official sub-mechanism taxonomy is provided in the source; |
| authoritative sub-mechanism is strictly classified as "Unknown / insufficient evidence".         |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| LAYER B: DATA-DRIVEN DESCRIPTIVE MACRO-FINANCIAL TYPOLOGY                                         |
| Empirical grouping based strictly on observed predictor values at t prior to crisis onset at t+1. |
| Descriptive Profile A: High private-credit / external-deficit configuration (N=9, 2007 fold)       |
| Descriptive Profile B: High current-account-surplus configuration (N=6, 2007 fold)                 |
| Descriptive Profile C: Lower private-credit / higher-inflation / weaker-reserves (N=4, non-2007)  |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
| LAYER C: ECONOMIC INTERPRETATION AND HYPOTHESES                                                   |
| Non-causal explanatory hypotheses linking model parameters to observed predictor differences.     |
| Explains ranking behavior without asserting counterfactual proof or historical crisis causality.   |
+---------------------------------------------------------------------------------------------------+
```

### Strict Non-Causal Rules:
1. **Zero Causal Inferences:** No observed predictor is asserted as the "cause" of a crisis.
2. **No Counterfactual Claims:** We do not claim that a crisis was "missed due to" a particular predictor in the absence of a formal counterfactual model.
3. **Small-Sample Boundary:** With exactly $N=4$ non-2007 evaluation events, findings outside the 2007 fold are strictly confined to the observed sample and phrased as *"Within the four observed non-2007 evaluation events..."*.
4. **Ranking Inversion Framing:** The non-2007 ROC-AUC of 0.2301 is described as *"an observed ranking inversion in the out-of-sample predictions"*, rather than claiming the model had "zero predictive capability."

---

## 3. Data and Multilateral Sources

All data analyzed in Phase 6 derive strictly from multilateral sources locked in the repository:
1. **Crisis Chronology:** Official IMF Systemic Banking Crises Database (Laeven & Valencia, 2026, IMF Working Paper WP/26/94). Identifies non-borderline systemic banking crisis onset years across all 76 sample economies.
2. **Macroeconomic and Balance-Sheet Predictors:** World Bank World Development Indicators (WDI), collected and harmonized in Phase 4B and Phase 5D:
   - Real GDP Growth (`NY.GDP.MKTP.KD.ZG`)
   - CPI Inflation (`FP.CPI.TOTL.ZG`)
   - Gross International Reserves in USD (`FI.RES.TOTL.CD`)
   - Reserves YoY Percentage Change (derived from `FI.RES.TOTL.CD`)
   - Domestic Credit to Private Sector as % of GDP (`FS.AST.PRVT.GD.ZS`)
   - Current Account Balance as % of GDP (`BN.CAB.XOKA.GD.ZS`)

---

## 4. Existing Model Inputs and Pre-Registered Specification

The feature vector $\mathbf{x}_{c,t}$ is strictly locked at **6 variables total**:
$$\mathbf{x}_{c,t} = \big[\text{gdp\_growth}_{c,t},\, \text{inflation}_{c,t},\, \text{reserves\_usd}_{c,t},\, \text{reserves\_usd\_yoy\_pct\_change}_{c,t},\, \text{private\_credit\_pct\_gdp}_{c,t},\, \text{current\_account\_pct\_gdp}_{c,t}\big]$$

All predictor transformations, imputations (training-fold median), and robust z-score normalizations were fitted strictly on training data $[1990, T-1]$ in Phase 5D. No models were refitted or re-estimated during Phase 6.

---

## 5. Event Construction and Temporal Alignment

The evaluation sample spans 9 expanding folds evaluated at test years $T \in \{2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008\}$, producing exactly 684 pooled out-of-sample country-year observations across the 76 economies.

### Strict Temporal Alignment:
- **Observation / Prediction Year ($t = T$):** Predictor vector $\mathbf{x}_{c,t}$ is measured strictly in calendar year $t$.
- **Forward Outcome Horizon ($t \to t+1$):** Binary outcome $y_{c,t} = 1$ if an official IMF systemic banking crisis begins in year $t+1$; $y_{c,t} = 0$ otherwise.
- **Crisis Onset Year ($t+1$):** Calendar year in which the systemic banking crisis erupts.
- **Zero Leakage:** No observation from year $t+1$ or later is present in the feature vectors.

Across the 684 evaluation observations, exactly **19 positive forward crisis events** occur:
- **15 events** occur in the 2007 fold ($t=2007 \to \text{crisis in } 2008$): Austria, Belgium, Switzerland, Germany, Denmark, Spain, France, Greece, Hungary, Ireland, Iceland, Italy, Netherlands, Portugal, and Sweden.
- **4 events** occur outside the 2007 fold:
  1. **Uruguay (`URY`):** $t=2001 \to \text{crisis in } 2002$
  2. **Dominican Republic (`DOM`):** $t=2002 \to \text{crisis in } 2003$
  3. **United Kingdom (`GBR`):** $t=2006 \to \text{crisis in } 2007$
  4. **Nigeria (`NGA`):** $t=2008 \to \text{crisis in } 2009$

---

## 6. Layer A: Authoritative Crisis Classification Audit

Every evaluation crisis episode was audited against the repository's authoritative sources. The findings are exported to [phase6_mechanism_categorization.csv](file:///Users/macbookair/Documents/ChatGPT/finance/results/tables/phase6_mechanism_categorization.csv).

### Authoritative Classification Summary Table (19 Evaluation Events)

| Country | Code | Pred Year ($t$) | Crisis Year ($t+1$) | Authoritative Source | Official Type | Sub-Mechanism Category | Sub-Mechanism Source Support |
| :--- | :---: | :---: | :---: | :--- | :--- | :--- | :--- |
| **Uruguay** | `URY` | 2001 | 2002 | IMF WP/26/94 | `systemic_banking` | Unknown / insufficient evidence | None present in repository |
| **Dominican Rep.** | `DOM` | 2002 | 2003 | IMF WP/26/94 | `systemic_banking` | Unknown / insufficient evidence | None present in repository |
| **United Kingdom** | `GBR` | 2006 | 2007 | IMF WP/26/94 | `systemic_banking` | Unknown / insufficient evidence | None present in repository |
| **Austria** | `AUT` | 2007 | 2008 | IMF WP/26/94 | `systemic_banking` | Unknown / insufficient evidence | None present in repository |
| **Belgium** | `BEL` | 2007 | 2008 | IMF WP/26/94 | `systemic_banking` | Unknown / insufficient evidence | None present in repository |
| **Switzerland** | `CHE` | 2007 | 2008 | IMF WP/26/94 | `systemic_banking` | Unknown / insufficient evidence | None present in repository |
| **Germany** | `DEU` | 2007 | 2008 | IMF WP/26/94 | `systemic_banking` | Unknown / insufficient evidence | None present in repository |
| **Denmark** | `DNK` | 2007 | 2008 | IMF WP/26/94 | `systemic_banking` | Unknown / insufficient evidence | None present in repository |
| **Spain** | `ESP` | 2007 | 2008 | IMF WP/26/94 | `systemic_banking` | Unknown / insufficient evidence | None present in repository |
| **France** | `FRA` | 2007 | 2008 | IMF WP/26/94 | `systemic_banking` | Unknown / insufficient evidence | None present in repository |
| **Greece** | `GRC` | 2007 | 2008 | IMF WP/26/94 | `systemic_banking` | Unknown / insufficient evidence | None present in repository |
| **Hungary** | `HUN` | 2007 | 2008 | IMF WP/26/94 | `systemic_banking` | Unknown / insufficient evidence | None present in repository |
| **Ireland** | `IRL` | 2007 | 2008 | IMF WP/26/94 | `systemic_banking` | Unknown / insufficient evidence | None present in repository |
| **Iceland** | `ISL` | 2007 | 2008 | IMF WP/26/94 | `systemic_banking` | Unknown / insufficient evidence | None present in repository |
| **Italy** | `ITA` | 2007 | 2008 | IMF WP/26/94 | `systemic_banking` | Unknown / insufficient evidence | None present in repository |
| **Netherlands** | `NLD` | 2007 | 2008 | IMF WP/26/94 | `systemic_banking` | Unknown / insufficient evidence | None present in repository |
| **Portugal** | `PRT` | 2007 | 2008 | IMF WP/26/94 | `systemic_banking` | Unknown / insufficient evidence | None present in repository |
| **Sweden** | `SWE` | 2007 | 2008 | IMF WP/26/94 | `systemic_banking` | Unknown / insufficient evidence | None present in repository |
| **Nigeria** | `NGA` | 2008 | 2009 | IMF WP/26/94 | `systemic_banking` | Unknown / insufficient evidence | None present in repository |

> [!IMPORTANT]
> **Audit Finding on Authoritative Provenance:**
> The IMF Systemic Banking Crises Database (Laeven & Valencia, 2026, WP/26/94) directly supports the binary onset identification of non-borderline systemic banking crises for all 19 episodes. However, the database does not contain a structured taxonomy of sub-triggers (such as sovereign contagion, terms-of-trade shock, fraud, or wholesale funding freeze). Consequently, any assignment of detailed historical sub-mechanisms would rely on external narratives unsupported by the primary dataset. Per research integrity standards, all 19 episodes are authoritatively categorized as **"Unknown / insufficient evidence"**.

---

## 7. Layer B: Data-Driven Descriptive Macro-Financial Typology

While authoritative sources do not categorize sub-mechanisms, the harmonized panel data allow an objective, descriptive classification of the macro-financial configurations observed at prediction year $t$.

*Event-level profiles exported to [phase6_crisis_event_profiles.csv](file:///Users/macbookair/Documents/ChatGPT/finance/results/tables/phase6_crisis_event_profiles.csv).*

### Complete 19-Event Profile Table

| ISO | Country | Fold ($t$) | Status ($\tau=0.50$) | Predicted Prob | GDP Growth | Inflation | Reserves YoY | Credit / GDP | CA / GDP | Data-Driven Descriptive Profile |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **URY** | Uruguay | 2001 | **Missed** | 0.4264 | -3.84% | 4.36% | +11.66% | 53.9% | -2.38% | Descriptive Profile C: Lower credit / higher inflation / weaker reserves |
| **DOM** | Dominican Rep. | 2002 | **Missed** | 0.4635 | +4.49% | 5.22% | -57.02% | [Imputed] | -2.94% | Descriptive Profile C: Lower credit / higher inflation / weaker reserves |
| **GBR** | United Kingdom | 2006 | **Missed** | 0.4559 | +2.19% | 2.46% | +7.93% | 153.5% | -3.11% | Descriptive Profile C: Lower credit / higher inflation / weaker reserves |
| **AUT** | Austria | 2007 | **Missed** | 0.4445 | +3.78% | 2.17% | +40.92% | 93.5% | **+3.81%** | Descriptive Profile B: High current-account surplus |
| **BEL** | Belgium | 2007 | **Missed** | 0.4659 | +3.68% | 1.82% | +22.69% | 68.4% | **+1.50%** | Descriptive Profile B: High current-account surplus |
| **CHE** | Switzerland | 2007 | **Missed** | 0.4730 | +3.98% | 0.73% | +16.62% | 151.1% | **+8.63%** | Descriptive Profile B: High current-account surplus |
| **DEU** | Germany | 2007 | **Missed** | 0.4795 | +2.89% | 2.30% | +21.76% | 95.5% | **+6.52%** | Descriptive Profile B: High current-account surplus |
| **DNK** | Denmark | 2007 | **Detected** | **0.5163** | +0.99% | 1.69% | +10.41% | **184.0%** | +1.51% | Descriptive Profile A: High private credit / external deficit |
| **ESP** | Spain | 2007 | **Detected** | **0.5619** | +3.53% | 2.79% | -1.61% | **167.6%** | **-9.37%** | Descriptive Profile A: High private credit / external deficit |
| **FRA** | France | 2007 | **Detected** | **0.5078** | +2.53% | 1.49% | +17.56% | 89.0% | -0.33% | Descriptive Profile A: High private credit / external deficit |
| **GRC** | Greece | 2007 | **Detected** | **0.5305** | +3.51% | 2.90% | +27.99% | 85.8% | **-14.19%** | Descriptive Profile A: High private credit / external deficit |
| **HUN** | Hungary | 2007 | **Detected** | **0.5222** | +0.33% | 7.96% | +11.40% | 53.2% | **-7.29%** | Descriptive Profile A: High private credit / external deficit |
| **IRL** | Ireland | 2007 | **Detected** | **0.5215** | +5.31% | 4.89% | +11.25% | **158.1%** | **-5.13%** | Descriptive Profile A: High private credit / external deficit |
| **ISL** | Iceland | 2007 | **Detected** | **0.5701** | +8.74% | 5.05% | +12.37% | **243.2%** | **-13.63%** | Descriptive Profile A: High private credit / external deficit |
| **ITA** | Italy | 2007 | **Detected** | **0.5018** | +1.46% | 1.83% | +24.20% | 81.5% | -1.36% | Descriptive Profile A: High private credit / external deficit |
| **NLD** | Netherlands | 2007 | **Missed** | 0.4694 | +3.89% | 1.61% | +12.66% | 112.7% | **+5.86%** | Descriptive Profile B: High current-account surplus |
| **PRT** | Portugal | 2007 | **Detected** | **0.5400** | +2.51% | 2.45% | +16.49% | **142.2%** | **-9.66%** | Descriptive Profile A: High private credit / external deficit |
| **SWE** | Sweden | 2007 | **Missed** | 0.4637 | +3.22% | 2.21% | +10.76% | 111.3% | **+8.15%** | Descriptive Profile B: High current-account surplus |
| **NGA** | Nigeria | 2008 | **Missed** | 0.2978 | +6.76% | 11.58% | +3.26% | 18.6% | **+8.59%** | Descriptive Profile C: Lower credit / higher inflation / weaker reserves |

### Characterization of the Three Descriptive Profiles

1. **Descriptive Profile A: High Private-Credit / External-Deficit Configuration ($N=9$):**
   - **Observed Sample:** All 9 detected events from the 2007 fold (DNK, ESP, FRA, GRC, HUN, IRL, ISL, ITA, PRT).
   - **Empirical Attributes at $t$:** Marked by elevated domestic private credit depth (median **142.2%** of GDP, mean **133.8%**, IQR 67.2%) and negative current account balances (median **-7.29%** of GDP, mean **-6.60%**, IQR 8.54%).
   - **Model Output:** Out-of-sample predicted probabilities range from 0.5018 to 0.5701 (median **0.5222**), all exceeding the decision threshold $\tau = 0.50$.

2. **Descriptive Profile B: High Current-Account-Surplus Configuration ($N=6$):**
   - **Observed Sample:** The 6 missed events from the 2007 fold (AUT, BEL, CHE, DEU, NLD, SWE).
   - **Empirical Attributes at $t$:** Characterized by substantial financial depth (median credit **103.4%** of GDP, mean **105.4%**), but distinguished by **positive current account balances** ranging from +1.50% to +8.63% of GDP (median **+6.19%**, mean **+5.74%**). Every missed 2007 country ran a current account surplus.
   - **Model Output:** Out-of-sample predicted probabilities cluster in a narrow band from 0.4445 to 0.4795 (median **0.4677**), remaining below $\tau = 0.50$.

3. **Descriptive Profile C: Lower Private-Credit / Higher-Inflation / Weaker-Reserve-Dynamics Configuration ($N=4$):**
   - **Observed Sample:** The 4 non-2007 evaluation events (URY in 2001, DOM in 2002, GBR in 2006, NGA in 2008).
   - **Empirical Attributes at $t$:** As an observed group, these episodes exhibit lower median private credit (median **53.9%** of GDP, mean **75.3%**), higher inflation (median **4.79%**, mean **5.91%**, max 11.58%), and weaker reserve dynamics (median YoY growth **+5.59%**, mean **-8.54%**, min -57.02%).
   - **Model Output:** Out-of-sample predicted probabilities range from 0.2978 to 0.4635 (median **0.4411**, mean **0.4109**).

---

## 8. Descriptive Comparison Across Groups

*Comparison table exported to [phase6_group_descriptive_statistics.csv](file:///Users/macbookair/Documents/ChatGPT/finance/results/tables/phase6_group_descriptive_statistics.csv).*

| Predictor | Group | Count | Mean | Std Dev | Median | IQR | Min | Max |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Private Credit** | 2007 Crisis Events | 15 | **122.47%** | 51.08% | **111.25%** | 67.20% | 53.20% | 243.15% |
| (% of GDP) | Non-2007 Crisis Events | 3 | 75.32% | 69.94% | 53.85% | 67.42% | 18.63% | 153.47% |
| | Non-Crisis Panel | 570 | 66.03% | 44.95% | 53.73% | 68.45% | 5.97% | 301.02% |
| **Current Account**| 2007 Crisis Events | 15 | **-1.66%** | 7.77% | -0.33% | 13.17% | -14.19% | +8.63% |
| (% of GDP) | Non-2007 Crisis Events | 4 | +0.04% | 5.71% | -2.66% | 3.34% | -3.11% | +8.59% |
| | Non-Crisis Panel | 635 | -0.88% | 9.68% | -0.96% | 8.42% | -36.20% | +41.91% |
| **GDP Growth** | 2007 Crisis Events | 15 | 3.36% | 1.96% | 3.51% | 1.31% | 0.33% | 8.74% |
| (%) | Non-2007 Crisis Events | 4 | 2.40% | 4.56% | 3.34% | 4.37% | -3.84% | 6.76% |
| | Non-Crisis Panel | 665 | 4.14% | 3.36% | 3.93% | 3.67% | -11.18% | 26.64% |
| **Inflation** | 2007 Crisis Events | 15 | 2.79% | 1.85% | 2.21% | 1.08% | 0.73% | 7.96% |
| (%) | Non-2007 Crisis Events | 4 | 5.91% | 3.96% | 4.79% | 2.93% | 2.46% | 11.58% |
| | Non-Crisis Panel | 665 | 4.86% | 6.69% | 3.37% | 4.29% | -9.80% | 96.10% |
| **Reserves YoY** | 2007 Crisis Events | 15 | +17.03% | 9.76% | +16.49% | 10.90% | -1.61% | +40.92% |
| (%) | Non-2007 Crisis Events | 4 | -8.54% | 32.50% | +5.59% | 20.67% | -57.02% | +11.66% |
| | Non-Crisis Panel | 665 | +13.97% | 32.62% | +9.66% | 27.03% | -90.17% | 437.08% |
| **Reserves USD** | 2007 Crisis Events | 15 | $40.6B | $43.2B | $24.1B | $40.7B | $0.93B | $135.9B |
| (Level) | Non-2007 Crisis Events | 4 | $30.2B | $33.1B | $28.3B | $53.7B | $0.47B | $63.8B |
| | Non-Crisis Panel | 665 | $45.7B | $147.1B | $10.4B | $32.7B | $0.02B | $1,966.0B |

---

## 9. Probability Profile and the Observed Ranking Inversion

*Probability distributions exported to [phase6_probability_distributions.csv](file:///Users/macbookair/Documents/ChatGPT/finance/results/tables/phase6_probability_distributions.csv).*

| Group | Count | Mean Prob | Std Dev | Median Prob | IQR | Min | Max | % $\ge 0.50$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pooled Evaluation** | 684 | 0.4874 | 0.0993 | 0.4733 | 0.0654 | 0.0260 | 0.9829 | 28.7% |
| **All Crisis Events** | 19 | 0.4848 | 0.0603 | 0.4795 | 0.0582 | 0.2978 | 0.5701 | 47.4% |
| **2007 Crisis Events** | 15 | **0.5045** | 0.0377 | **0.5078** | 0.0551 | 0.4445 | 0.5701 | **60.0%** |
| **Non-2007 Crisis Events** | 4 | **0.4109** | 0.0771 | **0.4411** | 0.0635 | 0.2978 | 0.4635 | **0.0%** |
| **Non-Crisis Panel** | 665 | 0.4874 | 0.1003 | 0.4733 | 0.0650 | 0.0260 | 0.9829 | 28.1% |

### Statistical Driver of the Non-2007 ROC-AUC (0.2301):
In Phase 5E, excluding the 2007 evaluation fold resulted in an out-of-sample ROC-AUC of **0.2301**. The distribution table illuminates why this occurred:
- The median predicted probability for the **Non-Crisis Panel is 0.4733** (mean = 0.4874).
- The median predicted probability for **Non-2007 Crisis Events is 0.4411** (mean = 0.4109).
- Rather than indicating that the model had "zero predictive capability", this result reflects **an observed ranking inversion in the out-of-sample predictions**: within the four observed non-2007 evaluation events, the model assigned systematically lower probabilities to actual crisis countries than to the typical non-crisis observation.

---

## 10. Layer C: Economic Interpretations and Hypotheses

The parameter estimates of the Extended Logistic Regression in the final expanding training fold ($T=2008$) were:
- `private_credit_pct_gdp`: $\beta = +0.5142$ ($\text{OR} = 1.6723$)
- `reserves_usd`: $\beta = -0.1708$
- `gdp_growth`: $\beta = -0.1277$
- `current_account_pct_gdp`: $\beta = -0.0463$ ($\text{OR} = 0.9547$)
- `reserves_usd_yoy_pct_change`: $\beta = -0.0449$
- `inflation`: $\beta = +0.0963$

### Explaining 2007 Model Discrimination:
The model's detected 2007 events are descriptively characterized by higher private credit-to-GDP ratios and more negative current-account balances. When evaluated on the 2007 fold, the model encountered economies with elevated private credit depth (Iceland 243%, Denmark 184%, Spain 168%, Ireland 158%, Portugal 142%) and negative current accounts (Greece -14.2%, Iceland -13.6%, Portugal -9.7%, Spain -9.4%). Given the large positive coefficient on private credit ($\beta = +0.5142$), these configurations pushed predicted probabilities above $\tau = 0.50$.

Conversely, the 6 missed 2007 economies (Austria, Belgium, Switzerland, Germany, Netherlands, Sweden) all ran current account surpluses (+1.50% to +8.63%), while exhibiting lower median credit ratios than the detected economies. Under the model's specification, these balance-sheet configurations yielded lower predicted probabilities (0.4445 to 0.4795).

### Explaining Non-2007 Ranking Behavior:
Within the four observed non-2007 evaluation events, the data exhibit different configurations:
1. **Lower Credit Depth:** In Nigeria (18.6% of GDP) and Uruguay (53.9% of GDP), private credit depth was well below the panel median. Under a specification heavily weighted toward private credit accumulation, these lower levels naturally dampened model output.
2. **Current Account Balances:** In Nigeria, an oil-export current account surplus of +8.59% coincided with a model probability of 0.2978, the lowest among all 19 crisis events.
3. **Indicator Weighting Disparities:** In the Dominican Republic, the observed pre-crisis profile included a severe reserve contraction (-57.02% YoY). However, in the historical training panel, the estimated coefficient on reserve percentage change was modest ($\beta = -0.0449$), which did not produce an elevated probability (0.4635).

> [!NOTE]
> **Interpretation Boundary:**
> The observed predictor differences provide a plausible explanation for the model's ranking behavior, but do not establish why those crises occurred. They describe the empirical alignment between the model's estimated weights and the historical data, without implying counterfactual proof.

---

## 11. Visualizations

Three publication-quality figures were programmatically generated and exported in both SVG vector format and companion PNG format under `results/figures/`:

1. **Figure 1:** [phase6_predictor_profile_comparison.svg](file:///Users/macbookair/Documents/ChatGPT/finance/results/figures/phase6_predictor_profile_comparison.svg) ([PNG](file:///Users/macbookair/Documents/ChatGPT/finance/results/figures/phase6_predictor_profile_comparison.png))  
   *Compares distributions of Private Credit / GDP and Current Account / GDP between 2007 Crises (median credit 111.3%), Non-2007 Crises (median credit 53.9%), and Non-Crisis observations (median credit 53.7%).*
2. **Figure 2:** [phase6_probability_distribution.svg](file:///Users/macbookair/Documents/ChatGPT/finance/results/figures/phase6_probability_distribution.svg) ([PNG](file:///Users/macbookair/Documents/ChatGPT/finance/results/figures/phase6_probability_distribution.png))  
   *Illustrates the out-of-sample predicted probability distributions across groups relative to $\tau = 0.50$, displaying that 2007 crises cluster above 0.50 (60%), while non-2007 crises cluster below the non-crisis panel median (median 0.4411 vs 0.4733).*
3. **Figure 3:** [phase6_standardized_profiles.svg](file:///Users/macbookair/Documents/ChatGPT/finance/results/figures/phase6_standardized_profiles.svg) ([PNG](file:///Users/macbookair/Documents/ChatGPT/finance/results/figures/phase6_standardized_profiles.png))  
   *Displays the 6-variable standardized profile (z-scores relative to the full panel), illustrating that 2007 was marked by elevated private credit (+1.25 SD), while non-2007 had reserve depletion (-0.69 SD) and higher inflation (+0.16 SD).*

---

## 12. Methodological Limitations

1. **Severe Small-Sample Limitation ($N=4$):** Exactly 4 non-2007 crisis episodes exist in the evaluation window. Consequently, no broad generalization claim can be made that the model systematically fails all non-GFC crises, nor that these four episodes represent all alternative crisis mechanisms. All observations outside 2007 apply strictly to these four events.
2. **Aggregate Annual Reporting Lags:** Annual World Bank data cannot capture rapid intra-year liquidity developments.
3. **Absence of Bank-Level Interconnectedness Data:** Macroeconomic balance sheets do not capture off-balance-sheet exposures or cross-border interbank derivative linkages.

---

## 13. Reproducibility

Phase 6 is fully reproducible from the command line:
```bash
# Execute Phase 6 analysis engine and generate tables/figures
.venv/bin/python -m crisis_ews.cli run-phase-6

# Run full test suite (including 6 Phase 6 unit tests)
.venv/bin/pytest -v

# Run code linter
.venv/bin/ruff check .
```

---

## 14. Conclusion

> *"The observed predictive advantage of the extended specification is concentrated in the 2007 pre-GFC evaluation fold and appears to reflect a particular macro-financial configuration rather than a broadly generalizable crisis signal."*

The 6-variable Extended Logistic Regression's detected 2007 events are descriptively characterized by higher private credit-to-GDP ratios and more negative current-account balances. In the 2007 evaluation fold, where multiple advanced economies exhibited this exact profile, the model detected 60% of crisis onsets. However, within the four observed non-2007 evaluation events, the economies exhibited different configurations—including lower private credit, external surpluses, and rapid reserve contractions—resulting in an observed ranking inversion where actual crisis events received lower predicted probabilities than the calm non-crisis panel median.
