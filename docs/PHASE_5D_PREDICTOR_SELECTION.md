# Phase 5D: Predictor Selection & Data Coverage Audit

**Project:** Global Financial Crisis Early Warning System  
**Repository:** `global-financial-crisis-ews`  
**Phase:** 5D — Pre-Model Predictor Selection & Data Coverage Audit  
**Status:** COMPLETE (Pre-model research design and empirical coverage audit)  
**Country Universe:** 76 World Bank economies (complete core series 1990–2025)  
**Observation Window:** 1990–2025 (2,736 country-years)  
**Validation Design:** Historical expanding window ($T \in [2000, 2008]$, 684 test observations, 19 forward crisis events)  
**Target Definition:** $y(c, t) = 1$ if qualifying IMF systemic banking crisis onset starts in calendar year $t+1$, 0 otherwise  
**Baseline Checkpoint:** LOCKED at commit `8aadf76`  

---

## 1. Research Question

Phase 5D addresses the central empirical question:

> *"Does adding a small, theoretically motivated set of macro-financial vulnerability indicators improve one-year-ahead systemic banking crisis prediction beyond the parsimonious flow macroeconomic baseline?"*

### Motivation and Core Principles
In Phase 5B and Phase 5C, out-of-sample evaluations demonstrated that a four-variable macroeconomic flow baseline (`gdp_growth`, `inflation`, `reserves_usd`, `reserves_usd_yoy_pct_change`) achieved limited discriminatory power ($\text{PR-AUC} = 0.0278$, approximating the unconditional event rate of $2.78\%$, with $208$ false alarms for Logistic Regression and $161$ for Random Forest).

However, rather than running an unconstrained data-mining search over dozens of variables or using model-based feature selection (such as LASSO or recursive feature elimination), Phase 5D constructs **one pre-specified Extended Predictor Set** based strictly on:
1. Economic and macroprudential theory of banking crises;
2. Empirical data availability and international comparability across the 76-country panel;
3. Leakage-safe, contemporaneous transformations;
4. Objective, pre-registered inclusion rules formulated **before any predictive models are estimated**.

---

## 2. Economic Motivation

Systemic banking panics are fundamentally balance-sheet, leverage, and liquidity phenomena. Economic theory and historical empirical literature (Kaminsky and Reinhart 1999; Borio and Lowe 2002; Gourinchas and Obstfeld 2012; Schularick and Taylor 2012; Aldasoro et al. 2018) identify three critical vulnerability channels that are omitted from the flow macroeconomic baseline:

1. **Domestic Leverage & Credit Overhang (Private Credit / GDP):**
   - *Mechanism:* Rapid accumulation of private sector debt relative to economic output increases private sector debt-servicing ratios and leaves the banking sector vulnerable to asset-price declines, corporate defaults, and collateral liquidation spirals.
   - *Expected Direction:* Positive ($+$): higher private credit-to-GDP ratios increase systemic vulnerability.
2. **External Imbalance & Sudden-Stop Vulnerability (Current Account / GDP):**
   - *Mechanism:* Large current account deficits reflect domestic expenditure exceeding national income, financed by foreign capital inflows. Deficit economies are structurally exposed to sudden stops or reversals in foreign financing, which deplete foreign reserves, trigger currency depreciation, and precipitate banking runs.
   - *Expected Direction:* Negative ($-$, where deficit is negative): larger deficits (more negative values) increase crisis vulnerability.
3. **Domestic Macroeconomic Distress & Debt Repayment Capacity (Unemployment Rate):**
   - *Mechanism:* Elevated or accelerating unemployment reduces household disposable income and corporate revenues, leading to widespread debt service delinquency, surging non-performing loans (NPLs), and the erosion of bank capital.
   - *Expected Direction:* Positive ($+$): higher unemployment reflects broader real-sector distress.

---

## 3. Candidate Variables & Data Sources

We audited six candidate indicators across the 76-country universe for 1990–2025, prioritizing official, traceable multilateral databases:

| Candidate Name | World Bank Indicator Code | Source Database | Primary Source Agency | Native Units | Earliest Available Year | Latest Available Year | Requires Transformation? |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **Private credit / GDP** | `FS.AST.PRVT.GD.ZS` | World Development Indicators (WDI) | IMF International Financial Statistics (IFS) / OECD | % of GDP | 1960 | 2024 | No (Level) |
| **Private credit growth** | `FS.AST.PRVT.GD.ZS` (derived) | WDI / IMF IFS | IMF IFS / OECD / Derived | % YoY change | 1961 | 2024 | Yes (Annual % change) |
| **Current account / GDP** | `BN.CAB.XOKA.GD.ZS` | World Development Indicators (WDI) | IMF Balance of Payments (BPM6) / OECD | % of GDP | 1970 | 2024 | No (Level) |
| **Exchange rate depreciation** | `PA.NUS.FCRF` (derived) | World Development Indicators (WDI) | IMF International Financial Statistics (IFS) | % YoY change | 1960 | 2024 | Yes (Annual % change) |
| **Unemployment (ILO)** | `SL.UEM.TOTL.ZS` | World Development Indicators (WDI) | ILO Modelled Estimates (ILOEST) | % of total labor force | 1991 | 2024 | No (Level) |
| **Unemployment (National)** | `SL.UEM.TOTL.NE.ZS` | World Development Indicators (WDI) | ILO Labour Force Statistics (LFS) | % of total labor force | 1968 | 2024 | No (Level) |

---

## 4. Transformation Rules & Timing Protocol

All transformations adhere strictly to the project's temporal contract:
- **No Forward Information:** The value of predictor $X_{c, t}$ uses only information observed at calendar year $t$ or earlier to predict crisis onset at $t+1$.
- **No Cross-Sectional Leakage:** Transformations are computed strictly within each individual economy's time series (`groupby('country_code')`).
- **No Label Conditioning:** Zero crisis outcome labels are used in transformation or feature definition.

### Exact Mathematical Formulations

1. **Private Credit / GDP (`private_credit_pct_gdp`):**
   $$X_{c, t}^{\text{cred}} = \text{Credit/GDP}_{c, t}$$
   Used directly in levels (% of GDP) as reported by the World Bank.

2. **Private Credit Growth (`private_credit_growth`):**
   $$X_{c, t}^{\text{cred\_growth}} = \left( \frac{\text{Credit/GDP}_{c, t} - \text{Credit/GDP}_{c, t-1}}{\text{Credit/GDP}_{c, t-1}} \right) \times 100$$
   Requires non-zero positive denominator. If $\text{Credit/GDP}_{c, t-1}$ is missing or $\le 0$, the value is set to `NaN`.

3. **Current Account Balance / GDP (`current_account_pct_gdp`):**
   $$X_{c, t}^{\text{ca}} = \text{Current Account/GDP}_{c, t}$$
   Used directly in levels (% of GDP) as reported under IMF BPM6 standards.

4. **Exchange Rate Depreciation (`exchange_rate_depreciation`):**
   Let $E_{c, t}$ denote the period-average official exchange rate in Local Currency Units per US Dollar ($\text{LCU} / \text{USD}$).
   $$X_{c, t}^{\text{deprec}} = \left( \frac{E_{c, t} - E_{c, t-1}}{E_{c, t-1}} \right) \times 100$$
   - $X_{c, t} > 0 \implies$ Local currency depreciates against the USD (requires more LCU to purchase 1 USD).
   - $X_{c, t} < 0 \implies$ Local currency appreciates against the USD.

5. **Unemployment Rate (`unemployment_rate_ilo`):**
   $$X_{c, t}^{\text{unemp}} = \text{Unemployment Rate}_{c, t}$$
   Used directly in levels (% of total labor force) from harmonized ILO modeled estimates.

---

## 5. Coverage Audit

The candidate indicators were merged onto the locked 76-country panel ($2,736$ country-years from 1990 to 2025). The table below reports the complete coverage audit across the entire sample and within the locked 2000–2008 validation window ($9 \times 76 = 684$ test observations, 19 forward crisis events):

| Candidate Name | Indicator Code | Available Countries | Full Coverage (1990–2025) | Available Obs (Full) | Validation Coverage (2000–2008) | Available Obs (2000–2008) | 2000–2008 Crises Retained | Total Crises Retained (1990–2025) | Eligible? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`private_credit_pct_gdp`** | `FS.AST.PRVT.GD.ZS` | 76 / 76 | 81.91% | 2,241 / 2,736 | **85.96%** | 588 / 684 | **18 / 19 (94.7%)** | 35 / 44 (79.5%) | **YES** |
| **`private_credit_growth`** | `FS.AST.PRVT.GD.ZS` (derived) | 76 / 76 | 79.09% | 2,164 / 2,736 | 82.16% | 562 / 684 | 18 / 19 (94.7%) | 34 / 44 (77.3%) | **NO** |
| **`current_account_pct_gdp`** | `BN.CAB.XOKA.GD.ZS` | 76 / 76 | 94.81% | 2,594 / 2,736 | **95.61%** | 654 / 684 | **19 / 19 (100.0%)** | 44 / 44 (100.0%) | **YES** |
| **`exchange_rate_depreciation`** | `PA.NUS.FCRF` (derived) | 76 / 76 | 96.86% | 2,650 / 2,736 | 99.85% | 683 / 684 | 19 / 19 (100.0%) | 44 / 44 (100.0%) | **NO** |
| **`unemployment_rate_ilo`** | `SL.UEM.TOTL.ZS` | 74 / 76 | 94.66% | 2,590 / 2,736 | **97.37%** | 666 / 684 | **19 / 19 (100.0%)** | 39 / 44 (88.6%) | **YES** |
| **`unemployment_rate_national`** | `SL.UEM.TOTL.NE.ZS` | 76 / 76 | 77.85% | 2,130 / 2,736 | 80.12% | 548 / 684 | 18 / 19 (94.7%) | 36 / 44 (81.8%) | **NO** |

*Table source: [`phase5d_candidate_coverage.csv`](file:///Users/macbookair/Documents/ChatGPT/finance/results/tables/phase5d_candidate_coverage.csv).*

---

## 6. Objective Inclusion Rule

Before examining correlation structures or potential model impacts, we pre-registered five non-negotiable eligibility criteria:

1. **Theoretical Linkage:** The candidate must measure an established systemic banking-crisis vulnerability channel (credit boom, external imbalance, currency mismatch, or debtor distress).
2. **Traceable Official Source:** The indicator must originate from an official international statistical repository (World Bank WDI, IMF IFS, ILO).
3. **Leakage-Free Construction:** The transformation must rely strictly on contemporaneous or historical observations ($\le t$).
4. **Validation Period Coverage Threshold ($\ge 85.0\%$):**
   - The indicator must be observed in at least **$85.0\%$ of country-years** across the locked 2000–2008 expanding evaluation window ($684$ observations).
   - It must retain at least **$90.0\%$ (at least 17 of 19) of the out-of-sample positive crisis events** during 2000–2008.
5. **Data Continuity & Non-Redundancy:**
   - The candidate must not suffer from catastrophic structural definition breaks (such as currency redenomination jumps).
   - It must not exhibit extreme linear collinearity ($|r| > 0.90$) with an existing baseline predictor.

---

## 7. Redundancy & Collinearity Assessment

We calculated full pairwise Pearson and Spearman correlation matrices across the 4 baseline predictors and candidate indicators on the 76-country panel:

### Pearson Correlation Matrix

| Variable | `gdp_growth` | `inflation` | `reserves_usd` | `reserves_yoy` | `private_credit` | `credit_growth` | `current_account` | `fx_depreciation` | `unemployment_ilo` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`gdp_growth`** | 1.0000 | -0.0354 | 0.0283 | 0.0272 | -0.1338 | -0.2558 | 0.0881 | -0.0144 | -0.1174 |
| **`inflation`** | -0.0354 | 1.0000 | -0.0128 | 0.0645 | -0.0357 | 0.0346 | -0.0073 | **0.9585** | -0.0075 |
| **`reserves_usd`** | 0.0283 | -0.0128 | 1.0000 | -0.0237 | 0.2903 | 0.0020 | 0.1485 | -0.0167 | -0.1018 |
| **`reserves_yoy`** | 0.0272 | 0.0645 | -0.0237 | 1.0000 | -0.0841 | -0.0553 | 0.0821 | 0.0608 | -0.0001 |
| **`private_credit`** | -0.1338 | -0.0357 | 0.2903 | -0.0841 | 1.0000 | 0.0337 | 0.1643 | -0.0126 | -0.0836 |
| **`credit_growth`** | -0.2558 | 0.0346 | 0.0020 | -0.0553 | 0.0337 | 1.0000 | -0.0889 | 0.0479 | -0.0804 |
| **`current_account`** | 0.0881 | -0.0073 | 0.1485 | 0.0821 | 0.1643 | -0.0889 | 1.0000 | 0.0086 | -0.1626 |
| **`fx_depreciation`**| -0.0144 | **0.9585** | -0.0167 | 0.0608 | -0.0126 | 0.0479 | 0.0086 | 1.0000 | 0.0070 |
| **`unemployment_ilo`**| -0.1174 | -0.0075 | -0.1018 | -0.0001 | -0.0836 | -0.0804 | -0.1626 | 0.0070 | 1.0000 |

*Table source: [`phase5d_correlations_pearson.csv`](file:///Users/macbookair/Documents/ChatGPT/finance/results/tables/phase5d_correlations_pearson.csv).*

### Key Econometric Findings from Correlation Structure
1. **Collinearity Between Exchange Rate Depreciation and Inflation ($r = 0.9585$):**
   Under purchasing power parity (PPP), high-inflation and hyperinflation regimes (such as Angola and Brazil in the 1990s) experience currency depreciations that match domestic price increases almost one-for-one. Including both variables introduces severe multicollinearity into linear models without supplying distinct information.
2. **Independence of Eligible Candidates:**
   The three eligible candidates (`private_credit_pct_gdp`, `current_account_pct_gdp`, `unemployment_rate_ilo`) exhibit modest correlations with all existing baseline features:
   - `private_credit_pct_gdp` with `reserves_usd`: $r = +0.2903$ (financial depth is higher in wealthier economies).
   - `current_account_pct_gdp` with `reserves_usd`: $r = +0.1485$ (surplus nations accumulate FX buffers).
   - `unemployment_rate_ilo` with `gdp_growth`: $r = -0.1174$ (reflects Okun's law: lower output growth coincides with higher unemployment).
   - All pairwise correlations among the eligible candidates remain strictly below $|r| < 0.20$.

---

## 8. Excluded Variables & Explicit Rationales

Three candidate indicators failed the pre-registered inclusion criteria and are excluded from the Extended Predictor Set:

### 1. `exchange_rate_depreciation` (EXCLUDED)
- **Primary Reason — Catastrophic Structural Data Break:**  
  The underlying World Bank series `PA.NUS.FCRF` reports exchange rates in Local Currency Units per USD. For the 12 Eurozone economies in our 76-country universe (AUT, BEL, DEU, ESP, FIN, FRA, GRC, IRL, ITA, NLD, PRT, CYP), national currencies were replaced by the Euro in 1999 (GRC in 2001, CYP in 2008). In 1999, Italy's LCU/USD jumped from 1,736 ITL to 0.938 EUR, creating an artificial **$-99.95\%$ depreciation spike**; Spain dropped by **$-99.37\%$**; France by **$-84.10\%$**; Germany by **$-46.68\%$**. Because 1999 is the immediate training year preceding test fold $T=2000$, these artificial conversion discontinuities introduce severe measurement error into expanding-window model estimation.
- **Secondary Reason — Extreme Collinearity:**  
  Across the panel, annual exchange rate depreciation exhibits a Pearson correlation of **$+0.9585$ with `inflation`**, violating Criterion 5.

### 2. `private_credit_growth` (EXCLUDED)
- **Primary Reason — Coverage Failure:**  
  Because annual percentage growth requires two consecutive years of non-missing credit observations, coverage in the 2000–2008 evaluation window drops to **$82.16\%$ (562 / 684 observations)**, failing the pre-registered $85.0\%$ threshold.
- **Secondary Reason — Concept Redundancy:**  
  Credit growth heavily overlaps with the level of private credit to GDP while introducing additional missingness.

### 3. `unemployment_rate_national` (EXCLUDED)
- **Primary Reason — Coverage Failure:**  
  National labor force surveys achieve only **$80.12\%$ coverage in 2000–2008 (548 / 684 observations)**, failing the $85.0\%$ threshold.
- **Secondary Reason — Definitional Heterogeneity:**  
  National unemployment definitions vary widely across emerging and developing economies (e.g., treatment of agricultural and informal labor), whereas the ILO modeled estimate (`SL.UEM.TOTL.ZS`) applies a standardized, harmonized econometrically consistent definition.

---

## 9. Final Pre-Specified Extended Predictor Set

The final Extended Predictor Set consists of exactly **7 predictors**: the 4 locked baseline predictors plus **3 additional, pre-registered macro-financial indicators**:

### Baseline Predictors (4 features, locked)
1. `gdp_growth`: GDP growth (annual %) — Real business cycle flow
2. `inflation`: Inflation, consumer prices (annual %) — Monetary/macroeconomic instability
3. `reserves_usd`: Total reserves (includes gold, current US$) — FX buffer scale
4. `reserves_usd_yoy_pct_change`: Annual % change in total reserves — Liquidity buffer acceleration/drain

### Additional Extended Predictors (3 features, pre-specified)
5. `private_credit_pct_gdp`: Domestic credit to private sector (% of GDP)  
   - *Source:* World Bank WDI (`FS.AST.PRVT.GD.ZS`)  
   - *Concept:* Domestic leverage and financial depth  
   - *Validation Coverage:* $85.96\%$ (588 / 684), crisis retention $94.7\%$ (18 / 19)  
   - *Missingness Handling:* Fold-isolated training median imputation (matching baseline architecture)
6. `current_account_pct_gdp`: Current account balance (% of GDP)  
   - *Source:* World Bank WDI (`BN.CAB.XOKA.GD.ZS`, IMF BPM6)  
   - *Concept:* External imbalance and sudden-stop vulnerability  
   - *Validation Coverage:* $95.61\%$ (654 / 684), crisis retention $100.0\%$ (19 / 19)  
   - *Missingness Handling:* Fold-isolated training median imputation
7. `unemployment_rate_ilo`: Total unemployment rate (% of total labor force, ILO modeled)  
   - *Source:* World Bank WDI (`SL.UEM.TOTL.ZS`)  
   - *Concept:* Real-economy distress and borrower default pressure  
   - *Validation Coverage:* $97.37\%$ (666 / 684), crisis retention $100.0\%$ (19 / 19)  
   - *Missingness Handling:* Fold-isolated training median imputation (imputing 1990 where ILO series begins in 1991)

---

## 10. Research-Integrity & Pre-Registration Checklist

- [x] **Zero Model Fitting:** No predictive models (Logistic Regression, Random Forest, or others) were estimated during Phase 5D.
- [x] **Zero Test Performance Conditioning:** Variables were screened and selected strictly on economic theory, data quality, and coverage metrics—not on out-of-sample performance metrics.
- [x] **Country Universe Preserved:** The 76-country universe remains completely unchanged.
- [x] **Validation Design Preserved:** Historical expanding-window evaluation ($T \in [2000, 2008]$) remains unchanged.
- [x] **Target Definition Preserved:** Forward systemic banking crisis onset ($t \to t+1$) remains unchanged.
- [x] **Baseline Intact:** Phase 5B baseline outputs and Phase 5C sensitivity outputs remain unaltered.
- [x] **Reproducibility:** Dedicated module [`predictor_audit.py`](file:///Users/macbookair/Documents/ChatGPT/finance/src/crisis_ews/evaluation/predictor_audit.py), CLI subcommand `run-phase-5d`, and 56 passing unit tests.
