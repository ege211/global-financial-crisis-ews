# Phase 4B Validation Design and Temporal Gate Lock

**Date Locked:** 2026-09-04  
**Status:** Pre-registered and locked before any predictive model training.

---

## 1. Fixed Candidate Pool, Inclusion Rule, and Final Sample

* **Candidate Universe:** All 217 World Bank economies with non-aggregate regional identifiers (`region != "Aggregates"`) in the official country metadata.
* **Objective Inclusion Rule:** Complete annual coverage (1990–2025, 36 years) for all three core input series:
  1. GDP growth (`NY.GDP.MKTP.KD.ZG`)
  2. Inflation (`FP.CPI.TOTL.ZG`)
  3. Total international reserves (`FI.RES.TOTL.CD`)
* **Pre-outcome Filtering:** The inclusion rule was evaluated and applied solely on raw data availability **before** linking crisis labels or estimating any predictive model.
* **Final Primary Sample:** Exactly **76 countries** meet the complete-core requirement. The remaining 141 economies are excluded with explicit, auditable reasons recorded in [`results/tables/expanded_country_coverage.csv`](../results/tables/expanded_country_coverage.csv).
* **Modeled Panel Structure:** 76 countries × 36 years = **2,736 country-year observations**.

---

## 2. Supervised Target Contract

* **Target Definition:** For country $c$ in annual observation year $t$:
  $$y(c, t) = 1 \quad \text{if a qualifying systemic banking-crisis onset starts in calendar year } t+1, \quad 0 \text{ otherwise.}$$
* **Lookahead Prevention:** The contemporaneous crisis year $t$ is strictly excluded from early-warning features.
* **Chronology Source:** Laeven & Valencia (2026), *Systemic Banking Crises Database: 1970–2025*, IMF Working Paper 2026/094.
* **Exclusion of Borderline Episodes:** The three explicitly identified borderline events in WP/26/94 (Nicaragua 2018, Sri Lanka 2023, Vietnam 2022) are excluded from the primary target.
* **Resulting Labels:** 44 positive forward labels occur within the 1990–2025 feature panel (positive rate: 1.61%).

---

## 3. Empirical Crisis Distribution & Post-2008 Finding

An audit of positive forward labels across the 76-country panel by year reveals a stark structural distribution:

| Period | Feature Years ($t$) | Total Country-Years | Positive Labels ($y=1$) | Positive Rate | Onset Episodes Represented |
|---|---|---:|---:|---:|---|
| Pre-GFC Historical | 1990–2006 | 1,292 | 27 | 2.09% | 1991–2007 onsets (e.g. Nordic, Asian, Tequila crises) |
| Global Financial Crisis | 2007 | 76 | 15 | 19.74% | 2008 systemic banking crises across 15 economies |
| Immediate GFC Aftermath | 2008 | 76 | 1 | 1.32% | Nigeria 2009 onset |
| Post-2008 Regime | 2009–2025 | 1,292 | 1 | 0.08% | Cyprus 2011 onset ($t=2010$) only |
| Post-2010 Regime | 2011–2025 | 1,140 | 0 | 0.00% | Zero onsets in the 76-country complete-data sample |

---

## 4. Temporal Validation Gate Decision (Task C Compliance)

### The Methodological Problem
A standard prospective holdout block (e.g. 2011–2025, or 2015–2025) contains **0 positive crisis events** in countries with complete World Bank macro series. Even extending the holdout to 2009–2025 yields only **1 positive event** across 1,292 country-years.
* Area Under the ROC Curve (ROC-AUC) is mathematically undefined on a single-class test fold.
* Precision-Recall AUC (PR-AUC), sensitivity, and calibration metrics are either uncomputable or subject to infinite relative error on $N_{\text{pos}} \le 1$.

### Locked Validation Structure
Under strict research integrity rules (Task C):
1. **No Manufactured Holdout:** We do not retroactively modify the crisis definition, relax the complete-core inclusion rule, or alter the $t \to t+1$ target to force artificial positive events into a post-2008 holdout.
2. **Defensible Historical Expanding-Window Evaluation:**
   * Minimum training window: 10 years ($1990 \le \text{year} \le 1999$).
   * First test fold: $T = 2000$.
   * Expanding annual test folds: $T \in [2000, 2008]$.
   * In each fold $T$, models are trained strictly on observations with $\text{year} < T$. Preprocessing (median imputation, standardization) is fitted exclusively on the training fold.
3. **Descriptive Post-2008 Low-Onset Regime:**
   * Predictions for 2009–2025 are tracked separately as a descriptive out-of-sample evaluation on a low-incidence macro regime.
   * Model outputs over 2009–2025 will be reported primarily in terms of false alarm rates and specificity, with explicit acknowledgement that true positive sensitivity is unmeasurable during this zero/near-zero event regime.
4. **Predictive Models Remain Locked:** In accordance with Phase 4B restrictions, no predictive models have been fitted. Model training and evaluation will take place exclusively in Phase 5 under this locked protocol.
