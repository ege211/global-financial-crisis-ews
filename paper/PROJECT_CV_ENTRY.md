# Curriculum Vitae & Application Entries: Global Financial Crisis EWS

This document provides standardized, fact-checked descriptions of the research project for graduate school applications (Warwick, UCL, Durham, Bath, Manchester, KCL), quantitative finance resumes, and academic CVs.

All claims reflect locked empirical results from the repository ($N=684$ out-of-sample country-years across 76 economies, 1990–2025 panel, 9 expanding-window folds, IMF Laeven & Valencia ground truth).

---

### Version A: One-Line Summary
*Ideal for single-line project listings or LinkedIn project summaries.*

> **Global Financial Crisis Early Warning System (Independent Research):** Designed a leak-free, expanding-window temporal validation framework across a 76-country panel (1990–2025) using World Bank and IMF data, demonstrating that credit-augmented logistic regression achieves 47.37% out-of-sample recall for systemic banking crises but exhibits regime-specific concentration in the 2007 pre-GFC fold.

---

### Version B: Two-Bullet Concise Description
*Ideal for standard CVs, resumes, and graduate school activity sections.*

> **Global Financial Crisis Early Warning System | Quantitative Macro-Finance Project** *(2026)*
> - Engineered an end-to-end, leak-free early warning pipeline predicting $t \to t+1$ systemic banking crisis onsets across 76 economies using World Bank WDI and IMF (Laeven & Valencia, 2026) data, evaluated across nine expanding-window temporal folds ($N=684$ country-years).
> - Evaluated linear logistic regression against random forest ensembles under severe rare-event class imbalance (2.78% base rate), finding that private credit depth improved pooled PR-AUC by +23.7% and recall to 47.37%, while identifying that random forests suffered from base-rate probability compression and the linear model's predictive advantage was concentrated in the 2007 pre-GFC shock wave.

---

### Version C: Four-Bullet Detailed Application Description
*Ideal for academic CVs, postgraduate research statements, and technical project appendices.*

> **Global Financial Crisis Early Warning System: Macro-Financial Predictability & Temporal Generalization** *(2026)*
> - **Empirical Architecture & Data Pipeline:** Assembled a balanced 76-country macroeconomic panel (1990–2025, $N=2,736$) using the World Bank API and ground-truth systemic banking crisis dates from IMF WP/26/94; structured a leak-free $t \to t+1$ forward target evaluated over nine sequential expanding-window temporal folds ($T=2000,\dots,2008$; $N=684$ test country-years, 19 crisis onsets).
> - **Model Specification & Rare-Event Optimization:** Implemented cost-sensitive Logistic Regression and balanced Random Forest benchmarks; demonstrated that augmenting macroeconomic flows with private credit depth ($\beta = +0.5142, \text{OR} = 1.6723$) and current account balance increased pooled PR-AUC from 0.0278 to 0.0344 (+23.7%), ROC-AUC to 0.5734, and out-of-sample recall from 21.05% to 47.37% at $\tau = 0.50$.
> - **Non-Parametric Ensemble Diagnostics:** Diagnosed why Random Forest ensembles failed to outperform linear models despite lower Brier score (0.1347 vs. 0.2482), identifying that tree averaging under 2.78% class prevalence compresses predicted probabilities toward the base rate, suppressing operational crisis signals (missing 17 of 19 events, 10.53% recall).
> - **Regime Stability & Mechanism Dissection:** Conducted fold-by-fold temporal audits and institutional case studies, proving that predictive gains were concentrated in the 2007 pre-GFC fold (15/19 pooled crises; 60.0% recall, 31.03% precision), whereas out-of-sample recall collapsed to 0.0% across the remaining eight non-2007 folds ($0/4$ crises) due to unmodeled cross-border wholesale contagion and idiosyncratic emerging-market shocks.

---

### Key Verification Metrics Reference Table
*Quick reference for interviews and application queries:*

| Metric / Parameter | Value / Detail | Methodological Rationale |
| :--- | :--- | :--- |
| **Economies Analyzed** | 76 countries (28 advanced, 48 emerging) | Objective, continuous coverage 1990–2025 |
| **Out-of-Sample Window** | 2000–2008 ($N = 684$ country-years) | Strictly prospective expanding-window evaluation |
| **Crises in Test Set** | 19 events (unconditional base rate: 2.78%) | Ground truth from IMF Laeven & Valencia (WP/26/94) |
| **Primary Model** | Extended Logistic Regression (6 variables) | Cost-sensitive weighted loss, threshold $\tau = 0.50$ |
| **Pooled Discrimination** | PR-AUC: 0.0344 (+23.7% over baseline), ROC-AUC: 0.5734 | PR-AUC prioritized over ROC-AUC for class imbalance |
| **Pooled Recall / Precision** | Recall: 47.37% (9/19), Precision: 4.59% (9/196) | False alarms: 187; misses: 10 |
| **2007 Fold Performance** | Recall: 60.0% (9/15), Precision: 31.03%, PR-AUC: 0.2647 | 15 of 19 pooled crises clustered in 2007 fold |
| **Non-2007 Performance** | Recall: 0.0% (0/4), PR-AUC: 0.0051, ROC-AUC: 0.2301 | Fails on cross-border wholesale & idiosyncratic crises |
| **Random Forest Contrast** | Recall: 10.53% (2/19), Brier: 0.1347 | Probability compression toward 2.78% base rate |
| **Test Suite & Tooling** | 80 automated tests, Ruff clean, full CLI | `pytest` coverage of leak prevention and synthesis |
