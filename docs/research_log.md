# Research log

## 2026-09-04 — initial architecture

- Created a reproducible MVP with an official World Bank ingestion path and no bundled economic values or crisis labels.
- Chose a one-year annual forward label for systemic banking-crisis onsets. The chronology must be supplied from a cited reputable source before analysis.
- Selected regularized logistic regression as the baseline and a constrained random forest only as a comparison.
- Key limitation: World Bank API values are revised historical values, so this baseline is not a vintage-real-time backtest.
- No data have been downloaded, no chronology has been imported, and no performance results exist at this point.

Future entries should record data-source versions, exclusion decisions, revised hypotheses, failed runs, and all robustness specifications before results are interpreted.

## 2026-09-04 — Phase 1 audit and limited raw-data retrieval

- Audited the collector, labels, transformations, temporal evaluation, and documentation; recorded findings in `docs/PHASE_1_AUDIT.md`.
- Verified the official World Bank API endpoint and downloaded an ignored seven-country, 1995–2017 raw JSON validation sample. Details, checksums, and coverage are recorded in `docs/PIPELINE_VALIDATION_DATA_AUDIT.md`.
- Rejected private-credit/GDP from the *pipeline-validation profile* only because of observed structural coverage gaps for DEU, IDN, and TUR. This was done before labels/performance were available.
- Recommended the 2026 IMF Laeven–Valencia systemic banking-crisis chronology, but did not import or transcribe any dates.
- The environment lacks Python 3.11+ and every project dependency. No clean panel, labels, model evaluation, or empirical result has been generated.

## 2026-09-04 — Phase 2 real validation dataset

- Downloaded and SHA-256 verified the official IMF WP/26/94 data ZIP and imported only the exact crosswalk-matched rows for the seven-country profile.
- Built a real 1995–2017 World Bank panel from preserved official JSON: 161 country-year rows and 483 non-missing input variable cells.
- Constructed the one-year-forward systemic-banking-crisis label. There are three positive labels in the feature period and none in the pre-specified 2008–2017 test block.
- Stopped before Logistic Regression and Random Forest. The test block has no positive cases, so the intended predictive metrics are not estimable. No split, label, threshold, country, or feature definition was changed to obtain a score.

## 2026-09-04 — Phase 3 expanded-sample coverage audit

- Created a 55-country candidate pool before expanded-sample model results, with explicit IMF-name crosswalk and World Bank metadata.
- Downloaded the three Phase 2-approved core series for 1990–2025, retained unmodified raw JSON, and imported 72 traceable non-borderline chronology episodes across the candidate pool.
- Applied a complete-core-coverage inclusion rule before fitting any expanded-sample model. This produced a 44-country, 1,584-row primary panel; 11 candidates were excluded solely for incomplete core coverage.
- The primary panel has 38 forward-positive labels, but all occur no later than 2008. A post-2008 contiguous final holdout has zero positives, so Phase 3 stopped before model-performance evaluation. This does not alter the Phase 2 historical result or split.

## 2026-09-04 — Phase 4A candidate universe expansion & methodology lock

- Expanded the research scope beyond the 55-country Phase 3 candidate pool to the complete universe of 217 World Bank non-aggregate economies.
- Implemented deterministic typography normalization and an explicit 9-country override table (`config/imf_2026_world_bank_name_overrides.csv`) to resolve 100% of the 120 IMF source country names to World Bank ISO-3 codes without unmapped entries.
- Downloaded all-country World Bank metadata and core raw JSON payloads (1990–2025) for GDP growth, inflation, and international reserves to `data/raw/expanded_research/`.
- Committed checkpoint `f61e9be` and drafted `docs/PHASE_4B_METHODOLOGY_LOCK.md`, pre-registering fixed inclusion rules, variable definitions, and explicit prohibitions against post-hoc outcome conditioning.

## 2026-09-04 — Phase 4B objective universe audit & validation design lock

- Applied the pre-specified complete-core coverage rule across all 217 candidate economies before looking at crisis labels or model metrics. Exactly 76 economies met complete annual coverage (1990–2025) across all three series; 141 were excluded with documented reasons in `results/tables/expanded_country_coverage.csv`.
- Extracted and mapped the complete IMF WP/26/94 database: 161 non-borderline systemic banking crisis episodes globally, with 3 author-identified borderline episodes excluded (Nicaragua 2018, Sri Lanka 2023, Vietnam 2022).
- Constructed the 76-country, 2,736-row modeling panel with $t \to t+1$ forward target. Exactly 44 positive labels exist across the 36-year period (1.61% positive rate).
- Empirical finding for validation design: Across 2009–2025 (1,292 country-years), there is only 1 positive forward label (Cyprus $t=2010$ forecasting 2011 onset; Nigeria 2009 onset corresponds to $t=2008$). Years 2011–2025 have 0 positive forward labels in the 76-country complete-data universe.
- Validation design lock: In compliance with Task C rules, no artificial post-2008 holdout was manufactured. The temporal validation design is locked as a historical expanding-window backtest (training starting 1990, evaluation folds $T \in [2000, 2008]$). The post-2008 period is documented as a structural limitation (a post-GFC low-onset regime).
- Software quality: Resolved KeyError in `country_coverage_report`, corrected unquoted CSV fields in overrides table, enabled system site packages and setuptools discovery in `.venv`, integrated CLI subcommand `run-expanded-research`, and added unit tests in `tests/test_phase4b_expanded.py`. All 22 tests pass.
- In strict adherence to research-integrity constraints, zero predictive models were trained. Phase 5 model estimation remains locked.

## 2026-09-04 — Phase 5A modeling infrastructure & pre-fit audit

- Configured the 4 pre-registered core predictors (`gdp_growth`, `inflation`, `reserves_usd`, `reserves_usd_yoy_pct_change`) in `config/expanded_research.yaml`.
- Extended the expanding-window infrastructure in `src/crisis_ews/evaluation/walk_forward.py` and `src/crisis_ews/cli.py` with `test_end_year: 2008` bounding to isolate the 9 historical evaluation folds ($T \in [2000, 2008]$) from the post-2008 low-onset regime.
- Built a dedicated pre-fit audit suite in `src/crisis_ews/evaluation/pre_fit_audit.py` and CLI subcommand `pre-fit-audit` (alias `audit-pre-fit`).
- Verified modeling panel integrity: 76 countries, 2,736 rows, 44 positive forward labels ($t \to t+1$), zero duplicate observations.
- Audited the 9 expanding-window folds: 684 cumulative test rows, 19 positive crisis labels; every training fold contains $\ge 24$ positive events, ensuring valid dual-class training.
- Confirmed zero data leakage across all folds ($\max(\text{train\_year}) < \min(\text{test\_year})$) and verified fold-isolated preprocessing (median imputation and scaling fitted strictly on training rows).
- Evaluated pre-fit feature distributions and multicollinearity: all pairwise Pearson and Spearman correlations among predictors are strictly below $|0.15|$. Missingness is exactly 0 across all features except 76 values in `reserves_usd_yoy_pct_change` strictly confined to 1990 due to the 1-year lag shift.
- Exported audit tables: `results/tables/expanded_pre_fit_folds.csv`, `results/tables/expanded_pre_fit_features.csv`, and `results/tables/expanded_pre_fit_correlations.csv`.
- Documented the pre-fit methodology and hypotheses in `docs/PHASE_5A_PRE_FIT_AUDIT.md`.
- In strict adherence to research integrity, ZERO predictive models were trained and ZERO performance metrics were calculated.

## 2026-09-04 — Phase 5B initial model estimation

- Executed the first out-of-sample model estimation under the locked expanding-window protocol ($T \in [2000, 2008]$) on the 76-country, 1990–2025 panel (684 test observations, 19 positive forward crisis events).
- Evaluated the pre-specified baseline: regularized $L_2$ Logistic Regression ($C=1.0$, `class_weight='balanced'`, fold-fitted median imputation and standardization). Out-of-sample pooled results: PR-AUC = 0.0278, ROC-AUC = 0.5106, Precision = 0.0189, Recall = 0.2105, F1 = 0.0346, Brier = 0.2512.
- Evaluated the pre-specified comparator: constrained Random Forest (`n_estimators=400`, `max_depth=4`, `min_samples_leaf=5`, `class_weight='balanced'`, fold-fitted median imputation). Out-of-sample pooled results: PR-AUC = 0.0256, ROC-AUC = 0.4712, Precision = 0.0242, Recall = 0.2105, F1 = 0.0435, Brier = 0.1524.
- In test year 2007 (anticipating the 2008 GFC, 15 crisis countries), Logistic Regression achieved an intra-fold ROC-AUC of 0.6590 and PR-AUC of 0.2743, but generated 10 false alarms (Recall = 20.0%, False Negative Rate = 80.0%).
- Honest econometric finding: Neither model demonstrates robust predictive discrimination over the 2000–2008 backtest as a whole; PR-AUC approximates the unconditional event rate (0.0278), and high false positive counts (208 for LR, 161 for RF) highlight that 4 core flow macroeconomic variables alone cannot reliably anticipate banking panics without credit, leverage, or balance-sheet vulnerability indicators.
- Feature interpretation: Logistic Regression coefficients in the final fold ($T=2008$, trained 1990–2007) align directionally with economic hypotheses: real GDP growth is counter-cyclical ($\beta = -0.1125, \text{OR} = 0.8936$), inflation is pro-cyclical ($\beta = +0.1778, \text{OR} = 1.1946$), and reserve growth is protective ($\beta = -0.0704, \text{OR} = 0.9320$). Tree importances place highest weight on reserve change (41.9%) and inflation (23.0%). All interpretations are explicitly non-causal.
- Out-of-sample calibration: Class weighting shifts predicted probabilities toward $\approx 0.50$, resulting in negative Brier Skill Scores relative to unweighted climatology ($\text{BSS} = -8.30$ for LR, $-4.64$ for RF).
- Software quality & testing: Implemented `src/crisis_ews/evaluation/initial_models.py`, added `run-initial-models` CLI command, generated predictions in `results/model_outputs/`, exported comparison and fold tables in `results/tables/`, and added 11 unit tests in `tests/test_phase5b_models.py`. All 42 tests pass.
- In strict adherence to research integrity, ZERO hyperparameter tuning, ZERO threshold optimization, and ZERO post-hoc validation changes were performed.

## 2026-09-04 — Phase 5C robustness & sensitivity analysis

- Executed the four pre-specified, independent robustness and sensitivity analyses across the locked 76-country panel and 9 expanding validation folds ($T \in [2000, 2008]$), without modifying or overwriting the locked Phase 5B baseline.
- Analysis A (Inflation Outlier Sensitivity): Implemented fold-isolated 1st/99th percentile winsorization of `inflation` learned strictly from training folds. Random Forest results are numerically invariant (PR-AUC = 0.0256, ROC-AUC = 0.4712) due to decision tree invariance to monotonic ordinal order statistics. Logistic Regression shows variance compression: false alarms drop from 208 to 128 (Brier improves from 0.2512 to 0.2223), but True Positives fall from 4 to 1 (Recall = 0.0526) and ROC-AUC drops to 0.3893.
- Analysis B (Class-Weight Sensitivity): Tested unweighted loss optimization (`class_weight=None`). Both Logistic Regression and Random Forest predict probabilities $< 0.50$ across all 684 test observations, generating 0 warnings (0 TP, 0 FP, 19 FN, 665 TN). This demonstrates the "Brier paradox" in severe class imbalance (unconditional rate 2.78%): unweighted models achieve artificially low Brier scores ($\approx 0.028$) by predicting tranquility everywhere, rendering them completely useless as early warning systems.
- Analysis C (Decision Threshold Sensitivity): Evaluated $\tau \in \{0.25, 0.50, 0.75\}$ on locked Phase 5B predictions. At $\tau=0.25$, Logistic Regression captures 100% of crises (19/19) but flags 683 of 684 country-years (FPR = 99.85%), creating complete alarm fatigue. At $\tau=0.75$, both models miss 100% of crises (FNR = 100.0%). Random Forest at $\tau=0.25$ detects 6 of 19 crises (Recall = 31.58%) with FPR = 48.72% (324 false alarms).
- Analysis D (Fold-Level Stability): Documented the severe temporal clustering of systemic crises. Of the 19 test events, 15 (78.9%) occur in test year 2007 (anticipating the 2008 GFC). In 4 folds (2000, 2003, 2004, 2005), there are 0 positive events; ROC-AUC and PR-AUC are mathematically undefined and recorded honestly as "Not estimable". In 3 folds (2001, 2002, 2006), exactly 1 crisis occurs, inducing wild metric swings. In 2007, Logistic Regression achieves intra-fold ROC-AUC = 0.6590 and PR-AUC = 0.2743 with 3/15 detections and 10 false alarms.
- Exported four audited tables to `results/tables/`: `expanded_robustness_inflation.csv`, `expanded_robustness_class_weight.csv`, `expanded_threshold_sensitivity.csv`, and `expanded_fold_stability.csv`. Preserved separate sensitivity prediction files in `results/model_outputs/`.
- Documented full methodology and interpretations in `docs/PHASE_5C_ROBUSTNESS_RESULTS.md`. Integrated CLI subcommand `run-phase-5c` (aliases `robustness`, `robustness-analysis`) and added 7 unit tests in `tests/test_phase5c_robustness.py`. All 49 repository tests pass.
- Research integrity confirmed: NO baseline changes, NO parameter tuning, NO threshold optimization, NO post-hoc feature selection.

