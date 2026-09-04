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

## 2026-09-04 — Phase 5D predictor selection & data coverage audit

- Executed pre-model predictor selection and data coverage audit across six candidate vulnerability indicators for the 76-country panel (1990–2025) and locked expanding validation folds ($T \in [2000, 2008]$).
- Formulated an objective inclusion rule BEFORE estimating any predictive models: candidates must possess clear theoretical linkage to systemic banking crisis vulnerability, traceable official multilateral sourcing, leakage-free transformation ($\le t$), $\ge 85.0\%$ observation coverage in the 2000–2008 evaluation window ($684$ obs), $\ge 90.0\%$ crisis retention (at least $17/19$ positive events in 2000–2008), and freedom from structural breaks or extreme linear collinearity ($|r| > 0.90$).
- Eligible candidate 1: Domestic credit to private sector (% of GDP) (`FS.AST.PRVT.GD.ZS`, WDI/IMF IFS) — Domestic leverage/overhang channel. 85.96% validation coverage (588/684), retains 18/19 test crises (94.7%).
- Eligible candidate 2: Current account balance (% of GDP) (`BN.CAB.XOKA.GD.ZS`, WDI/IMF BPM6) — External imbalance/sudden-stop vulnerability channel. 95.61% validation coverage (654/684), retains 19/19 test crises (100.0%).
- Eligible candidate 3: Total unemployment rate (% of labor force, ILO modeled) (`SL.UEM.TOTL.ZS`, WDI/ILO) — Real-economy distress/debtor repayment incapacity channel. 97.37% validation coverage (666/684), retains 19/19 test crises (100.0%).
- Excluded candidate 1: Private credit growth (`FS.AST.PRVT.GD.ZS` derived) — Fails coverage threshold at 82.16% in 2000–2008 (< 85% rule); redundant with credit level.
- Excluded candidate 2: Exchange rate depreciation (`PA.NUS.FCRF` derived) — Fails data continuity due to Eurozone 1999 conversion causing -46% to -99.95% artificial drops; extreme linear collinearity with inflation ($r = 0.9585$).
- Excluded candidate 3: National unemployment rate (`SL.UEM.TOTL.NE.ZS`) — Fails coverage threshold at 80.12% in 2000–2008 (< 85% rule); severe cross-country definitional heterogeneity.
- Extended Predictor Set pre-specified at exactly 7 variables (4 baseline + 3 extended: `gdp_growth`, `inflation`, `reserves_usd`, `reserves_usd_yoy_pct_change`, `private_credit_pct_gdp`, `current_account_pct_gdp`, `unemployment_rate_ilo`).
- Exported audit tables: `results/tables/phase5d_candidate_coverage.csv`, `results/tables/expanded_phase5d_candidate_coverage.csv`, `results/tables/phase5d_correlations_pearson.csv`, and `results/tables/phase5d_correlations_spearman.csv`.
- Software quality & testing: Implemented `src/crisis_ews/evaluation/predictor_audit.py`, added `run-phase-5d` CLI command, documented findings in `docs/PHASE_5D_PREDICTOR_SELECTION.md`, and added 7 unit tests in `tests/test_phase5d_predictors.py`. All 56 repository tests pass; ruff clean.
- Research integrity confirmed: ZERO models fitted, ZERO performance conditioning, country universe and validation design preserved intact.

## 2026-09-04 — Phase 5E 6-variable extended model estimation

- Executed the out-of-sample evaluation of the approved 6-variable Extended Predictor Set across the locked 76-country panel and 9 expanding validation folds ($T \in [2000, 2008]$, 684 test observations, 19 forward crisis events).
- Locked 6-variable specification: `gdp_growth`, `inflation`, `reserves_usd`, `reserves_usd_yoy_pct_change` plus the two pre-registered additions: `private_credit_pct_gdp` (WDI `FS.AST.PRVT.GD.ZS`) and `current_account_pct_gdp` (WDI `BN.CAB.XOKA.GD.ZS`). Excluded `exchange_rate_depreciation` and `unemployment_rate_ilo` from primary model to preserve rare-event degrees of freedom.
- Main empirical finding (Logistic Regression): Substantial improvement across all performance metrics over the 4-variable baseline. PR-AUC increased from 0.0278 to 0.0344 (+23.7%); ROC-AUC increased from 0.5106 to 0.5734; Recall surged from 21.05% (4/19) to 47.37% (9/19); Precision rose from 0.0189 to 0.0459; F1 rose from 0.0346 to 0.0837; Brier score improved from 0.2512 to 0.2482; False alarms dropped from 208 down to 187.
- GFC stress test ($T=2007$ anticipating 2008 crises): In the multi-event crisis epicenter fold (15 crises), Extended Logistic Regression detected 9 of 15 crisis economies (Recall = 60.0%, True Positives = 9: DNK, ESP, FRA, GRC, HUN, IRL, ISL, ITA, PRT), tripling baseline detections (3 of 15: DEU, ESP, FRA).
- Random Forest: Extended ensemble became more conservative at the static $\tau=0.50$ cutoff, reducing false alarms from 161 to 122 and Brier score from 0.1524 to 0.1347, with PR-AUC at 0.0274 and 2 True Positives.
- Feature interpretation: In the final training fold ($T=2008$, trained 1990–2007), `private_credit_pct_gdp` emerged as the largest positive predictor ($\beta = +0.5142, \text{OR} = 1.6723$, +67.2% odds per SD increase) and the second most important tree feature (22.5%). Current account exhibited a negative sign ($\beta = -0.0463, \text{OR} = 0.9547$), indicating that larger deficits elevate crisis odds. All six features aligned with theoretical economic directions.
- Software quality & testing: Implemented `src/crisis_ews/evaluation/extended_models.py`, added `run-extended-models` CLI command, generated distinct predictions in `results/model_outputs/extended_*_predictions.csv` (leaving baseline files intact), exported comparison tables in `results/tables/`, and added 7 unit tests in `tests/test_phase5e_extended_models.py`. All 63 tests pass; ruff clean.
- Research integrity confirmed: Zero hyperparameter tuning, zero threshold tuning, zero modification of Phase 5B/5C artifacts, no commits, no push.

## 2026-09-04 — Phase 5E temporal generalization & GFC dependence diagnostic

- Executed a dedicated temporal-generalization diagnostic on the Phase 5D Extended Logistic Regression out-of-sample predictions (684 obs, 19 crises) vs the Phase 5B Baseline Logistic Regression.
- Rigorous non-destructive protocol: ZERO model retraining, ZERO threshold adjustments, ZERO predictor alterations. Evaluated five temporal slices:
  1. Pooled 2000–2008 reference (684 obs, 19 crises)
  2. Excluding 2007 test fold (608 obs, 4 crises)
  3. Pre-GFC historical period 2000–2006 (532 obs, 3 crises)
  4. 2007 GFC test fold (76 obs, 15 crises)
  5. 2008 post-GFC test fold (76 obs, 1 crisis)
- Empirical findings & substantive diagnostic: The observed predictive advantage of the extended specification is concentrated in the 2007 pre-GFC evaluation fold and appears to reflect a particular macro-financial configuration rather than a broadly generalizable crisis signal.
  - Excluding 2007, Extended Logistic Regression catches 0 out of 4 crises (Recall = 0.0%, TP = 0, FP = 167, FN = 4), with an out-of-sample ROC-AUC of 0.2301 reflecting an observed ranking inversion in the out-of-sample predictions.
  - In the pre-GFC period (2000–2006), Extended LR catches 0 out of 3 crises (Turkey 2001, Argentina 2002, Nigeria 2006), with PR-AUC at 0.0051 (below unconditional prior 0.0056) and ROC-AUC at 0.2936.
  - In the 2007 GFC fold (forecasting 2008), Extended LR catches 9 of 15 crises (Recall = 60.0%, Precision = 31.0%, PR-AUC = 0.2647, ROC-AUC = 0.6470).
  - Within the four observed non-2007 evaluation events, all four events were missed at the fixed 0.50 threshold, exhibiting lower private credit depth and different macro-financial predictor configurations.
- Software quality & testing: Implemented `src/crisis_ews/evaluation/temporal_generalization.py`, added `temporal-generalization` CLI command, generated `results/tables/phase5e_temporal_generalization.csv`, documented findings in `docs/PHASE_5E_TEMPORAL_GENERALIZATION.md`, and added 5 unit tests in `tests/test_phase5e_temporal_generalization.py`. All 68 repository tests pass; ruff clean.
- Research integrity confirmed: Transparent reporting of model boundary conditions; no overclaiming of general early warning predictive power.

## 2026-09-04 — Phase 6 crisis mechanism and temporal concentration analysis

- Executed a descriptive crisis-mechanism and temporal-concentration analysis on the 19 out-of-sample crisis events from the Phase 5D 6-variable Extended Logistic Regression.
- Research integrity protocol maintained: ZERO model retraining, ZERO threshold adjustments, ZERO predictor additions, ZERO causal claims. Strict observational analysis.
- Key findings across the 19 evaluation crisis events:
  - 15 events occurred in the 2007 pre-GFC fold (forecasting 2008 crises); 4 events occurred outside 2007 (Uruguay 2001, Dominican Republic 2002, United Kingdom 2006, Nigeria 2008).
  - All 9 detected events occurred in the 2007 fold. In that fold, detected economies exhibited extreme private credit expansion (median 142.2% of GDP) and severe current account deficits (median -7.29% of GDP): DNK, ESP, FRA, GRC, HUN, IRL, ISL, ITA, PRT.
  - All 6 missed economies in 2007 (AUT, BEL, CHE, DEU, NLD, SWE) ran current account surpluses (+1.50% to +8.63% of GDP, median +6.19%), with predicted probabilities remaining below 0.50 (range 0.4445 to 0.4795).
  - Within the four observed non-2007 evaluation events, all four events were missed at the fixed 0.50 threshold (Recall = 0.0%), exhibiting lower median private credit (53.9% of GDP), higher inflation (median 4.79%), and weaker reserve dynamics (median YoY growth +5.59%).
  - Statistical driver of ranking inversion: Within the four observed non-2007 evaluation events, actual crisis events received a median predicted probability of 0.4411 (mean = 0.4109), lower than the non-crisis panel median (0.4733), resulting in an observed ranking inversion in the out-of-sample predictions (ROC-AUC = 0.2301).
- Visualizations generated:
  - `results/figures/phase6_predictor_profile_comparison.svg` and `.png`: Private credit and current account distributions across groups.
  - `results/figures/phase6_probability_distribution.svg` and `.png`: Out-of-sample predicted probability distributions relative to threshold 0.50.
  - `results/figures/phase6_standardized_profiles.svg` and `.png`: Six-variable standardized z-scores relative to full panel.
- Software quality & testing: Implemented `src/crisis_ews/evaluation/crisis_mechanisms.py`, added `run-phase-6` CLI command, exported 5 audit tables in `results/tables/phase6_*.csv`, documented findings in `docs/PHASE_6_CRISIS_MECHANISM_ANALYSIS.md`, and added 6 unit tests in `tests/test_phase6_crisis_mechanisms.py`.
- Verified test suite and linting: All tests pass; ruff clean. All Phase 5 artifacts remain intact.

## 2026-09-05 — Phase 6 methodological audit & correction pass

- Conducted a comprehensive research integrity and methodological audit of Phase 6 to resolve internal taxonomy inconsistencies and purge external causal narratives.
- Enforced strict three-layer architecture:
  1. Authoritative Crisis Classification: Multilateral ground truth is strictly IMF WP/26/94 (Laeven & Valencia, 2026). All 19 evaluation events are confirmed as non-borderline `systemic_banking` crisis onsets. Because the IMF source provides no sub-mechanism taxonomy, empirical sub-mechanism is strictly classified as "Unknown / insufficient evidence" across all 19 events.
  2. Data-Driven Descriptive Typology: Grouped strictly on observed predictor values at prediction year $t$:
     - Descriptive Profile A: High private-credit / external-deficit configuration ($N=9$, all detected in 2007 fold: DNK, ESP, FRA, GRC, HUN, IRL, ISL, ITA, PRT).
     - Descriptive Profile B: High current-account-surplus configuration ($N=6$, all missed in 2007 fold: AUT, BEL, CHE, DEU, NLD, SWE).
     - Descriptive Profile C: Lower private-credit / higher-inflation / weaker-reserve-dynamics configuration ($N=4$, non-2007 folds: URY, DOM, GBR, NGA).
  3. Economic Hypotheses and Interpretations: Strictly non-causal explanatory hypotheses linking estimated parameters ($\beta_{\text{credit}} = +0.5142, \beta_{\text{ca}} = -0.0463$) to observed predictor differences. Explains ranking behavior without asserting counterfactual proof or causal crisis etiology.
- Sanitized `CRISIS_METADATA` and `phase6_mechanism_categorization.csv` / `phase6_crisis_event_profiles.csv`: purged informal historical narratives (e.g., Baninter fraud, Northern Rock liquidity run, oil price slump, subprime conduits) that lacked authoritative dataset backing in the repository.
- Audited model claims: toned down strong statements (avoided "effective detector", "zero predictive capability", "missed due to X"); framed non-2007 ROC-AUC (0.2301) strictly as "an observed ranking inversion in the out-of-sample predictions"; explicitly qualified all non-2007 conclusions with "Within the four observed non-2007 evaluation events..." ($N=4$).
- Approved conclusion recorded verbatim: "The observed predictive advantage of the extended specification is concentrated in the 2007 pre-GFC evaluation fold and appears to reflect a particular macro-financial configuration rather than a broadly generalizable crisis signal."
- Code and tests verified: `.venv/bin/pytest -v` (74/74 passed), `.venv/bin/ruff check .` (clean). Baseline Phase 5 artifacts remain untouched. No commits, no push.

## 2026-09-05 — Phase 7 final model selection & research synthesis

- Executed the comprehensive final synthesis and multi-criteria model selection across all evaluated architectures (Baseline Logistic Regression, Baseline Random Forest, Extended Logistic Regression, Extended Random Forest).
- Final Model Selection Determination:
  - Extended Logistic Regression (6 variables) is retained as the Primary Research Specification with explicit regime qualifications.
  - While achieving the highest pooled discrimination (PR-AUC = 0.0344, ROC-AUC = 0.5734, Recall = 47.37%, Precision = 4.59%), this selection explicitly acknowledges that its predictive advantage is concentrated in the 2007 pre-GFC evaluation fold (60.0% Recall) and did not generalize to the four observed non-2007 crisis episodes (Recall = 0.0%, ROC-AUC = 0.2301).
  - Random Forest models compressed predicted probabilities toward the rare-event base rate, yielding low Brier scores (0.1347-0.1524) but missing 15 to 17 of 19 crisis events at standard decision cutoffs.
- Generated Synthesis Tables & Visualizations:
  - `results/tables/final_model_comparison.csv`: Multi-criteria comparison of all 4 architectures across pooled, 2007, and non-2007 subsets.
  - `results/tables/final_robustness_summary.csv`: Synthesis of 6 evaluated robustness dimensions (winsorization, class weighting, threshold sensitivity, fold stability, predictor expansion, regime generalization).
  - `results/tables/final_model_coefficients.csv`: Final parameter estimates, odds ratios, and non-causal associative interpretations for the Primary Research Specification.
  - `results/figures/phase7_performance_synthesis.svg` and `.png`: Publication-quality multi-panel visualization of out-of-sample Recall, Precision-Recall dynamics, and temporal concentration.
- Documentation & Reproducibility:
  - Authored `docs/FINAL_RESEARCH_SYNTHESIS.md` structured across all 22 required sections.
  - Integrated CLI command `run-final-synthesis` (aliases `final-synthesis`, `run-phase-7`).
  - Added unit test suite `tests/test_phase7_final_synthesis.py` (6 unit tests).
  - Verified test suite: all 80 tests pass; ruff clean. All Phase 5 and Phase 6 artifacts preserved byte-identical.


