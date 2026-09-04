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
