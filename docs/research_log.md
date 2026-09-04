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
