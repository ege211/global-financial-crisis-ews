# Phase 1 data and label audit

Audit date: 2026-09-04. Scope: repository design and source-access audit; no full-sample model has been trained.

## What was inspected

All package modules, CLI commands, YAML configuration, tests, raw/interim/processed data contracts, dashboard, and existing methodology documentation were reviewed. No notebooks exist yet; this is acceptable at this phase because the reproducible CLI is the authoritative path.

## Already sound

- Annual country-year panel architecture, ISO-3 identifiers, editable country and feature configuration.
- Regularized logistic regression as the primary baseline and a constrained random forest only as a comparator.
- Expanding temporal folds with a fresh preprocessing pipeline per fold. Scaling and median imputation are fitted within the fold's training data.
- A forward annual target that excludes the crisis-onset year, class-weighted training, and rare-event-oriented metrics.
- Explicit refusal to bundle undocumented crisis dates or report imagined model performance.

## Corrections made during this audit

1. The World Bank collector now preserves the unmodified API JSON response in `data/raw/`; parsed long-form data is stored only in `data/interim/`.
2. Pagination is handled explicitly instead of assuming one API page.
3. The pipeline now produces a full requested-window country coverage report, including complete-case years, crisis-label counts, and missingness over all requested country-year-variable cells.
4. Label construction now selects an explicit configured crisis type (`systemic_banking`) rather than mixing a future composite chronology into the target.
5. New tests cover crisis/pre-crisis/post-crisis/non-crisis years, countries without events, repeated events, overlapping horizons, and training-only preprocessing.

## Remaining methodological constraints

- WDI values are revised historical series. They approximate, but do not reproduce, the information vintage available at year end. Results using them must be described as historical backtests with a revision-risk limitation.
- Annual observations cannot establish a precise twelve-month real-time information set. The estimand is therefore: *given the annual values recorded for year t, does a documented systemic banking crisis start in calendar year t+1?*
- Parsed interim data omit API records with null values, while preserved raw JSON retains them. Missingness is calculated against the complete requested country-year-variable grid, so nulls are not silently treated as zeros.
- The present 28-country sample is configured before seeing labels or performance. It must be retained, expanded, or excluded only through a documented data-quality rule.

## Gate before empirical modeling

Do not run a full panel or interpret model metrics until the chronology has been manually imported from the audited source, source/version metadata are recorded, and the generated country coverage report has been reviewed.
