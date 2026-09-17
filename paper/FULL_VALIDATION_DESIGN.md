# Full-sample validation design and Phase 3 gate

## Fixed data and target

The Phase 3 primary panel contains 44 countries, annual 1990–2025 observations, and the three complete core inputs specified in `docs/PHASE_3_RESEARCH_PLAN.md`. The target remains `y(c,t)=1` when a non-borderline systemic banking-crisis onset occurs in year `t+1`. No Phase 2 artifact, test period, label, or feature was changed.

## Intended temporal procedure

The model architecture remains an expanding window. A model fitted on all rows through year `t` may predict only rows in `t+1`; the fold pipeline fits imputation and scaling on the training rows alone. Random splitting, full-sample scaling, and test-set threshold selection remain prohibited.

## Empirical validation gate

The expanded label audit found 38 positive labels, but every one occurs from 1990 through 2008; the 2009–2025 period has zero positives in this pre-specified country/label/feature sample. Thus a later contiguous final holdout cannot estimate rare-event discrimination or calibration. Selecting an earlier test window solely to include positive outcomes after viewing their dates would not create a genuine untouched final test.

Accordingly, Phase 3 does **not** authorize Logistic Regression or Random Forest performance claims. The prospective final test period is **not estimable** in this dataset. An expanding historical backtest through 2008 may be pre-specified as a later descriptive sensitivity exercise, but it must not be described as final out-of-sample evidence and is outside the current Phase 3 results.

## Required next design decision

Before models are fitted, choose one of the following and record it as a new research phase: (1) expand the candidate universe using the same fixed chronology and coverage rules to seek later non-borderline onsets; or (2) explicitly frame a pre-2009 expanding backtest as descriptive historical evidence, with no final holdout claim. Neither option permits changing the Phase 2 or Phase 3 primary label definition.
