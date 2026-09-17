# Phase 3 research plan — recorded before expanded-sample model evaluation

## Question and hypothesis

Can a small, economically motivated set of annual macro-financial indicators provide useful early-warning information about the next calendar year's non-borderline systemic banking-crisis onset? The working hypothesis is associational and predictive, not causal: weak growth, elevated inflation, and falling foreign-exchange reserves may be associated with higher subsequent crisis risk.

## Candidate sample and inclusion rule

The candidate pool contains 55 countries across advanced, emerging-market, and developing economies, selected for geographic and income variation before expanded-sample model results are examined. It is in `config/full_research_countries.yaml`.

The coverage audit was completed before any expanded-sample model result. The primary panel rule is now fixed as: (1) an accepted World Bank country code; (2) all 36 annual 1990–2025 observations present for each of GDP growth, inflation, and reserves; and (3) no critical validation error. This stronger complete-core rule was selected from coverage alone to avoid primary-specification imputation of the raw inputs and yields 44 countries. The rule does not condition on a country having a crisis. Every excluded candidate remains in the coverage report with an explicit exclusion reason; this rule must not be changed after performance is observed.

## Labels and timing

The source is Laeven & Valencia (2026), IMF WP/26/94, using only non-borderline systemic banking-crisis onset years. Primary target: `y(c,t)=1` if a qualifying onset is in `t+1`, otherwise `0`. The same target and chronology source used in Phase 2 are retained.

## Features and transformations

Primary features are GDP growth (level), inflation (level), and total-reserves year-over-year percent change. The first reserve-change value within each country remains missing; it is not zero-filled. Additional variables are not part of the Phase 3 primary specification until their separate coverage and timing audit is approved.

## Validation and models

The design remains expanding-window temporal forecasting. The final holdout boundary will be set in `docs/FULL_VALIDATION_DESIGN.md` from the expanded data's date/label coverage, before evaluating any model. The planned baseline is the existing class-weighted L2 logistic regression. The only comparator is the existing depth-constrained random forest. No XGBoost, hyperparameter search, or test-set threshold tuning is authorized.

## Primary measures and robustness

Primary measures, when defined, are PR-AUC, recall, precision, false-negative rate, Brier score, and ROC-AUC. Planned robustness checks are feature-subset comparison, threshold reporting without selecting on final holdout results, and prospective alternate-horizon specifications. No result has been examined under this plan.
