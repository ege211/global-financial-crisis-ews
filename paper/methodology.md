# Methodology

## Question and interpretation

The project asks whether selected macro-financial indicators provide useful early-warning information for systemic banking-crisis onsets. It estimates associations useful for prediction; it does not identify causal effects.

## Timing and labels

Rows are country-year observations. With annual features for year `t`, the primary target is whether an imported, documented systemic banking-crisis onset occurs in `t+1`. The current year is excluded, so post-onset data are not described as an early warning. The configured horizon can be changed, with all changes logged.

## Feature availability and leakage limits

Feature transformations use current and previous years within the same country only. Train/test splits are chronological. Median imputation and standardization occur inside a sklearn pipeline fitted solely on each training fold. The source’s revised annual data and unmodeled publication lags are a documented limitation, not something the code hides.

## Evaluation

The evaluation is an expanding window: each test year is predicted by a model trained only on earlier years. Outputs store country, year, outcome, probability, predicted class, model, training period, and feature version. Metrics include ROC-AUC, PR-AUC, precision, recall, F1, confusion-matrix cells, Brier score, and false-negative rate. AUC values are omitted when a test aggregation contains only one class.

For the Phase 2 validation profile, a pre-model audit found no positive labels in its pre-specified 2008–2017 test block. No model output is interpreted or manufactured for that profile; metric fields are reported as “Not estimable.”

## Models and uncertainty

The primary model is class-weighted, L2-regularized logistic regression. Its standardized coefficients and odds ratios are interpretable as predictive associations, not causal effects. Classical p-values and confidence intervals are intentionally not emitted for the penalized estimator; they would require a separately specified inference/sensitivity model and should never be fabricated. A depth-constrained random forest is a secondary nonlinear comparison.

## Robustness protocol

Robustness runs must be pre-specified and saved with `save_experiment_manifest` before inspecting outcomes. Planned checks: feature subsets, horizons, country subsets, threshold choices, model comparison, and exclusion of major global-crisis years. The final out-of-sample period must not be repeatedly tuned against.
