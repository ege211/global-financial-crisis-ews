# Validation design

## Target and information set

For a country-year row `i,t`, define the initial target as:

`Y(i,t; h) = 1` if a non-borderline systemic banking crisis begins in country `i` in any year `t + 1, ..., t + h`; otherwise `0`.

The primary specification has `h = 1`. It is read as “given the annual information recorded for year `t`, predict whether an onset begins during calendar year `t+1`.” It is **not** a claim of exact twelve-month, monthly, or real-time prediction, because WDI publication lags and revision vintages are not yet modelled.

## Expanding window

For each test year `T`, fit a new pipeline only on rows with `year < T` and predict every available country in `year == T`. With a 1990–2005 training window, the first test is 2006; the next train window ends in 2006 and tests 2007. The implementation does not use a random split.

Each fold fits its own median imputer and (for logistic regression) standard scaler. The test fold is never used to fit transformations, select features, set a threshold, or tune hyperparameters.

## Evaluation protocol for later phases

After the coverage audit, pre-specify a final untouched out-of-sample period, preferably a contiguous block with enough crisis onsets to make analysis meaningful. Earlier expanding folds can support specification development only. The final period must not be repeatedly used to choose countries, features, thresholds, or hyperparameters.

If insufficient onsets exist in the final block, report that limitation and use descriptive historical backtesting rather than pretending a reliable holdout estimate exists. Aggregated metrics must identify the years and countries included. Folds or aggregated test sets with a single outcome class do not have a defined ROC-AUC and should report it as unavailable.

## Phase 2 validation-profile result

For the fixed seven-country, 1995–2017 profile, the specified minimum 10 training years and 2008 first test year produce a 2008–2017 evaluation block. Its labels are all zero: the only 2008 onset is associated with the 2007 feature row under the one-year-forward target. This means the test set contains no positive cases and cannot estimate ROC-AUC, PR-AUC, recall, F1, or a meaningful calibration curve. The Phase 2 pipeline therefore stops after the data and label audit; it does not alter the split retrospectively to manufacture an evaluable score.

## Model implementation audit

The logistic pipeline is median imputation → standardization → class-weighted L2 logistic regression. The random-forest pipeline is median imputation → constrained class-weighted forest; it intentionally does not standardize because tree splits are scale-invariant. Both output probabilities and are freshly constructed per fold. No XGBoost or feature-selection routine is included at this phase.
