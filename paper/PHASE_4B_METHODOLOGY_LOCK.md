# Phase 4B methodology lock

Date locked: 2026-09-04, before any expanded-universe model fitting or model-performance inspection.

## Fixed choices

- **Crisis source:** Laeven & Valencia (2026), IMF Working Paper 2026/094, *Systemic Banking Crises Database: 1970–2025*.
- **Crisis definition:** systemic banking-crisis onset, excluding the source’s explicitly identified borderline episodes.
- **Target:** `y(c,t)=1` only when a qualifying onset starts in `t+1`; crisis-year features are not assigned a contemporaneous positive target.
- **Core predictors:** annual World Bank GDP growth, inflation, and reserve growth only.
- **Country universe:** all World Bank economies with a non-aggregate region identifier in the downloaded country metadata, whether or not they have a recorded crisis.
- **Primary inclusion:** valid country code, no critical data-quality failure, and complete annual 1990–2025 raw coverage for all three core inputs. The resulting sample is all countries satisfying this rule; it is not capped and does not condition on outcomes.
- **Transformation and preprocessing:** reserve growth uses only `t` and `t-1`; missingness is retained; fold imputation/scaling are training-only.
- **Validation philosophy:** expanding windows only; no random temporal split, no future data in training, and no test-informed threshold, feature, or country choice.

## Explicit prohibitions

No country can be added or removed because it has a later crisis, favorable model outcome, or preferred metric. No historical label, split, or Phase 2/3 artifact may be revised to create event coverage. XGBoost and hyperparameter optimization are outside Phase 4B.

## Mapping protocol

IMF country names are normalized deterministically against World Bank metadata. Nine non-identical official display-name matches use the documented mapping overrides in `config/imf_2026_world_bank_name_overrides.csv`. Any unmatched IMF source country is a stop condition, not an invitation to guess a code.
