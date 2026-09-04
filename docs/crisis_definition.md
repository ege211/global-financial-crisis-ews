# Crisis definition

## Initial target

The initial target is an **annual, systemic banking crisis onset**. A country-year `t` receives `crisis_within_horizon = 1` only when the imported chronology records a systemic banking crisis start in `t + 1` (or within `t + 1` through `t + h` for a configured `h`-year horizon). The crisis year itself is excluded from the early-warning window.

This MVP does not bundle crisis dates. Before running `ews prepare`, the researcher must enter a cited chronology in a copy of `data/raw/crisis_chronology_template.csv`. Each row needs the country ISO-3 code, onset year, crisis type, source title, and direct source URL. The loader rejects incomplete entries and duplicates.

## Recommended source and scope decision

Use the IMF systemic banking crisis chronology published by Laeven and Valencia, selecting only non-borderline systemic-banking-crisis onset years and preserving the exact source/version in the CSV. The currently recommended vintage is their 2026 *Systemic Banking Crises Database: 1970–2025*; see `docs/CRISIS_DATASET_AUDIT.md`. It is the researcher’s responsibility to verify coverage, definitions, and licensing before transcription. A broader composite (banking, currency, sovereign debt) is intentionally not silently substituted, because that changes the estimand and class balance.

The annual timing convention creates an unavoidable limitation: annual World Bank indicators may have publication lags and revisions. Predictions should therefore be described as a historical annual backtest under an information-set approximation, not as real-time deployable forecasts.

## Included and excluded events

Only events explicitly coded as systemic banking crises in the chosen source are included in the initial label. Currency-only events, sovereign-debt-only events, and stress episodes absent from the source are excluded. Any future change to these rules must be recorded in `docs/research_log.md`, assigned a new label version, and rerun without overwriting previous results.
