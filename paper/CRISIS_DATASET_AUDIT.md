# Crisis dataset audit

## Recommended initial chronology

**Laeven, Luc and Fabian Valencia (2026), “Systemic Banking Crises Database: 1970–2025,” IMF Working Paper 2026/094.** The IMF describes this as the update of the Laeven–Valencia systemic-banking-crisis database through the end of 2025. Its 2018 predecessor documents 151 systemic-banking-crisis episodes over 1970–2017 and explains that it updates earlier 2008 and 2013 vintages. See the [2026 IMF publication](https://www.elibrary.imf.org/abstract/journals/001/2026/094/article-A001-en.xml) and the [2018 methodology/publication page](https://www.imf.org/en/publications/wp/issues/2018/09/14/systemic-banking-crises-revisited-46232).

This is the recommended initial target because it is an IMF-authored, internationally comparable chronology with a specifically defined systemic banking-crisis category. It is preferable to an unstructured composite, which would combine events with different onset and transmission mechanisms.

## Definition and coverage

The 2026 publication reports 164 systemic banking-crisis episodes, including borderline episodes, from 1970 through 2025. The initial project label must use **only non-borderline systemic banking-crisis onset years**, unless a pre-specified robustness exercise explicitly changes that choice. Do not use crisis duration, policy response, fiscal cost, output loss, currency-crisis, or sovereign-debt-crisis fields as predictors or labels in the baseline.

Country availability and exact dates must be taken from the source's published appendix/supplement, not inferred from news reports or secondary websites. The chronology applies to a broader global country universe than the current sample; mappings from its country names to the project's ISO-3 codes require a reviewed crosswalk.

## Import procedure

1. Download the publication, appendix, and any author/IMF-provided machine-readable supplement directly from the IMF source. Record the exact URL, version, and retrieval date in `docs/research_log.md`.
2. Confirm the source's criteria for inclusion, start-date convention, and treatment of borderline episodes. Preserve its country name and original start year in a private audit worksheet.
3. Create `data/raw/crisis_chronology.csv` from the repository template. Include only fields with an explicit source basis; use `crisis_type=systemic_banking`.
4. Preserve a source citation and URL on every imported row. Add source version, original country name, inclusion decision, and crosswalk rationale in additional columns where useful.
5. Run `ews prepare --chronology data/raw/crisis_chronology.csv`. Review the country coverage and label-count reports before considering any model run.

## Access and redistribution

The IMF publication page is publicly accessible, but the repository does **not** redistribute the chronology. Copyright, any supplement-specific terms, and the availability of a downloadable machine-readable table must be checked at import time. If the data cannot be redistributed, retain it locally under `data/raw/` (which is ignored by Git) and distribute only the import instructions and aggregate metadata.

## Limitations

- Crisis-start years are annual labels; they do not identify an exact information cutoff within a year.
- The source has author-defined inclusion criteria. It is a defensible chronology, not ground truth about every financial-stress event.
- The current 2026 source must be version-pinned. Earlier 2018 and 2020 vintages should not be mixed with it.
- Full 1970–2025 coverage does not imply every indicator or country has a comparable macro-financial series for that period.
