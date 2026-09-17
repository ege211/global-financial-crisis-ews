# Phase 4B Data and Universe Audit

**Audit Date:** 2026-09-04  
**Scope:** Full World Bank candidate universe expansion, complete-core inclusion audit, and IMF 1970–2025 systemic banking crisis chronology mapping.

---

## 1. Candidate Universe Definition

The candidate universe comprises all **217 non-aggregate economies** defined in the official World Bank API country metadata (`region.value != "Aggregates"`). No candidate economy was added or excluded based on crisis incidence or economic performance.

### Raw Source Payloads & Verification

All raw data payloads downloaded from official sources are cryptographically verified:

| File | Source | SHA-256 |
|---|---|---|
| `world_bank_all_country_metadata.json` | World Bank API V2 | `d29d57f8adf954c5e2a1520a02fb2c7b45575d8db3bd327a9dff47d66914231c` |
| `world_bank_gdp_growth_all.json` | World Bank API (`NY.GDP.MKTP.KD.ZG`) | `2f6be823439cb6d1e5ee17332f6ad2568edf332f053d721f95332bee9abd8527` |
| `world_bank_inflation_all.json` | World Bank API (`FP.CPI.TOTL.ZG`) | `c68b5c8f3c17d99bb8286e6be0cab14ef8c4374696e3dce7666a920a53eb095b` |
| `world_bank_reserves_usd_all.json` | World Bank API (`FI.RES.TOTL.CD`) | `d1201a381d89951965901f038414c25e424225eda5dcb587fc238db5ad573526` |
| `SYSTEMIC_BANKING_CRISES_DATABASE_2026.xlsx` | IMF WP/26/94 (Extracted) | `ac2741b634bf230a90ebee077ce8d9a7db0c57339579c4d290c75056a0c24013` |
| `imf_wp2694_dataset.zip` | IMF Official Data Archive | `70bdf38ddaff014021b523811b813840187bf4f8b78f8ffc3a1d5bd0fd7979ae` |

---

## 2. Pre-Outcome Inclusion Audit

Each of the 217 candidate economies was evaluated against the primary inclusion rule:
* Valid non-aggregate ISO-3 country code.
* Zero critical data quality failures (range violations, duplicates, invalid years).
* Complete 36-year raw annual series (1990–2025) for GDP growth, inflation, and total reserves.

### Findings
* **Included Economies:** Exactly **76 economies** (35.0% of candidates) possess complete, uninterrupted 1990–2025 data across all three core variables.
* **Excluded Economies:** **141 economies** (65.0% of candidates) were excluded solely due to incomplete core series.
  * Incomplete reserves only: 50 economies
  * Incomplete inflation only: 33 economies
  * Incomplete GDP growth only: 13 economies
  * Multiple incomplete series: 45 economies
* Full country-by-country breakdown with specific exclusion reasons is stored in [`results/tables/expanded_country_coverage.csv`](../results/tables/expanded_country_coverage.csv).

---

## 3. IMF Chronology Mapping & Borderline Treatment

* **Source:** Laeven, Luc and Fabian Valencia (2026), *Systemic Banking Crises Database: 1970–2025*, IMF Working Paper 2026/094.
* **Extraction:** 164 total systemic banking crisis onset episodes across 120 unique source country names.
* **Country Resolution:** 100% of the 120 source country names map deterministically to World Bank ISO-3 codes using typography normalization and 9 documented overrides in [`config/imf_2026_world_bank_name_overrides.csv`](../config/imf_2026_world_bank_name_overrides.csv). Zero unmapped country names remain.
* **Borderline Episodes:** 3 episodes explicitly designated as borderline by the authors in Table 1 / Appendix I were excluded:
  * Nicaragua (2018)
  * Sri Lanka (2023)
  * Vietnam (2022)
* **Retained Chronology:** 161 non-borderline systemic banking crisis episodes globally from 1976 through 2019.
* **Episodes in Included Countries:** 68 episodes across all historical years, with 46 starting $\ge 1990$.

---

## 4. Supervised Panel & Forward Label Statistics

* **Panel Dimensions:** 76 countries × 36 years (1990–2025) = **2,736 country-year observations**.
* **Supervised Target ($t \to t+1$):**
  * Total positive labels: **44** (positive label rate: 1.61%).
  * Countries with at least one positive label: 41.
  * Countries without any positive label: 35.
* **Temporal Distribution:**
  * 1990–2006: 27 positive labels.
  * 2007 (forecasting 2008 GFC onsets): 15 positive labels across 15 economies.
  * 2008 (forecasting Nigeria 2009 onset): 1 positive label.
  * 2010 (forecasting Cyprus 2011 onset): 1 positive label.
  * 2009, 2011–2025: **0 positive labels**.

Detailed tables are preserved under [`results/tables/`](../results/tables/):
* [`expanded_country_coverage.csv`](../results/tables/expanded_country_coverage.csv)
* [`expanded_crisis_chronology_summary.csv`](../results/tables/expanded_crisis_chronology_summary.csv)
* [`expanded_label_statistics.csv`](../results/tables/expanded_label_statistics.csv)
* [`expanded_modeling_sample_summary.csv`](../results/tables/expanded_modeling_sample_summary.csv)
* [`expanded_missingness.csv`](../results/tables/expanded_missingness.csv)
* [`expanded_data_quality.csv`](../results/tables/expanded_data_quality.csv)
* [`expanded_validation_metrics.csv`](../results/tables/expanded_validation_metrics.csv)
