# Data sources and availability

## Macro-financial indicators

The first ingestion stage calls the official [World Bank Indicators API](https://datahelpdesk.worldbank.org/knowledgebase/articles/889392-about-the-indicators-api-documentation) for annual World Development Indicators listed in `config/variables.yaml`. Untouched JSON API payloads are retained in `data/raw/`; parsed long-form staging data is stored in `data/interim/`, and only then pivoted into a processed panel. `world_bank_provenance.json` records indicator codes, request templates, selected countries/years, and download time.

The API returns current revised historical values. It does not reconstruct the vintage that would have been published at each historical prediction date. This is a meaningful look-ahead risk. Results must say that clearly and should not be called a real-time forecast until vintage-aware sources or publication-lag rules are implemented.

## Crisis chronology

No dates are supplied by this repository. The researcher must supply and cite an external, reputable chronology in the required CSV structure. The selected source, version, access date, coverage, coding choices, and any country-code mapping must be logged before any result is interpreted.

## Other data sources

BIS credit and debt series, IMF International Financial Statistics, IMF Financial Soundness Indicators, and national central-bank series are possible future extensions. Each requires a collector, versioned metadata, availability analysis, and an explicit decision about reporting lags. They are not represented as if currently present.

## Phase 3 source version record

The Phase 3 coverage audit uses three World Bank Indicators API series (`NY.GDP.MKTP.KD.ZG`, `FP.CPI.TOTL.ZG`, and `FI.RES.TOTL.CD`) for the 55-country candidate list, 1990–2025. Unmodified response JSON, API source metadata, retrieval scope, and SHA-256 checksums are retained under the ignored `data/raw/full_research/` directory. The crisis chronology is imported from the same official Laeven–Valencia 2026 workbook and traceable through `results/tables/full_crisis_chronology_summary.csv`.
