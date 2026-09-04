# Global Financial Crisis Early Warning System

A research-grade, reproducible starting point for investigating whether annual macro-financial indicators contain early-warning information about **documented systemic banking-crisis onsets**. It is deliberately conservative: it does not include invented labels, fabricated results, or claims of causality.

> This is an academic research model and not investment, economic-policy, or financial advice.

## What is implemented

- Official World Bank annual-data ingestion with raw-file preservation and provenance.
- Editable country, indicator, model, horizon, and feature configuration.
- Schema/range/duplicate validation and a country-variable coverage report.
- Explicit crisis-chronology contract; the researcher supplies a cited source rather than trusting unexplained dates.
- Leakage-conscious annual transformations and forward target construction.
- Train-only imputation/scaling inside expanding-window folds.
- Logistic-regression baseline, constrained random-forest comparator, rare-event metrics, and auditable prediction outputs.
- A small read-only Streamlit research dashboard, activated only after results exist.

## Setup

Requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,dashboard]'
pytest
```

## Reproduce the data stages

```bash
# 1. Download annual WDI data. Raw and interim files are kept separate.
ews download --start-year 1990 --end-year 2025

# 2. Copy the template and populate it only from a cited, reputable chronology.
cp data/raw/crisis_chronology_template.csv data/raw/crisis_chronology.csv
# Edit data/raw/crisis_chronology.csv; see docs/crisis_definition.md.

# 3. Validate, construct features and create annual forward labels.
ews prepare --chronology data/raw/crisis_chronology.csv

# 4. Run out-of-sample expanding-window evaluation.
ews evaluate --model logistic_regression
ews evaluate --model random_forest

# Optional: inspect generated historical outputs.
streamlit run dashboard/app.py
```

### Pipeline validation profile

This limited 7-country, 1995–2017 profile is an engineering/data-quality run, **not** evidence about crisis predictability. It isolates all generated files under named `pipeline_validation` directories:

```bash
ews --profile config/pipeline_validation.yaml download
ews --profile config/pipeline_validation.yaml prepare --chronology data/raw/crisis_chronology.csv
ews --profile config/pipeline_validation.yaml evaluate --model logistic_regression
```

The commands must be run from the repository root (or supply `--root /absolute/path`). No model results are meaningful until the crisis chronology has been independently reviewed.

## Important limitations

- World Bank historical values may be revised; the current pipeline does not reconstruct original release vintages or enforce source-specific publication lags.
- Annual timing is a coarse approximation to a 12-month real-time decision problem.
- The available sample may have few crisis onsets, so uncertainty and false negatives matter more than headline accuracy.
- A feature’s coefficient or importance is a predictive association, never evidence that it causes a crisis.

See [methodology](docs/methodology.md), [crisis definition](docs/crisis_definition.md), [data sources](docs/data_sources.md), and the [research log](docs/research_log.md) before interpreting outputs.
