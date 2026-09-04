# Phase 2 runbook — seven-country pipeline validation

This runbook recreates the real-data audit for AUS, BRA, DEU, IDN, MEX, TUR, and ZAF, 1995–2017. It is a pipeline-validation exercise, not evidence that crises can be predicted.

## Environment

Use Python 3.11 or newer. The system Python on the original development machine was 3.9.6, so it is not suitable.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

## Inputs

1. Download the official IMF WP/26/94 associated ZIP from the URL in `docs/CRISIS_LABEL_AUDIT.md` to `data/raw/pipeline_validation/imf_wp2694_dataset.zip`.
2. Verify its SHA-256 against the value recorded in that audit before use.
3. World Bank raw JSON is obtained using the configured profile. The API responses must remain under `data/raw/pipeline_validation/`.

## Reproduction commands

```bash
# Download only the profile's GDP growth, inflation, and reserves inputs.
ews --profile config/pipeline_validation.yaml download

# Extract the one official workbook, create the reviewed profile chronology, and audit it.
ews --profile config/pipeline_validation.yaml import-imf-2026 \
  --dataset-zip data/raw/pipeline_validation/imf_wp2694_dataset.zip

# Validate, build annual transformations, and construct t -> t+1 labels.
ews --profile config/pipeline_validation.yaml prepare \
  --chronology data/raw/pipeline_validation/crisis_chronology.csv
```

The immutable sources are the IMF ZIP and World Bank JSON. Parsed observations belong in `data/interim/`; the labeled panel belongs in `data/processed/`; raw inputs must never be overwritten by those stages.

## Evaluation gate

Before calling `ews evaluate`, inspect the generated label summary. For this exact profile, the pre-specified 2008–2017 test window has no positive labels. Consequently logistic regression and random forest must remain **not run** and their metrics **not estimable**. The correct reproducible outcome is the completed data/label pipeline plus this stop decision, not forced model output.

Any future evaluation redesign needs a prospectively documented sample/horizon/holdout design before looking at new model results. It must not overwrite this validation run.
