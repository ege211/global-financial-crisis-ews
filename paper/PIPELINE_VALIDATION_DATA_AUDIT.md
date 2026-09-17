# Pipeline-validation raw-data audit

This is an **engineering/data-quality audit**, not empirical analysis and not a model result.

## Retrieval

On 2026-09-04, raw JSON responses were downloaded directly from the official World Bank V2 Indicators API for AUS, BRA, DEU, IDN, MEX, TUR, and ZAF, years 1995–2017. The files reside in ignored `data/raw/pipeline_validation/` paths. Every response reported source ID `2` (World Development Indicators), a single page, 161 requested country-year records, and API metadata `lastupdated=2026-07-13`.

SHA-256 checksums:

| File | SHA-256 |
|---|---|
| `world_bank_gdp_growth.json` | `8fd2bffa7974c2ddb7fdcf489e064ad22d61711312b8f68b6bf79c20728377be` |
| `world_bank_inflation.json` | `5001f815b28a85e1560e475ddec166772d80f7ffc293b3b3ac26e4c565615d63` |
| `world_bank_private_credit_pct_gdp.json` | `c2775d697af1de03cf0ecb1261d8039474e7e51aef8081480d19be93871cd404` |
| `world_bank_reserves_usd.json` | `fe8cb281f35851e966e7afecc64d2467036c7622e1381c8085099fcb0505e614` |

## Raw API validation

| Series | Non-null | Duplicate country-years | Invalid ISO-3 codes | Years outside 1995–2017 | Configured-range violations |
|---|---:|---:|---:|---:|---:|
| GDP growth | 161 | 0 | 0 | 0 | 0 |
| Inflation | 161 | 0 | 0 | 0 | 0 |
| Private credit / GDP | 126 | 0 | 0 | 0 | 0 |
| Total reserves | 161 | 0 | 0 | 0 | 0 |

The observed ranges were GDP growth −13.13% to 10.98%, inflation −0.69% to 89.11%, private credit/GDP 12.24% to 142.42%, and reserves USD 2.34bn to 373.96bn. These are range checks only; they are not a claim that every economically extreme observation is erroneous or valid.

## Coverage finding and profile decision

GDP growth, inflation, and total reserves have 23 non-null years for every sampled country. Private credit/GDP has 17 non-null years for DEU (2001–2017), 9 for IDN (2009–2017), and 10 for TUR (2008–2017); MEX has 21 (1997–2017). This is structural missingness, not a value to impute.

Therefore `config/pipeline_validation.yaml` excludes private credit/GDP from its three-feature engineering profile. This rule was set using source coverage only, before importing any crisis labels or evaluating model performance. The main research configuration remains unchanged pending its full coverage audit.

## Phase 2 execution update

The raw download and independent JSON checks completed. Phase 2 subsequently parsed these exact payloads into the real seven-country 1995–2017 panel: 161 country-year rows and 483 non-missing input cells. The derived panel, source provenance, coverage reports, and labels are in the ignored `data/interim/`, `data/processed/`, and `data/metadata/` pipeline-validation directories; the summarized outputs are under `results/tables/`.

The pre-specified 2008–2017 evaluation block has no positive forward labels. That makes predictive metrics not estimable, so no Logistic Regression or Random Forest run was performed. See `docs/CRISIS_LABEL_AUDIT.md` for the label audit and stop decision.
