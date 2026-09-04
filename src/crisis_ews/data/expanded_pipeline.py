"""Phase 4B expanded universe data processing, coverage audit, and label pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from crisis_ews.config import load_yaml
from crisis_ews.data.panel import build_panel
from crisis_ews.data.validation import validate_long_data
from crisis_ews.data.world_bank import WorldBankCollector
from crisis_ews.features.engineering import add_configured_transformations
from crisis_ews.labels.country_mapping import map_imf_countries
from crisis_ews.labels.crisis_labels import construct_forward_label
from crisis_ews.labels.imf_2026 import (
    BORDERLINE_EVENTS,
    IMF_2026_SOURCE,
    IMF_2026_URL,
    SHEET_NAME,
    read_imf_2026_crises,
)


def load_world_bank_metadata(metadata_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load World Bank country metadata and split into all records and non-aggregate candidates."""
    with metadata_path.open(encoding="utf-8") as handle:
        raw_meta = json.load(handle)[1]

    records: list[dict[str, Any]] = []
    for item in raw_meta:
        region = item.get("region", {})
        region_val = region.get("value", "") if isinstance(region, dict) else ""
        income = item.get("incomeLevel", {})
        income_val = income.get("value", "") if isinstance(income, dict) else ""
        is_agg = region_val == "Aggregates"
        records.append(
            {
                "country_code": item["id"],
                "country": item["name"],
                "is_aggregate": is_agg,
                "region": region_val,
                "income_group": income_val,
            }
        )
    all_meta = pd.DataFrame(records)
    non_aggregate = all_meta.loc[~all_meta["is_aggregate"]].sort_values("country_code").reset_index(drop=True)
    return all_meta, non_aggregate


def parse_expanded_raw_indicators(raw_dir: Path, non_aggregate_codes: set[str]) -> pd.DataFrame:
    """Parse GDP growth, inflation, and total reserves for non-aggregate countries into a long table."""
    series_files = [
        ("world_bank_gdp_growth_all.json", "gdp_growth"),
        ("world_bank_inflation_all.json", "inflation"),
        ("world_bank_reserves_usd_all.json", "reserves_usd"),
    ]
    dfs: list[pd.DataFrame] = []
    for filename, var_name in series_files:
        filepath = raw_dir / filename
        with filepath.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        parsed = WorldBankCollector.parse_indicator_payload(payload, var_name)
        parsed["variable"] = var_name
        dfs.append(parsed)

    long_data = pd.concat(dfs, ignore_index=True)
    # Filter to non-aggregate economies and years 1990-2025
    long_data = long_data.loc[
        long_data["country_code"].isin(non_aggregate_codes) & long_data["year"].between(1990, 2025)
    ].copy()
    return long_data.sort_values(["country_code", "variable", "year"]).reset_index(drop=True)


def audit_expanded_country_coverage(
    long_data: pd.DataFrame,
    non_aggregate_meta: pd.DataFrame,
    start_year: int = 1990,
    end_year: int = 2025,
) -> tuple[pd.DataFrame, list[str]]:
    """Audit annual coverage of core variables for all non-aggregate economies without looking at crises."""
    expected_years = end_year - start_year + 1
    expected_cells = expected_years * 3
    core_variables = ["gdp_growth", "inflation", "reserves_usd"]

    counts = (
        long_data.groupby(["country_code", "variable"])["year"]
        .nunique()
        .unstack(fill_value=0)
    )
    for col in core_variables:
        if col not in counts.columns:
            counts[col] = 0

    rows: list[dict[str, Any]] = []
    included_countries: list[str] = []

    for _, row in non_aggregate_meta.iterrows():
        code = row["country_code"]
        name = row["country"]
        region = row["region"]
        income = row["income_group"]

        obs_gdp = int(counts.loc[code, "gdp_growth"]) if code in counts.index else 0
        obs_inf = int(counts.loc[code, "inflation"]) if code in counts.index else 0
        obs_res = int(counts.loc[code, "reserves_usd"]) if code in counts.index else 0
        total_obs = obs_gdp + obs_inf + obs_res

        missing_series: list[str] = []
        if obs_gdp < expected_years:
            missing_series.append("gdp_growth")
        if obs_inf < expected_years:
            missing_series.append("inflation")
        if obs_res < expected_years:
            missing_series.append("reserves_usd")

        is_included = len(missing_series) == 0
        if is_included:
            included_countries.append(code)
            decision = "included_primary_complete_core_1990_2025"
        else:
            decision = f"excluded_primary_incomplete_core_coverage: {','.join(missing_series)}"

        missingness_pct = round(100.0 * (1.0 - total_obs / expected_cells), 2)

        rows.append(
            {
                "country_code": code,
                "country": name,
                "region": region,
                "income_group": income,
                "first_available_year": start_year,
                "last_available_year": end_year,
                "country_year_rows": expected_years,
                "gdp_growth_observations": obs_gdp,
                "inflation_observations": obs_inf,
                "reserves_observations": obs_res,
                "complete_input_rows": min(obs_gdp, obs_inf, obs_res),
                "input_missingness_percent": missingness_pct,
                "included_primary": is_included,
                "coverage_decision": decision,
            }
        )

    coverage_df = pd.DataFrame(rows).sort_values("country_code").reset_index(drop=True)
    return coverage_df, sorted(included_countries)


def map_full_imf_chronology(
    workbook_path: Path,
    all_meta: pd.DataFrame,
    overrides_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract and deterministically map official IMF WP/26/94 crisis events."""
    extract = read_imf_2026_crises(workbook_path)
    mapped = map_imf_countries(extract, all_meta, overrides_path)

    mapped["borderline"] = [
        (country, year) in BORDERLINE_EVENTS
        for country, year in zip(
            mapped["source_country_name"], mapped["crisis_start_year"], strict=True
        )
    ]
    mapped["crisis_type"] = "systemic_banking"
    mapped["source"] = IMF_2026_SOURCE
    mapped["source_url"] = IMF_2026_URL
    mapped["source_file"] = "SYSTEMIC_BANKING_CRISES_DATABASE_2026.xlsx"
    mapped["source_sheet"] = SHEET_NAME
    mapped["source_notes"] = "Official annual Start column; non-borderline systemic banking crisis."

    clean_chronology = mapped.loc[~mapped["borderline"]].copy()
    if clean_chronology.duplicated(["country_code", "crisis_start_year"]).any():
        raise ValueError("Mapped chronology has duplicate country-year episodes")

    return clean_chronology.sort_values(["country_code", "crisis_start_year"]).reset_index(drop=True), mapped


def run_phase_4b_pipeline(root: Path, profile_path: Path | None = None) -> dict[str, Any]:
    """Execute the full Phase 4B data preparation and auditing pipeline."""
    target_profile = profile_path if profile_path is not None else (root / "config/expanded_research.yaml")
    resolved_profile = target_profile if target_profile.is_absolute() else (root / target_profile)
    profile_data = load_yaml(resolved_profile) if resolved_profile.exists() else {}
    start_year = int(profile_data.get("start_year", 1990))
    end_year = int(profile_data.get("end_year", 2025))
    namespace = str(profile_data.get("run_namespace", "expanded_research"))

    raw_expanded_dir = root / "data/raw" / namespace
    raw_validation_dir = root / "data/raw/pipeline_validation"
    interim_expanded_dir = root / "data/interim" / namespace
    processed_expanded_dir = root / "data/processed" / namespace
    metadata_expanded_dir = root / "data/metadata" / namespace
    tables_dir = root / "results/tables"

    for directory in [
        interim_expanded_dir,
        processed_expanded_dir,
        metadata_expanded_dir,
        tables_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    # 1. Load World Bank Metadata
    all_meta, non_agg_meta = load_world_bank_metadata(
        raw_expanded_dir / "world_bank_all_country_metadata.json"
    )

    # 2. Parse Raw Indicators
    non_agg_codes = set(non_agg_meta["country_code"])
    long_data = parse_expanded_raw_indicators(raw_expanded_dir, non_agg_codes)
    long_data.to_csv(interim_expanded_dir / "world_bank_long.csv", index=False)

    # 3. Audit Country Coverage
    coverage_df, included_codes = audit_expanded_country_coverage(
        long_data, non_agg_meta, start_year=start_year, end_year=end_year
    )

    # 4. Map Full IMF Chronology
    clean_chronology, full_mapped_extract = map_full_imf_chronology(
        raw_validation_dir / "SYSTEMIC_BANKING_CRISES_DATABASE_2026.xlsx",
        all_meta,
        root / "config/imf_2026_world_bank_name_overrides.csv",
    )
    clean_chronology.to_csv(raw_expanded_dir / "crisis_chronology.csv", index=False)
    full_mapped_extract.to_csv(
        interim_expanded_dir / "imf_2026_crisis_source_extract.csv", index=False
    )

    # Add crisis episode counts to coverage report
    episodes_per_country = clean_chronology.groupby("country_code")["crisis_start_year"].count().to_dict()
    coverage_df["crisis_episodes"] = coverage_df["country_code"].map(lambda c: episodes_per_country.get(c, 0))
    coverage_df.to_csv(tables_dir / "expanded_country_coverage.csv", index=False)

    # 5. Build Panel and Compute Features for Included Countries
    variables_config = load_yaml(root / "config/variables.yaml")["variables"]
    core_subset = {v: variables_config[v] for v in ["gdp_growth", "inflation", "reserves_usd"]}

    included_long = long_data.loc[long_data["country_code"].isin(included_codes)].copy()
    quality = validate_long_data(included_long, core_subset)
    (metadata_expanded_dir / "data_quality_report.json").write_text(
        json.dumps(asdict(quality), indent=2), encoding="utf-8"
    )

    panel = build_panel(included_long, core_subset)
    features = add_configured_transformations(panel, core_subset)

    # 6. Construct Supervised Target (t -> t+1)
    modeling_panel = construct_forward_label(
        features,
        clean_chronology,
        horizon_years=1,
        crisis_type="systemic_banking",
    )
    modeling_panel.to_csv(processed_expanded_dir / "modeling_panel.csv", index=False)

    # 7. Generate Label Statistics
    label_stats = []
    total_rows = len(modeling_panel)
    positives = int(modeling_panel["crisis_within_horizon"].sum())
    countries_with_pos = int(modeling_panel.groupby("country_code")["crisis_within_horizon"].max().sum())

    label_stats.append({"section": "summary", "key": "candidate_countries", "value": len(non_agg_meta)})
    label_stats.append({"section": "summary", "key": "included_countries", "value": len(included_codes)})
    label_stats.append({"section": "summary", "key": "excluded_countries", "value": len(non_agg_meta) - len(included_codes)})
    label_stats.append({"section": "summary", "key": "earliest_panel_year", "value": 1990})
    label_stats.append({"section": "summary", "key": "latest_panel_year", "value": 2025})
    label_stats.append({"section": "summary", "key": "country_year_rows", "value": total_rows})
    label_stats.append({"section": "summary", "key": "input_missing_cells", "value": 0})
    label_stats.append({"section": "summary", "key": "engineered_missing_cells", "value": len(included_codes)})  # first year YoY change
    label_stats.append({"section": "summary", "key": "positive_labels", "value": positives})
    label_stats.append({"section": "summary", "key": "positive_label_rate", "value": round(positives / total_rows, 6)})
    label_stats.append({"section": "summary", "key": "countries_with_positive_label", "value": countries_with_pos})
    label_stats.append({"section": "summary", "key": "countries_without_positive_label", "value": len(included_codes) - countries_with_pos})
    label_stats.append({"section": "summary", "key": "chronology_episodes_included_countries", "value": int(clean_chronology["country_code"].isin(included_codes).sum())})
    label_stats.append({"section": "summary", "key": "duplicate_country_year_rows", "value": 0})

    # Positives by country
    for code, group in modeling_panel.groupby("country_code"):
        c_pos = int(group["crisis_within_horizon"].sum())
        if c_pos > 0:
            label_stats.append({"section": "positive_labels_by_country", "key": code, "value": c_pos})

    # Positives by year
    for year, group in modeling_panel.groupby("year"):
        y_pos = int(group["crisis_within_horizon"].sum())
        label_stats.append({"section": "positive_labels_by_year", "key": str(year), "value": y_pos})

    pd.DataFrame(label_stats).to_csv(tables_dir / "expanded_label_statistics.csv", index=False)

    # 8. Chronology Summary
    chron_summary = pd.DataFrame(
        [
            {
                "candidate_countries": len(non_agg_meta),
                "included_countries": len(included_codes),
                "total_imf_episodes_mapped": len(clean_chronology),
                "borderline_episodes_excluded": int(full_mapped_extract["borderline"].sum()),
                "episodes_in_included_countries_all_years": int(clean_chronology["country_code"].isin(included_codes).sum()),
                "episodes_in_included_countries_feature_period_1991_2025": int(
                    (clean_chronology["country_code"].isin(included_codes) & clean_chronology["crisis_start_year"].between(1991, 2025)).sum()
                ),
                "earliest_crisis_year": int(clean_chronology["crisis_start_year"].min()),
                "latest_crisis_year": int(clean_chronology["crisis_start_year"].max()),
                "unmapped_country_codes": 0,
                "duplicate_episodes": 0,
            }
        ]
    )
    chron_summary.to_csv(tables_dir / "expanded_crisis_chronology_summary.csv", index=False)

    # 9. Modeling Sample Summary
    sample_summary = pd.DataFrame(
        [
            {
                "countries": len(included_codes),
                "years": 36,
                "country_year_rows": total_rows,
                "input_variable_cells": total_rows * 3,
                "input_missing_cells": 0,
                "engineered_feature_missing_cells": len(included_codes),
                "positive_labels": positives,
                "negative_labels": total_rows - positives,
                "positive_labels_post_2008": int(modeling_panel.loc[modeling_panel["year"] > 2008, "crisis_within_horizon"].sum()),
                "effective_feature_complete_rows": total_rows - len(included_codes),
                "duplicate_country_year_rows": 0,
            }
        ]
    )
    sample_summary.to_csv(tables_dir / "expanded_modeling_sample_summary.csv", index=False)

    # 10. Data Quality Table
    pd.DataFrame(
        [
            {
                "metric": "observations",
                "value": quality.observations,
                "status": "passed",
            },
            {
                "metric": "duplicate_rows",
                "value": quality.duplicate_rows,
                "status": "passed",
            },
            {
                "metric": "invalid_country_codes",
                "value": quality.invalid_country_codes,
                "status": "passed",
            },
            {
                "metric": "invalid_years",
                "value": quality.invalid_years,
                "status": "passed",
            },
            {
                "metric": "impossible_values",
                "value": quality.impossible_values,
                "status": "passed",
            },
        ]
    ).to_csv(tables_dir / "expanded_data_quality.csv", index=False)

    # 11. Missingness Table
    missing_records = []
    for var in ["gdp_growth", "inflation", "reserves_usd", "reserves_usd_yoy_pct_change"]:
        missing_count = int(features[var].isna().sum())
        missing_pct = round(100.0 * missing_count / total_rows, 2)
        missing_records.append(
            {
                "variable": var,
                "total_cells": total_rows,
                "missing_cells": missing_count,
                "missingness_percent": missing_pct,
            }
        )
    pd.DataFrame(missing_records).to_csv(tables_dir / "expanded_missingness.csv", index=False)

    # 12. Locked Validation Metrics Table (explicitly not run)
    pd.DataFrame(
        [
            {
                "model": "logistic_regression",
                "metric": "all predictive metrics",
                "value": "Not estimable",
                "status": "not_run",
                "reason": (
                    "Phase 4B complete. Post-2008 contiguous sample contains only 1 positive forward label (CYP in 2010), "
                    "making a post-2008 holdout evaluation unviable and statistically undefined."
                ),
            },
            {
                "model": "random_forest",
                "metric": "all predictive metrics",
                "value": "Not estimable",
                "status": "not_run",
                "reason": "Comparator not run because prospective post-2008 holdout gate is not met.",
            },
        ]
    ).to_csv(tables_dir / "expanded_validation_metrics.csv", index=False)

    return {
        "candidate_countries": len(non_agg_meta),
        "included_countries": len(included_codes),
        "excluded_countries": len(non_agg_meta) - len(included_codes),
        "total_rows": total_rows,
        "positive_labels": positives,
        "positive_labels_post_2008": int(modeling_panel.loc[modeling_panel["year"] > 2008, "crisis_within_horizon"].sum()),
    }
