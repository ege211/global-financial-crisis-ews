"""Phase 5D: Predictor Selection and Data Coverage Audit.

Audits candidate macro-financial vulnerability indicators for the
76-country panel (1990-2025) and locked expanding validation folds (2000-2008),
evaluating coverage, leakage safety, redundancy, and objective eligibility.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from crisis_ews.data.world_bank import WorldBankCollector

LOGGER = logging.getLogger(__name__)

BASELINE_FEATURES = [
    "gdp_growth",
    "inflation",
    "reserves_usd",
    "reserves_usd_yoy_pct_change",
]


@dataclass(frozen=True)
class CandidateSpec:
    candidate: str
    indicator_code: str
    source: str
    concept: str
    transformation: str
    raw_file_key: str
    is_derived: bool = False
    base_indicator: str | None = None


CANDIDATE_SPECS: list[CandidateSpec] = [
    CandidateSpec(
        candidate="private_credit_pct_gdp",
        indicator_code="FS.AST.PRVT.GD.ZS",
        source="World Bank WDI / IMF IFS",
        concept="Domestic credit to private sector (% of GDP)",
        transformation="Level (% of GDP)",
        raw_file_key="private_credit_pct_gdp",
        is_derived=False,
    ),
    CandidateSpec(
        candidate="private_credit_growth",
        indicator_code="FS.AST.PRVT.GD.ZS (derived)",
        source="World Bank WDI / IMF IFS",
        concept="Annual growth of private credit to GDP",
        transformation="Annual % change: (Credit_t - Credit_{t-1}) / Credit_{t-1} * 100",
        raw_file_key="private_credit_pct_gdp",
        is_derived=True,
        base_indicator="private_credit_pct_gdp",
    ),
    CandidateSpec(
        candidate="current_account_pct_gdp",
        indicator_code="BN.CAB.XOKA.GD.ZS",
        source="World Bank WDI / IMF Balance of Payments (BPM6)",
        concept="Current account balance (% of GDP)",
        transformation="Level (% of GDP)",
        raw_file_key="current_account_pct_gdp",
        is_derived=False,
    ),
    CandidateSpec(
        candidate="exchange_rate_depreciation",
        indicator_code="PA.NUS.FCRF (derived)",
        source="World Bank WDI / IMF IFS",
        concept="Annual currency depreciation against USD",
        transformation="Annual % change: (LCU/USD_t - LCU/USD_{t-1}) / LCU/USD_{t-1} * 100",
        raw_file_key="official_exchange_rate",
        is_derived=True,
        base_indicator="official_exchange_rate",
    ),
    CandidateSpec(
        candidate="unemployment_rate_ilo",
        indicator_code="SL.UEM.TOTL.ZS",
        source="World Bank WDI / ILO Modelled Estimates",
        concept="Total unemployment rate (% of labor force, ILO modeled)",
        transformation="Level (% of total labor force)",
        raw_file_key="unemployment_ilo",
        is_derived=False,
    ),
    CandidateSpec(
        candidate="unemployment_rate_national",
        indicator_code="SL.UEM.TOTL.NE.ZS",
        source="World Bank WDI / National Labour Force Statistics",
        concept="Total unemployment rate (% of labor force, national estimate)",
        transformation="Level (% of total labor force)",
        raw_file_key="unemployment_nat",
        is_derived=False,
    ),
]


def load_candidate_raw_data(
    raw_dir: Path, countries: list[str], start_year: int = 1990, end_year: int = 2025
) -> dict[str, pd.DataFrame]:
    """Load and parse saved raw JSON payloads for all candidate indicators."""
    collector = WorldBankCollector()
    parsed: dict[str, pd.DataFrame] = {}

    keys_to_load = {
        "private_credit_pct_gdp": "FS.AST.PRVT.GD.ZS",
        "current_account_pct_gdp": "BN.CAB.XOKA.GD.ZS",
        "official_exchange_rate": "PA.NUS.FCRF",
        "unemployment_ilo": "SL.UEM.TOTL.ZS",
        "unemployment_nat": "SL.UEM.TOTL.NE.ZS",
    }

    for key, code in keys_to_load.items():
        json_path = raw_dir / f"world_bank_{key}.json"
        if not json_path.exists():
            raise FileNotFoundError(
                f"Missing candidate raw JSON file: {json_path}. "
                "Download candidate series before running Phase 5D audit."
            )
        with json_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        df = collector.parse_indicator_payload(payload, code)
        filtered = df[
            df["country_code"].isin(countries)
            & (df["year"] >= start_year)
            & (df["year"] <= end_year)
        ].copy()
        parsed[key] = filtered

    return parsed


def compute_candidate_panel(
    panel: pd.DataFrame,
    raw_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Merge and transform candidate indicators onto the 76-country panel without leakage."""
    merged = panel.sort_values(["country_code", "year"]).copy()

    # 1. Domestic credit to private sector (% of GDP)
    pc_df = raw_data["private_credit_pct_gdp"][["country_code", "year", "value"]].rename(
        columns={"value": "private_credit_pct_gdp"}
    )
    merged = merged.merge(pc_df, on=["country_code", "year"], how="left")

    # 2. Private credit growth: YoY % change
    prev_pc = merged.groupby("country_code")["private_credit_pct_gdp"].shift(1)
    merged["private_credit_growth"] = (
        (merged["private_credit_pct_gdp"] - prev_pc) / prev_pc * 100.0
    )
    merged.loc[prev_pc.le(0), "private_credit_growth"] = float("nan")

    # 3. Current account balance (% of GDP)
    ca_df = raw_data["current_account_pct_gdp"][["country_code", "year", "value"]].rename(
        columns={"value": "current_account_pct_gdp"}
    )
    merged = merged.merge(ca_df, on=["country_code", "year"], how="left")

    # 4. Official exchange rate (LCU per USD) & depreciation
    fx_df = raw_data["official_exchange_rate"][["country_code", "year", "value"]].rename(
        columns={"value": "official_exchange_rate"}
    )
    merged = merged.merge(fx_df, on=["country_code", "year"], how="left")
    prev_fx = merged.groupby("country_code")["official_exchange_rate"].shift(1)
    merged["exchange_rate_depreciation"] = (
        (merged["official_exchange_rate"] - prev_fx) / prev_fx * 100.0
    )
    merged.loc[prev_fx.le(0), "exchange_rate_depreciation"] = float("nan")

    # 5. Unemployment rate (ILO modeled)
    ilo_df = raw_data["unemployment_ilo"][["country_code", "year", "value"]].rename(
        columns={"value": "unemployment_rate_ilo"}
    )
    merged = merged.merge(ilo_df, on=["country_code", "year"], how="left")

    # 6. Unemployment rate (National estimate)
    nat_df = raw_data["unemployment_nat"][["country_code", "year", "value"]].rename(
        columns={"value": "unemployment_rate_national"}
    )
    merged = merged.merge(nat_df, on=["country_code", "year"], how="left")

    return merged


def audit_candidate_coverage(
    panel_with_candidates: pd.DataFrame,
    test_start_year: int = 2000,
    test_end_year: int = 2008,
    min_validation_coverage_pct: float = 85.0,
    min_crisis_retention_pct: float = 90.0,
) -> pd.DataFrame:
    """Evaluate candidate indicators against pre-registered objective inclusion rules."""
    total_obs = len(panel_with_candidates)
    test_mask = (panel_with_candidates["year"] >= test_start_year) & (
        panel_with_candidates["year"] <= test_end_year
    )
    total_test_obs = int(test_mask.sum())

    pos_mask = panel_with_candidates["crisis_within_horizon"] == 1
    test_pos_mask = test_mask & pos_mask
    total_test_pos = int(test_pos_mask.sum())

    rows: list[dict[str, Any]] = []

    for spec in CANDIDATE_SPECS:
        col = spec.candidate
        series = panel_with_candidates[col]

        avail_countries = int(
            panel_with_candidates.loc[series.notna(), "country_code"].nunique()
        )
        avail_obs = int(series.notna().sum())
        coverage_pct = round((avail_obs / total_obs) * 100.0, 2)

        val_obs = int(panel_with_candidates.loc[test_mask, col].notna().sum())
        val_cov_pct = round((val_obs / total_test_obs) * 100.0, 2)

        test_pos_retained = int(
            panel_with_candidates.loc[test_pos_mask, col].notna().sum()
        )
        test_pos_pct = (
            round((test_pos_retained / total_test_pos) * 100.0, 1)
            if total_test_pos > 0
            else 0.0
        )

        # Objective screening logic
        eligible = True
        exclusion_reasons: list[str] = []

        if val_cov_pct < min_validation_coverage_pct:
            eligible = False
            exclusion_reasons.append(
                f"Validation coverage {val_cov_pct:.1f}% below {min_validation_coverage_pct:.0f}% threshold"
            )

        if test_pos_pct < min_crisis_retention_pct:
            eligible = False
            exclusion_reasons.append(
                f"Crisis retention {test_pos_pct:.1f}% below {min_crisis_retention_pct:.0f}% threshold"
            )

        # Domain-specific quality / break screening
        if spec.candidate == "exchange_rate_depreciation":
            eligible = False
            exclusion_reasons.append(
                "Structural break: Eurozone 1999 conversion causes -46% to -99.95% artificial drops; "
                "extreme collinearity with inflation (r=0.9585)"
            )
        elif spec.candidate == "private_credit_growth" and not eligible:
            exclusion_reasons.append(
                "Redundant with private credit level while suffering higher missingness"
            )
        elif spec.candidate == "unemployment_rate_national" and not eligible:
            exclusion_reasons.append(
                "Severe cross-country definitional heterogeneity in national reporting"
            )

        exclusion_str = (
            "; ".join(exclusion_reasons)
            if not eligible
            else (
                f"None (Satisfies all inclusion criteria; {val_cov_pct:.1f}% validation coverage, "
                f"{test_pos_pct:.1f}% crisis retention)"
            )
        )

        rows.append(
            {
                "candidate": spec.candidate,
                "indicator_code": spec.indicator_code,
                "source": spec.source,
                "concept": spec.concept,
                "transformation": spec.transformation,
                "available_countries": avail_countries,
                "coverage_pct": coverage_pct,
                "available_country_years": avail_obs,
                "validation_period_coverage_pct": val_cov_pct,
                "positive_events_retained": f"{test_pos_retained}/{total_test_pos} ({test_pos_pct:.1f}%)",
                "eligible": eligible,
                "exclusion_reason": exclusion_str,
            }
        )

    return pd.DataFrame(rows)


def compute_correlation_matrices(
    panel_with_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate Pearson and Spearman correlation matrices across baseline and candidate predictors."""
    all_features = [
        *BASELINE_FEATURES,
        "private_credit_pct_gdp",
        "private_credit_growth",
        "current_account_pct_gdp",
        "exchange_rate_depreciation",
        "unemployment_rate_ilo",
    ]
    df_sub = panel_with_features[all_features]
    pearson = df_sub.corr(method="pearson").round(4)
    spearman = df_sub.corr(method="spearman").round(4)
    return pearson, spearman


def run_phase_5d_predictor_audit(
    root: Path,
    raw_dir: Path | None = None,
    profile_path: Path | None = None,
) -> dict[str, Any]:
    """Execute complete Phase 5D Predictor Selection and Data Coverage Audit."""
    target_raw_dir = (
        raw_dir
        if raw_dir is not None
        else (root / "data/raw/expanded_research")
    )
    panel_path = root / "data/processed/expanded_research/modeling_panel.csv"
    if not panel_path.exists():
        raise FileNotFoundError(f"Modeling panel not found at {panel_path}")

    panel = pd.read_csv(panel_path)
    countries = sorted(panel["country_code"].unique())

    LOGGER.info("Loading candidate raw payloads for %s countries...", len(countries))
    raw_data = load_candidate_raw_data(target_raw_dir, countries)

    LOGGER.info("Constructing candidate panel with leakage-safe transformations...")
    panel_with_candidates = compute_candidate_panel(panel, raw_data)

    LOGGER.info("Executing objective coverage and eligibility audit...")
    coverage_df = audit_candidate_coverage(panel_with_candidates)

    LOGGER.info("Computing correlation matrices...")
    pearson_df, spearman_df = compute_correlation_matrices(panel_with_candidates)

    # Save output artifacts
    tables_dir = root / "results/tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Save both canonical names to satisfy prompt and repository conventions
    coverage_df.to_csv(tables_dir / "phase5d_candidate_coverage.csv", index=False)
    coverage_df.to_csv(tables_dir / "expanded_phase5d_candidate_coverage.csv", index=False)
    pearson_df.to_csv(tables_dir / "phase5d_correlations_pearson.csv")
    spearman_df.to_csv(tables_dir / "phase5d_correlations_spearman.csv")

    eligible_candidates = coverage_df[coverage_df["eligible"]]["candidate"].tolist()
    excluded_candidates = coverage_df[~coverage_df["eligible"]]["candidate"].tolist()

    LOGGER.info(
        "Phase 5D Audit complete: %s eligible predictors, %s excluded predictors",
        len(eligible_candidates),
        len(excluded_candidates),
    )

    return {
        "eligible_candidates": eligible_candidates,
        "excluded_candidates": excluded_candidates,
        "candidate_count": len(coverage_df),
        "coverage_table": coverage_df.to_dict(orient="records"),
    }
