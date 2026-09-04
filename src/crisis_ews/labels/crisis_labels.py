"""Label construction from an explicitly sourced historical crisis chronology."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CHRONOLOGY_COLUMNS = {"country_code", "crisis_start_year", "crisis_type", "source", "source_url"}


def read_crisis_chronology(path: str | Path) -> pd.DataFrame:
    """Read an annual crisis chronology and reject undocumented or duplicate events.

    No observations are bundled with the repository: users must supply a reputable,
    cited chronology under ``data/raw/`` before modeling.
    """
    chronology = pd.read_csv(path)
    missing = CHRONOLOGY_COLUMNS - set(chronology.columns)
    if missing:
        raise ValueError(f"Crisis chronology missing required columns: {sorted(missing)}")
    chronology["crisis_start_year"] = pd.to_numeric(
        chronology["crisis_start_year"], errors="raise"
    ).astype(int)
    if chronology[list(CHRONOLOGY_COLUMNS)].isna().any().any():
        raise ValueError("Crisis chronology contains missing required metadata")
    if chronology.duplicated(["country_code", "crisis_start_year", "crisis_type"]).any():
        raise ValueError("Crisis chronology contains duplicate country-year-type events")
    return chronology.sort_values(["country_code", "crisis_start_year"]).reset_index(drop=True)


def construct_forward_label(
    panel: pd.DataFrame,
    chronology: pd.DataFrame,
    horizon_years: int = 1,
    crisis_type: str = "systemic_banking",
) -> pd.DataFrame:
    """Create ``crisis_within_horizon`` for years t+1 through t+horizon.

    For annual features observed in year t, a label of 1 means a documented crisis
    starts in the following calendar year(s). The contemporaneous year is deliberately
    excluded to avoid presenting post-onset values as early warnings.
    """
    if horizon_years < 1:
        raise ValueError("horizon_years must be at least one")
    selected = chronology.loc[chronology["crisis_type"].eq(crisis_type)].copy()
    if selected.empty:
        raise ValueError(f"Chronology contains no events with crisis_type={crisis_type!r}")
    event_years = selected.groupby("country_code")["crisis_start_year"].agg(set).to_dict()
    labeled = panel.copy()
    labeled["crisis_within_horizon"] = [
        int(any((year + offset) in event_years.get(country, set()) for offset in range(1, horizon_years + 1)))
        for country, year in zip(labeled["country_code"], labeled["year"], strict=True)
    ]
    labeled["prediction_horizon_years"] = horizon_years
    labeled["crisis_type"] = crisis_type
    return labeled
