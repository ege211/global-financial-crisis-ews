"""Controlled import of the official IMF WP/26/94 crisis workbook."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

IMF_2026_SOURCE = "Laeven and Valencia (2026), IMF WP/26/94"
IMF_2026_URL = "https://www.imf.org/-/media/files/publications/wp/2026/datasets/wp2694.zip"
SHEET_NAME = "Crisis Resolution and Outcomes"
# Explicitly identified as borderline in WP/26/94, Table 1 / Appendix I.
BORDERLINE_EVENTS = {("Nicaragua", 2018), ("Sri Lanka", 2023), ("Vietnam", 2022)}


def read_imf_2026_crises(workbook_path: str | Path) -> pd.DataFrame:
    """Read official annual onset data without changing source timing or rows.

    The source sheet's ``Start`` column is annual. Country-name footnote markers are
    removed only in a separate matching field; the original cell remains preserved.
    """
    raw = pd.read_excel(workbook_path, sheet_name=SHEET_NAME, usecols=["Country", "Start"])
    raw = raw.rename(columns={"Country": "source_country_name_raw", "Start": "crisis_start_year"})
    raw = raw.dropna(subset=["source_country_name_raw", "crisis_start_year"]).copy()
    raw["crisis_start_year"] = pd.to_numeric(raw["crisis_start_year"], errors="raise").astype(int)
    raw["source_country_name"] = (
        raw["source_country_name_raw"].str.replace(r"\s+\d+/$", "", regex=True).str.strip()
    )
    if raw.duplicated(["source_country_name", "crisis_start_year"]).any():
        raise ValueError("Official source extract contains duplicate cleaned country-year events")
    return raw.sort_values(["source_country_name", "crisis_start_year"]).reset_index(drop=True)


def import_validation_chronology(
    source_extract: pd.DataFrame, crosswalk_path: str | Path
) -> pd.DataFrame:
    """Filter source rows through a reviewed profile-specific name-to-ISO crosswalk.

    This deliberately does not infer ISO codes for the full global chronology.
    Countries in the crosswalk with no source row remain valid no-crisis countries.
    """
    crosswalk = pd.read_csv(crosswalk_path, dtype=str)
    required = {"country_code", "source_country_name"}
    missing = required - set(crosswalk.columns)
    if missing:
        raise ValueError(f"Crosswalk missing columns: {sorted(missing)}")
    if crosswalk.duplicated("country_code").any() or crosswalk.duplicated("source_country_name").any():
        raise ValueError("Crosswalk must be one-to-one for the validation profile")
    mapped = source_extract.merge(crosswalk, on="source_country_name", how="inner", validate="many_to_one")
    chronology = mapped[["country_code", "crisis_start_year", "source_country_name_raw"]].copy()
    chronology["borderline"] = [
        (country, year) in BORDERLINE_EVENTS
        for country, year in zip(
            mapped["source_country_name"], mapped["crisis_start_year"], strict=True
        )
    ]
    chronology = chronology.loc[~chronology["borderline"]].copy()
    chronology["crisis_type"] = "systemic_banking"
    chronology["source"] = IMF_2026_SOURCE
    chronology["source_url"] = IMF_2026_URL
    chronology["source_file"] = "SYSTEMIC_BANKING_CRISES_DATABASE_2026.xlsx"
    chronology["source_sheet"] = SHEET_NAME
    chronology["source_notes"] = "Official annual Start column; non-borderline profile event."
    if chronology.duplicated(["country_code", "crisis_start_year"]).any():
        raise ValueError("Mapped chronology has duplicate country-year episodes")
    return chronology.sort_values(["country_code", "crisis_start_year"]).reset_index(drop=True)


def chronology_summary(chronology: pd.DataFrame, profile_countries: list[str]) -> pd.DataFrame:
    """Return one-row, auditable profile chronology statistics."""
    years = chronology["crisis_start_year"]
    return pd.DataFrame(
        [
            {
                "profile_countries": len(profile_countries),
                "countries_with_episodes": int(chronology["country_code"].nunique()),
                "systemic_banking_episodes": len(chronology),
                "borderline_episodes_in_primary": int(chronology["borderline"].sum()),
                "earliest_crisis_year": int(years.min()) if not years.empty else None,
                "latest_crisis_year": int(years.max()) if not years.empty else None,
                "duplicate_episodes": int(chronology.duplicated(["country_code", "crisis_start_year"]).sum()),
                "missing_country_codes": int(chronology["country_code"].isna().sum()),
                "overlapping_same_country_year_episodes": int(
                    chronology.duplicated(["country_code", "crisis_start_year"], keep=False).sum()
                ),
            }
        ]
    )
