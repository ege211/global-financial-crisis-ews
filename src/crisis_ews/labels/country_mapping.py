"""Auditable mapping from IMF chronology country names to World Bank ISO-3 codes."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd


def normalized_name(value: str) -> str:
    """Normalize display-name typography only; semantic exceptions require overrides."""
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", ascii_value)


def map_imf_countries(
    source_extract: pd.DataFrame, world_bank_metadata: pd.DataFrame, overrides_path: str | Path
) -> pd.DataFrame:
    """Map every source-country name or fail loudly on an unresolved mapping.

    ``world_bank_metadata`` must contain ``country_code``, ``country``, and
    ``is_aggregate``. Mapping never uses crisis years or model outcomes.
    """
    required_source = {"source_country_name"}
    required_metadata = {"country_code", "country", "is_aggregate"}
    if required_source - set(source_extract.columns):
        raise ValueError("Source extract lacks source_country_name")
    if required_metadata - set(world_bank_metadata.columns):
        raise ValueError("World Bank metadata lacks required columns")
    candidates = world_bank_metadata.loc[~world_bank_metadata["is_aggregate"]].copy()
    candidates["normalized_country"] = candidates["country"].map(normalized_name)
    if candidates["normalized_country"].duplicated().any():
        raise ValueError("World Bank normalized country names are ambiguous")
    automatic = candidates.set_index("normalized_country")["country_code"].to_dict()
    overrides = pd.read_csv(overrides_path, dtype=str)
    if overrides.duplicated("source_country_name").any():
        raise ValueError("Country-name override table has duplicate source names")
    override_map = overrides.set_index("source_country_name")["world_bank_country_code"].to_dict()

    mapped = source_extract.copy()
    mapped["country_code"] = [
        override_map.get(name, automatic.get(normalized_name(name)))
        for name in mapped["source_country_name"]
    ]
    unresolved = sorted(mapped.loc[mapped["country_code"].isna(), "source_country_name"].unique())
    if unresolved:
        raise ValueError(f"Unresolved IMF-to-World-Bank country mappings: {unresolved}")
    mapped["mapping_method"] = mapped["source_country_name"].map(
        lambda name: "documented_override" if name in override_map else "normalized_exact_name"
    )
    return mapped
