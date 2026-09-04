"""Reproducible collector for the World Bank Indicators API."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

BASE_URL = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"


@dataclass(frozen=True)
class WorldBankCollector:
    """Download annual WDI observations, retaining a raw API response per series."""

    timeout_seconds: int = 30
    retries: int = 3

    def fetch_indicator_payload(
        self, countries: Iterable[str], indicator: str, start_year: int, end_year: int
    ) -> list[object]:
        """Return the unmodified JSON records received from all API pages."""
        country_arg = ";".join(sorted(set(countries)))
        params: dict[str, object] = {
            "date": f"{start_year}:{end_year}",
            "format": "json",
            "per_page": 20000,
            "page": 1,
        }
        payloads: list[object] = []
        while True:
            response = self._get(BASE_URL.format(country=country_arg, indicator=indicator), params)
            payload = response.json()
            if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
                raise ValueError(f"World Bank returned no usable payload for {indicator}")
            payloads.append(payload)
            metadata = payload[0]
            if not isinstance(metadata, dict):
                raise ValueError(f"World Bank returned malformed metadata for {indicator}")
            if int(metadata.get("page", 1)) >= int(metadata.get("pages", 1)):
                return payloads
            params["page"] = int(metadata["page"]) + 1

    @staticmethod
    def parse_indicator_payload(payloads: list[object], indicator: str) -> pd.DataFrame:
        """Convert a saved World Bank payload into a tidy staging table without imputation."""
        # A direct API download is a single ``[metadata, observations]`` payload;
        # the collector's paginated cache is a list of those payloads. Both preserve
        # the exact API response and are valid raw-data contracts.
        if (
            len(payloads) == 2
            and isinstance(payloads[0], dict)
            and isinstance(payloads[1], list)
        ):
            payloads = [payloads]
        records: list[dict[str, object]] = []
        for payload in payloads:
            if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
                raise ValueError(f"Malformed saved World Bank payload for {indicator}")
            for item in payload[1]:
                if item["value"] is None:
                    continue
                records.append(
                    {
                        "country_code": item["countryiso3code"],
                        "country_name": item["country"]["value"],
                        "year": int(item["date"]),
                        "indicator_code": indicator,
                        "value": float(item["value"]),
                    }
                )
        return pd.DataFrame(
            records, columns=["country_code", "country_name", "year", "indicator_code", "value"]
        )

    def fetch_indicator(
        self, countries: Iterable[str], indicator: str, start_year: int, end_year: int
    ) -> pd.DataFrame:
        """Download then parse an indicator; used for interactive inspection and tests.

        ``download_dataset`` is preferred for research runs because it preserves the
        raw payload alongside the parsed staging data.
        """
        return self.parse_indicator_payload(
            self.fetch_indicator_payload(countries, indicator, start_year, end_year), indicator
        )

    def download_dataset(
        self,
        countries: Iterable[str],
        variables: dict[str, dict[str, object]],
        start_year: int,
        end_year: int,
        raw_directory: str | Path,
    ) -> pd.DataFrame:
        """Download configured indicators and cache raw API responses as CSV plus provenance JSON."""
        raw_dir = Path(raw_directory)
        raw_dir.mkdir(parents=True, exist_ok=True)
        frames: list[pd.DataFrame] = []
        provenance: dict[str, object] = {"source": "World Bank Indicators API", "series": {}}
        for variable_name, spec in variables.items():
            indicator = str(spec["indicator"])
            payloads = self.fetch_indicator_payload(countries, indicator, start_year, end_year)
            with (raw_dir / f"world_bank_{variable_name}.json").open("w", encoding="utf-8") as handle:
                json.dump(payloads, handle, indent=2)
            frame = self.parse_indicator_payload(payloads, indicator)
            frame["variable"] = variable_name
            frames.append(frame)
            provenance["series"][variable_name] = {
                "indicator": indicator,
                "url": BASE_URL.format(country="{countries}", indicator=indicator),
                "retrieved_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "requested_country_codes": sorted(set(countries)),
                "requested_start_year": start_year,
                "requested_end_year": end_year,
            }
        with (raw_dir / "world_bank_provenance.json").open("w", encoding="utf-8") as handle:
            json.dump(provenance, handle, indent=2)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _get(self, url: str, params: dict[str, object]) -> requests.Response:
        # Kept local so saved-payload parsing remains usable in offline audit environments.
        import requests

        last_error: requests.RequestException | None = None
        for attempt in range(self.retries):
            try:
                response = requests.get(url, params=params, timeout=self.timeout_seconds)
                response.raise_for_status()
                return response
            except requests.RequestException as error:
                last_error = error
                if attempt + 1 < self.retries:
                    time.sleep(2**attempt)
        raise RuntimeError(f"World Bank request failed after {self.retries} attempts: {url}") from last_error
