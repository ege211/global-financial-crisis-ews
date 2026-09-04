import csv
from pathlib import Path


def test_full_research_candidate_and_crosswalk_codes_are_unique_and_aligned() -> None:
    root = Path(__file__).parents[1]
    countries = [
        line.strip()[2:]
        for line in (root / "config/full_research_countries.yaml").read_text(encoding="utf-8").splitlines()
        if line.startswith("  - ")
    ]
    with (root / "config/full_research_crisis_crosswalk.csv").open(encoding="utf-8", newline="") as handle:
        crosswalk_codes = [row["country_code"] for row in csv.DictReader(handle)]
    assert 40 <= len(countries) <= 60
    assert len(countries) == len(set(countries))
    assert countries == crosswalk_codes
