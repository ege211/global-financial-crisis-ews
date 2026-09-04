"""Tests for Phase 4B expanded universe, mapping, and label integrity."""

from pathlib import Path

import pandas as pd

from crisis_ews.data.expanded_pipeline import (
    load_world_bank_metadata,
    map_full_imf_chronology,
)
from crisis_ews.data.validation import country_coverage_report


def test_imf_mapping_resolves_all_source_countries_without_unmapped() -> None:
    root = Path(__file__).parents[1]
    all_meta, _ = load_world_bank_metadata(
        root / "data/raw/expanded_research/world_bank_all_country_metadata.json"
    )
    clean_chron, full_extract = map_full_imf_chronology(
        root / "data/raw/pipeline_validation/SYSTEMIC_BANKING_CRISES_DATABASE_2026.xlsx",
        all_meta,
        root / "config/imf_2026_world_bank_name_overrides.csv",
    )
    assert len(full_extract) == 164
    assert full_extract["country_code"].notna().all()
    assert full_extract["country_code"].nunique() == 120
    assert not clean_chron.duplicated(["country_code", "crisis_start_year"]).any()


def test_borderline_events_are_identified_and_excluded() -> None:
    root = Path(__file__).parents[1]
    all_meta, _ = load_world_bank_metadata(
        root / "data/raw/expanded_research/world_bank_all_country_metadata.json"
    )
    clean_chron, full_extract = map_full_imf_chronology(
        root / "data/raw/pipeline_validation/SYSTEMIC_BANKING_CRISES_DATABASE_2026.xlsx",
        all_meta,
        root / "config/imf_2026_world_bank_name_overrides.csv",
    )
    borderline_rows = full_extract.loc[full_extract["borderline"]]
    assert len(borderline_rows) == 3
    borderline_pairs = set(zip(borderline_rows["source_country_name"], borderline_rows["crisis_start_year"]))
    assert borderline_pairs == {("Nicaragua", 2018), ("Sri Lanka", 2023), ("Vietnam", 2022)}
    # Verify none of them appear in clean_chronology
    clean_pairs = set(zip(clean_chron["source_country_name"], clean_chron["crisis_start_year"]))
    assert not clean_pairs.intersection(borderline_pairs)


def test_expanded_universe_inclusion_rule_criteria() -> None:
    root = Path(__file__).parents[1]
    _, non_agg_meta = load_world_bank_metadata(
        root / "data/raw/expanded_research/world_bank_all_country_metadata.json"
    )
    coverage_path = root / "results/tables/expanded_country_coverage.csv"
    assert coverage_path.exists()
    coverage_df = pd.read_csv(coverage_path)
    assert len(coverage_df) == 217
    assert len(non_agg_meta) == 217

    included = coverage_df.loc[coverage_df["included_primary"]]
    assert len(included) == 76
    assert (included["gdp_growth_observations"] == 36).all()
    assert (included["inflation_observations"] == 36).all()
    assert (included["reserves_observations"] == 36).all()
    assert (included["input_missingness_percent"] == 0.0).all()

    excluded = coverage_df.loc[~coverage_df["included_primary"]]
    assert len(excluded) == 141
    assert excluded["coverage_decision"].str.startswith("excluded_primary_incomplete_core_coverage").all()


def test_forward_label_contract_expanded_panel() -> None:
    root = Path(__file__).parents[1]
    panel_path = root / "data/processed/expanded_research/modeling_panel.csv"
    assert panel_path.exists()
    panel = pd.read_csv(panel_path)
    assert len(panel) == 76 * 36  # 2736 rows

    # Total positive labels
    total_positives = panel["crisis_within_horizon"].sum()
    assert total_positives == 44

    # Post-2008 forward labels:
    # 2008 has 1 positive (Nigeria crisis in 2009)
    # 2010 has 1 positive (Cyprus crisis in 2011)
    # All other years > 2008 have 0 positives
    post_2008 = panel.loc[panel["year"] > 2008]
    assert post_2008["crisis_within_horizon"].sum() == 1
    cyprus_2010 = panel.loc[(panel["country_code"] == "CYP") & (panel["year"] == 2010)]
    assert cyprus_2010["crisis_within_horizon"].item() == 1

    post_2010 = panel.loc[panel["year"] > 2010]
    assert post_2010["crisis_within_horizon"].sum() == 0


def test_validation_coverage_counts_safely_handles_missing_variable_columns() -> None:
    long_data = pd.DataFrame(
        {
            "country_code": ["AAA", "AAA"],
            "country_name": ["A", "A"],
            "year": [2000, 2001],
            "variable": ["x", "x"],
            "value": [1.0, 2.0],
        }
    )
    panel = pd.DataFrame(
        {"country_code": ["AAA"], "year": [2000], "x": [1.0], "crisis_within_horizon": [0]}
    )
    report = country_coverage_report(long_data, panel, ["AAA", "BBB"], ["x", "y"], 2000, 2001)
    assert report.loc[report.country_code.eq("AAA"), "complete_case_country_years"].item() == 0
    assert report.loc[report.country_code.eq("AAA"), "missingness_percent"].item() == 50.0
    assert report.loc[report.country_code.eq("BBB"), "missingness_percent"].item() == 100.0


def test_cli_expanded_research_subcommand_registered_and_executable(monkeypatch, caplog) -> None:
    import logging

    from crisis_ews.cli import main

    root = Path(__file__).parents[1]
    caplog.set_level(logging.INFO)
    monkeypatch.setattr("sys.argv", ["ews", "--root", str(root), "run-expanded-research"])
    main()
    assert "Phase 4B expanded research completed" in caplog.text
    assert "76 included" in caplog.text
