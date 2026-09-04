import pandas as pd

from crisis_ews.labels.imf_2026 import import_validation_chronology


def test_imf_import_uses_explicit_crosswalk_and_preserves_start_year(tmp_path) -> None:
    source = pd.DataFrame(
        {
            "source_country_name_raw": ["Türkiye", "Türkiye 5/"],
            "source_country_name": ["Türkiye", "Türkiye"],
            "crisis_start_year": [1982, 2000],
        }
    )
    crosswalk = tmp_path / "crosswalk.csv"
    crosswalk.write_text("country_code,source_country_name\nTUR,Türkiye\n", encoding="utf-8")
    output = import_validation_chronology(source, crosswalk)
    assert output[["country_code", "crisis_start_year"]].to_dict("records") == [
        {"country_code": "TUR", "crisis_start_year": 1982},
        {"country_code": "TUR", "crisis_start_year": 2000},
    ]
    assert not output["borderline"].any()


def test_imf_import_excludes_explicit_borderline_episode(tmp_path) -> None:
    source = pd.DataFrame(
        {
            "source_country_name_raw": ["Vietnam"],
            "source_country_name": ["Vietnam"],
            "crisis_start_year": [2022],
        }
    )
    crosswalk = tmp_path / "crosswalk.csv"
    crosswalk.write_text("country_code,source_country_name\nVNM,Vietnam\n", encoding="utf-8")
    assert import_validation_chronology(source, crosswalk).empty
