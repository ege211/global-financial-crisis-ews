import pandas as pd

from crisis_ews.labels.crisis_labels import construct_forward_label


def test_forward_label_excludes_contemporaneous_crisis_year() -> None:
    panel = pd.DataFrame({"country_code": ["AAA", "AAA", "AAA"], "year": [2007, 2008, 2009]})
    chronology = pd.DataFrame(
        {
            "country_code": ["AAA"],
            "crisis_start_year": [2008],
            "crisis_type": ["systemic_banking"],
            "source": ["Example source"],
            "source_url": ["https://example.com"],
        }
    )
    labeled = construct_forward_label(panel, chronology, horizon_years=1)
    assert labeled["crisis_within_horizon"].tolist() == [1, 0, 0]


def test_labels_cover_multiple_events_overlapping_windows_and_never_crisis_country() -> None:
    panel = pd.DataFrame(
        {
            "country_code": ["AAA"] * 7 + ["BBB"] * 7,
            "year": list(range(2000, 2007)) * 2,
        }
    )
    chronology = pd.DataFrame(
        {
            "country_code": ["AAA", "AAA", "AAA"],
            "crisis_start_year": [2002, 2004, 2004],
            "crisis_type": ["systemic_banking", "systemic_banking", "currency"],
            "source": ["Source"] * 3,
            "source_url": ["https://example.com"] * 3,
        }
    )
    one_year = construct_forward_label(panel, chronology, horizon_years=1)
    aaa = one_year.loc[one_year.country_code.eq("AAA"), "crisis_within_horizon"].tolist()
    bbb = one_year.loc[one_year.country_code.eq("BBB"), "crisis_within_horizon"].tolist()
    # Pre-crisis years 2001 and 2003 are positive; onset (2002/2004), post-crisis
    # years, currency-only events, and countries with no crisis are not positives.
    assert aaa == [0, 1, 0, 1, 0, 0, 0]
    assert bbb == [0] * 7

    two_year = construct_forward_label(panel, chronology, horizon_years=2)
    assert two_year.loc[two_year.country_code.eq("AAA"), "crisis_within_horizon"].tolist() == [1, 1, 1, 1, 0, 0, 0]


def test_first_and_last_available_years_do_not_create_out_of_range_labels() -> None:
    panel = pd.DataFrame({"country_code": ["AAA", "AAA"], "year": [2000, 2001]})
    chronology = pd.DataFrame(
        {
            "country_code": ["AAA"],
            "crisis_start_year": [2001],
            "crisis_type": ["systemic_banking"],
            "source": ["Source"],
            "source_url": ["https://example.com"],
        }
    )
    output = construct_forward_label(panel, chronology)
    assert output["crisis_within_horizon"].tolist() == [1, 0]
