import pandas as pd

from crisis_ews.features.engineering import add_configured_transformations


def test_yoy_transformation_does_not_cross_country_boundaries() -> None:
    panel = pd.DataFrame(
        {
            "country_code": ["AAA", "AAA", "BBB"],
            "year": [2000, 2001, 2000],
            "reserves": [100.0, 150.0, 200.0],
        }
    )
    variables = {"reserves": {"transform": "yoy_pct_change"}}
    output = add_configured_transformations(panel, variables)
    assert pd.isna(output.loc[output.country_code.eq("AAA") & output.year.eq(2000), "reserves_yoy_pct_change"].item())
    assert output.loc[output.country_code.eq("AAA") & output.year.eq(2001), "reserves_yoy_pct_change"].item() == 50.0
    assert pd.isna(output.loc[output.country_code.eq("BBB"), "reserves_yoy_pct_change"].item())


def test_yoy_transformation_at_t_does_not_depend_on_t_plus_one() -> None:
    base = pd.DataFrame(
        {"country_code": ["AAA", "AAA", "AAA"], "year": [2000, 2001, 2002], "reserves": [100.0, 150.0, 999.0]}
    )
    changed_future = base.copy()
    changed_future.loc[changed_future.year.eq(2002), "reserves"] = 1.0
    variables = {"reserves": {"transform": "yoy_pct_change"}}
    first = add_configured_transformations(base, variables)
    second = add_configured_transformations(changed_future, variables)
    assert first.loc[first.year.eq(2001), "reserves_yoy_pct_change"].item() == second.loc[second.year.eq(2001), "reserves_yoy_pct_change"].item()
