# Data dictionary (Phase 1 audited configuration)

The machine-readable authority is `config/variables.yaml`; this document explains the initial feature set. All listed series are annual World Development Indicators queried through the World Bank V2 Indicators API. The [official API documentation](https://datahelpdesk.worldbank.org/knowledgebase/articles/898599-indicator-api-queries) confirms that indicator responses include code, unit, source metadata, and source notes; those metadata should be saved with each research download.

| Variable | WDI code | Units / frequency | Modeling transformation | Rationale and key risk |
|---|---|---:|---|---|
| GDP growth | `NY.GDP.MKTP.KD.ZG` | %, annual | Level | Weak activity may be associated with vulnerability; revised national accounts create vintage risk. |
| Inflation | `FP.CPI.TOTL.ZG` | %, annual | Level | Price instability proxy; country methodology and high-inflation outliers require inspection. |
| Domestic credit | `FS.AST.DOMS.GD.ZS` | % GDP, annual | Level | Financial-sector credit depth/leverage proxy; it is not credit growth. |
| Private credit | `FS.AST.PRVT.GD.ZS` | % GDP, annual | Level | Private-sector indebtedness proxy; coverage must be confirmed country by country. |
| Total reserves | `FI.RES.TOTL.CD` | current USD, annual | Within-country YoY % change | External-buffer proxy; values are nominal and zero bases produce missing change values. |
| Current account | `BN.CAB.XOKA.GD.ZS` | % GDP, annual | Level | External-imbalance proxy; potentially revised. |
| Real interest rate | `FR.INR.RINR` | %, annual | Level | Monetary/financial-conditions proxy; definitions and coverage vary. |
| Official exchange rate | `PA.NUS.FCRF` | LCU/USD, annual | Within-country YoY % change | Depreciation/pressure proxy; regime changes limit comparability. |

The current feature set is **provisional**. It is economically motivated and suitable for an audit download, not yet confirmed as empirically viable. The generated country coverage report decides whether a series/country survives into the initial modeling panel. Missing values remain missing during panel construction. The model’s `SimpleImputer(strategy="median")` is fitted inside each expanding training fold only. No zero-filling, interpolation, or full-sample imputation is used.

## Phase 3 primary specification

The expanded primary panel uses only GDP growth, inflation, and the year-over-year change in total reserves. It requires complete raw coverage for all three annual input series for every year from 1990 through 2025. This rule yields 44 countries and no missing raw-input cells; reserve growth has one mechanically unavailable first observation per country. Domestic/private credit, current account, exchange rate, and interest-rate variables remain outside the primary specification pending separate source, coverage, and timing audits. They must not be reintroduced because of model performance.
