# Crisis label audit — seven-country pipeline-validation sample

Audit date: 2026-09-04. This audit imports only the seven configured profile countries from the original IMF source; it does not replace or alter the official workbook.

## Source and import record

- **Source:** Laeven, Luc and Fabian Valencia, *Systemic Banking Crises Database: 1970–2025*, IMF Working Paper 2026/094.
- **Authors:** Luc Laeven and Fabian Valencia.
- **Publication date:** 15 May 2026.
- **Database version:** 2026 update, 1970–2025.
- **Official data ZIP:** `wp2694.zip`, downloaded from <https://www.imf.org/-/media/files/publications/wp/2026/datasets/wp2694.zip> on 2026-09-04.
- **SHA-256 of the original ZIP:** `70bdf38ddaff014021b523811b813840187bf4f8b78f8ffc3a1d5bd0fd7979ae`.
- **Source workbook:** `SYSTEMIC_BANKING_CRISES_DATABASE_2026.xlsx`.
- **Sheet / columns:** `Crisis Resolution and Outcomes`; `Country`, `Start`.
- **Date representation:** annual `Start` year. It is used because the World Bank feature panel is annual; the project makes no claim of monthly or real-time precision.

The source defines a banking crisis using significant banking-system distress plus significant intervention, with the first year both conditions are met treated as systemic. The source identifies three global borderline episodes (Nicaragua 2018, Sri Lanka 2023, Vietnam 2022). The primary import excludes them. None belongs to this seven-country profile.

## Exact imported profile chronology

The reviewed crosswalk maps seven profile ISO-3 codes to source names. Five countries have source episodes; Australia and South Africa have no source episode and are retained as no-crisis countries. The import contains **8 non-borderline episodes**: BRA (1990, 1994), DEU (2008), IDN (1997), MEX (1981, 1994), and TUR (1982, 2000). There are no duplicate country-year episodes, overlaps, missing ISO codes, or unmatched crosswalk names.

The complete machine-readable audit—including country/year/decade counts—is [crisis_chronology_summary.csv](../results/tables/crisis_chronology_summary.csv). The imported rows are stored in the ignored raw-data area with the original archive; no dates were reconstructed from secondary sources.

## Forward-label definition

For country `c` and annual feature row `t`:

`y(c,t) = 1` if a qualifying non-borderline systemic banking-crisis onset occurs in `t + 1`; otherwise `y(c,t) = 0`.

Thus a 1997 Indonesia onset gives the 1996 row a positive label, not the 1997 row. In the 1995–2017 panel, three onsets produce three positives: IDN 1997, TUR 2000, and DEU 2008. Events before 1995 remain in the source chronology but do not create an in-window label.

## Evaluation limitation and stop decision

The profile's pre-specified expanding evaluation begins in 2008. It contains **zero positive labels** from 2008–2017 because the 2008 onset's forward label belongs to 2007. ROC-AUC and PR-AUC are therefore not estimable for the intended test window. This is a sample-design limitation, not a reason to move dates, expand the target window, or tune the model after seeing outcomes. Logistic regression and random forest are deliberately not run for this validation profile.
