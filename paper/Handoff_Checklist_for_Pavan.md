# WindCast Climate – handoff to Pavan (MPCE submission)

Target journal: Journal of Modern Power Systems and Clean Energy (MPCE), ScholarOne, Original Paper (10-page limit; a slight excess is accepted).

## Files
- `WindCast_Climate_Paper_MPCE_Format.docx` – main paper (11 pages incl. references, two-column, MPCE template style).
- `WindCast_Climate_Supplementary_Material.docx` – everything cut from the main paper (tables S-I to S-XVIII, figures S1–S6).
- `Cover_Letter_MPCE.docx` – cover letter.
- Everything in **red** in these files is yours to fill or confirm. Delete the square brackets when done.

## 1. Fill in the red items (main paper)
1. Author line: IEEE student membership for you and Jayadharshini (or delete); supervisor's full name.
2. First-page footnote: manuscript dates (received/revised/accepted/CrossCheck/online) – the journal fills these, leave as the template if unknown; funding statement; affiliation, e-mail addresses, corresponding author; DOI (assigned by the journal).
3. Acknowledgment section: supervisor's CRediT roles; competing-interests statement; repository URL, final commit hash, code licence and archived DOI (e.g. Zenodo).
4. References: accessed dates / URLs for the online sources (TimescaleDB, CEA data portal, Open-Meteo, IWPA, MNRE Lok Sabha reply); confirm Cutler et al. 2007 details and the edition of Wilks.
5. Author biographies at the end (B.Tech details, current position, interests; supervisor's Ph.D.). Check the MPCE template for whether it asks for a photograph.
6. Supplementary Material: the test totals in Section S3 (run `pytest -q` and enter the final count).

## 2. Check against the MPCE site before submitting
- Article processing charge / open-access terms and the 10-page rule on mpce.info (we could not open the site automatically).
- Figures are Times New Roman where we could control them, but they were exported as images; MPCE asks for source/vector files (EPS/PDF/VSD) – export the figures again from the plotting scripts at 600 dpi or as PDF if the editor asks.
- Upload figures individually as well as inside the manuscript.

## 3. Code repository (Jayadharshini does the push)
Run `pytest -q`, then commit and push: `paper/`, `dashboard/app.py`, `dashboard/extremes.py`, `etl/fetch_forecast_weather.py`, `etl/fetch_nexgddp_daily.py`, `models/forecast_weather_test.py`, `models/forecast_weather_scores.csv`, `models/forecast_weather_errors.csv`, `climate/capacity_check.py`, `climate/capacity_check.csv`, `climate/variability_test.csv`, the new tests, `data/golden/tn_wind_capacity_mw.csv` and the `AI_USE.md` line. Do NOT add `data/raw/`. Then put the final commit hash into the paper and cover letter.

## 4. What changed in this version (so you are not surprised)
- Main paper cut from 20 to 11 pages. Removed from the main text but kept in the Supplementary Material: positioning table, data-source/schema/fold tables, DM pairwise tests, per-model results, ramp calendar, low/high-day scores, linear-model extremes, baselines, thresholds, fleet-growth table, daily climate-model tables, forecast-vs-ERA5 table, database latency, dashboard.
- Dashboard screenshots removed from the main paper (MPCE does not accept screenshots); they are only in the supplement.
- References renumbered by first citation, full journal titles, at most three authors then *et al.*; 45 references remain (uncited ones dropped).
- Appendices removed from the main paper; Acknowledgment now carries contributions, competing interests, data/code availability and the AI-use statement.
- Numbers were not changed; every figure quoted in the text comes from the committed result files.

## 5. Before upload
Read the PDF once from start to end (export from Word), check that tables sit near their first mention, and that no red text or square-bracket placeholders remain. Submit through ScholarOne.
