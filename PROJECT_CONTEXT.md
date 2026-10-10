# PROJECT_CONTEXT.md — read this fully before writing any code

Put this file in the repo root. It is the single source of truth for every AI coding agent and both teammates.
If code and this file disagree, this file wins, unless a pull request has changed it.

Repo name: `windcast-climate`. Project name: WindCast Climate.

## 1. What we are building (one paragraph)

A DBMS mini project that doubles as a research paper. We forecast **daily wind power generation for Tamil Nadu, India**
1 to 3 days ahead, store everything in **PostgreSQL + TimescaleDB**, then ask a climate question: how much does
expected wind generation change by **2030 and 2050** under **SSP2-4.5** and **SSP5-8.5**, and how much of the future
weather lies **outside the range the model was trained on**. A **Streamlit** dashboard reads live from the database.

Working paper title: *Is Your Wind Forecast Climate-Proof? A Database-Backed Framework Coupling Operational
Forecasting with CMIP6 Covariate-Shift Analysis for Tamil Nadu, India.*

Claim wording (agreed): we study **forecast validity under climate shift**. We do NOT claim the weather-to-power
relationship "drifts"; a power curve does not change with climate, the input weather does.

## 2. Team and ownership

| Person | Owns | Never edits without a PR the other approves |
| --- | --- | --- |
| Pavan | Database schema and DDL, Docker setup, ETL (ERA5, CEA), NEX-GDDP extraction and change factors, materialized views, dashboard | `models/`, `climate/` |
| Jayadharshini | EDA, persistence / linear / Random Forest / XGBoost models, time-based validation, climate-shift module and diagnostic, figures | `db/`, `etl/` |

Both must be able to explain the other's half. Swap code reviews on every merge. `main` is protected; one branch per person.

## 3. Sprint (7 days) and gates

Day 1 lock scope (done: Tamil Nadu chosen) · Day 2 golden dataset + baseline · Day 3 DB load, model reads from DB ·
Day 4 RF/XGBoost, change factors, output table shape · Day 5 climate module, results freeze · Day 6 dashboard on real
results, benchmark, fresh-clone test · Day 7 paper skeleton and demo dry run.

Gates: G0 scope and schema frozen · G1 golden dataset loads · G2 model reads the database · G3 `climate_impact`
shape frozen · G4 results freeze (numbers in the paper stop changing) · G5 fresh-clone test by the other person · G6 demo dry run.

## 4. Data facts (verified, do not assume otherwise)

**Generation target — CEA state-wise daily renewable generation (India Data Portal, open licence).**

- File: `state-control-renewable-energy-generation.csv`. Columns: `id, date, state_name, state_code, region, wind_energy, solar_energy, other_renewable_energy, total_renewable_energy`.
- Range **2020-05-28 to 2025-04-27** (not 2019–2024). 1,776 days present of 1,796 in the span, so **20 days are missing**
  (15 of them in 2021). Do not fill silently; keep them as NULL and report the count.
- 2 duplicate (date, state) rows exist; keep the first.
- `wind_energy` is believed to be **million units (MU) per day**; unit not yet officially confirmed.
- Tamil Nadu: mean about 32.7 MU/day; strong seasonality (about 7 MU/day in November, about 73 MU/day in July, peak June–September);
  full-year means 31 / 33 / 36 / 32 MU/day for 2021–2024, so little capacity-growth trend.
- Installed capacity per year is NOT in this file. Capacity factor = generation / (installed capacity MW × 24 / 1000) needs a
  separate capacity source (MNRE / CEA / wind industry yearly figures). If unavailable, fall back to detrending by year and say so.

**Weather inputs — Open-Meteo Historical API (ERA5 reanalysis).** Hourly from 1940; we aggregate to daily.
Proposed sampling points (approximate, Pavan to confirm): Muppandal/Kayathar belt about 8.25N 77.55E; Palghat gap /
Udumalpet–Palladam belt about 10.6N 77.25E; Theni/Cumbum belt about 9.75N 77.3E. State feature = mean of the three points.

**Climate projections — NEX-GDDP-CMIP6 (NASA, CC0).** Daily, 0.25 degree, 34 climate models, scenarios `ssp245` and
`ssp585`, historical to 2014 and scenarios 2015–2100. Variables available to us: `sfcWind`, `tas`, `hurs`, `huss`.
**No surface pressure.** It is already bias-corrected against a reference dataset that is not ERA5.

## 5. Hard rules (agents must follow these)

1. **Daily resolution only.** The coupling to CMIP6 requires daily data. No hourly modelling.
2. **Feature contract.** Model inputs are ONLY: daily mean wind speed, daily mean temperature, daily mean humidity
   (relative humidity `hurs` as the common variable), plus lagged generation / capacity factor, plus calendar terms
   (day-of-year sine/cosine). **Pressure is forbidden** because it does not exist in the climate data.
3. **Target.** Capacity factor if yearly Tamil Nadu installed capacity is found by Day 3; otherwise `generation_mu` directly. **No detrending by year** (it would remove real wind-year variability; yearly means 2021–2024 show little capacity trend). Decided at G0, 2026-10-10.
4. **Validation is time-based.** Walk-forward or chronological split. **Never a random split.** No shuffling. No leakage
   from future rows into features.
5. **Always compare against persistence** (yesterday's value). A model that does not beat persistence is not reported as good.
6. **Climate coupling method.** Primary: **delta-change**, computed **monthly** (12 factors per climate model / scenario / window; wind = ratio future/baseline, temperature and humidity = additive difference). Annual rows (`month` NULL) kept for summaries only. Direct daily NEX-GDDP feed is NOT stored for now (revisit in week 2). Method detail: — compute each climate model's change between a 1995–2014 baseline and
   2030 / 2050 windows (about 20-year windows), apply that change to ERA5 weather, feed the trained model.
   Secondary check: feed NEX-GDDP values directly and compare. Report the **spread across climate models** as the uncertainty.
7. **Shift diagnostic.** Report the share of future days whose features fall outside the training range
   (per-feature percentile bounds and/or a distance measure).
8. **Never invent data, results or citations.** Every number in the paper must come from code in this repo. If data is missing, say so.
9. **Reproducibility.** One command (`docker compose up`, then a documented pipeline command) must rebuild everything from a
   fresh clone. Pin dependencies. Seeds fixed.
10. **Schema is frozen after G0.** Changes need a PR approved by the other person and an update to this file.

## 6. Database contract (FROZEN at G0 on 2026-10-10 — DDL in `db/init/`, details in `db/README.md`)

Daily grain. Units: wind m/s at 10 m, temperature deg C, humidity % RH, generation MU/day.

| Table | Key | Columns |
| --- | --- | --- |
| `stations` | `station_id` | `name`, `kind` (`state_aggregate` / `weather_point`), `latitude`, `longitude`, `region`. ID 1 = TN_STATE; 2–4 = wind-belt points |
| `raw_weather_readings` (hypertable) | (`station_id`, `ts`) | `wind_speed`, `temperature`, `humidity`, `pressure` (nullable, never a model input), `n_hours`, `source`, `ingested_at` |
| `power_output` (hypertable) | (`station_id`, `ts`) | `generation_mu` (NULL = missing CEA day), `capacity_mw`, `capacity_factor`, `source`, `ingested_at` |
| `climate_scenarios` | `scenario_id`; unique (`name`, `target_year`, `month`, `climate_model`, `station_id`) | `name` (`historical` / `ssp245` / `ssp585`), `target_year`, `window_start`, `window_end`, `month` (NULL = annual), `projected_wind_speed`, `projected_temperature`, `projected_humidity` |
| `forecast_runs` | `run_id` | `model_version`, `run_timestamp`, `horizon_days` (1–3), `params` (JSONB: seed, features, train window) |
| `model_predictions` | `prediction_id`; unique (`run_id`, `station_id`, `forecast_date`) | `predicted_value`, `lower`, `upper` |
| `climate_impact` | `impact_id`; unique (`region`, `scenario`, `target_year`, `climate_model`, `metric`) | `value`, `p10`, `p90`. Shape frozen at G3 |

Model input: **`mv_daily_features`** — one row per day per region: mean of the 3 wind-belt points (`wind_speed`, `temperature`,
`humidity`, `n_points`) LEFT JOINed to TN_STATE `generation_mu`, `capacity_mw`, `capacity_factor`. Missing generation stays NULL;
weather days outside the CEA range also have NULL generation, so filter on it. `mv_weekly_features` gives weekly means.
**Models read views, not raw tables, from Gate G2 on.** Lags and calendar terms are built in model code.

## 7. Repo layout

```
db/          DDL, migrations, views            (Pavan)
etl/         ERA5, CEA, NEX-GDDP loaders       (Pavan)
models/      baseline, RF/XGBoost, validation  (Jayadharshini)
climate/     change factors, shift diagnostic  (Jayadharshini)
dashboard/   Streamlit app                     (Pavan)
data/        raw/ (gitignored) and golden/     (golden = Day 2 hand-off file)
paper/       figures and tables generated by code
PROJECT_CONTEXT.md   this file
README.md    one-command reproduction
```

## 8. Working agreement for AI agents

- Read this whole file first, then summarise it back in 5 lines before writing code.
- Make the smallest change that does the task. Do not touch folders owned by the other teammate.
- After writing code, **run it** and show the output. Do not claim a result you did not run.
- Explain what you did in plain language; the human is a beginner in ML and climate data and must understand it for a viva.
- When a decision is not covered here, stop and ask; do not guess silently.
- Log in `AI_USE.md` one line per session: tool, task, date (venues may require an AI-use disclosure).

## 9. Open items (update when resolved)

- Installed wind capacity per year for Tamil Nadu (source and values).
- Confirm CEA unit (MU/day) from the CEA report.
- Confirm sampling coordinates for the three Tamil Nadu wind belts.
- Venue decision happens at G4.
