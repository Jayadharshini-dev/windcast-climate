# ETL (owner: Pavan)

Loads the raw data into the database and exports the golden dataset (gate G1).

```bash
pip install -r requirements.txt
docker compose up -d
python -m etl.run_phase2
```

Needs `data/raw/state-control-renewable-energy-generation.csv` (CEA, India Data Portal) and internet
access to Open-Meteo on the first run. Downloads are cached in `data/raw/era5/`. Re-running is
safe, because rows are upserted, not duplicated.

| Script | Loads | Into |
| --- | --- | --- |
| `load_cea.py` | Tamil Nadu daily `wind_energy` (MU/day) | `power_output`, station 1 |
| `load_era5.py` | ERA5 hourly → daily means, 3 wind-belt points | `raw_weather_readings`, stations 2–4 |
| `run_phase2.py` | both of the above, refreshes views, writes `data/golden/` | — |

## Data decisions

- **Missing CEA days → NULL rows**, never filled (20 days absent from the file).
- **All-zero CEA days → NULL.** On 2020-06-18 and 2022-10-30, wind, solar and total renewable are
  all exactly 0, while the neighbouring days show normal values (e.g. 86 → 0 → 75 MU in June).
  These are reporting gaps, not calm days. That makes **22 NULL days** in total.
- **Duplicates:** the 2 duplicate (date, state) rows in the file are Jharkhand, not Tamil Nadu.
  The loader still keeps the first row if any ever appear.
- **ERA5 via Open-Meteo with `models=era5`**: pure ERA5 at 0.25°, the same grid as NEX-GDDP-CMIP6.
  Each point snaps to the nearest grid cell, and the report lists the actual cell used.
- **Wind at 10 m in m/s** (`wind_speed_unit=ms`), to match NEX-GDDP `sfcWind`.
- **Daily means use India calendar days** (`timezone=Asia/Kolkata`), so each weather value covers the
  same 24 hours as that date's CEA generation figure.
- **Weather range 2020-05-01 → 2025-04-30**: the CEA span plus a margin for lag features.
- `capacity_mw` / `capacity_factor` are left NULL until yearly TN capacity data is found (Phase 3).

## Reading data in model code (gate G2)

```python
from etl.features import read_features

df = read_features()                          # every day in the span
train = read_features(require_generation=True)  # only days with a generation value
sub = read_features(start="2021-01-01", end="2023-12-31")
```

Run from the repo root so `etl` is importable. The connection comes from `DATABASE_URL` or `.env`
(see `etl/db.py`); with the default docker-compose it is `localhost:5432`, or `5433` if you set
`DB_HOST_PORT=5433`.

One row per day, sorted by date, 2020-05-01 → 2025-04-30 (1,826 rows): `ts, region, wind_speed, temperature, humidity, n_points,
generation_mu, capacity_mw, capacity_factor`. Weather is the equal-weight mean of the 3 wind-belt
points. `generation_mu` is NaN on 52 days and never filled: 22 missing days inside the CEA span plus the 30
weather-only margin days around it (kept for lag features). `read_features(require_generation=True)`
returns the 1,774 rows with a value. `capacity_*` are NaN until
yearly capacity data is loaded; until then the target is `generation_mu`. Lags and day-of-year
terms are **not** included, so build them in the model code after the time split. Pressure is
deliberately not exposed (rule 2).

Check it works: `pytest tests/test_features.py` (skips if the database is empty).

## Climate data: NEX-GDDP change factors (Phase 4)

```bash
earthengine authenticate                                   # once, opens a browser
python -m etl.run_phase4 --ee-project YOUR_PROJECT_ID      # fetch from Google Earth Engine, load, export
python -m etl.run_phase4 --skip-fetch                      # load the committed CSV only (no Google account needed)
```

**One-time Earth Engine setup:** register at https://code.earthengine.google.com/register (free for
non-commercial use), create or pick a Google Cloud project there, and use its project id above.

| Script | Does |
| --- | --- |
| `fetch_nexgddp.py` | For each climate model, scenario window and calendar month, asks Earth Engine for the mean of the daily NEX-GDDP values (`sfcWind`, `tas`, `hurs`) at the 3 wind-belt points. Resumable. Writes `data/golden/nexgddp_monthly_means.csv` |
| `load_nexgddp.py` | Validates, converts Kelvin to deg C, and upserts into `climate_scenarios` (12 monthly rows + 1 annual row with `month` NULL per model/window/point) |
| `run_phase4.py` | Applies `db/migrations/*.sql`, runs both, exports `change_factors_region_monthly.csv` and `climate_load_report.md` |

**Why Earth Engine:** the raw NEX-GDDP files are global maps, tens of GB per model for our windows.
Earth Engine averages them on Google's servers and returns only a few numbers per request.

**Design (agreed at G0):**
- Baseline **1995–2014** (`historical`, label 2005); futures **2021–2040** (label 2030) and **2041–2060** (label 2050) under `ssp245` and `ssp585`.
- **Monthly** change factors per climate model: wind = ratio future/baseline, temperature and humidity = additive. Each model is compared with its own baseline only.
- Provisional ensemble of **10 models** (`MODELS` in `fetch_nexgddp.py`): chosen for institutional spread, **not** tested for skill over South India. The spread across models is the uncertainty range.
- No pressure (not in NEX-GDDP, rule 2). No direct daily NEX-GDDP feed for now (revisit in week 2).
- Annual rows are the plain mean of the 12 monthly means (months equally weighted).
- NEX-GDDP can be masked on ocean/coastal cells. The fetch checks the 3 points first and stops with a clear message if one has no data.

**What the model team reads:** the views `v_change_factors` (per point) and `v_change_factors_region`
(state level: ratio/difference of the equal-weight point means, matching how `mv_daily_features` is built),
or the CSV export. Humidity plus a positive delta can exceed 100%: clip to 0–100 after applying it.

Tests: `pytest tests/test_nexgddp.py` (synthetic data, rolled back; Earth Engine calls themselves are not covered).

## Output for the model team

- `data/golden/tn_daily_golden.csv`: one row per day 2020-05-28 → 2025-04-27, exported from `mv_daily_features`.
- `data/golden/load_report.md`: every count and check, generated by the script.
