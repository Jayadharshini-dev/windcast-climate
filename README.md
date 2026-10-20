# WindCast Climate
Database-backed wind power forecasting and CMIP6 climate-shift analysis for Tamil Nadu, India.
Stack: PostgreSQL + TimescaleDB (Docker), Python, scikit-learn / XGBoost, Streamlit.
See `PROJECT_CONTEXT.md` for the full project context and rules.

## Run it from a fresh clone (no accounts, no internet after install)

Needs Docker Desktop and Python 3.9 to 3.12 (3.12 recommended; numpy 2.0.2 has no wheel for 3.13).

```bash
git clone https://github.com/Jayadharshini-dev/windcast-climate.git
cd windcast-climate
cp .env.example .env                       # once; if port 5432 is busy, add DB_HOST_PORT=5433 to .env
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
docker compose up -d                       # starts the database and creates the tables
python -m etl.run_all                      # builds the whole database and checks it (about 10 seconds)
streamlit run dashboard/app.py             # opens the dashboard at http://localhost:8501
```

`python -m etl.run_all` (which runs `etl.rebuild`) fills the database from the small CSV files committed in `data/seed/`
(CEA generation, ERA5 weather) and `data/golden/` (NEX-GDDP climate means), then checks that the
result matches `data/golden/tn_daily_golden.csv`. If it prints `Done. Database is ready.`, the
database is correct. Run `pytest` afterwards to check everything (about 36 tests).

If the database is on port 5433, put `DATABASE_URL=postgresql://windcast:windcast_dev@localhost:5433/windcast`
in `.env`.

### The full pipeline, end to end

```
data/seed + data/golden  ->  database  ->  models (read the database)  ->  climate analysis  ->  database  ->  dashboard
```

`python -m etl.run_all --retrain` (about 5 minutes) does all of it: rebuilds the database, trains and validates the
forecast models on data read from the database, applies the 10 climate models' change factors, writes the results back
into `forecast_runs`, `model_predictions` and `climate_impact`, and the dashboard reads them. The models are seeded, so the
numbers reproduce the committed files in `models/` and `climate/` to within rounding (RMSE within 0.02 MU/day).

## Folders

| Folder | Owner | What is in it |
| --- | --- | --- |
| `db/` | Pavan | schema, views, migrations: see `db/README.md` |
| `etl/` | Pavan | loaders, rebuild, benchmark: see `etl/README.md` |
| `dashboard/` | Pavan | Streamlit app (`app.py`) and every SQL query it runs (`queries.py`) |
| `models/`, `climate/` | Jayadharshini | forecasting models, climate-shift analysis |
| `data/seed/`, `data/golden/` | Pavan | committed data that makes the rebuild work offline |
| `paper/` | both | benchmark numbers and paper material |

## Refreshing the data from the original sources (optional)

Only needed if the sources change; the committed seed is enough to reproduce everything.

```bash
python -m etl.run_phase2                                   # CEA csv in data/raw/ + Open-Meteo download
python -m etl.export_seed                                  # rewrite data/seed/ from them
pip install -r requirements-fetch.txt                      # adds earthengine-api (Python 3.10+)
python -m etl.run_phase4 --ee-project YOUR_PROJECT_ID      # NEX-GDDP from Google Earth Engine
python -m etl.benchmark_queries                            # query timings -> paper/benchmark_results.md
```

## Dashboard

Five views, all read-only against the database, for three kinds of reader:

| View | For | What it shows |
| --- | --- | --- |
| Overview | everyone | the seasonal wind curve, three plain answers (when it blows, how well we forecast, what climate may do) with how sure we are, the daily record |
| Public | citizens, students, journalists | month picker, the 2030/2050 climate outlook in plain words, what the study does not tell you, where the weather comes from |
| Researchers | academics, the team | every climate model's answer, wind versus temperature drivers, monthly change factors, forecast skill tables, CSV downloads |
| Planners | grid and energy officials | forecast against actual with an error band, seasonal planning calendar, scenario risk ranges, one-page summary download |
| Data and method | everyone | data quality, sources, method in four lines, limits |

`python -m etl.rebuild` also loads the committed model results (`models/trees_holdout_predictions.csv`) and climate
impact results (`climate/climate_impact.csv`) into `forecast_runs`, `model_predictions` and `climate_impact`, so every
view has data on a fresh clone. The forecasts shown are back-tests on 2024 onward, not live forecasts.
The look is a light theme pinned in `.streamlit/config.toml`; styling is in `dashboard/theme.py`.
