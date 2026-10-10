# Database (owner: Pavan)

PostgreSQL 16 + TimescaleDB 2.17, run with Docker. Daily grain, all dates in UTC.

## Start it

```bash
cp .env.example .env          # once
docker compose up -d          # first start runs db/init/*.sql in order
docker compose exec db psql -U windcast -d windcast -c '\dt'
```

Rebuild from scratch (deletes all data): `docker compose down -v && docker compose up -d`

The `db/init` scripts only run when the data volume is new. After the schema is frozen,
changes go in a new numbered file in `db/migrations/` and through a PR.

## Files

| File | What it does |
| --- | --- |
| `init/01_extensions.sql` | Enables TimescaleDB, sets the database timezone to UTC |
| `init/02_schema.sql` | The 7 tables |
| `init/03_seed_stations.sql` | The state record + 3 wind-belt sampling points (fixed IDs 1–4) |
| `init/04_views.sql` | `mv_daily_features`, `mv_weekly_features` (what the model reads) |

## Tables at a glance

| Table | Grain / key | Notes |
| --- | --- | --- |
| `stations` | `station_id` | `kind` = `weather_point` (IDs 2–4) or `state_aggregate` (ID 1, `TN_STATE`) |
| `raw_weather_readings` | (`station_id`, `ts`) — hypertable | ERA5 daily means per point. m/s at 10 m, °C, % RH. `pressure` stored but never a model input |
| `power_output` | (`station_id`, `ts`) — hypertable | CEA state generation on station 1. Missing CEA days = row with `generation_mu` NULL |
| `climate_scenarios` | unique (`name`, `target_year`, `month`, `climate_model`, `station_id`) | NEX-GDDP window means. `historical` = 1995–2014 baseline. `month` NULL = annual |
| `forecast_runs` | `run_id` | `params` JSONB holds seed, features, train window |
| `model_predictions` | unique (`run_id`, `station_id`, `forecast_date`) | `lower` ≤ `predicted_value` ≤ `upper` enforced |
| `climate_impact` | unique (`region`, `scenario`, `target_year`, `climate_model`, `metric`) | Shape frozen at G3. `climate_model = 'ensemble'` for the summary row |

How the pieces join: weather is stored per sampling point, generation per state.
`mv_daily_features` averages the weather points of a region for each day and joins the state's
generation. Its `n_points` column shows how many points had wind data that day (3 = complete).

## Units the ETL must deliver

- Open-Meteo returns wind in **km/h by default** — request `wind_speed_unit=ms`.
- Wind must be **10 m** (`wind_speed_10m`), because NEX-GDDP `sfcWind` is 10 m. Do not use 100 m.
- NEX-GDDP `tas` is in **Kelvin** — convert to °C before loading.
- Daily values = mean of the 24 hourly values for that UTC date; `n_hours` records how many there were.

## Changes from the draft contract (PROJECT_CONTEXT.md §6)

1. **`reading_id` and `output_id` removed; primary key is (`station_id`, `ts`).** TimescaleDB requires
   every unique key on a hypertable to include the time column, so a standalone ID cannot be the PK.
   The composite key is also the `(station_id, ts)` index the contract asks for, and it makes
   duplicate days impossible (ETL uses `ON CONFLICT DO NOTHING`, which keeps the first row).
2. **`stations.kind` added** to separate the state record from weather sampling points.
3. **`climate_scenarios`: `window_start`, `window_end`, `month` added; `name` also allows `historical`.**
   Delta-change needs the baseline values in the same table, and the window years make the averaging period explicit.
4. **`forecast_runs.params` (JSONB) added** for reproducibility (seed, features, training window).
5. **`raw_weather_readings.n_hours`, `source`, `ingested_at`; `power_output.source`, `ingested_at` added** for data lineage.
6. **Surrogate PKs and unique keys added** to `climate_scenarios`, `model_predictions`, `climate_impact` (the draft had none on the last).

## Open questions to settle at G0

- **Delta-change grain:** annual or monthly change factors? The table supports both (`month` NULL or 1–12). Jayadharshini's call.
- **Direct NEX-GDDP feed (secondary check):** needs daily values per model, not window means — roughly
  34 models × 2 scenarios × 40 years × 365 days × 3 points ≈ 3 million rows. Proposal: a separate
  `climate_daily` hypertable added only if we run that check.
- **Sampling coordinates** in `03_seed_stations.sql` are approximate and still need confirming.
