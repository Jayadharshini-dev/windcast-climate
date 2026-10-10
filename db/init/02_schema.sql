-- 02_schema.sql — WindCast Climate schema (G0 candidate)
-- Grain: DAILY. Units are fixed per column (see comments) — ETL must convert to these.
-- Contract source: PROJECT_CONTEXT.md section 6. Deviations from the draft are listed in db/README.md.

------------------------------------------------------------------------------
-- 1. stations: every location we store data for.
--    'weather_point'   = an ERA5 / NEX-GDDP sampling point in a wind belt
--    'state_aggregate' = the whole state (CEA generation is reported per state)
------------------------------------------------------------------------------
CREATE TABLE stations (
    station_id  SMALLINT     PRIMARY KEY,
    name        TEXT         NOT NULL UNIQUE,
    kind        TEXT         NOT NULL CHECK (kind IN ('weather_point', 'state_aggregate')),
    latitude    NUMERIC(7,4) CHECK (latitude  BETWEEN -90  AND 90),
    longitude   NUMERIC(7,4) CHECK (longitude BETWEEN -180 AND 180),
    region      TEXT         NOT NULL,                 -- e.g. 'Tamil Nadu'
    -- a weather point must have coordinates; the state aggregate need not
    CHECK (kind = 'state_aggregate' OR (latitude IS NOT NULL AND longitude IS NOT NULL))
);

------------------------------------------------------------------------------
-- 2. raw_weather_readings: daily ERA5 (Open-Meteo) values per weather point.
--    Hypertable. PK is (station_id, ts) because TimescaleDB requires every
--    unique key to include the time column.
------------------------------------------------------------------------------
CREATE TABLE raw_weather_readings (
    station_id   SMALLINT    NOT NULL REFERENCES stations(station_id),
    ts           DATE        NOT NULL,
    wind_speed   DOUBLE PRECISION CHECK (wind_speed  >= 0   AND wind_speed  < 60),   -- m/s, daily mean at 10 m
    temperature  DOUBLE PRECISION CHECK (temperature BETWEEN -20 AND 55),            -- deg C, daily mean at 2 m
    humidity     DOUBLE PRECISION CHECK (humidity    BETWEEN 0   AND 100),           -- %, daily mean relative humidity at 2 m
    pressure     DOUBLE PRECISION CHECK (pressure    BETWEEN 500 AND 1100),          -- hPa, nullable, NOT a model input (rule 2)
    n_hours      SMALLINT    CHECK (n_hours BETWEEN 0 AND 24),                  -- hourly values behind the daily mean
    source       TEXT        NOT NULL DEFAULT 'era5_open_meteo',
    ingested_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (station_id, ts)
);
SELECT create_hypertable('raw_weather_readings', by_range('ts', INTERVAL '1 year'));

------------------------------------------------------------------------------
-- 3. power_output: CEA daily wind generation for the state.
--    Missing CEA days are stored as rows with generation_mu = NULL (rule: never fill silently).
------------------------------------------------------------------------------
CREATE TABLE power_output (
    station_id       SMALLINT    NOT NULL REFERENCES stations(station_id),
    ts               DATE        NOT NULL,
    generation_mu    DOUBLE PRECISION CHECK (generation_mu >= 0),                    -- million units (GWh) per day; NULL = missing
    capacity_mw      DOUBLE PRECISION CHECK (capacity_mw > 0),                       -- installed wind capacity, nullable
    capacity_factor  DOUBLE PRECISION CHECK (capacity_factor BETWEEN 0 AND 1),       -- generation_mu / (capacity_mw*24/1000)
    source           TEXT        NOT NULL DEFAULT 'cea_india_data_portal',
    ingested_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (station_id, ts)
);
SELECT create_hypertable('power_output', by_range('ts', INTERVAL '1 year'));

------------------------------------------------------------------------------
-- 4. climate_scenarios: NEX-GDDP-CMIP6 window means per climate model.
--    'historical' rows hold the 1995–2014 baseline that delta-change factors are
--    computed against. month = NULL means an annual mean; 1–12 = that calendar month.
------------------------------------------------------------------------------
CREATE TABLE climate_scenarios (
    scenario_id            INTEGER  GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name                   TEXT     NOT NULL CHECK (name IN ('historical', 'ssp245', 'ssp585')),
    target_year            SMALLINT NOT NULL CHECK (target_year BETWEEN 1950 AND 2100), -- window centre label, e.g. 2030
    window_start           SMALLINT NOT NULL,
    window_end             SMALLINT NOT NULL,
    month                  SMALLINT CHECK (month BETWEEN 1 AND 12),
    climate_model          TEXT     NOT NULL,                                     -- e.g. 'ACCESS-CM2'
    station_id             SMALLINT NOT NULL REFERENCES stations(station_id),
    projected_wind_speed   DOUBLE PRECISION CHECK (projected_wind_speed >= 0),           -- m/s (sfcWind, 10 m)
    projected_temperature  DOUBLE PRECISION,                                                  -- deg C (tas converted from K)
    projected_humidity     DOUBLE PRECISION CHECK (projected_humidity BETWEEN 0 AND 100),-- % (hurs)
    CHECK (window_start <= target_year AND target_year <= window_end),
    UNIQUE NULLS NOT DISTINCT (name, target_year, month, climate_model, station_id)
);

------------------------------------------------------------------------------
-- 5. forecast_runs: one row per model training/forecast run.
------------------------------------------------------------------------------
CREATE TABLE forecast_runs (
    run_id         INTEGER     GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    model_version  TEXT        NOT NULL,                                     -- e.g. 'persistence', 'xgb_v1'
    run_timestamp  TIMESTAMPTZ NOT NULL DEFAULT now(),
    horizon_days   SMALLINT    NOT NULL CHECK (horizon_days BETWEEN 1 AND 3),
    params         JSONB                                                     -- seed, features, train window, hyperparameters
);

------------------------------------------------------------------------------
-- 6. model_predictions: one predicted value per run, station and date.
------------------------------------------------------------------------------
CREATE TABLE model_predictions (
    prediction_id    BIGINT   GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id           INTEGER  NOT NULL REFERENCES forecast_runs(run_id) ON DELETE CASCADE,
    station_id       SMALLINT NOT NULL REFERENCES stations(station_id),
    forecast_date    DATE     NOT NULL,
    predicted_value  DOUBLE PRECISION NOT NULL,                                      -- same unit as the target (CF or detrended MU)
    lower            DOUBLE PRECISION,
    upper            DOUBLE PRECISION,
    UNIQUE (run_id, station_id, forecast_date),
    CHECK (lower IS NULL OR lower <= predicted_value),
    CHECK (upper IS NULL OR upper >= predicted_value)
);
CREATE INDEX idx_model_predictions_station_date ON model_predictions (station_id, forecast_date);

------------------------------------------------------------------------------
-- 7. climate_impact: final results table read by the dashboard and the paper.
--    Shape is frozen at gate G3 (Jayadharshini's module writes it).
--    climate_model = 'ensemble' for the cross-model summary row.
------------------------------------------------------------------------------
CREATE TABLE climate_impact (
    impact_id      INTEGER  GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    region         TEXT     NOT NULL,
    scenario       TEXT     NOT NULL CHECK (scenario IN ('ssp245', 'ssp585')),
    target_year    SMALLINT NOT NULL,
    climate_model  TEXT     NOT NULL,
    metric         TEXT     NOT NULL,          -- e.g. 'mean_generation_change_pct', 'share_days_out_of_range'
    value          DOUBLE PRECISION NOT NULL,
    p10            DOUBLE PRECISION,
    p90            DOUBLE PRECISION,
    UNIQUE (region, scenario, target_year, climate_model, metric)
);
