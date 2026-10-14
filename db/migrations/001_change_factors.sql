-- 001_change_factors.sql — delta-change factors from climate_scenarios (no table changes, views only).
-- Safe to run repeatedly (CREATE OR REPLACE). Applied automatically by `python -m etl.run_phase4`;
-- to apply by hand:  docker compose exec -T db psql -U windcast -d windcast < db/migrations/001_change_factors.sql
--
-- climate_scenarios holds NEX-GDDP window means per climate model, calendar month and weather point:
--   name = 'historical'  -> 1995-2014 baseline (target_year 2005)
--   name = 'ssp245' / 'ssp585' -> 2021-2040 (target_year 2030) and 2041-2060 (target_year 2050)
-- Change factors compare each future window with that model's OWN baseline (never across models):
--   wind        -> ratio     future / baseline   (apply by multiplying ERA5 wind)
--   temperature -> additive  future - baseline   (deg C, apply by adding)
--   humidity    -> additive  future - baseline   (% points, apply by adding; clip to 0-100 afterwards)
-- month = NULL rows are annual summaries; 1-12 are the monthly factors the model should apply.

-- Per weather point (stations 2-4).
CREATE OR REPLACE VIEW v_change_factors AS
SELECT f.name                  AS scenario,
       f.target_year,
       f.window_start,
       f.window_end,
       f.month,
       f.climate_model,
       f.station_id,
       f.projected_wind_speed  / NULLIF(h.projected_wind_speed, 0)  AS wind_ratio,
       f.projected_temperature - h.projected_temperature            AS temp_delta_c,
       f.projected_humidity    - h.projected_humidity               AS humidity_delta_pct
FROM climate_scenarios f
JOIN climate_scenarios h
  ON  h.name          = 'historical'
  AND h.climate_model = f.climate_model
  AND h.station_id    = f.station_id
  AND h.month IS NOT DISTINCT FROM f.month
WHERE f.name IN ('ssp245', 'ssp585');

-- Per region: the state-level feature is the equal-weight mean of the weather points, so the state
-- factor is the ratio / difference of the point-MEANS (not the mean of point ratios).
CREATE OR REPLACE VIEW v_change_factors_region AS
SELECT s.region,
       f.name                  AS scenario,
       f.target_year,
       f.window_start,
       f.window_end,
       f.month,
       f.climate_model,
       count(*)                AS n_points,
       avg(f.projected_wind_speed)  / NULLIF(avg(h.projected_wind_speed), 0) AS wind_ratio,
       avg(f.projected_temperature) - avg(h.projected_temperature)           AS temp_delta_c,
       avg(f.projected_humidity)    - avg(h.projected_humidity)              AS humidity_delta_pct
FROM climate_scenarios f
JOIN stations s ON s.station_id = f.station_id AND s.kind = 'weather_point'
JOIN climate_scenarios h
  ON  h.name          = 'historical'
  AND h.climate_model = f.climate_model
  AND h.station_id    = f.station_id
  AND h.month IS NOT DISTINCT FROM f.month
WHERE f.name IN ('ssp245', 'ssp585')
GROUP BY s.region, f.name, f.target_year, f.window_start, f.window_end, f.month, f.climate_model;
