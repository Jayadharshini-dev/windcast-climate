-- 04_views.sql — the data the model reads (from gate G2 on, models read views, not raw tables).
-- Refresh after each ETL load:
--   REFRESH MATERIALIZED VIEW CONCURRENTLY mv_daily_features;
--   REFRESH MATERIALIZED VIEW CONCURRENTLY mv_weekly_features;

-- One row per day for the state: weather averaged over the wind-belt points,
-- joined to that day's state generation. Lags and calendar terms are built in
-- model code (so leakage rules stay in one place).
CREATE MATERIALIZED VIEW mv_daily_features AS
WITH state_weather AS (
    SELECT w.ts,
           s.region,
           AVG(w.wind_speed)   AS wind_speed,
           AVG(w.temperature)  AS temperature,
           AVG(w.humidity)     AS humidity,
           COUNT(w.wind_speed) AS n_points            -- < 3 means a point was missing that day
    FROM raw_weather_readings w
    JOIN stations s ON s.station_id = w.station_id AND s.kind = 'weather_point'
    GROUP BY w.ts, s.region
)
SELECT sw.ts,
       sw.region,
       sw.wind_speed,
       sw.temperature,
       sw.humidity,
       sw.n_points,
       p.generation_mu,
       p.capacity_mw,
       p.capacity_factor
FROM state_weather sw
JOIN stations st ON st.region = sw.region AND st.kind = 'state_aggregate'
LEFT JOIN power_output p ON p.station_id = st.station_id AND p.ts = sw.ts
WITH NO DATA;

CREATE UNIQUE INDEX ux_mv_daily_features ON mv_daily_features (region, ts);

-- Weekly means (Monday-start weeks) for EDA and the dashboard.
CREATE MATERIALIZED VIEW mv_weekly_features AS
SELECT date_trunc('week', ts)::date AS week_start,
       region,
       AVG(wind_speed)        AS wind_speed,
       AVG(temperature)       AS temperature,
       AVG(humidity)          AS humidity,
       AVG(generation_mu)     AS generation_mu,
       COUNT(generation_mu)   AS n_days_with_generation
FROM mv_daily_features
GROUP BY 1, 2
WITH NO DATA;

CREATE UNIQUE INDEX ux_mv_weekly_features ON mv_weekly_features (region, week_start);

REFRESH MATERIALIZED VIEW mv_daily_features;
REFRESH MATERIALIZED VIEW mv_weekly_features;
