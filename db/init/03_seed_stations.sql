-- 03_seed_stations.sql
-- Fixed IDs so ETL and model code can refer to them reliably.
-- Coordinates are APPROXIMATE (PROJECT_CONTEXT.md section 9 open item) — Pavan to confirm
-- against wind-farm locations, then update here before G0.
INSERT INTO stations (station_id, name, kind, latitude, longitude, region) VALUES
    (1, 'TN_STATE',              'state_aggregate', NULL,    NULL,    'Tamil Nadu'),
    (2, 'TN_Muppandal_Kayathar', 'weather_point',   8.2500, 77.5500, 'Tamil Nadu'),
    (3, 'TN_Palghat_Udumalpet',  'weather_point',  10.6000, 77.2500, 'Tamil Nadu'),
    (4, 'TN_Theni_Cumbum',       'weather_point',   9.7500, 77.3000, 'Tamil Nadu');
