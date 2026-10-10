-- 01_extensions.sql
-- TimescaleDB adds hypertables (automatic time-partitioning) on top of PostgreSQL.
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Store and compare all times in UTC. Daily data uses DATE, so this mainly
-- affects run timestamps (forecast_runs.run_timestamp, ingested_at).
DO $$
BEGIN
  EXECUTE format('ALTER DATABASE %I SET timezone TO %L', current_database(), 'UTC');
END $$;
