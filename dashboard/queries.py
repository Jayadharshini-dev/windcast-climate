"""All SQL the dashboard runs, in one place (also what etl/benchmark_queries.py times).

Every function takes an open psycopg connection and returns a DataFrame. The dashboard never
writes to the database.
"""
from __future__ import annotations

import pandas as pd

REGION = "Tamil Nadu"


def _frame(conn, sql: str, params: tuple = ()) -> pd.DataFrame:
    cur = conn.execute(sql, params)
    return pd.DataFrame(cur.fetchall(), columns=[c.name for c in cur.description])


def daily_features(conn, start=None, end=None) -> pd.DataFrame:
    df = _frame(
        conn,
        """SELECT ts, wind_speed, temperature, humidity, n_points, generation_mu
           FROM mv_daily_features
           WHERE region = %s AND (%s::date IS NULL OR ts >= %s) AND (%s::date IS NULL OR ts <= %s)
           ORDER BY ts""",
        (REGION, start, start, end, end),
    )
    df["ts"] = pd.to_datetime(df["ts"])
    return df.astype({c: float for c in df.columns if c not in ("ts", "n_points")})


def weekly_features(conn) -> pd.DataFrame:
    df = _frame(conn, "SELECT * FROM mv_weekly_features WHERE region = %s ORDER BY week_start", (REGION,))
    return df


def data_quality(conn) -> dict:
    row = conn.execute(
        """SELECT count(*), count(generation_mu), count(*) FILTER (WHERE n_points = 3),
                  min(ts), max(ts), min(ts) FILTER (WHERE generation_mu IS NOT NULL),
                  max(ts) FILTER (WHERE generation_mu IS NOT NULL)
           FROM mv_daily_features WHERE region = %s""",
        (REGION,),
    ).fetchone()
    return {
        "days": row[0], "days_with_generation": row[1], "days_all_3_points": row[2],
        "first_day": row[3], "last_day": row[4], "gen_first": row[5], "gen_last": row[6],
    }


def table_counts(conn) -> pd.DataFrame:
    parts = " UNION ALL ".join(
        f"SELECT '{t}' AS table_name, count(*) AS rows FROM {t}"
        for t in ("stations", "raw_weather_readings", "power_output", "climate_scenarios",
                  "forecast_runs", "model_predictions", "climate_impact")
    )
    return _frame(conn, parts)


def monthly_climatology(conn) -> pd.DataFrame:
    df = _frame(
        conn,
        """SELECT extract(month FROM ts)::int AS month,
                  avg(wind_speed) AS wind_speed, avg(generation_mu) AS generation_mu
           FROM mv_daily_features WHERE region = %s GROUP BY 1 ORDER BY 1""",
        (REGION,),
    )
    return df.astype(float)


def change_factors(conn, scenario: str, target_year: int, month: int | None) -> pd.DataFrame:
    """One row per climate model (state level). month=None gives the annual row."""
    df = _frame(
        conn,
        """SELECT climate_model, wind_ratio, temp_delta_c, humidity_delta_pct
           FROM v_change_factors_region
           WHERE region = %s AND scenario = %s AND target_year = %s
             AND month IS NOT DISTINCT FROM %s
           ORDER BY climate_model""",
        (REGION, scenario, target_year, month),
    )
    return df


def monthly_factors(conn, scenario: str, target_year: int) -> pd.DataFrame:
    """12 monthly values per climate model, for the spaghetti chart."""
    return _frame(
        conn,
        """SELECT month, climate_model, wind_ratio, temp_delta_c, humidity_delta_pct
           FROM v_change_factors_region
           WHERE region = %s AND scenario = %s AND target_year = %s AND month IS NOT NULL
           ORDER BY climate_model, month""",
        (REGION, scenario, target_year),
    )


def annual_factors_all(conn) -> pd.DataFrame:
    return _frame(
        conn,
        """SELECT scenario, target_year, climate_model, wind_ratio, temp_delta_c, humidity_delta_pct
           FROM v_change_factors_region WHERE region = %s AND month IS NULL
           ORDER BY scenario, target_year, climate_model""",
        (REGION,),
    )


def forecast_runs(conn) -> pd.DataFrame:
    return _frame(
        conn,
        """SELECT r.run_id, r.model_version, r.run_timestamp, r.horizon_days, r.params,
                  count(p.prediction_id) AS n_predictions,
                  min(p.forecast_date) AS first_date, max(p.forecast_date) AS last_date
           FROM forecast_runs r LEFT JOIN model_predictions p USING (run_id)
           GROUP BY r.run_id ORDER BY r.run_id DESC""",
    )


def predictions_vs_actual(conn, run_id: int, station_id: int = 1) -> pd.DataFrame:
    df = _frame(
        conn,
        """SELECT p.forecast_date AS ts, p.predicted_value, p.lower, p.upper, f.generation_mu AS actual
           FROM model_predictions p
           LEFT JOIN mv_daily_features f ON f.ts = p.forecast_date AND f.region = %s
           WHERE p.run_id = %s AND p.station_id = %s ORDER BY p.forecast_date""",
        (REGION, run_id, station_id),
    )
    df["ts"] = pd.to_datetime(df["ts"])
    return df.astype({c: float for c in df.columns if c != "ts"})


def climate_impact(conn) -> pd.DataFrame:
    return _frame(
        conn,
        """SELECT region, scenario, target_year, climate_model, metric, value, p10, p90
           FROM climate_impact WHERE region = %s ORDER BY scenario, target_year, metric, climate_model""",
        (REGION,),
    )


# ---------------------------------------------------------------- added for the audience views
def stations(conn) -> pd.DataFrame:
    df = _frame(conn, "SELECT station_id, name, latitude, longitude FROM stations WHERE kind = 'weather_point' ORDER BY 1")
    return df.astype({"latitude": float, "longitude": float})


def monthly_by_year(conn) -> pd.DataFrame:
    """Mean generation and wind per calendar month in each year, for the planning calendar."""
    df = _frame(
        conn,
        """SELECT extract(year FROM ts)::int AS year, extract(month FROM ts)::int AS month,
                  avg(generation_mu) AS generation_mu, avg(wind_speed) AS wind_speed, count(generation_mu) AS n_days
           FROM mv_daily_features WHERE region = %s GROUP BY 1, 2 ORDER BY 1, 2""",
        (REGION,),
    )
    return df.astype({"generation_mu": float, "wind_speed": float})


def forecast_skill(conn) -> pd.DataFrame:
    """Error of every stored forecast run on its test days (days with an actual value)."""
    df = _frame(
        conn,
        """SELECT r.run_id, r.model_version AS model, r.horizon_days, count(*) AS n_days,
                  sqrt(avg(power(p.predicted_value - f.generation_mu, 2))) AS rmse,
                  avg(abs(p.predicted_value - f.generation_mu)) AS mae
           FROM forecast_runs r
           JOIN model_predictions p USING (run_id)
           JOIN mv_daily_features f ON f.ts = p.forecast_date AND f.region = %s
           WHERE f.generation_mu IS NOT NULL
           GROUP BY r.run_id, r.model_version, r.horizon_days ORDER BY r.horizon_days, rmse""",
        (REGION,),
    )
    return df.astype({"rmse": float, "mae": float})


def runs_available(conn) -> pd.DataFrame:
    return _frame(
        conn,
        """SELECT r.run_id, r.model_version AS model, r.horizon_days
           FROM forecast_runs r WHERE EXISTS (SELECT 1 FROM model_predictions p WHERE p.run_id = r.run_id)
           ORDER BY r.horizon_days, r.model_version""",
    )


def change_factors_download(conn) -> pd.DataFrame:
    return _frame(
        conn,
        """SELECT scenario, target_year, month, climate_model, wind_ratio, temp_delta_c, humidity_delta_pct
           FROM v_change_factors_region WHERE region = %s
           ORDER BY scenario, target_year, climate_model, month NULLS LAST""",
        (REGION,),
    )
