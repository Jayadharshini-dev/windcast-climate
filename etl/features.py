"""The one function the model code uses to read data (gate G2: models read the database, via views).

    from etl.features import read_features
    df = read_features()                          # all days
    df = read_features(require_generation=True)   # only days with a generation value (training rows)

Returns a pandas DataFrame, one row per day, sorted by date, from 2020-05-01 to 2025-04-30 (1,826 rows: the
weather range). Generation exists only for 2020-05-28 to 2025-04-27, so the 30 margin days around that span
(kept for lag features) always have generation_mu = NaN. Use require_generation=True for training rows.
Columns:

    ts              datetime64[ns]  the date (India calendar day)
    region          str             'Tamil Nadu'
    wind_speed      float  m/s, 10 m, mean of the 3 wind-belt points (equal weight)
    temperature     float  deg C, 2 m, mean of the 3 points
    humidity        float  % relative humidity, 2 m, mean of the 3 points
    n_points        int    how many of the 3 points had data that day (3 = complete)
    generation_mu   float  million units per day; NaN = no value, never filled: 22 missing days inside the CEA
                           span + the 30 weather-only margin days = 52 NaN rows (1,774 rows have a value)
    capacity_mw     float  NaN until yearly capacity data is loaded
    capacity_factor float  NaN until yearly capacity data is loaded

What is NOT in here, on purpose: lag features and calendar terms (day-of-year sin/cos). Build them in
the model code, after the time-based split, so no future row can leak into a feature (rule 4).
Pressure is not exposed at all (rule 2).
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from etl.db import connect

COLUMNS = [
    "ts", "region", "wind_speed", "temperature", "humidity", "n_points",
    "generation_mu", "capacity_mw", "capacity_factor",
]
FLOAT_COLUMNS = ["wind_speed", "temperature", "humidity", "generation_mu", "capacity_mw", "capacity_factor"]


def read_features(
    region: str = "Tamil Nadu",
    start: str | dt.date | None = None,
    end: str | dt.date | None = None,
    require_generation: bool = False,
    conn=None,
) -> pd.DataFrame:
    """Read daily features from the materialized view mv_daily_features.

    start / end   inclusive date bounds (e.g. '2021-01-01'); None = no bound
    require_generation   True drops days where generation_mu is NULL
    conn          an open psycopg connection; if None, one is opened and closed here
    """
    sql = f"SELECT {', '.join(COLUMNS)} FROM mv_daily_features WHERE region = %s"
    params: list = [region]
    if start is not None:
        sql += " AND ts >= %s"
        params.append(start)
    if end is not None:
        sql += " AND ts <= %s"
        params.append(end)
    if require_generation:
        sql += " AND generation_mu IS NOT NULL"
    sql += " ORDER BY ts"

    own = conn is None
    conn = conn or connect()
    try:
        cur = conn.execute(sql, params)
        df = pd.DataFrame(cur.fetchall(), columns=COLUMNS)
    finally:
        if own:
            conn.close()

    if df.empty:
        raise ValueError(
            f"mv_daily_features has no rows for region={region!r} between {start} and {end}. "
            "Has the data been loaded (python -m etl.run_phase2)? Is the region name spelled exactly?"
        )

    df["ts"] = pd.to_datetime(df["ts"])
    df[FLOAT_COLUMNS] = df[FLOAT_COLUMNS].astype("float64")  # NULL -> NaN, also for all-NULL columns
    df["n_points"] = df["n_points"].astype("int64")
    return df
