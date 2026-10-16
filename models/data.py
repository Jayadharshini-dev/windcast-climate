"""One place where model code gets its daily data (gate G2: read the database, fall back to the golden CSV).

    from models.data import load_daily
    df = load_daily()                  # DataFrame indexed by date (daily), columns wind_speed, temperature,
                                       # humidity, generation_mu (NaN = missing day, never filled)

Order of attempts (source="auto"): the database view via etl.features.read_features(); if the database is not
reachable (Docker not running), the golden CSV data/golden/tn_daily_golden.csv. Force one with source="db" or
source="csv", or with the environment variable WINDCAST_SOURCE=db|csv. A CSV path can be passed as csv_path.
The two sources hold the same numbers; the database also has 30 weather-only margin days with no generation.
"""
import os
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN = REPO_ROOT / "data" / "golden" / "tn_daily_golden.csv"
KEEP = ["wind_speed", "temperature", "humidity", "generation_mu"]


def _from_csv(path):
    df = pd.read_csv(path, parse_dates=["ts"])
    return df


def _from_db():
    from etl.features import read_features  # imported here so the CSV route needs no database driver

    return read_features()


def load_daily(source=None, csv_path=None, verbose=True):
    source = (source or os.environ.get("WINDCAST_SOURCE") or "auto").lower()
    df, used = None, None
    if csv_path is not None:
        df, used = _from_csv(csv_path), f"CSV {Path(csv_path).name}"
    elif source in ("auto", "db"):
        try:
            df, used = _from_db(), "database view mv_daily_features"
        except Exception as e:  # database down, driver missing, view empty ...
            if source == "db":
                raise
            if verbose:
                print(f"[data] database not available ({type(e).__name__}); using the golden CSV instead.")
    if df is None:
        df, used = _from_csv(GOLDEN), f"CSV {GOLDEN.name}"
    if verbose:
        print(f"[data] source: {used}")
    df = df.sort_values("ts").set_index("ts")[KEEP].asfreq("D")
    return df
