"""Load the committed seed CSVs (data/seed/) into the database. No internet needed.

    python -m etl.load_seed

Uses the same upsert code as the real loaders, so the result is identical to loading the raw
downloads. Safe to re-run.
"""
import pandas as pd

from etl import load_cea, load_era5
from etl.db import ROOT, connect

SEED_DIR = ROOT / "data" / "seed"


def main() -> dict:
    cea = pd.read_csv(SEED_DIR / "cea_tn_daily.csv", parse_dates=["ts"])
    cea["ts"] = cea["ts"].dt.date
    n_cea = load_cea.load(cea)

    era5 = pd.read_csv(SEED_DIR / "era5_daily.csv", parse_dates=["ts"])
    era5["ts"] = era5["ts"].dt.date
    with connect() as conn:
        ids = dict(conn.execute("SELECT name, station_id FROM stations WHERE kind = 'weather_point'").fetchall())
    missing = set(era5["point"]) - set(ids)
    if missing:
        raise SystemExit(f"seed has points not in the stations table: {sorted(missing)}")
    n_era5 = {}
    for name, part in era5.groupby("point"):
        n_era5[name] = load_era5.load(ids[name], part.drop(columns="point"))
    return {"cea_rows": n_cea, "era5_rows": n_era5}


if __name__ == "__main__":
    print(main())
