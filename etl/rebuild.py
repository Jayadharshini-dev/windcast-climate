"""Rebuild the whole database from the repository alone (gate G5, one command).

    docker compose up -d
    python -m etl.rebuild

Steps: wait for the database -> apply migrations -> load the seed CSVs (CEA + ERA5) -> refresh
views -> load the committed NEX-GDDP monthly means -> load the committed model results -> check the result against the committed
golden file. Needs no internet and no accounts. Re-running is safe.
"""
import sys
import time

import pandas as pd

from etl import load_nexgddp, load_results, load_seed, run_phase2, run_phase4
from etl.db import ROOT, connect

GOLDEN = ROOT / "data" / "golden" / "tn_daily_golden.csv"


def wait_for_db(seconds: int = 60) -> None:
    for i in range(seconds):
        try:
            with connect() as conn:
                conn.execute("SELECT 1")
            return
        except Exception as exc:  # database still starting
            last = exc
            time.sleep(1)
    raise SystemExit(f"database not reachable after {seconds}s: {last}\nIs `docker compose up -d` running?")


def check_against_golden() -> str:
    golden = pd.read_csv(GOLDEN, parse_dates=["ts"])
    with connect() as conn:
        cur = conn.execute(
            "SELECT ts, region, wind_speed, temperature, humidity, n_points, generation_mu, capacity_mw, capacity_factor "
            "FROM mv_daily_features WHERE region = 'Tamil Nadu' AND ts BETWEEN %s AND %s ORDER BY ts",
            (golden["ts"].min().date(), golden["ts"].max().date()),
        )
        now = pd.DataFrame(cur.fetchall(), columns=[c.name for c in cur.description])
    now["ts"] = pd.to_datetime(now["ts"])
    now = now.round(4)
    if len(now) != len(golden):
        raise SystemExit(f"rebuild check FAILED: {len(now)} rows, golden has {len(golden)}")
    cols = ["wind_speed", "temperature", "humidity", "generation_mu", "n_points"]
    diff = (now[cols].astype(float) - golden[cols].astype(float)).abs().max().max()
    nan_match = (now[cols].isna().values == golden[cols].isna().values).all()
    if not nan_match or diff > 1e-3:
        raise SystemExit(f"rebuild check FAILED: max difference {diff}, NaN pattern match {nan_match}")
    return f"{len(now)} days match data/golden/tn_daily_golden.csv (max difference {diff:.5f})"


def main() -> None:
    print("1/6 waiting for the database ...")
    wait_for_db()
    print("2/6 applying migrations ...")
    run_phase4.apply_migrations()
    print("3/6 loading seed data (CEA generation + ERA5 weather) ...")
    print("   ", load_seed.main())
    run_phase2.refresh_views()
    print("4/6 loading NEX-GDDP climate change factors ...")
    print("   ", load_nexgddp.main())
    print("5/6 loading committed model and climate-impact results ...")
    print("   ", load_results.main())
    print("6/6 checking against the golden file ...")
    print("   ", check_against_golden())
    print("Done. Database is ready.")


if __name__ == "__main__":
    sys.exit(main())
