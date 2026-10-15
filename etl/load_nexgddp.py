"""Load the NEX-GDDP monthly window means (data/golden/nexgddp_monthly_means.csv) into climate_scenarios.

Checks and conversions (nothing is loaded silently):
  * tas Kelvin -> deg C; values outside plausible ranges stop the load
  * a (model, scenario, window) block is loaded only if all 12 months x all weather points are present;
    incomplete blocks are listed in the report and skipped
  * annual rows (month = NULL) are the mean of the 12 monthly means (months equally weighted)
Re-running is safe: rows are upserted on the table's unique key.
"""
from __future__ import annotations

import argparse
import pathlib

import pandas as pd

from etl.db import connect
from etl.fetch_nexgddp import OUT_CSV, VARIABLES

KELVIN = 273.15
RANGES = {  # plausible values for monthly means over South India (hard stop if violated)
    "sfcWind": (0.0, 15.0),   # m/s
    "temp_c": (15.0, 40.0),   # deg C
    "hurs": (10.0, 100.0),    # %
}
# n_days expected in a 20-year, one-month block: 560 (Feb, 365-day calendars) .. 620 (31-day months)
N_IMAGES_EXPECTED = (540, 640)


def prepare(df: pd.DataFrame, station_names: list[str]) -> tuple[pd.DataFrame, dict]:
    """Validate and convert. Returns (monthly rows for complete blocks, report dict)."""
    need = {"model", "scenario", "window_start", "window_end", "target_year", "month", "station",
            "n_images", *VARIABLES}
    if need - set(df.columns):
        raise ValueError(f"CSV is missing columns: {sorted(need - set(df.columns))}")
    df = df.copy()
    df["temp_c"] = df["tas"] - KELVIN

    for col, (lo, hi) in RANGES.items():
        bad = df[(df[col] < lo) | (df[col] > hi)]
        if len(bad):
            raise ValueError(f"{len(bad)} rows have {col} outside {lo}..{hi}; first: {bad.iloc[0].to_dict()}")
    unknown = set(df["station"]) - set(station_names)
    if unknown:
        raise ValueError(f"CSV has stations not in the database: {sorted(unknown)}")

    dupes = df.duplicated(["model", "scenario", "window_start", "month", "station"]).sum()
    if dupes:
        raise ValueError(f"{dupes} duplicate (model, scenario, window, month, station) rows in the CSV")

    key = ["model", "scenario", "window_start", "window_end", "target_year"]
    ok_blocks, incomplete = [], []
    for k, g in df.groupby(key):
        complete = g["month"].nunique() == 12 and g.groupby("month")["station"].nunique().eq(len(station_names)).all()
        (ok_blocks if complete else incomplete).append(k)
    odd_n = df[(df["n_images"] < N_IMAGES_EXPECTED[0]) | (df["n_images"] > N_IMAGES_EXPECTED[1])]

    keep = df.set_index(key).loc[ok_blocks].reset_index() if ok_blocks else df.iloc[0:0]
    # a future block needs the same model's baseline; report models that cannot produce change factors
    base_models = set(keep.loc[keep["scenario"] == "historical", "model"])
    usable = keep[(keep["scenario"] == "historical") | (keep["model"].isin(base_models))]
    report = {
        "rows_in_csv": len(df),
        "blocks_complete": len(ok_blocks),
        "blocks_incomplete_skipped": [dict(zip(key, k)) for k in incomplete],
        "rows_with_unusual_n_images": len(odd_n),
        "models_with_complete_baseline": sorted(base_models),
        "models_without_complete_baseline_excluded": sorted(set(df["model"]) - base_models),
    }
    return usable, report


def with_annual(monthly: pd.DataFrame) -> pd.DataFrame:
    """Add month = NULL annual-summary rows (mean of the 12 monthly means)."""
    key = ["model", "scenario", "window_start", "window_end", "target_year", "station"]
    annual = monthly.groupby(key, as_index=False)[["sfcWind", "temp_c", "hurs"]].mean()
    annual["month"] = None
    out = pd.concat([monthly[key + ["month", "sfcWind", "temp_c", "hurs"]], annual], ignore_index=True)
    return out


def upsert(conn, rows: pd.DataFrame) -> int:
    """Write rows into climate_scenarios. Does NOT commit: the caller decides (tests roll back)."""
    ids = dict(conn.execute("SELECT name, station_id FROM stations").fetchall())
    params = [
        (r.scenario, int(r.target_year), int(r.window_start), int(r.window_end),
         None if pd.isna(r.month) else int(r.month), r.model, ids[r.station],
         float(r.sfcWind), float(r.temp_c), float(r.hurs))
        for r in rows.itertuples(index=False)
    ]
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO climate_scenarios
                   (name, target_year, window_start, window_end, month, climate_model, station_id,
                    projected_wind_speed, projected_temperature, projected_humidity)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (name, target_year, month, climate_model, station_id) DO UPDATE
               SET window_start = EXCLUDED.window_start, window_end = EXCLUDED.window_end,
                   projected_wind_speed = EXCLUDED.projected_wind_speed,
                   projected_temperature = EXCLUDED.projected_temperature,
                   projected_humidity = EXCLUDED.projected_humidity
            """,
            params,
        )
    return len(params)


def main(csv_path: pathlib.Path = OUT_CSV) -> dict:
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} not found. Run the fetch first (python -m etl.run_phase4 --ee-project ...) "
                         "or pull the committed file from git.")
    df = pd.read_csv(csv_path)
    with connect() as conn:
        names = [n for (n,) in conn.execute("SELECT name FROM stations WHERE kind = 'weather_point'")]
        monthly, report = prepare(df, names)
        if monthly.empty:
            raise SystemExit("No complete blocks to load. See the report: " + str(report))
        report["rows_upserted"] = upsert(conn, with_annual(monthly))
        conn.commit()
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=pathlib.Path, default=OUT_CSV)
    for k, v in main(p.parse_args().csv).items():
        print(f"{k}: {v}")
