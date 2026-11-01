"""Fetch NEX-GDDP-CMIP6 monthly window means at the 3 wind-belt points from Google Earth Engine.

Why Earth Engine: the raw NEX-GDDP files are global maps (tens of GB per model for our windows).
Earth Engine hosts the same dataset (NASA/GDDP-CMIP6, 0.25 degree, daily) and averages it on
Google's servers, so only a few numbers per request come back.

For every (climate model, scenario window, calendar month) it returns the mean of the daily values
over the window, sampled at the grid cell containing each weather point:

    historical  1995-2014  (baseline; target_year label 2005)
    ssp245      2021-2040 (2030) and 2041-2060 (2050)
    ssp585      2021-2040 (2030) and 2041-2060 (2050)

Variables: sfcWind (m/s, 10 m), tas (K), hurs (%). No pressure: it does not exist in NEX-GDDP (rule 2).

Output: data/golden/nexgddp_monthly_means.csv (small, committed, so a fresh clone can load it
without an Earth Engine account:  python -m etl.run_phase4 --skip-fetch).
The run is resumable: finished (model, scenario, window, month) blocks are skipped on re-run.

One-time setup (in your own Terminal, because it opens a browser):
    earthengine authenticate
    python -m etl.run_phase4 --ee-project YOUR_GOOGLE_CLOUD_PROJECT_ID
"""
from __future__ import annotations

import argparse
import csv
import math
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from etl.db import ROOT, connect

DATASET = "NASA/GDDP-CMIP6"
VARIABLES = ["sfcWind", "tas", "hurs"]

# Provisional ensemble: spread across institutions, NOT tested for skill over South India.
# Jayadharshini to confirm. Names must match the `model` property in Earth Engine (checked at start).
MODELS = [
    "ACCESS-CM2", "CanESM5", "CMCC-ESM2", "EC-Earth3", "GFDL-ESM4",
    "IPSL-CM6A-LR", "MIROC6", "MPI-ESM1-2-HR", "MRI-ESM2-0", "NorESM2-MM",
]
# (scenario, window_start, window_end, target_year label)
WINDOWS = [
    ("historical", 1995, 2014, 2005),
    ("ssp245", 2021, 2040, 2030),
    ("ssp245", 2041, 2060, 2050),
    ("ssp585", 2021, 2040, 2030),
    ("ssp585", 2041, 2060, 2050),
]
OUT_CSV = ROOT / "data" / "golden" / "nexgddp_monthly_means.csv"
FIELDS = ["model", "scenario", "window_start", "window_end", "target_year", "month",
          "station", "n_images", "sfcWind", "tas", "hurs"]


# ---------------------------------------------------------------- pure helpers (unit-tested, no Earth Engine)
def rows_from_features(features: list[dict], model: str, scenario: str, window_start: int,
                       window_end: int, target_year: int, month: int) -> list[dict]:
    """Turn the 'features' list returned by Earth Engine into CSV rows. Fails loudly on missing values."""
    rows = []
    for f in features:
        p = f["properties"]
        missing = [v for v in VARIABLES if p.get(v) is None]
        if missing or not p.get("n_images"):
            raise ValueError(
                f"No value for {missing or 'all variables'} at point {p.get('station')!r} "
                f"({model}, {scenario}, {window_start}-{window_end}, month {month}, n_images={p.get('n_images')}). "
                "NEX-GDDP can be masked on ocean/coastal cells: if this is a coastal point, move its "
                "coordinates slightly inland in db/init/03_seed_stations.sql (and tell the team)."
            )
        rows.append({
            "model": model, "scenario": scenario, "window_start": window_start, "window_end": window_end,
            "target_year": target_year, "month": month, "station": p["station"],
            "n_images": int(p["n_images"]),
            **{v: float(p[v]) for v in VARIABLES},
        })
    return rows


def grid_cell_center(lat: float, lon: float) -> tuple[float, float]:
    """Centre of the 0.25-degree NEX-GDDP cell containing the point (grid centres at x.125, x.375, ...)."""
    return (math.floor(lat / 0.25) * 0.25 + 0.125, math.floor(lon / 0.25) * 0.25 + 0.125)


def read_done(path: pathlib.Path) -> set[tuple]:
    """(model, scenario, window_start, month) blocks already in the CSV, for resuming."""
    if not path.exists():
        return set()
    with path.open() as fh:
        return {(r["model"], r["scenario"], int(r["window_start"]), int(r["month"])) for r in csv.DictReader(fh)}


def append_rows(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


def load_points() -> list[tuple[str, float, float]]:
    with connect() as conn:
        return [(n, float(la), float(lo)) for n, la, lo in conn.execute(
            "SELECT name, latitude, longitude FROM stations WHERE kind = 'weather_point' ORDER BY station_id")]


# ---------------------------------------------------------------- Earth Engine calls (need network + account)
def init_ee(project: str):
    try:
        import ee
    except ImportError as exc:
        raise SystemExit("earthengine-api is not installed. Run: pip install -r requirements-fetch.txt (Python 3.10+)") from exc
    try:
        ee.Initialize(project=project)
    except Exception as exc:  # not authenticated, wrong project id, API not enabled
        raise SystemExit(
            f"Earth Engine is not ready ({exc}).\n"
            "  1. Register (free, non-commercial): https://code.earthengine.google.com/register\n"
            "  2. Run in Terminal:  earthengine authenticate\n"
            "  3. Re-run with --ee-project YOUR_PROJECT_ID (the Google Cloud project you registered)."
        ) from exc
    return ee


def points_collection(ee, points):
    return ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([lon, lat]), {"station": name}) for name, lat, lon in points
    ])


def _collection(ee, model: str, scenario: str):
    return (ee.ImageCollection(DATASET)
            .filter(ee.Filter.eq("model", model))
            .filter(ee.Filter.eq("scenario", scenario)))


def check_models(ee, models: list[str]) -> None:
    """Every requested model must exist for all three scenarios, or we stop before doing any work."""
    probes = {"historical": ("2000-07-01", "2000-07-02"), "ssp245": ("2030-07-01", "2030-07-02"),
              "ssp585": ("2030-07-01", "2030-07-02")}
    for scenario, (d0, d1) in probes.items():
        have = set(ee.ImageCollection(DATASET).filter(ee.Filter.eq("scenario", scenario))
                   .filterDate(d0, d1).aggregate_array("model").getInfo())
        missing = [m for m in models if m not in have]
        if missing:
            raise SystemExit(f"Model(s) {missing} not found for scenario {scenario} in {DATASET}.\n"
                             f"Available names: {sorted(have)}\nEdit MODELS in etl/fetch_nexgddp.py.")


def _reduce(ee, image, points_fc, proj, use_crs: bool):
    if use_crs:  # sample on the dataset's own 0.25 degree grid
        return image.reduceRegions(collection=points_fc, reducer=ee.Reducer.first(), crs=proj)
    return image.reduceRegions(collection=points_fc, reducer=ee.Reducer.first(), scale=27830)


def preflight(ee, points_fc, points, model: str) -> None:
    """Sample one day at the 3 points: stops early if a point is on a masked (ocean) cell."""
    img = _collection(ee, model, "historical").filterDate("2000-07-01", "2000-07-04").first().select(VARIABLES)
    feats = _reduce(ee, img, points_fc, img.projection(), True).getInfo()["features"]
    print(f"Preflight ({model}, one day in 2000):")
    for f in feats:
        p = f["properties"]
        name = p["station"]
        lat, lon = next((a, b) for n, a, b in points if n == name)
        print(f"  {name}: requested ({lat}, {lon}) -> expected grid cell centre {grid_cell_center(lat, lon)}; "
              f"values { {v: p.get(v) for v in VARIABLES} }")
        if any(p.get(v) is None for v in VARIABLES):
            raise SystemExit(f"Point {name} has no data (masked cell, probably ocean/coast). "
                             "Move it slightly inland in db/init/03_seed_stations.sql and tell the team.")


def fetch_block(ee, points_fc, model, scenario, y0, y1, target_year, month, retries=3) -> list[dict]:
    col = (_collection(ee, model, scenario)
           .filterDate(f"{y0}-01-01", f"{y1 + 1}-01-01")
           .filter(ee.Filter.calendarRange(month, month, "month"))
           .select(VARIABLES))
    n = col.size()
    proj = col.first().projection()
    mean_img = col.mean()
    last = None
    for attempt in range(retries):
        for use_crs in (True, False):  # fall back to a nominal scale if the crs form is rejected
            try:
                fc = _reduce(ee, mean_img, points_fc, proj, use_crs).map(lambda f: f.set("n_images", n))
                return rows_from_features(fc.getInfo()["features"], model, scenario, y0, y1, target_year, month)
            except ValueError:
                raise  # missing data is not transient
            except Exception as exc:  # network / quota / EEException
                last = exc
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Earth Engine failed for {model} {scenario} {y0}-{y1} month {month}: {last}")


def main(project: str, models: list[str] = MODELS, workers: int = 4, out_csv: pathlib.Path = OUT_CSV) -> dict:
    ee = init_ee(project)
    points = load_points()
    points_fc = points_collection(ee, points)
    check_models(ee, models)
    preflight(ee, points_fc, points, models[0])

    done = read_done(out_csv)
    tasks = [(m, s, y0, y1, ty, mo) for m in models for (s, y0, y1, ty) in WINDOWS for mo in range(1, 13)
             if (m, s, y0, mo) not in done]
    print(f"{len(tasks)} blocks to fetch ({len(done)} already in {out_csv.name}).")
    t0, finished = time.time(), 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_block, ee, points_fc, *t): t for t in tasks}
        for fut in as_completed(futures):
            append_rows(out_csv, fut.result())  # only this thread writes the file
            finished += 1
            if finished % 20 == 0 or finished == len(tasks):
                print(f"  {finished}/{len(tasks)} blocks, {time.time() - t0:.0f}s")
    return {"models": models, "blocks_fetched": finished, "csv": str(out_csv)}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ee-project", required=True, help="your Google Cloud project id registered for Earth Engine")
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    print(main(a.ee_project, workers=a.workers))
