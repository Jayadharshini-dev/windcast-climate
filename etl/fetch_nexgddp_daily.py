"""Fetch DAILY NEX-GDDP-CMIP6 values at the 3 wind-belt points from Google Earth Engine (one CSV per model, scenario, year).

Why: the monthly-mean pipeline (etl/fetch_nexgddp.py) cannot show how day-to-day weather changes, and its windows
(1995-2014, 2021-2040, 2041-2060) do not overlap the ERA5 record (2020-2025), so the projected wind cannot be checked
against ERA5. Daily values fix both:
  * "validation": ssp245 for 2020-2025. SSP experiments start in 2015, so this period overlaps ERA5; the two
    cannot match day by day (a climate model is not a weather forecast) but their distributions can be compared.
  * "variability": historical 1995-2014 plus ssp585 2041-2060 for a few models that span the spread, so the change in
    day-to-day behaviour (ramps, spells) can be read from the model's own daily output instead of being assumed away.

Run from the repo root, after `earthengine authenticate` (needs a Google Cloud project registered for Earth Engine):
    python -m etl.fetch_nexgddp_daily --ee-project YOUR_PROJECT --plan validation     # 10 models x 6 years = 60 calls
    python -m etl.fetch_nexgddp_daily --ee-project YOUR_PROJECT --plan variability    # 4 models x 40 years = 160 calls
    python -m etl.fetch_nexgddp_daily --ee-project YOUR_PROJECT --plan variability_all   # all 10 models: 240 more calls
    python -m etl.fetch_nexgddp_daily --ee-project YOUR_PROJECT --models MIROC6 --spec ssp585:2041-2060   # custom

One call returns one model-scenario-year: about 365 days x 3 points. Files go to data/raw/nexgddp_daily/ (git-ignored:
they are large) as  <model>_<scenario>_<year>.csv  with columns  date,station,sfcWind,tas,hurs  (tas in kelvin).
Finished files are skipped on re-run, so the job can be stopped and resumed.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from etl.db import ROOT
from etl.fetch_nexgddp import MODELS, VARIABLES, _collection, _reduce, check_models, init_ee, load_points, points_collection, preflight

OUT_DIR = ROOT / "data" / "raw" / "nexgddp_daily"
FIELDS = ["date", "station", "sfcWind", "tas", "hurs"]
PLANS = {
    "validation": (MODELS, "ssp245:2020-2025"),
    "variability": (["CMCC-ESM2", "EC-Earth3", "GFDL-ESM4", "MIROC6"], "historical:1995-2014,ssp585:2041-2060"),
    # all ten models: files of the four above are skipped, so this fetches the other six (240 calls)
    "variability_all": (MODELS, "historical:1995-2014,ssp585:2041-2060"),
}


# ---------------------------------------------------------------- pure helpers (unit-tested, no Earth Engine)
def parse_spec(spec: str) -> list[tuple[str, int, int]]:
    """'historical:1995-2014,ssp585:2041-2060' -> [('historical', 1995, 2014), ('ssp585', 2041, 2060)]."""
    out = []
    for part in spec.split(","):
        scen, years = part.strip().split(":")
        y0, y1 = (int(v) for v in years.split("-"))
        if scen not in ("historical", "ssp245", "ssp585") or y1 < y0:
            raise ValueError(f"bad spec part: {part!r}")
        out.append((scen, y0, y1))
    return out


def out_path(model: str, scenario: str, year: int, out_dir: pathlib.Path = OUT_DIR) -> pathlib.Path:
    return out_dir / f"{model}_{scenario}_{year}.csv"


def rows_from_daily_features(features: list[dict]) -> list[dict]:
    """Earth Engine features -> CSV rows. Fails loudly on a missing value (masked cell) instead of writing a gap."""
    rows = []
    for f in features:
        p = f["properties"]
        miss = [k for k in ["date", "station", *VARIABLES] if p.get(k) is None]
        if miss:
            raise ValueError(f"missing {miss} in {p}")
        rows.append({"date": p["date"], "station": p["station"], **{v: float(p[v]) for v in VARIABLES}})
    return sorted(rows, key=lambda r: (r["date"], r["station"]))


def write_rows(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)  # a half-written file never looks finished


# ---------------------------------------------------------------- Earth Engine call (needs network + account)
def fetch_year(ee, points_fc, model: str, scenario: str, year: int, retries: int = 3) -> list[dict]:
    col = _collection(ee, model, scenario).filterDate(f"{year}-01-01", f"{year + 1}-01-01").select(VARIABLES)
    proj = col.first().projection()

    def per_image(img):
        date = img.date().format("YYYY-MM-dd")
        return _reduce(ee, img, points_fc, proj, True).map(lambda f: f.set("date", date))

    last = None
    for attempt in range(retries):
        try:
            return rows_from_daily_features(col.map(per_image).flatten().getInfo()["features"])
        except ValueError:
            raise
        except Exception as exc:  # network / quota / EEException
            last = exc
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Earth Engine failed for {model} {scenario} {year}: {last}")


def main(project: str, models: list[str], spec: str, workers: int = 4, out_dir: pathlib.Path = OUT_DIR) -> dict:
    ee = init_ee(project)
    points = load_points()
    points_fc = points_collection(ee, points)
    check_models(ee, models)
    preflight(ee, points_fc, points, models[0])
    tasks = [(m, s, y) for m in models for (s, y0, y1) in parse_spec(spec) for y in range(y0, y1 + 1)
             if not out_path(m, s, y, out_dir).exists()]
    print(f"{len(tasks)} model-years to fetch into {out_dir}")
    t0, done = time.time(), 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(fetch_year, ee, points_fc, *t): t for t in tasks}
        for fut in as_completed(futs):
            m, s, y = futs[fut]
            rows = fut.result()
            write_rows(out_path(m, s, y, out_dir), rows)  # only this thread writes
            done += 1
            if done % 10 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} ({time.time() - t0:.0f}s)")
    return {"fetched": done, "dir": str(out_dir)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ee-project", required=True)
    ap.add_argument("--plan", choices=sorted(PLANS))
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--spec")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if not a.plan and not (a.models and a.spec):
        ap.error("give --plan, or both --models and --spec")
    m, s = PLANS[a.plan] if a.plan else (None, None)
    print(main(a.ee_project, a.models or m, a.spec or s, a.workers))
