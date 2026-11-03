"""Archived weather FORECASTS for the 2024-2025 holdout, from the Open-Meteo Previous Runs API.

Why: the robustness test of models/robustness.py degrades observed weather with simulated random errors. Real forecast
errors are different (correlated from day to day, tied to weather systems). This script downloads what a numerical
weather prediction model actually forecast, d days ahead, for the days of the holdout, so the forecasting models can be
tested on real forecast weather (models/forecast_weather_test.py).

Source: https://previous-runs-api.open-meteo.com/v1/forecast, variables  <name>_previous_day<d>  = the value forecast
d days before the valid hour (d = 1, 2, 3). Archives start in 2024 for most models, which covers the holdout (2024-01-01
onward). Wind in m/s at 10 m, India calendar days (Asia/Kolkata), hourly values averaged to daily means, equal-weight
mean of the three wind-belt points: the same choices as the ERA5 loader (etl/load_era5.py), so forecast and ERA5 are
comparable. A day is kept only if all 24 hours exist at all three points.

Run from the repo root (needs the database for the point coordinates, and internet):
    python -m etl.fetch_forecast_weather                       # ECMWF IFS 0.25 degree, 2024-01-01 .. 2025-04-30
    python -m etl.fetch_forecast_weather --model gfs_global    # another model, if you want a second opinion

Raw responses are cached in data/raw/forecast_weather/ (git-ignored); the daily table is written to
data/raw/forecast_weather/forecast_<model>.csv with columns  ts, wind_d1, temp_d1, hum_d1, wind_d2, ... hum_d3.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import pandas as pd
import requests

from etl.db import ROOT
from etl.fetch_nexgddp import load_points

API = "https://previous-runs-api.open-meteo.com/v1/forecast"
CACHE_DIR = ROOT / "data" / "raw" / "forecast_weather"
BASE = {"wind": "wind_speed_10m", "temp": "temperature_2m", "hum": "relative_humidity_2m"}
LEADS = (1, 2, 3)
TIMEZONE = "Asia/Kolkata"
DEFAULT_MODEL, DEFAULT_START, DEFAULT_END = "ecmwf_ifs025", "2024-01-01", "2025-04-30"
CHUNK_DAYS = 15


def hourly_names() -> list[str]:
    return [f"{v}_previous_day{d}" for v in BASE.values() for d in LEADS]


def date_chunks(start: str, end: str, days: int = CHUNK_DAYS) -> list[tuple[str, str]]:
    s, e, out = pd.Timestamp(start), pd.Timestamp(end), []
    while s <= e:
        t = min(s + pd.Timedelta(days=days - 1), e)
        out.append((s.strftime("%Y-%m-%d"), t.strftime("%Y-%m-%d")))
        s = t + pd.Timedelta(days=1)
    return out


def _get(params: dict, name: str, start: str) -> dict:
    """One request with patient retries: the free API is sometimes slow (timeout), rate-limited (429) or overloaded (503)."""
    for attempt in range(12):
        try:
            r = requests.get(API, params=params, timeout=(15, 180))
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            if attempt == 11:
                raise RuntimeError(f"Open-Meteo did not answer after 12 tries: {exc}") from exc
            print(f"[forecast] {name} {start}: no answer, retrying ({attempt + 1}/11)")
            time.sleep(15 * (attempt + 1))
            continue
        if r.status_code == 200:
            try:
                data = r.json()
            except ValueError:  # empty or cut-off answer from a busy server: treat as a failed try
                if attempt == 11:
                    raise RuntimeError("Open-Meteo kept returning an empty answer")
                print(f"[forecast] {name} {start}: empty answer, retrying ({attempt + 1}/11)")
                time.sleep(30)
                continue
            time.sleep(4)  # be gentle with the free API
            return data
        if r.status_code in (429, 500, 502, 503, 504) and attempt < 11:
            print(f"[forecast] {name} {start}: server busy ({r.status_code}), waiting 90 s ({attempt + 1}/11)")
            time.sleep(90)
            continue
        raise RuntimeError(f"Open-Meteo answered {r.status_code}: {r.text[:300]}")  # 400 = a real mistake (bad model, dates)
    raise RuntimeError("unreachable")


def fetch(name: str, lat: float, lon: float, start: str, end: str, model: str, cache_dir: pathlib.Path = CACHE_DIR) -> dict:
    """One light request per variable (three lead times each), merged, so no single request is heavy."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{name}_{model}_{start}_{end}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    merged: dict | None = None
    for v in BASE.values():
        params = {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
                  "hourly": ",".join(f"{v}_previous_day{d}" for d in LEADS),
                  "models": model, "wind_speed_unit": "ms", "temperature_unit": "celsius", "timezone": TIMEZONE}
        part = _get(params, name, start)
        if merged is None:
            merged = part
        else:
            merged["hourly"].update({k: x for k, x in part["hourly"].items() if k != "time"})
            merged.setdefault("hourly_units", {}).update(part.get("hourly_units", {}))
    cache.write_text(json.dumps(merged))
    return merged


def to_daily(data: dict, lead: int) -> pd.DataFrame:
    """One point, one lead: daily means of the three variables; NaN unless all 24 hours are present."""
    h = pd.DataFrame(data["hourly"])
    units = data.get("hourly_units", {})
    wu = units.get(f"wind_speed_10m_previous_day{lead}")
    if wu not in (None, "m/s"):
        raise ValueError(f"wind unit is {wu}, expected m/s")
    h["ts"] = pd.to_datetime(h["time"]).dt.normalize()
    cols = {k: f"{v}_previous_day{lead}" for k, v in BASE.items()}
    g = h.groupby("ts")
    out = pd.DataFrame({k: g[c].mean() for k, c in cols.items()})
    full = g[cols["wind"]].count().eq(24) & g[cols["temp"]].count().eq(24) & g[cols["hum"]].count().eq(24)
    return out.where(full)


def state_mean(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Equal-weight mean over points; a day is NaN if any point is missing."""
    cat = pd.concat(frames, keys=range(len(frames)))
    return cat.groupby(level=1).agg(lambda s: s.mean() if s.notna().sum() == len(frames) else float("nan"))


def build_table(per_point: list[dict]) -> pd.DataFrame:
    """per_point: one dict per point of {lead: daily frame}. Returns the state-mean table with one column per variable and lead."""
    cols = {}
    for lead in LEADS:
        sm = state_mean([p[lead] for p in per_point])
        for k in BASE:
            cols[f"{k}_d{lead}"] = sm[k]
    t = pd.DataFrame(cols).sort_index()
    t.index.name = "ts"
    return t


def main(model: str = DEFAULT_MODEL, start: str = DEFAULT_START, end: str = DEFAULT_END) -> pathlib.Path:
    points = load_points()
    per_point = []
    for name, lat, lon in points:
        parts = {lead: [] for lead in LEADS}
        for s, e in date_chunks(start, end):
            data = fetch(name, lat, lon, s, e, model)
            for lead in LEADS:
                parts[lead].append(to_daily(data, lead))
            time.sleep(1)
        per_point.append({lead: pd.concat(parts[lead]).sort_index() for lead in LEADS})
        print(f"[forecast] {name}: {len(per_point[-1][1])} days")
    t = build_table(per_point)
    path = CACHE_DIR / f"forecast_{model}.csv"
    t.to_csv(path)
    print(f"{path}: {len(t)} days, complete days: " + ", ".join(f"d{l}={int(t[f'wind_d{l}'].notna().sum())}" for l in LEADS))
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END)
    a = ap.parse_args()
    main(a.model, a.start, a.end)
