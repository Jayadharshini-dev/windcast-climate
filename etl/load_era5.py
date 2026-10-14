"""Load daily ERA5 weather for the three wind-belt points into raw_weather_readings.

Source: Open-Meteo Historical API with models=era5 (ERA5 reanalysis, 0.25 degree grid, same
resolution as NEX-GDDP-CMIP6). Hourly values are averaged to daily means.

Choices (documented in etl/README.md):
  * wind_speed_10m in m/s (10 m to match NEX-GDDP sfcWind; Open-Meteo defaults to km/h)
  * days are India calendar days (timezone=Asia/Kolkata) so each daily mean covers the same
    24 hours as the CEA generation figure for that date
  * surface_pressure is stored but never used as a model input (rule 2)
Raw API responses are cached in data/raw/era5/ (gitignored) so re-runs do not re-download.
Re-running is safe: rows are upserted on (station_id, ts).
"""
import argparse
import json
import pathlib
import time

import pandas as pd
import requests

from etl.db import ROOT, connect

API = "https://archive-api.open-meteo.com/v1/archive"
CACHE_DIR = ROOT / "data" / "raw" / "era5"
HOURLY = ["wind_speed_10m", "temperature_2m", "relative_humidity_2m", "surface_pressure"]
TIMEZONE = "Asia/Kolkata"
SOURCE = "era5_open_meteo"
# Covers the CEA span (2020-05-28 to 2025-04-27) plus a few weeks either side for lag features.
DEFAULT_START, DEFAULT_END = "2020-05-01", "2025-04-30"


def fetch(name: str, lat: float, lon: float, start: str, end: str, cache_dir: pathlib.Path) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{name}_{start}_{end}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    params = {
        "latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
        "hourly": ",".join(HOURLY), "models": "era5",
        "wind_speed_unit": "ms", "temperature_unit": "celsius", "timezone": TIMEZONE,
    }
    for attempt in range(4):
        r = requests.get(API, params=params, timeout=120)
        if r.status_code == 200:
            data = r.json()
            cache.write_text(json.dumps(data))
            return data
        if attempt == 3:
            r.raise_for_status()
        time.sleep(5 * (attempt + 1))  # rate limit / transient error: back off and retry
    raise RuntimeError("unreachable")


def to_daily(data: dict) -> pd.DataFrame:
    h = pd.DataFrame(data["hourly"])
    units = data.get("hourly_units", {})
    if units.get("wind_speed_10m") not in (None, "m/s"):
        raise ValueError(f"wind unit is {units['wind_speed_10m']}, expected m/s")
    h["ts"] = pd.to_datetime(h["time"]).dt.date
    daily = h.groupby("ts").agg(
        wind_speed=("wind_speed_10m", "mean"),
        temperature=("temperature_2m", "mean"),
        humidity=("relative_humidity_2m", "mean"),
        pressure=("surface_pressure", "mean"),
        n_hours=("wind_speed_10m", "count"),
    ).reset_index()
    return daily


def load(station_id: int, daily: pd.DataFrame) -> int:
    def val(x):
        return None if pd.isna(x) else float(x)

    rows = [
        (station_id, r.ts, val(r.wind_speed), val(r.temperature), val(r.humidity), val(r.pressure), int(r.n_hours), SOURCE)
        for r in daily.itertuples(index=False)
    ]
    with connect() as conn, conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO raw_weather_readings
                   (station_id, ts, wind_speed, temperature, humidity, pressure, n_hours, source)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (station_id, ts) DO UPDATE
               SET wind_speed = EXCLUDED.wind_speed, temperature = EXCLUDED.temperature,
                   humidity = EXCLUDED.humidity, pressure = EXCLUDED.pressure,
                   n_hours = EXCLUDED.n_hours, source = EXCLUDED.source, ingested_at = now()
            """,
            rows,
        )
    return len(rows)


def main(start: str = DEFAULT_START, end: str = DEFAULT_END, cache_dir: pathlib.Path = CACHE_DIR) -> dict:
    with connect() as conn:
        points = conn.execute(
            "SELECT station_id, name, latitude, longitude FROM stations "
            "WHERE kind = 'weather_point' ORDER BY station_id"
        ).fetchall()
    stats = {}
    for station_id, name, lat, lon in points:
        data = fetch(name, float(lat), float(lon), start, end, cache_dir)
        daily = to_daily(data)
        n = load(station_id, daily)
        stats[name] = {
            "requested_lat_lon": (float(lat), float(lon)),
            "era5_grid_lat_lon": (data.get("latitude"), data.get("longitude")),
            "days_loaded": n,
            "days_with_fewer_than_24h": int((daily["n_hours"] < 24).sum()),
            "mean_wind_ms": round(float(daily["wind_speed"].mean()), 2),
            "mean_temp_c": round(float(daily["temperature"].mean()), 2),
            "mean_rh_pct": round(float(daily["humidity"].mean()), 1),
        }
    return stats


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    a = p.parse_args()
    for k, v in main(a.start, a.end).items():
        print(k, v)
