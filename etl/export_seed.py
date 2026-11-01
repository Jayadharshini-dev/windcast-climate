"""Write the small seed CSVs that let anyone rebuild the database with no internet.

    python -m etl.export_seed

Reads the raw downloads (CEA csv + cached Open-Meteo JSON, both gitignored) and writes
  data/seed/cea_tn_daily.csv      ts, generation_mu           (NULL = missing/reporting gap)
  data/seed/era5_daily.csv        point, ts, wind_speed, temperature, humidity, pressure, n_hours
The seed holds exactly what load_cea / load_era5 would put in the database, so a rebuild from the
seed is identical to a rebuild from the raw files. Only needed when the raw data changes.
"""
import json
import pathlib

import pandas as pd

from etl import load_cea, load_era5
from etl.db import ROOT

SEED_DIR = ROOT / "data" / "seed"
POINTS = ["TN_Muppandal_Kayathar", "TN_Palghat_Udumalpet", "TN_Theni_Cumbum"]


def main(cea_csv: pathlib.Path = load_cea.CEA_FILE, era5_dir: pathlib.Path = load_era5.CACHE_DIR) -> None:
    SEED_DIR.mkdir(parents=True, exist_ok=True)
    cea, _ = load_cea.prepare(cea_csv)
    cea.to_csv(SEED_DIR / "cea_tn_daily.csv", index=False)

    frames = []
    for name in POINTS:
        path = era5_dir / f"{name}_{load_era5.DEFAULT_START}_{load_era5.DEFAULT_END}.json"
        daily = load_era5.to_daily(json.loads(path.read_text()))
        daily.insert(0, "point", name)
        frames.append(daily)
    era5 = pd.concat(frames, ignore_index=True)
    era5.to_csv(SEED_DIR / "era5_daily.csv", index=False, float_format="%.6f")
    print(f"cea_tn_daily.csv: {len(cea)} rows; era5_daily.csv: {len(era5)} rows")


if __name__ == "__main__":
    main()
