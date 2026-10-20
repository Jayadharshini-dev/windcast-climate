"""One command from an empty database to a working project.

    docker compose up -d
    python -m etl.run_all               # about 10 seconds: rebuild the database, load the committed results
    python -m etl.run_all --retrain     # about 5 minutes: also retrain the models and redo the climate analysis

Then:  streamlit run dashboard/app.py

Default (fast) path
  etl.rebuild: migrations, seed data, climate change factors, the committed model and climate results,
  and a check against the golden file.

--retrain adds the real modelling pipeline, reading the data from the database:
  1. python -m models.trees              -> models/trees_*.csv   (walk-forward validation, holdout predictions)
  2. python -m climate.climate_impact    -> climate/climate_impact*.csv  (delta-change on the 10 climate models)
  3. etl.load_results                    -> forecast_runs, model_predictions, climate_impact in the database
The models are seeded. Retraining from the database reproduces the committed files to within rounding of the golden CSV
(forecast RMSE within 0.02 MU/day, climate averages within 0.1 percentage point), but not bit for bit, so `git status`
will show small changes in models/ and climate/. Discard them with `git checkout models climate` if you only wanted to check.
"""
import argparse
import os
import subprocess
import sys
import time

from etl import load_results, rebuild
from etl.db import ROOT, connect


def step(title: str, cmd: list[str]) -> None:
    print(f"\n>>> {title}\n    $ {' '.join(cmd)}", flush=True)
    t = time.time()
    env = {**os.environ, "WINDCAST_SOURCE": "db"}  # force models to read the database, never the CSV fallback
    r = subprocess.run(cmd, cwd=ROOT, env=env)
    if r.returncode != 0:
        raise SystemExit(f"step failed ({title}), exit code {r.returncode}")
    print(f"    done in {time.time() - t:.0f}s", flush=True)


def summary() -> None:
    with connect() as conn:
        rows = {t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
                for t in ("power_output", "raw_weather_readings", "climate_scenarios",
                          "forecast_runs", "model_predictions", "climate_impact")}
    print("\nRows in the database:", rows)
    print("Next:  streamlit run dashboard/app.py")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--retrain", action="store_true", help="retrain the models and redo the climate analysis (slow)")
    a = ap.parse_args()
    rebuild.main()
    if a.retrain:
        step("Train and validate the forecast models", [sys.executable, "-m", "models.trees"])
        step("Apply the climate change factors", [sys.executable, "-m", "climate.climate_impact"])
        print("\n>>> Loading the new results into the database")
        print("   ", load_results.main())
    summary()


if __name__ == "__main__":
    main()
