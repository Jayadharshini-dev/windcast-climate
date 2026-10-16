"""Load the modelling team's committed results into the database so the dashboard can show them.

    python -m etl.load_results

  models/trees_holdout_predictions.csv  -> forecast_runs + model_predictions  (one run per model and horizon)
  climate/climate_impact.csv            -> climate_impact                      (same rows as `climate_impact --write-db`)

Notes
  * forecast_runs.horizon_days is limited to 1-3 by the frozen schema, so the same-day weather-only models
    (horizon 0) are not stored; their numbers stay in models/trees_results_all_folds.csv.
  * Predictions are the 2024-01-01 onward holdout, trained on earlier data only. lower/upper are left empty
    (the models give point forecasts); the dashboard derives an error band from the holdout errors.
  * Safe to re-run: runs from this file are replaced, climate_impact rows are upserted.
"""
import json

import pandas as pd

from etl.db import ROOT, connect

PRED_CSV = ROOT / "models" / "trees_holdout_predictions.csv"
IMPACT_CSV = ROOT / "climate" / "climate_impact.csv"
SOURCE = "models/trees_holdout_predictions.csv"
STATION_ID = 1  # TN_STATE


def load_predictions(path=PRED_CSV) -> dict:
    df = pd.read_csv(path, parse_dates=["ts"])
    df = df[df["horizon_days"].between(1, 3)]
    n_runs = n_rows = 0
    with connect() as conn:
        conn.execute("DELETE FROM forecast_runs WHERE params->>'source' = %s", (SOURCE,))
        for (model, h), g in df.groupby(["model", "horizon_days"]):
            params = {"source": SOURCE, "split": "holdout", "train_end": "2023-12-31",
                      "test_start": str(g["ts"].min().date()), "test_end": str(g["ts"].max().date())}
            run_id = conn.execute(
                "INSERT INTO forecast_runs (model_version, horizon_days, params) VALUES (%s, %s, %s) RETURNING run_id",
                (model, int(h), json.dumps(params))).fetchone()[0]
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO model_predictions (run_id, station_id, forecast_date, predicted_value) VALUES (%s,%s,%s,%s)",
                    [(run_id, STATION_ID, r.ts.date(), float(r.predicted_mu)) for r in g.itertuples()])
            n_runs += 1
            n_rows += len(g)
        conn.commit()
    return {"forecast_runs": n_runs, "model_predictions": n_rows}


def load_impact(path=IMPACT_CSV) -> dict:
    df = pd.read_csv(path)
    sql = ("INSERT INTO climate_impact (region, scenario, target_year, climate_model, metric, value, p10, p90) "
           "VALUES (%s,%s,%s,%s,%s,%s,%s,%s) "
           "ON CONFLICT (region, scenario, target_year, climate_model, metric) "
           "DO UPDATE SET value = EXCLUDED.value, p10 = EXCLUDED.p10, p90 = EXCLUDED.p90")
    recs = [(r.region, r.scenario, int(r.target_year), r.climate_model, r.metric, float(r.value),
             None if pd.isna(r.p10) else float(r.p10), None if pd.isna(r.p90) else float(r.p90))
            for r in df.itertuples()]
    with connect() as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, recs)
        conn.commit()
    return {"climate_impact": len(recs)}


def main() -> dict:
    return {**load_predictions(), **load_impact()}


if __name__ == "__main__":
    print(main())
