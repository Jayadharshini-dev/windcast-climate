"""Load CEA daily wind generation for Tamil Nadu into power_output (station 1, TN_STATE).

Rules (PROJECT_CONTEXT.md section 4):
  * duplicate (date, state) rows -> keep the first
  * missing days -> stored as rows with generation_mu = NULL, never filled
  * a day where wind, solar AND total renewable are all exactly 0 is a reporting failure,
    not a calm day, so it is also stored as NULL and listed in the report
Re-running is safe: rows are upserted on (station_id, ts).
"""
import argparse
import pathlib

import pandas as pd

from etl.db import ROOT, connect

CEA_FILE = ROOT / "data" / "raw" / "state-control-renewable-energy-generation.csv"
STATE = "Tamil Nadu"
STATE_STATION_ID = 1
SOURCE = "cea_india_data_portal"


def prepare(csv_path: pathlib.Path) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(csv_path)
    tn = raw[raw["state_name"] == STATE].copy()
    tn["date"] = pd.to_datetime(tn["date"]).dt.date

    n_dupes = int(tn.duplicated(["date", "state_name"]).sum())
    tn = tn.drop_duplicates(["date", "state_name"], keep="first")

    all_zero = (tn["wind_energy"] == 0) & (tn["solar_energy"] == 0) & (tn["total_renewable_energy"] == 0)
    zero_days = sorted(tn.loc[all_zero, "date"])
    tn.loc[all_zero, "wind_energy"] = None

    span = pd.date_range(min(tn["date"]), max(tn["date"]), freq="D").date
    out = pd.DataFrame({"ts": span}).merge(
        tn[["date", "wind_energy"]].rename(columns={"date": "ts", "wind_energy": "generation_mu"}),
        on="ts", how="left",
    )
    absent = sorted(set(span) - set(tn["date"]))

    stats = {
        "first_day": str(span[0]),
        "last_day": str(span[-1]),
        "days_in_span": len(span),
        "days_in_file": len(tn),
        "days_absent_from_file": len(absent),
        "absent_by_year": pd.Series([d.year for d in absent]).value_counts().sort_index().to_dict(),
        "all_zero_days_set_null": [str(d) for d in zero_days],
        "null_generation_total": int(out["generation_mu"].isna().sum()),
        "duplicate_rows_dropped_tn": n_dupes,
        "duplicate_rows_in_file_all_states": int(raw.duplicated(["date", "state_name"]).sum()),
        "mean_mu_per_day": round(float(out["generation_mu"].mean()), 2),
    }
    return out, stats


def load(df: pd.DataFrame) -> int:
    rows = [
        (STATE_STATION_ID, r.ts, None if pd.isna(r.generation_mu) else float(r.generation_mu), SOURCE)
        for r in df.itertuples(index=False)
    ]
    with connect() as conn, conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO power_output (station_id, ts, generation_mu, source)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (station_id, ts) DO UPDATE
               SET generation_mu = EXCLUDED.generation_mu,
                   source        = EXCLUDED.source,
                   ingested_at   = now()
            """,
            rows,
        )
    return len(rows)


def main(csv_path: pathlib.Path = CEA_FILE) -> dict:
    df, stats = prepare(csv_path)
    stats["rows_upserted"] = load(df)
    return stats


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=pathlib.Path, default=CEA_FILE)
    for k, v in main(p.parse_args().csv).items():
        print(f"{k}: {v}")
