"""Capacity-growth check: does the climate-shift result survive when generation is expressed per unit of capacity?

Run from the repo root:   python -m climate.capacity_check

The target in the paper is generation (MU/day), not capacity factor, because yearly capacity was not available
when the schema was frozen. Tamil Nadu's operational wind capacity grew by about 8% between March 2020 and March 2023
(data/golden/tn_wind_capacity_mw.csv: reported values to March 2023, an ASSUMED constant addition afterwards - replace
those two rows with reported figures). Fleet growth could be absorbed by the weather-only model as if it were weather.

What this script does
  1. Linearly interpolates capacity by date and rescales generation to a fixed fleet:  g_adj(t) = g(t) * C_ref / C(t),
     with C_ref = capacity on 2023-03-31 (so the numbers stay in MU/day of today's fleet).
  2. Reports the holdout RMSE of the weather-only XGBoost trained on the adjusted series (train to 2023-12-31, test
     2024-01-01 onward), against the unadjusted series.
  3. Re-runs the climate-shift analysis (same 10 models, 2 scenarios, 2 windows, monthly factors) with a weather-only
     XGBoost trained on the adjusted series, and compares the ensemble change with the primary result.

Output: climate/capacity_check.csv  (primary vs capacity-adjusted ensemble change, per case) and a printed summary.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from climate.climate_impact import FEATS, load_factors, shifted_weather, to_features
from models.data import REPO_ROOT, load_daily
from models.trees import SEED, TARGET, build, fit_weather_only

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent
CAP_CSV = REPO_ROOT / "data" / "golden" / "tn_wind_capacity_mw.csv"
REF_DATE = "2023-03-31"


def capacity_series(index, cap=None):
    """Daily capacity (MW) by linear interpolation between the dated values; flat outside the first/last date."""
    cap = pd.read_csv(CAP_CSV, parse_dates=["date"]) if cap is None else cap
    s = cap.set_index("date")["capacity_mw"].astype(float)
    t0 = s.index[0]
    x = (s.index - t0).days.values
    xi = (index - t0).days.values
    return pd.Series(np.interp(xi, x, s.values), index=index)


def adjust_generation(df, cap=None):
    """Generation expressed as if the fleet had its 2023-03-31 size on every day."""
    c = capacity_series(df.index, cap)
    ref = capacity_series(pd.DatetimeIndex([REF_DATE]), cap).iloc[0]
    out = df.copy()
    out[TARGET] = df[TARGET] * ref / c
    return out


def holdout_rmse(df):
    X = build(df, 0)
    d = X.join(df[TARGET].rename("y")).dropna()
    tr, te = d[d.index <= "2023-12-31"], d[d.index >= "2024-01-01"]
    m, cols = fit_weather_only(df[df.index <= "2023-12-31"], "xgb")
    p = np.clip(m.predict(te[cols]), 0, None)
    return float(np.sqrt(((te["y"] - p) ** 2).mean()))


def ensemble_changes(df):
    weather = df[FEATS].dropna()
    train = df.dropna(subset=FEATS + [TARGET])
    m, cols = fit_weather_only(train, "xgb")
    pred = lambda X: np.clip(m.predict(X[cols]), 0, None)
    base = pred(to_features(weather)).mean()
    fac = load_factors()
    rows = []
    for (scen, year), g in fac.groupby(["scenario", "target_year"]):
        ch = [100 * (pred(to_features(shifted_weather(weather, f))).mean() / base - 1) for _, f in g.groupby("climate_model")]
        rows.append(dict(scenario=scen, target_year=int(year), mean=float(np.mean(ch)), p10=float(np.percentile(ch, 10)),
                         p90=float(np.percentile(ch, 90)), models_up_pct=float(100 * np.mean(np.array(ch) > 0))))
    return pd.DataFrame(rows)


def main():
    np.random.seed(SEED)
    df = load_daily()
    adj = adjust_generation(df)
    c = capacity_series(df.index)
    print(f"[capacity] {c.iloc[0]:.0f} MW at {df.index[0].date()} -> {c.iloc[-1]:.0f} MW at {df.index[-1].date()} "
          f"(+{100 * (c.iloc[-1] / c.iloc[0] - 1):.1f}%)")
    r0, r1 = holdout_rmse(df), holdout_rmse(adj)
    print(f"[holdout] weather-only XGBoost RMSE, unadjusted {r0:.2f}, capacity-adjusted {r1:.2f} (MU/day of a fixed fleet)")
    a, b = ensemble_changes(df), ensemble_changes(adj)
    res = a.merge(b, on=["scenario", "target_year"], suffixes=("_primary", "_capacity_adjusted"))
    res["difference_points"] = res["mean_capacity_adjusted"] - res["mean_primary"]
    res.to_csv(OUT / "capacity_check.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.round(2).to_string(index=False))
    return res


if __name__ == "__main__":
    main()
