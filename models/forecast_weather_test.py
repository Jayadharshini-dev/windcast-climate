"""Skill with REAL weather forecasts instead of simulated errors.

Run from the repo root, after  python -m etl.fetch_forecast_weather :
    python -m models.forecast_weather_test [--model ecmwf_ifs025]

The models are trained exactly as before (observed ERA5 weather, data to 2023-12-31, hyper-parameters of
models/trees_chosen_params.csv) and tested on the 2024+ holdout, but the target-day weather they are given is the
weather FORECAST issued d days earlier (d = max(h, 1)): the forecast a real operator would have had.

Reported, per horizon and model
  * RMSE with observed weather (the optimistic figure of the main tables), with forecast weather, and the reference
    (persistence for h >= 1, climatology for h = 0), all on the same days;
  * the Diebold-Mariano test of the forecast-weather model against the reference;
  * the same for the noise-aware model of models/robustness.py;
  * a simple recalibration of the forecast to ERA5 (wind multiplied by one factor, temperature and humidity shifted by
    one constant, all fitted on 2024-01-01 .. 2024-06-30 only) and the scores on the days after the calibration window.
And the weather forecast's own error against ERA5 (bias, RMSE, relative wind error), so it can be placed on the
noise scale of the robustness test (the one-day models stop beating persistence at about 16-18% wind error there).

Outputs: models/forecast_weather_scores.csv, models/forecast_weather_errors.csv
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from etl.fetch_forecast_weather import CACHE_DIR
from models.data import load_daily
from models.robustness import NOISE_LEVELS, TRAIN_END, _chosen, _fit, dm_test, noisy_weather
from models.trees import SEED, TARGET, build

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent
CALIB_END = "2024-06-30"
WX = {"wind": "wind_speed", "temp": "temperature", "hum": "humidity"}


def lead_of(h: int) -> int:
    return max(h, 1)


def load_forecast(model: str, directory=None) -> pd.DataFrame:
    p = Path(directory or CACHE_DIR) / f"forecast_{model}.csv"
    if not p.exists():
        raise SystemExit(f"{p} not found. Run: python -m etl.fetch_forecast_weather --model {model}")
    return pd.read_csv(p, parse_dates=["ts"]).set_index("ts").asfreq("D")


def with_forecast(df: pd.DataFrame, fc: pd.DataFrame, lead: int, cal: dict | None = None) -> pd.DataFrame:
    """Copy of df whose weather columns are the forecast for the same day issued `lead` days earlier (optionally recalibrated)."""
    out = df.copy()
    for k, col in WX.items():
        v = fc[f"{k}_d{lead}"].reindex(out.index)
        if cal:
            v = v * cal["wind"] if k == "wind" else v + cal[k]
        out[col] = v
    out["humidity"] = out["humidity"].clip(0, 100)
    out["wind_speed"] = out["wind_speed"].clip(lower=0)
    return out


def calibration(df: pd.DataFrame, fc: pd.DataFrame, lead: int, end: str = CALIB_END) -> dict:
    """One wind factor and two additive constants that map the forecast onto ERA5 over days up to `end`."""
    both = pd.DataFrame({f"{k}_o": df[WX[k]] for k in WX}).join(pd.DataFrame({f"{k}_f": fc[f"{k}_d{lead}"] for k in WX})).dropna()
    both = both[both.index <= end]
    return {"wind": float(both["wind_o"].mean() / both["wind_f"].mean()), "temp": float((both["temp_o"] - both["temp_f"]).mean()),
            "hum": float((both["hum_o"] - both["hum_f"]).mean()), "n": int(len(both))}


def weather_error(df: pd.DataFrame, fc: pd.DataFrame, lead: int, start: str = "2024-01-01") -> dict:
    """Forecast against ERA5 on days from `start`."""
    r = {"lead_days": lead}
    for k, col in WX.items():
        d = pd.DataFrame({"o": df[col], "f": fc[f"{k}_d{lead}"]}).dropna()
        d = d[d.index >= start]
        e = d["f"] - d["o"]
        r[f"{k}_n"] = int(len(d))
        r[f"{k}_bias"] = float(e.mean())
        r[f"{k}_rmse"] = float(np.sqrt((e ** 2).mean()))
        r[f"{k}_corr"] = float(d["o"].corr(d["f"]))
        if k == "wind":
            r["wind_mean_obs"] = float(d["o"].mean())
            r["wind_rel_rmse_pct"] = float(100 * r["wind_rmse"] / d["o"].mean())
            r["wind_ratio"] = float(d["f"].mean() / d["o"].mean())
    return r


def _fit_noise_aware(df, h, kind):
    """Same recipe as models.robustness.run_noise_aware: observed weather plus three noisy copies (0.1, 0.2, 0.3)."""
    rf_p, xg_p = _chosen(h)
    rng = np.random.default_rng(SEED)
    parts = [df] + [noisy_weather(df, s, rng) for s in (0.1, 0.2, 0.3)]
    Xs, ys = [], []
    for part in parts:
        d = build(part, h).join(df[TARGET].rename("y")).dropna()
        d = d[d.index <= TRAIN_END]
        Xs.append(d.drop(columns="y"))
        ys.append(d["y"])
    Xtr, ytr = pd.concat(Xs), pd.concat(ys)
    m = (RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=1, **rf_p) if kind == "rf" else
         XGBRegressor(n_estimators=300, subsample=0.8, random_state=SEED, n_jobs=1, tree_method="hist", **xg_p))
    m.fit(Xtr, ytr)
    return m, list(Xtr.columns)


def reference_series(h: int) -> pd.Series:
    p = pd.read_csv(OUT / "trees_holdout_predictions.csv", parse_dates=["ts"])
    q = p[(p.horizon_days == h) & (p.model == ("persistence" if h >= 1 else "climatology"))]
    return q.set_index("ts")["predicted_mu"].sort_index()


def _rmse(a, b):
    return float(np.sqrt(((np.asarray(a) - np.asarray(b)) ** 2).mean()))


def score(y, p_obs, p_fc, ref, h, extra=None):
    """One row of scores on one set of days. DM statistic: negative = forecast-weather model better than the reference."""
    dm, pv = dm_test(y - p_fc, y - ref, h=max(h, 1))
    row = dict(n=len(y), rmse_observed_weather=_rmse(y, p_obs), rmse_forecast_weather=_rmse(y, p_fc), rmse_reference=_rmse(y, ref),
               dm_vs_reference=dm, p_vs_reference=pv)
    row.update(extra or {})
    return row


def run(model="ecmwf_ifs025", directory=None):
    df = load_daily()
    fc = load_forecast(model, directory)
    rows, errs = [], []
    for h in (0, 1, 2, 3):
        lead = lead_of(h)
        errs.append(dict(model=model, horizon_days=h, **weather_error(df, fc, lead)))
        cal = calibration(df, fc, lead)
        fdf, cdf = with_forecast(df, fc, lead), with_forecast(df, fc, lead, cal)
        ref = reference_series(h)
        for kind in ("rf", "xgb"):
            name = "random_forest" if kind == "rf" else "xgboost"
            m0, f0, test0 = _fit(df, h, kind)
            variants = {"plain": (m0, f0), "noise_aware": _fit_noise_aware(df, h, kind)}
            test_idx = test0.index
            for vname, (m, feats) in variants.items():
                Xo = build(df, h).loc[test_idx, feats]
                Xf = build(fdf, h).loc[test_idx, feats]
                Xc = build(cdf, h).loc[test_idx, feats]
                ok = Xo.notna().all(axis=1) & Xf.notna().all(axis=1) & Xc.notna().all(axis=1) & ref.reindex(test_idx).notna()
                idx = test_idx[ok.values]
                y = df[TARGET].reindex(idx).values
                po, pf, pc = (np.clip(m.predict(X.loc[idx]), 0, None) for X in (Xo, Xf, Xc))
                r_ = ref.reindex(idx).values
                base = dict(horizon_days=h, model=name, training=vname, lead_days=lead)
                rows.append({**base, "window": "all holdout days with a forecast", "variant": "raw forecast", **score(y, po, pf, r_, h)})
                after = idx > pd.Timestamp(CALIB_END)
                if after.sum() > 30:
                    rows.append({**base, "window": f"after {CALIB_END}", "variant": "raw forecast", **score(y[after], po[after], pf[after], r_[after], h)})
                    rows.append({**base, "window": f"after {CALIB_END}", "variant": "recalibrated forecast", **score(y[after], po[after], pc[after], r_[after], h)})
        print(f"[forecast-weather] h={h} done")
    res, er = pd.DataFrame(rows), pd.DataFrame(errs)
    res.to_csv(OUT / "forecast_weather_scores.csv", index=False)
    er.to_csv(OUT / "forecast_weather_errors.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n=== Weather forecast against ERA5 (holdout days) ===")
    print(er[["horizon_days", "lead_days", "wind_n", "wind_ratio", "wind_bias", "wind_rmse", "wind_rel_rmse_pct", "wind_corr", "temp_bias", "temp_rmse", "hum_bias", "hum_rmse"]].round(2).to_string(index=False))
    print("\n=== Holdout RMSE with real forecast weather (MU/day); DM < 0 means better than the reference ===")
    cols = ["horizon_days", "model", "training", "window", "variant", "n", "rmse_observed_weather", "rmse_forecast_weather", "rmse_reference", "dm_vs_reference", "p_vs_reference"]
    print(res[cols].round(3).to_string(index=False))
    return res, er


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ecmwf_ifs025")
    a = ap.parse_args()
    run(a.model)
