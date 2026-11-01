"""Robustness checks for the forecasting and ramp results (answers the three main reviewer objections).

Run from the repo root:   python -m models.robustness [--only noise|ramps|baselines|bias]

  1. noise     "Perfect weather forecast" objection. The models are trained on observed weather and tested on the
               holdout (2024+) with the TEST weather degraded by random errors of growing size, a stand-in for the
               error of a numerical weather prediction. Reported: holdout RMSE against the error level, and the
               error level at which the model stops beating its reference (persistence h>=1, climatology h=0).
               Noise scale s: wind speed x (1 + N(0, s)), temperature + N(0, 5 s) degC, humidity + N(0, 25 s) %.
               (s = 0.3 means about 30% wind error, 1.5 degC, 7.5 % humidity.) Errors are independent from day to
               day, which is a simplification; 20 random draws per level, seed 42. Lagged generation stays observed
               because it is known when a real forecast is issued.
  2. ramps     "Small samples / one threshold" objection. Ramp scores recomputed for ramp thresholds at the 80th,
               90th and 95th percentile of training-year changes, each with a 95% moving-block bootstrap interval
               (7-day blocks, 2000 resamples) on hit rate and CSI.
  3. baselines "Weak baselines" objection. Adds a monotone power-curve model (isotonic regression of generation on
               wind speed), an additive spline model (a GAM-like model) and, for h=1, SARIMAX with weather inputs
               (needs statsmodels; skipped if it is not installed). Compared with XGBoost by the Diebold-Mariano test.
  3b. noise_aware  Remedy for 1: models trained on observed plus noisy copies of the weather (see run_noise_aware).
  4. bias      "Capacity growth" objection. Mean forecast error by half-year on the holdout: a steady positive
               drift would show a fleet growing faster than the training period.

Everything trains on data up to 2023-12-31 only and tests on 2024-01-01 onward, with the hyper-parameters already
chosen by models.trees (models/trees_chosen_params.csv). Outputs go to models/robustness_*.csv and robustness_noise.png.
"""
import argparse
import ast
import math
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer
from xgboost import XGBRegressor

from models.data import load_daily
from models.ramps import Q_RAMP, k_day_change, ramp_threshold
from models.trees import SEED, TARGET, build, with_wind_powers

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent
TRAIN_END, TEST_START = "2023-12-31", "2024-01-01"
NOISE_LEVELS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
N_DRAWS = 20


# ----------------------------------------------------------------------------------------------- helpers
def noisy_weather(df, s, rng):
    """Degrade the weather columns by random errors of scale s (see module docstring). Returns a copy."""
    out = df.copy()
    n = len(out)
    out["wind_speed"] = (df["wind_speed"] * (1 + rng.normal(0, s, n))).clip(lower=0)
    out["temperature"] = df["temperature"] + rng.normal(0, 5 * s, n)
    out["humidity"] = (df["humidity"] + rng.normal(0, 25 * s, n)).clip(0, 100)
    return out


def dm_test(e1, e2, h=1):
    """Diebold-Mariano test, squared-error loss, Harvey-Leybourne-Newbold correction, Newey-West variance.
    Negative statistic favours model 1 (smaller errors). Returns (statistic, two-sided p)."""
    d = np.asarray(e1, float) ** 2 - np.asarray(e2, float) ** 2
    n = len(d)
    lags = int(math.floor(4 * (n / 100) ** (2 / 9)))
    dm = d - d.mean()
    var = (dm @ dm) / n
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        var += 2 * w * (dm[k:] @ dm[:-k]) / n
    stat = d.mean() / math.sqrt(var / n)
    hln = math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat *= hln
    return float(stat), float(2 * stats.t.sf(abs(stat), n - 1))


def block_bootstrap_idx(n, rng, block=7):
    """Indices for one moving-block bootstrap resample of n consecutive days."""
    n_blocks = int(math.ceil(n / block))
    starts = rng.integers(0, n - block + 1, n_blocks)
    return (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]


def _chosen(h):
    p = pd.read_csv(OUT / "trees_chosen_params.csv")
    r = p[(p.fold == "holdout_2024+") & (p.horizon_days == h)].iloc[0]
    return ast.literal_eval(r["rf"]), ast.literal_eval(r["xgb"])


def _fit(df, h, kind):
    rf_p, xg_p = _chosen(h)
    X = build(df, h)
    d = X.join(df[TARGET].rename("y")).dropna()
    tr = d[d.index <= TRAIN_END]
    feats = [c for c in d.columns if c != "y"]
    if kind == "rf":
        m = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=1, **rf_p)
    else:
        m = XGBRegressor(n_estimators=300, subsample=0.8, random_state=SEED, n_jobs=1, tree_method="hist", **xg_p)
    m.fit(tr[feats], tr["y"])
    return m, feats, d[d.index >= TEST_START]


def _reference_rmse(h):
    p = pd.read_csv(OUT / "trees_holdout_predictions.csv")
    name = "persistence" if h >= 1 else "climatology"
    q = p[(p.horizon_days == h) & (p.model == name)]
    return float(np.sqrt(((q.actual_mu - q.predicted_mu) ** 2).mean()))


# ----------------------------------------------------------------------------------------------- 1 noise
def run_noise(df):
    rows = []
    for h in (0, 1, 2, 3):
        ref = _reference_rmse(h)
        for kind in ("rf", "xgb"):
            m, feats, test = _fit(df, h, kind)
            for s in NOISE_LEVELS:
                rng = np.random.default_rng(SEED)
                rm = []
                for _ in range(1 if s == 0 else N_DRAWS):
                    Xp = build(noisy_weather(df, s, rng), h).loc[test.index, feats]
                    p = np.clip(m.predict(Xp), 0, None)
                    rm.append(float(np.sqrt(((test["y"] - p) ** 2).mean())))
                rows.append(dict(horizon_days=h, model="random_forest" if kind == "rf" else "xgboost", noise_scale=s,
                                 wind_error_pct=100 * s, rmse_mean=np.mean(rm), rmse_sd=np.std(rm),
                                 reference_rmse=ref, beats_reference=np.mean(rm) < ref))
            print(f"[noise] h={h} {kind} done")
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "robustness_noise.csv", index=False)

    # crossover: noise level where the model's RMSE reaches the reference (linear interpolation)
    cr = []
    for (h, m), g in res.groupby(["horizon_days", "model"]):
        g = g.sort_values("noise_scale")
        x, y, ref = g.noise_scale.values, g.rmse_mean.values, g.reference_rmse.iloc[0]
        c = np.nan if y[-1] < ref else float(np.interp(ref, y, x)) if (y > ref).any() and y[0] < ref else np.nan
        cr.append(dict(horizon_days=h, model=m, rmse_perfect_weather=y[0], reference_rmse=ref,
                       crossover_wind_error_pct=100 * c if not np.isnan(c) else np.nan,
                       rmse_at_30pct=float(g[g.noise_scale == 0.3].rmse_mean.iloc[0])))
    cr = pd.DataFrame(cr)
    cr.to_csv(OUT / "robustness_noise_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n=== Noise test: holdout RMSE with degraded weather (MU/day) ===")
    print(res.pivot_table(index=["horizon_days", "model"], columns="wind_error_pct", values="rmse_mean").round(2).to_string())
    print("\n=== Error level at which the model stops beating its reference ===")
    print(cr.round(2).to_string(index=False))

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=False, facecolor="#fcfcfb")
    for ax, h in zip(axes, (0, 1, 2, 3)):
        ax.set_facecolor("#fcfcfb")
        for m, c in (("random_forest", "#1baf7a"), ("xgboost", "#2a78d6")):
            g = res[(res.horizon_days == h) & (res.model == m)]
            ax.errorbar(g.wind_error_pct, g.rmse_mean, g.rmse_sd, color=c, marker="o", ms=3, label=m, lw=1.4)
        ax.axhline(_reference_rmse(h), color="#eb6834", ls="--", lw=1.2, label="persistence" if h else "climatology")
        ax.set_title("same-day (weather only)" if h == 0 else f"{h} day{'s' if h > 1 else ''} ahead", fontsize=9, loc="left")
        ax.set_xlabel("wind-speed error (%)")
        ax.grid(axis="y", color="#e6e5e1")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("holdout RMSE (MU per day)")
    axes[0].legend(frameon=False, fontsize=7)
    plt.tight_layout()
    plt.savefig(OUT / "robustness_noise_plain.png", dpi=150)
    return res, cr


def run_noise_aware(df):
    """Remedy for the perfect-weather objection: train on observed weather PLUS three noisy copies of it
    (noise scales 0.1, 0.2, 0.3 drawn once, seed 42), so the model learns to rely less on exact same-day weather.
    Tested exactly like run_noise (same test weather errors). Reported next to the plain models."""
    rows = []
    for h in (0, 1, 2, 3):
        ref = _reference_rmse(h)
        for kind in ("rf", "xgb"):
            rf_p, xg_p = _chosen(h)
            rng = np.random.default_rng(SEED)
            parts = [df] + [noisy_weather(df, s_, rng) for s_ in (0.1, 0.2, 0.3)]
            Xs, ys = [], []
            for part in parts:
                X = build(part, h)
                d = X.join(df[TARGET].rename("y")).dropna()
                d = d[d.index <= TRAIN_END]
                Xs.append(d.drop(columns="y"))
                ys.append(d["y"])
            Xtr, ytr = pd.concat(Xs), pd.concat(ys)
            if kind == "rf":
                m = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=1, **rf_p)
            else:
                m = XGBRegressor(n_estimators=300, subsample=0.8, random_state=SEED, n_jobs=1, tree_method="hist", **xg_p)
            m.fit(Xtr, ytr)
            feats = list(Xtr.columns)
            plain, _, test = _fit(df, h, kind)
            for s_ in NOISE_LEVELS:
                rng2 = np.random.default_rng(SEED)
                r_aw, r_pl = [], []
                for _ in range(1 if s_ == 0 else N_DRAWS):
                    Xp = build(noisy_weather(df, s_, rng2), h).loc[test.index, feats]
                    r_aw.append(float(np.sqrt(((test["y"] - np.clip(m.predict(Xp), 0, None)) ** 2).mean())))
                    r_pl.append(float(np.sqrt(((test["y"] - np.clip(plain.predict(Xp), 0, None)) ** 2).mean())))
                rows.append(dict(horizon_days=h, model="random_forest" if kind == "rf" else "xgboost", wind_error_pct=100 * s_,
                                 rmse_plain=np.mean(r_pl), rmse_noise_aware=np.mean(r_aw), reference_rmse=ref,
                                 noise_aware_beats_reference=np.mean(r_aw) < ref))
        print(f"[noise-aware] h={h} done")
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "robustness_noise_aware.csv", index=False)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), facecolor="#fcfcfb")
    for ax, h in zip(axes, (0, 1, 2, 3)):
        ax.set_facecolor("#fcfcfb")
        for m, c in (("random_forest", "#1baf7a"), ("xgboost", "#2a78d6")):
            g = res[(res.horizon_days == h) & (res.model == m)]
            ax.plot(g.wind_error_pct, g.rmse_plain, color=c, ls="--", marker="o", ms=3, lw=1.2, label=f"{m}, trained on observed weather")
            ax.plot(g.wind_error_pct, g.rmse_noise_aware, color=c, ls="-", marker="o", ms=3, lw=1.6, label=f"{m}, noise-aware training")
        ax.axhline(_reference_rmse(h), color="#eb6834", ls=":", lw=1.4, label="persistence" if h else "climatology")
        ax.set_title("same-day (weather only)" if h == 0 else f"{h} day{'s' if h > 1 else ''} ahead", fontsize=9, loc="left")
        ax.set_xlabel("wind-speed error (%)")
        ax.grid(axis="y", color="#e6e5e1")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("holdout RMSE (MU per day)")
    axes[1].legend(frameon=False, fontsize=6)
    plt.tight_layout()
    plt.savefig(OUT / "robustness_noise.png", dpi=150)
    pd.set_option("display.width", 200)
    print("\n=== Noise-aware training: holdout RMSE with degraded test weather (MU/day) ===")
    print(res.pivot_table(index=["horizon_days", "model"], columns="wind_error_pct", values=["rmse_plain", "rmse_noise_aware"]).round(2).to_string())
    return res


# ----------------------------------------------------------------------------------------------- 2 ramps
def _scores_np(t, p):
    hits = int(((t != 0) & (p == t)).sum())
    n_true, n_pred = int((t != 0).sum()), int((p != 0).sum())
    far_den = n_pred
    denom = n_true + n_pred - hits
    return (hits / n_true if n_true else np.nan, (n_pred - hits) / far_den if far_den else np.nan,
            hits / denom if denom else np.nan)


def run_ramps(df, n_boot=2000):
    gen = df[TARGET]
    pred = pd.read_csv(OUT / "trees_holdout_predictions.csv", parse_dates=["ts"])
    rng = np.random.default_rng(SEED)
    rows = []
    for (h, model), g in pred.groupby(["horizon_days", "model"]):
        if model in ("persistence", "climatology"):
            continue
        g = g.set_index("ts").sort_index()
        actual = gen.reindex(g.index)
        k = max(h, 1)
        if h >= 1:
            last = gen.shift(h, freq="D").reindex(g.index)
            tch, pch = actual - last, g["predicted_mu"] - last
        else:
            tch = actual - gen.shift(1, freq="D").reindex(g.index)
            pch = g["predicted_mu"] - g["predicted_mu"].shift(1, freq="D").reindex(g.index)
        ok = tch.notna() & pch.notna()
        tch, pch = tch[ok].values, pch[ok].values
        for q in (0.80, 0.90, 0.95):
            thr = ramp_threshold(gen, k, q=q)
            t = np.where(tch >= thr, 1, np.where(tch <= -thr, -1, 0))
            p = np.where(pch >= thr, 1, np.where(pch <= -thr, -1, 0))
            pod, far, csi = _scores_np(t, p)
            boots = np.array([_scores_np(t[i], p[i]) for i in (block_bootstrap_idx(len(t), rng) for _ in range(n_boot))])
            lo_hi = lambda col: tuple(np.nanpercentile(boots[:, col], [2.5, 97.5]))
            rows.append(dict(horizon_days=h, model=model, percentile=int(100 * q), threshold_mu=thr, n_real_ramps=int((t != 0).sum()),
                             pod=pod, pod_lo=lo_hi(0)[0], pod_hi=lo_hi(0)[1], far=far,
                             csi=csi, csi_lo=lo_hi(2)[0], csi_hi=lo_hi(2)[1]))
        print(f"[ramps] h={h} {model} done")
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "robustness_ramp_sensitivity.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n=== Ramp scores by threshold percentile with 95% block-bootstrap intervals ===")
    print(res.round(2).to_string(index=False))
    return res


# ----------------------------------------------------------------------------------------------- 3 baselines
def _gam(h):
    """Additive spline model (GAM-like): splines on weather and season, plain terms on generation lags."""
    X = build(pd.DataFrame(columns=["wind_speed", "temperature", "humidity", TARGET], index=pd.DatetimeIndex([])), h)
    cols = list(X.columns)
    spl = [c for c in cols if c in ("wind", "temp", "hum", "doy_sin", "doy_cos")]
    lin = [c for c in cols if c.startswith("gen_lag")]
    ct = ColumnTransformer([("s", SplineTransformer(n_knots=7, degree=3, extrapolation="constant"), spl),
                            ("l", "passthrough", lin)])
    return make_pipeline(ct, Ridge(alpha=1.0))


def run_baselines(df):
    pred = pd.read_csv(OUT / "trees_holdout_predictions.csv", parse_dates=["ts"])
    rows, errs = [], {}
    ref_models = {}
    for (h, m), g in pred.groupby(["horizon_days", "model"]):
        ref_models[(h, m)] = g.set_index("ts")["predicted_mu"]
    for h in (0, 1, 2, 3):
        X = build(df, h)
        d = X.join(df[TARGET].rename("y")).dropna()
        tr, te = d[d.index <= TRAIN_END], d[d.index >= TEST_START]
        feats = [c for c in d.columns if c != "y"]
        cand = {}
        if h == 0:
            iso = IsotonicRegression(out_of_bounds="clip", increasing=True).fit(tr["wind"], tr["y"])
            cand["power_curve"] = pd.Series(iso.predict(te["wind"]), index=te.index)
        gam = _gam(h)
        gam.fit(tr[feats], tr["y"])
        cand["gam_splines"] = pd.Series(np.clip(gam.predict(te[feats]), 0, None), index=te.index)
        if h == 1:
            try:
                from statsmodels.tsa.statespace.sarimax import SARIMAX

                ex = df[["wind_speed", "temperature", "humidity"]].assign(w2=lambda x: x.wind_speed ** 2)
                trn = df[df.index <= TRAIN_END]
                res = SARIMAX(trn[TARGET], exog=ex.loc[trn.index], order=(1, 0, 1), trend="c").fit(disp=False, maxiter=200)
                full = res.apply(df[TARGET], exog=ex)
                one = full.get_prediction(start=te.index[0], end=te.index[-1], exog=ex.loc[te.index[0]:te.index[-1]]).predicted_mean
                cand["sarimax_weather"] = one.reindex(te.index).clip(lower=0)
            except Exception as e:  # statsmodels missing or fit failed: keep going, say so
                print(f"[baselines] SARIMAX skipped ({type(e).__name__}: {e})")
        for name in ("linear", "random_forest", "xgboost"):
            if (h, name) in ref_models:
                cand[name] = ref_models[(h, name)].reindex(te.index)
        ref_name = "persistence" if h >= 1 else "climatology"
        cand[ref_name] = ref_models[(h, ref_name)].reindex(te.index)
        y = te["y"]
        xg_e = (y - cand["xgboost"]).dropna()
        for name, p in cand.items():
            e = (y - p).dropna()
            idx = e.index.intersection(xg_e.index)
            dm = dm_test(e[idx], xg_e[idx], max(h, 1)) if name != "xgboost" else (np.nan, np.nan)
            rows.append(dict(horizon_days=h, model=name, n=len(e), rmse=float(np.sqrt((e ** 2).mean())),
                             mae=float(e.abs().mean()), dm_vs_xgboost=dm[0], p_vs_xgboost=dm[1]))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "robustness_baselines.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n=== Extra baselines on the holdout (DM: positive = model worse than XGBoost) ===")
    print(res.round(3).to_string(index=False))
    return res


# ----------------------------------------------------------------------------------------------- 4 bias
def run_bias(df):
    pred = pd.read_csv(OUT / "trees_holdout_predictions.csv", parse_dates=["ts"])
    rows = []
    for (h, m), g in pred.groupby(["horizon_days", "model"]):
        if m not in ("xgboost", "random_forest"):
            continue
        g = g.assign(err=g.actual_mu - g.predicted_mu, half=g.ts.dt.year.astype(str) + "H" + ((g.ts.dt.month > 6) + 1).astype(str))
        for half, q in g.groupby("half"):
            rows.append(dict(horizon_days=h, model=m, period=half, n=len(q), mean_error_mu=q.err.mean(),
                             mean_actual_mu=q.actual_mu.mean(), bias_pct_of_mean=100 * q.err.mean() / q.actual_mu.mean()))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "robustness_bias.csv", index=False)
    print("\n=== Mean forecast error by half-year (actual - predicted; positive = under-forecast) ===")
    print(res.round(2).to_string(index=False))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["noise", "noise_aware", "ramps", "baselines", "bias"])
    a = ap.parse_args()
    np.random.seed(SEED)
    df = load_daily()
    for name, fn in (("noise", run_noise), ("noise_aware", run_noise_aware), ("ramps", run_ramps), ("baselines", run_baselines), ("bias", run_bias)):
        if a.only in (None, name):
            fn(df)


if __name__ == "__main__":
    main()
