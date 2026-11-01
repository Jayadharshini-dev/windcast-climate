"""Two checks that need DAILY NEX-GDDP-CMIP6 output (fetch it first with etl/fetch_nexgddp_daily.py).

Run from the repo root:
    python -m climate.daily_checks --validate        # 1. NEX-GDDP against ERA5 over 2020-2025 (ssp245)
    python -m climate.daily_checks --variability     # 2. change in ramps / low days / spells from the model's own daily output
    python -m climate.daily_checks                   # both

1. Validation. SSP experiments start in 2015, so ssp245 2020-2025 overlaps the ERA5 record. A climate model is not a
   weather forecast, so days cannot match; we compare distributions: mean bias, spread (standard deviation ratio),
   wind-speed quantiles, the correlation of the monthly cycle, the lag-1 day-to-day autocorrelation, and the change in
   mean predicted generation when the weather-only model is fed raw NEX weather instead of ERA5.
2. Variability. The monthly delta-change method keeps today's day-to-day weather and so cannot show whether ramps or
   low-wind spells change. Here the model's own daily output is used instead, after quantile delta mapping (Cannon et
   al. 2015) to ERA5, per calendar month: wind multiplicatively, temperature and humidity additively. The same mapping
   is applied to the model's historical days (1995-2014) and to its future days (ssp585, 2041-2060), so both carry the
   model's own day-to-day sequence; ramp, low-day and spell thresholds are set on the historical series and the
   future series is compared with it. The change in mean generation is set beside the delta-change result.
   Caveat: climate-model day-to-day sequences are not observed sequences, so absolute counts are not comparable with
   the real record; only the change between the two periods is interpreted.

Outputs (climate/): nexgddp_vs_era5.csv, nexgddp_vs_era5.png, variability_test.csv
"""
import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from climate.climate_impact import FEATS, fit_models, to_features
from climate.extremes_impact import extreme_metrics, thresholds
from models.data import REPO_ROOT, load_daily
from models.trees import SEED, TARGET

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent
DAILY_DIR = REPO_ROOT / "data" / "raw" / "nexgddp_daily"
RENAME = {"sfcWind": "wind_speed", "tas": "temperature", "hurs": "humidity"}


# ---------------------------------------------------------------- loading
def load_nex_daily(model, scenario, years, directory=None):
    """State-mean daily weather for one model and scenario (equal weight over the 3 points), or None if incomplete."""
    directory = Path(directory or DAILY_DIR)
    frames = []
    for y in years:
        p = directory / f"{model}_{scenario}_{y}.csv"
        if not p.exists():
            return None
        frames.append(pd.read_csv(p))
    d = pd.concat(frames)
    g = d.groupby("date")[["sfcWind", "tas", "hurs"]].mean()
    g.index = pd.to_datetime(g.index)
    g["tas"] = g["tas"] - 273.15
    return g.rename(columns=RENAME).sort_index()


def models_available(scenario, years, directory=None):
    directory = Path(directory or DAILY_DIR)
    names = sorted({p.name.split("_")[0] for p in directory.glob("*.csv")}) if directory.exists() else []
    return [m for m in names if load_nex_daily(m, scenario, years, directory) is not None]


# ---------------------------------------------------------------- quantile delta mapping (per calendar month)
def qdm_month(fut, hist, obs, kind):
    """Quantile delta mapping for one calendar month (arrays). kind: 'mult' (wind) or 'add' (temperature, humidity)."""
    fut, hist, obs = (np.asarray(v, float) for v in (fut, hist, obs))
    tau = (pd.Series(fut).rank(method="average").values - 0.5) / len(fut)  # position of each future value in the future CDF
    q_obs, q_hist = np.quantile(obs, tau), np.quantile(hist, tau)
    if kind == "mult":
        delta = np.clip(fut / np.maximum(q_hist, 0.1), 0.2, 5.0)
        return q_obs * delta
    return q_obs + (fut - q_hist)


def qdm_series(fut, hist, obs):
    """Map a daily weather DataFrame (wind_speed, temperature, humidity) to ERA5 climate, month by month."""
    out = fut.copy()
    for m in range(1, 13):
        fm, hm, om = fut.index.month == m, hist.index.month == m, obs.index.month == m
        if not fm.any() or not hm.any() or not om.any():
            continue
        out.loc[fm, "wind_speed"] = qdm_month(fut.loc[fm, "wind_speed"], hist.loc[hm, "wind_speed"], obs.loc[om, "wind_speed"], "mult")
        out.loc[fm, "temperature"] = qdm_month(fut.loc[fm, "temperature"], hist.loc[hm, "temperature"], obs.loc[om, "temperature"], "add")
        out.loc[fm, "humidity"] = np.clip(qdm_month(fut.loc[fm, "humidity"], hist.loc[hm, "humidity"], obs.loc[om, "humidity"], "add"), 0, 100)
    return out


# ---------------------------------------------------------------- 1. validation
def lag1(x):
    a = x - x.groupby(x.index.month).transform("mean")  # anomalies from the monthly cycle
    return float(a.autocorr(1))


def compare_to_era5(nex, era5):
    """Distribution comparison of one model's daily weather with ERA5 over the common days (pure function)."""
    idx_n = nex.index.strftime("%m-%d")
    common = nex.index.intersection(era5.index)
    n, e = nex.loc[common], era5.loc[common]
    r = {"n_days": len(common)}
    for v in FEATS:
        r[f"{v}_mean_bias"] = float(n[v].mean() - e[v].mean())
        r[f"{v}_std_ratio"] = float(n[v].std() / e[v].std())
        r[f"{v}_lag1_nex"], r[f"{v}_lag1_era5"] = lag1(n[v]), lag1(e[v])
        mc = pd.concat([n[v].groupby(n.index.month).mean(), e[v].groupby(e.index.month).mean()], axis=1)
        r[f"{v}_monthly_cycle_corr"] = float(mc.corr().iloc[0, 1])
    r["wind_mean_ratio"] = float(n["wind_speed"].mean() / e["wind_speed"].mean())
    for q in (0.1, 0.5, 0.9, 0.99):
        r[f"wind_q{int(q * 100)}_ratio"] = float(n["wind_speed"].quantile(q) / e["wind_speed"].quantile(q))
    return r


def run_validation(pred_fn=None, directory=None):
    era5 = load_daily()[FEATS].dropna()
    rows = []
    models = models_available("ssp245", range(2020, 2026), directory)
    if not models:
        raise SystemExit("No complete ssp245 2020-2025 daily files found. Run: python -m etl.fetch_nexgddp_daily --plan validation")
    for m in models:
        nex = load_nex_daily(m, "ssp245", range(2020, 2026), directory)
        r = dict(model=m, **compare_to_era5(nex, era5))
        if pred_fn is not None:  # effect on predicted generation of using raw NEX weather instead of ERA5
            common = nex.index.intersection(era5.index)
            r["gen_pred_diff_pct_raw_nex_vs_era5"] = float(100 * (pred_fn(to_features(nex.loc[common])).mean() / pred_fn(to_features(era5.loc[common])).mean() - 1))
        rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "nexgddp_vs_era5.csv", index=False)
    pd.set_option("display.width", 220)
    keep = ["model", "n_days", "wind_mean_ratio", "wind_speed_std_ratio", "wind_speed_monthly_cycle_corr", "wind_q90_ratio", "temperature_mean_bias", "humidity_mean_bias"] + (["gen_pred_diff_pct_raw_nex_vs_era5"] if pred_fn else [])
    print("\n=== NEX-GDDP (ssp245) against ERA5, 2020-2025, state mean ===")
    print(res[keep].round(2).to_string(index=False))
    fig, ax = plt.subplots(figsize=(7, 3.8), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    mm = era5.groupby(era5.index.month)["wind_speed"].mean()
    for m in models:
        nex = load_nex_daily(m, "ssp245", range(2020, 2026), directory)
        ax.plot(range(1, 13), nex.groupby(nex.index.month)["wind_speed"].mean().values, color="#b8b7b2", lw=0.9)
    ax.plot(range(1, 13), mm.values, color="#2a78d6", lw=2.2, label="ERA5")
    ax.plot([], [], color="#b8b7b2", label="NEX-GDDP models")
    ax.set_xlabel("month")
    ax.set_ylabel("mean 10 m wind speed (m/s)")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#e6e5e1")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT / "nexgddp_vs_era5.png", dpi=150)
    return res


# ---------------------------------------------------------------- 2. variability
def variability_for_model(hist, fut, obs, pred_fn):
    """Mean-generation change and extreme-day metrics for one climate model (pure given pred_fn)."""
    h_corr, f_corr = qdm_series(hist, hist, obs), qdm_series(fut, hist, obs)
    ph = pd.Series(pred_fn(to_features(h_corr)), index=h_corr.index)
    pf = pd.Series(pred_fn(to_features(f_corr)), index=f_corr.index)
    thr = thresholds(ph)
    mh, mf = extreme_metrics(ph, thr), extreme_metrics(pf, thr)
    out = {"mean_generation_change_pct": float(100 * (pf.mean() / ph.mean() - 1))}
    for k in mh:
        out[f"{k}_hist"], out[f"{k}_future"] = mh[k], mf[k]
    return out


def run_variability(pred_fn, directory=None):
    obs = load_daily()[FEATS].dropna()
    hy, fy = range(1995, 2015), range(2041, 2061)
    models = [m for m in models_available("historical", hy, directory) if load_nex_daily(m, "ssp585", fy, directory) is not None]
    if not models:
        raise SystemExit("No model has both historical 1995-2014 and ssp585 2041-2060 daily files. Run: python -m etl.fetch_nexgddp_daily --plan variability")
    delta = {}
    f = OUT / "climate_impact.csv"
    if f.exists():
        d = pd.read_csv(f)
        d = d[(d.scenario == "ssp585") & (d.target_year == 2050) & (d.metric == "mean_generation_change_pct")]
        delta = dict(zip(d.climate_model, d.value))
    rows = []
    for m in models:
        r = variability_for_model(load_nex_daily(m, "historical", hy, directory), load_nex_daily(m, "ssp585", fy, directory), obs, pred_fn)
        rows.append(dict(model=m, delta_change_mean_generation_pct=delta.get(m, np.nan), **r))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "variability_test.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n=== Daily-output (QDM) vs monthly delta-change, ssp585 2041-2060 ===")
    print(res[["model", "delta_change_mean_generation_pct", "mean_generation_change_pct"]].round(2).to_string(index=False))
    for k in ("ramp_day_share_pct", "low_day_share_pct", "high_day_share_pct", "longest_low_spell_days", "spells_of_5plus_low_days_per_year"):
        print(f"\n{k}: historical -> future (per model)")
        print(res[["model", f"{k}_hist", f"{k}_future"]].round(2).to_string(index=False))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--variability", action="store_true")
    a = ap.parse_args()
    both = not (a.validate or a.variability)
    np.random.seed(SEED)
    df = load_daily()
    train = df.dropna(subset=FEATS + [TARGET])
    pred_fn = fit_models(train)["xgboost"]
    if a.validate or both:
        run_validation(pred_fn)
    if a.variability or both:
        run_variability(pred_fn)


if __name__ == "__main__":
    main()
