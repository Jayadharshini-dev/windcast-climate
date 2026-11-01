"""Extreme-day and ramp behaviour under climate shift (extends the delta-change analysis).

Run from the repo root:   python -m climate.extremes_impact [--model linear]   (linear = sensitivity check)

Question: apart from the AVERAGE generation change (climate_impact.py), does the climate shift make sudden changes
(ramps), very low-generation days and long low-generation spells more or less frequent?

Method (same weather-only XGBoost and the same 10 climate models x 2 scenarios x 2 windows as climate_impact.py)
  1. Predict a daily generation series for today's weather (baseline) and for each climate-shifted weather record.
  2. Thresholds are fixed on the BASELINE predicted series, so the baseline value is by construction about 10%:
       ramp day   : |prediction(t) - prediction(t-1)| >= 90th percentile of baseline |day-to-day change|
       low day    : prediction below the baseline 10th percentile
       high day   : prediction above the baseline 90th percentile
       low spell  : run of consecutive low days; we report the longest run in the record and the mean run length.
  3. Report the same quantities for each shifted record. A change of "+20%" in ramp days means ramp days become
     20% more frequent than they are today. Spread across the 10 climate models (p10, p90) is the uncertainty.

Honest limits (state them in the paper)
  - Delta-change keeps today's day-to-day weather sequence and only rescales it month by month. It therefore
    shows how the system's NONLINEAR response to a shifted mean changes extremes, but it cannot create new
    weather variability (more intense storms, longer calm spells). The numbers are a lower bound on any change
    in variability; treat them as "sensitivity to the mean shift", not a forecast of extremes.
  - The model predicts a smoother series than reality (it explains most, not all, of the variance), so these are
    changes in the model's extreme days, compared like with like (baseline vs shifted), never against observed counts.

Outputs (in climate/): extremes_impact.csv (long format like climate_impact.csv), extremes_baseline_reference.csv,
extremes_impact.png
"""
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from climate.climate_impact import FEATS, REGION, fit_models, load_factors, shifted_weather, to_features
from models.data import load_daily
from models.ramps import spell_lengths
from models.trees import SEED, TARGET

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent
Q_RAMP, Q_LOW, Q_HIGH = 0.90, 0.10, 0.90


def thresholds(base_pred):
    """Fixed from the baseline predicted series: (ramp size, low level, high level)."""
    d = base_pred.diff().abs()
    return float(d.quantile(Q_RAMP)), float(base_pred.quantile(Q_LOW)), float(base_pred.quantile(Q_HIGH))


def extreme_metrics(pred, thr):
    """Frequency metrics for one predicted daily series (a pandas Series with a daily DatetimeIndex)."""
    ramp_thr, low, high = thr
    d = pred.diff()
    is_low = pred < low
    spells = spell_lengths(is_low)
    return {
        "ramp_day_share_pct": float(100 * (d.abs() >= ramp_thr).sum() / d.notna().sum()),
        "up_ramp_day_share_pct": float(100 * (d >= ramp_thr).sum() / d.notna().sum()),
        "down_ramp_day_share_pct": float(100 * (d <= -ramp_thr).sum() / d.notna().sum()),
        "low_day_share_pct": float(100 * is_low.mean()),
        "high_day_share_pct": float(100 * (pred > high).mean()),
        "longest_low_spell_days": float(max(spells) if spells else 0),
        "mean_low_spell_days": float(np.mean(spells) if spells else 0),
        "spells_of_5plus_low_days_per_year": float(sum(1 for s in spells if s >= 5) / (len(pred) / 365.25)),
    }


def run(model="xgboost"):
    np.random.seed(SEED)
    df = load_daily()
    weather = df[FEATS].dropna()
    train = df.dropna(subset=FEATS + [TARGET])
    pred_fn = fit_models(train)[model]
    base = pd.Series(pred_fn(to_features(weather)), index=weather.index)
    thr = thresholds(base)
    ref = extreme_metrics(base, thr)
    print(f"[baseline] ramp size {thr[0]:.1f} MU, low < {thr[1]:.1f} MU, high > {thr[2]:.1f} MU "
          f"(from {len(base)} baseline days)")
    print("[baseline]", {k: round(v, 2) for k, v in ref.items()})

    fac_all = load_factors()
    rows = []
    for (scen, year), g in fac_all.groupby(["scenario", "target_year"]):
        per_model = []
        for cm, fac in g.groupby("climate_model"):
            p = pd.Series(pred_fn(to_features(shifted_weather(weather, fac))), index=weather.index)
            m = extreme_metrics(p, thr)
            per_model.append(m)
            for k, v in m.items():
                rows.append(dict(region=REGION, scenario=scen, target_year=int(year), climate_model=cm,
                                 metric=k, value=v, baseline=ref[k], p10=np.nan, p90=np.nan))
        pm = pd.DataFrame(per_model)
        for k in pm.columns:
            rows.append(dict(region=REGION, scenario=scen, target_year=int(year), climate_model="ensemble",
                             metric=k, value=float(pm[k].mean()), baseline=ref[k],
                             p10=float(np.percentile(pm[k], 10)), p90=float(np.percentile(pm[k], 90))))
    res = pd.DataFrame(rows)
    tag = "" if model == "xgboost" else f"_{model}"  # sensitivity runs never overwrite the primary files
    res.to_csv(OUT / f"extremes_impact{tag}.csv", index=False)
    pd.DataFrame([dict(model="xgboost", ramp_threshold_mu=thr[0], low_threshold_mu=thr[1],
                       high_threshold_mu=thr[2], n_days=len(base), **ref)]).to_csv(
        OUT / f"extremes_baseline_reference{tag}.csv", index=False)

    pd.set_option("display.width", 200)
    ens = res[res.climate_model == "ensemble"]
    print("\nENSEMBLE mean [p10, p90 across 10 climate models]  (baseline in brackets after the metric name)")
    for k in ref:
        print(f"\n{k}  (today: {ref[k]:.2f})")
        for _, r in ens[ens.metric == k].iterrows():
            print(f"   {r.scenario} {r.target_year}: {r.value:6.2f}  [{r.p10:6.2f}, {r.p90:6.2f}]")
    plot(res, ref, tag)
    return res


def plot(res, ref, tag=""):
    metrics = [("ramp_day_share_pct", "ramp days (% of days)"),
               ("low_day_share_pct", "low-generation days (% of days)"),
               ("longest_low_spell_days", "longest low spell (days)")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor="#fcfcfb")
    cases = [("ssp245", 2030), ("ssp245", 2050), ("ssp585", 2030), ("ssp585", 2050)]
    for ax, (k, label) in zip(axes, metrics):
        ax.set_facecolor("#fcfcfb")
        for i, (s, y) in enumerate(cases):
            d = res[(res.metric == k) & (res.scenario == s) & (res.target_year == y)]
            m, e = d[d.climate_model != "ensemble"], d[d.climate_model == "ensemble"].iloc[0]
            ax.scatter(np.full(len(m), i) + np.linspace(-0.18, 0.18, len(m)), m.value, color="#2a78d6", s=14, alpha=0.7)
            ax.hlines(e.value, i - 0.25, i + 0.25, color="#eb6834", lw=2)
        ax.axhline(ref[k], color="#7a7973", lw=0.9, ls="--", label="today")
        if k.endswith("share_pct"):
            ax.set_ylim(0, 1.2 * max(ax.get_ylim()[1] / 1.05, ref[k]))  # honest zero baseline for shares
        ax.set_xticks(range(4))
        ax.set_xticklabels([f"{s}\n{y}" for s, y in cases], fontsize=8)
        ax.set_title(label, loc="left", fontsize=10)
        ax.grid(axis="y", color="#e6e5e1")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Extremes under climate shift: dots = 10 climate models, orange bar = ensemble mean", x=0.01,
                 ha="left", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT / f"extremes_impact{tag}.png", dpi=150)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["xgboost", "linear"], default="xgboost",
                    help="weather-to-power model; linear is the sensitivity check (files get a _linear suffix)")
    run(ap.parse_args().model)
