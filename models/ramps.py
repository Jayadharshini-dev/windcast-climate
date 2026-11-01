"""Ramp (sudden change) and extreme-day analysis for Tamil Nadu wind generation.

Run from the repo root:   python -m models.ramps

Why this exists
  Average error (RMSE) hides what a grid operator fears most: a sudden jump or drop in wind generation, or a run of
  very low-wind days. This module asks a different question of the models we already built:
  "when generation really jumps or falls, did the forecast see it coming?"

Definitions (all thresholds come from the TRAINING years only, up to 2023-12-31, so the test years never leak in)
  ramp over k days   = generation(t) - generation(t-k)               (k = forecast horizon, 1..3)
  ramp event         = |ramp| >= the 90th percentile of |ramp| in the training years (up-ramp / down-ramp by sign)
  low-generation day = generation below the 10th percentile of training days
  high-generation day= generation above the 90th percentile of training days

How a forecast is judged
  Operational models (h = 1, 2, 3) know the last observed generation at day t-h, so the forecast ramp is
  prediction(t) - generation(t-h).  Persistence forecasts "no change", so it can never announce a ramp: that is the
  honest reference. The weather-only model (h = 0) has no last observation; its forecast ramp is
  prediction(t) - prediction(t-1), compared with the real one-day change.
  Scores:  hit rate (share of real ramps flagged with the right sign), false-alarm ratio (share of flagged ramps that
  were not real), CSI (hits / (hits + misses + false alarms), 1 is perfect), and RMSE on ramp days versus all other days.

Reads models/trees_holdout_predictions.csv (written by models.trees) and the daily data (models.data.load_daily).
Writes: models/ramps_scores.csv, models/ramps_events_holdout.csv, models/ramps_low_high_scores.csv,
models/ramps_by_month.csv, models/ramps_holdout.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models.data import load_daily

OUT = Path(__file__).resolve().parent
TARGET = "generation_mu"
TRAIN_END = "2023-12-31"
Q_RAMP, Q_LOW, Q_HIGH = 0.90, 0.10, 0.90


def k_day_change(gen, k):
    """Change over k calendar days; NaN if either end is a missing day."""
    return gen - gen.shift(k, freq="D").reindex(gen.index)


def ramp_threshold(gen, k, q=Q_RAMP, train_end=TRAIN_END):
    """Size of a 'sudden' change: q-th percentile of |k-day change| using training years only."""
    ch = k_day_change(gen, k)
    return float(ch[ch.index <= train_end].abs().quantile(q))


def ramp_labels(change, thr):
    """+1 up-ramp, -1 down-ramp, 0 no ramp (NaN stays NaN)."""
    out = pd.Series(0.0, index=change.index)
    out[change >= thr] = 1.0
    out[change <= -thr] = -1.0
    return out.where(change.notna())


def ramp_scores(true_lab, pred_lab):
    """Hit rate, false-alarm ratio and CSI. A hit needs the right direction."""
    m = true_lab.notna() & pred_lab.notna()
    t, p = true_lab[m], pred_lab[m]
    hits = int(((t != 0) & (p == t)).sum())
    n_true = int((t != 0).sum())
    n_pred = int((p != 0).sum())
    misses = n_true - hits
    false_alarms = n_pred - hits
    return dict(n_days=int(m.sum()), n_real_ramps=n_true, n_flagged=n_pred, hits=hits,
                hit_rate=hits / n_true if n_true else np.nan,
                false_alarm_ratio=false_alarms / n_pred if n_pred else np.nan,
                csi=hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) else np.nan)


def spell_lengths(flag):
    """Lengths of runs of consecutive True days (a calendar gap breaks a run)."""
    flag = flag.copy()
    lengths, run, prev = [], 0, None
    for ts, v in flag.items():
        if prev is not None and (ts - prev).days != 1:
            if run:
                lengths.append(run)
            run = 0
        if bool(v):
            run += 1
        elif run:
            lengths.append(run)
            run = 0
        prev = ts
    if run:
        lengths.append(run)
    return lengths


def main():
    df = load_daily()
    gen = df[TARGET].dropna()
    gen_full = df[TARGET]
    pred = pd.read_csv(OUT / "trees_holdout_predictions.csv", parse_dates=["ts"])

    rows, ev_rows, lh_rows = [], [], []
    lo = gen[gen.index <= TRAIN_END].quantile(Q_LOW)
    hi = gen[gen.index <= TRAIN_END].quantile(Q_HIGH)
    print(f"Training-year thresholds: low day < {lo:.1f} MU, high day > {hi:.1f} MU")

    for (h, model), g in pred.groupby(["horizon_days", "model"]):
        g = g.set_index("ts").sort_index()
        actual = gen_full.reindex(g.index)               # exact observed values (not the rounded CSV copy)
        k = max(h, 1)
        thr = ramp_threshold(gen_full, k)
        if h >= 1:
            last_obs = gen_full.shift(h, freq="D").reindex(g.index)
            true_ch = actual - last_obs
            pred_ch = g["predicted_mu"] - last_obs
        else:  # weather-only: day-over-day change of the forecast itself
            prev_pred = g["predicted_mu"].shift(1, freq="D").reindex(g.index)
            true_ch = actual - gen_full.shift(1, freq="D").reindex(g.index)
            pred_ch = g["predicted_mu"] - prev_pred
        tl, pl = ramp_labels(true_ch, thr), ramp_labels(pred_ch, thr)
        sc = ramp_scores(tl, pl)
        err = (actual - g["predicted_mu"]).abs()
        on = (tl != 0) & tl.notna()
        off = (tl == 0)
        sc.update(horizon_days=h, model=model, ramp_threshold_mu=thr,
                  rmse_on_ramp_days=float(np.sqrt((err[on] ** 2).mean())) if on.any() else np.nan,
                  rmse_other_days=float(np.sqrt((err[off] ** 2).mean())) if off.any() else np.nan)
        # direction only: of real ramps, how often did the forecast move the right way, whatever its size
        sc["right_direction_pct"] = float(100 * (np.sign(pred_ch[on]) == np.sign(true_ch[on])).mean()) if on.any() else np.nan
        rows.append(sc)

        # low / high generation days
        for name, real, flag in (("low", actual < lo, g["predicted_mu"] < lo),
                                 ("high", actual > hi, g["predicted_mu"] > hi)):
            mk = actual.notna()
            r, f = real[mk], flag[mk]
            hits = int((r & f).sum())
            lh_rows.append(dict(horizon_days=h, model=model, event=name, n_real=int(r.sum()), n_flagged=int(f.sum()),
                                hits=hits, hit_rate=hits / r.sum() if r.sum() else np.nan,
                                false_alarm_ratio=(f.sum() - hits) / f.sum() if f.sum() else np.nan))

        if model == "xgboost" and h == 1:
            ev = pd.DataFrame({"ts": g.index, "actual_mu": actual.values, "forecast_mu": g["predicted_mu"].values,
                               "real_change_mu": true_ch.values, "forecast_change_mu": pred_ch.values,
                               "real_label": tl.values, "forecast_label": pl.values})
            ev_rows.append(ev[ev.real_label != 0].assign(horizon_days=h, model=model))

    sc = pd.DataFrame(rows)
    cols = ["horizon_days", "model", "ramp_threshold_mu", "n_days", "n_real_ramps", "n_flagged", "hits", "hit_rate",
            "false_alarm_ratio", "csi", "right_direction_pct", "rmse_on_ramp_days", "rmse_other_days"]
    sc = sc[cols].sort_values(["horizon_days", "model"])
    sc.to_csv(OUT / "ramps_scores.csv", index=False)
    pd.DataFrame(lh_rows).sort_values(["event", "horizon_days", "model"]).to_csv(OUT / "ramps_low_high_scores.csv", index=False)
    if ev_rows:
        pd.concat(ev_rows).to_csv(OUT / "ramps_events_holdout.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n=== Ramp detection on the 2024+ holdout (event = change >= 90th percentile of training changes) ===")
    print(sc.round(2).to_string(index=False))
    print("\n=== Low / high generation days ===")
    print(pd.DataFrame(lh_rows).sort_values(["event", "horizon_days", "model"]).round(2).to_string(index=False))

    # when do real ramps happen? calendar-month profile over the whole record (day-to-day change, training threshold)
    thr1 = ramp_threshold(gen_full, 1)
    lab1 = ramp_labels(k_day_change(gen_full, 1), thr1).dropna()
    by_month = pd.DataFrame({"days": lab1.groupby(lab1.index.month).size(),
                             "up_ramps": (lab1 > 0).groupby(lab1.index.month).sum(),
                             "down_ramps": (lab1 < 0).groupby(lab1.index.month).sum()})
    by_month["ramp_share_pct"] = 100 * (by_month.up_ramps + by_month.down_ramps) / by_month.days
    by_month.index.name = "month"
    by_month.to_csv(OUT / "ramps_by_month.csv")
    print(f"\n=== Real one-day ramps by calendar month (threshold {thr1:.1f} MU, whole record) ===")
    print(by_month.round(1).to_string())

    # figure: XGBoost 1-day-ahead forecast through the holdout with real ramps marked, and bars of CSI
    g = pred[(pred.model == "xgboost") & (pred.horizon_days == 1)].set_index("ts").sort_index()
    ev = pd.concat(ev_rows) if ev_rows else pd.DataFrame()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [2.2, 1]}, facecolor="#fcfcfb")
    ax = axes[0]
    ax.set_facecolor("#fcfcfb")
    ax.plot(g.index, g["actual_mu"], color="#7a7973", lw=0.9, label="actual")
    ax.plot(g.index, g["predicted_mu"], color="#2a78d6", lw=0.9, label="XGBoost, 1 day ahead")
    if len(ev):
        up, dn = ev[ev.real_label > 0], ev[ev.real_label < 0]
        ax.scatter(up.ts, up.actual_mu, color="#1baf7a", s=14, zorder=3, label="real up-ramp")
        ax.scatter(dn.ts, dn.actual_mu, color="#eb6834", s=14, zorder=3, label="real down-ramp")
    ax.set_ylabel("generation (MU per day)")
    ax.set_title("Holdout 2024+: real sudden changes marked", loc="left", fontsize=10)
    ax.grid(axis="y", color="#e6e5e1")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax = axes[1]
    ax.set_facecolor("#fcfcfb")
    order = ["persistence", "climatology", "linear", "random_forest", "xgboost"]
    colors = {"persistence": "#7a7973", "climatology": "#7a7973", "linear": "#eda100",
              "random_forest": "#1baf7a", "xgboost": "#2a78d6"}
    w = 0.16
    for i, m in enumerate(order):
        xs, ys = [], []
        for j, h in enumerate((0, 1, 2, 3)):
            r = sc[(sc.horizon_days == h) & (sc.model == m)]
            if len(r):
                xs.append(j + (i - 2) * w)
                ys.append(float(r.csi.iloc[0]))
        ax.bar(xs, ys, w - 0.02, color=colors[m], label=m)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["same-day", "1 d", "2 d", "3 d"])
    ax.set_ylabel("CSI for ramps (1 = perfect)")
    ax.set_title("Ramp detection score", loc="left", fontsize=10)
    ax.grid(axis="y", color="#e6e5e1")
    ax.legend(frameon=False, fontsize=6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT / "ramps_holdout.png", dpi=150)


if __name__ == "__main__":
    main()
