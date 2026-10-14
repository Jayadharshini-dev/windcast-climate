"""Weather baselines for Tamil Nadu daily wind generation (horizons 1-3 days).

Run from the repo root:  python -m models.baseline_weather
Optional arguments:      python -m models.baseline_weather <golden_csv> <output_dir>

Models, for each horizon h (days ahead):
  persistence   : generation(t) predicted by generation(t-h)
  linear_gen    : linear regression on lagged generation (lags h..h+2) + day-of-year sin/cos
  linear_weather: linear_gen + wind, wind^2, wind^3, temperature, humidity (day t, perfect-prognosis)

Rules (PROJECT_CONTEXT.md section 5): daily data, no pressure, time-based split, always compare with persistence.
Evaluation uses ONE common set of test days per horizon (days where every feature exists), so all
three models are scored on exactly the same days.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED = 42
TRAIN_END = "2023-12-31"
TEST_START = "2024-01-01"
HORIZONS = (1, 2, 3)
TARGET = "generation_mu"

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "data" / "golden" / "tn_daily_golden.csv"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent


def load():
    df = pd.read_csv(DATA, parse_dates=["ts"]).sort_values("ts")
    df = df.set_index("ts").asfreq("D")  # explicit daily index; gaps stay NaN
    return df


def build(df, h):
    """Feature table for horizon h. Only generation from day t-h or earlier is used."""
    X = pd.DataFrame(index=df.index)
    for k in range(h, h + 3):
        X[f"gen_lag{k}"] = df[TARGET].shift(k)
    doy = df.index.dayofyear
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    X["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    X["wind"] = df["wind_speed"]
    X["wind2"] = df["wind_speed"] ** 2
    X["wind3"] = df["wind_speed"] ** 3
    X["temp"] = df["temperature"]
    X["hum"] = df["humidity"]
    return X


def metrics(y, p):
    return dict(RMSE=mean_squared_error(y, p) ** 0.5, MAE=mean_absolute_error(y, p), R2=r2_score(y, p))


def main():
    np.random.seed(SEED)
    df = load()
    n_all = len(df)
    n_target = int(df[TARGET].notna().sum())
    print(f"Loaded {DATA.name}: {n_all} days ({df.index.min().date()} to {df.index.max().date()}), "
          f"{n_target} with generation, {n_all - n_target} missing (kept as gaps, not filled).")
    both = df[["wind_speed", TARGET]].dropna()
    print(f"Same-day correlation wind vs generation: {both['wind_speed'].corr(both[TARGET]):.3f}\n")

    rows, preds = [], {}
    for h in HORIZONS:
        X = build(df, h)
        gen_cols = [f"gen_lag{k}" for k in range(h, h + 3)] + ["doy_sin", "doy_cos"]
        wx_cols = gen_cols + ["wind", "wind2", "wind3", "temp", "hum"]
        data = X.join(df[TARGET].rename("y"))
        valid = data.dropna(subset=["y"] + wx_cols)
        train = valid[valid.index <= TRAIN_END]
        test = valid[valid.index >= TEST_START]
        n_dropped_tr = int((data.index <= TRAIN_END).sum() - len(train))
        n_dropped_te = int((data.index >= TEST_START).sum() - len(test))
        print(f"h={h}: train rows {len(train)} (dropped {n_dropped_tr} with a missing value), "
              f"test rows {len(test)} (dropped {n_dropped_te})")

        p_persist = test[f"gen_lag{h}"]
        m_gen = LinearRegression().fit(train[gen_cols], train["y"])
        p_gen = pd.Series(m_gen.predict(test[gen_cols]), index=test.index)
        m_wx = LinearRegression().fit(train[wx_cols], train["y"])
        p_wx = pd.Series(m_wx.predict(test[wx_cols]), index=test.index)
        preds[h] = (test["y"], p_persist, p_gen, p_wx)

        base_rmse = metrics(test["y"], p_persist)["RMSE"]
        err_p = (test["y"] - p_persist).abs()
        for name, p in [("persistence", p_persist), ("linear_gen", p_gen), ("linear_weather", p_wx)]:
            m = metrics(test["y"], p)
            m["model"], m["horizon_days"], m["n_test"] = name, h, len(test)
            m["RMSE_skill_vs_persistence_%"] = 100 * (1 - m["RMSE"] / base_rmse)
            m["days_beating_persistence_%"] = (
                np.nan if name == "persistence" else 100 * float(((test["y"] - p).abs() < err_p).mean())
            )
            rows.append(m)
        if h == 1:
            coefs = dict(zip(wx_cols, m_wx.coef_.round(4)))

    res = pd.DataFrame(rows)[["model", "horizon_days", "n_test", "RMSE", "MAE", "R2",
                              "RMSE_skill_vs_persistence_%", "days_beating_persistence_%"]]
    res.to_csv(OUT / "baseline_weather_results.csv", index=False)
    print("\nResults on the test period (2024-01-01 onward), million units per day:")
    print(res.round(3).to_string(index=False))
    print(f"\nlinear_weather coefficients at h=1: {coefs}")

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False, facecolor="#fcfcfb")
    for ax, h in zip(axes, (1, 3)):
        y, pp, pg, pw = preds[h]
        ax.set_facecolor("#fcfcfb")
        ax.plot(y.index, y, color="#0b0b0b", lw=1.2, label="Actual")
        ax.plot(pp.index, pp, color="#eb6834", lw=1, alpha=0.8, label="Persistence")
        ax.plot(pw.index, pw, color="#2a78d6", lw=1.4, label="Linear + weather")
        ax.set_title(f"Tamil Nadu daily wind generation, {h}-day-ahead, test period", loc="left", fontsize=11)
        ax.set_ylabel("MU per day")
        ax.grid(axis="y", color="#e6e5e1")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(frameon=False, ncol=3, loc="upper left")
    plt.tight_layout()
    plt.savefig(OUT / "baseline_weather_actual_vs_predicted.png", dpi=150)
    print(f"\nSaved {OUT / 'baseline_weather_results.csv'} and baseline_weather_actual_vs_predicted.png")


if __name__ == "__main__":
    main()
