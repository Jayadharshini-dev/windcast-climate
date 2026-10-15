"""Random Forest and XGBoost for Tamil Nadu daily wind generation, with walk-forward validation.

Run from the repo root:  python -m models.trees
Optional arguments:      python -m models.trees <csv_path> <output_dir>   (default: database, else golden CSV)

Two families of model (both use only features that exist in ERA5 AND in the climate data: wind, temperature,
humidity, plus calendar terms; pressure is never used):

  operational h=1,2,3 : lagged generation (only values from day t-h or earlier) + weather of day t + calendar
  weather_only h=0    : weather of day t + calendar only, NO lagged generation.
                        This is the model the climate module will use, because future climate runs have no
                        "yesterday's generation" to feed in.

Baselines: persistence (h>=1) and monthly climatology (h=0). Linear regression is kept as the simple reference.

Validation (never random, never shuffled):
  - walk-forward folds with an expanding training window:
      fold 2022 : train 2020-05-28..2021-12-31, test 2022
      fold 2023 : train ..2022-12-31,           test 2023
      fold 2024+: train ..2023-12-31,           test 2024-01-01..end   <- headline "holdout"
  - hyper-parameters are tuned INSIDE each training window with TimeSeriesSplit, so the test period never
    influences model settings.
"""
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

from models.data import load_daily

warnings.filterwarnings("ignore")
SEED = 42
TARGET = "generation_mu"
HORIZONS = (0, 1, 2, 3)  # 0 = weather_only (climate-coupling model)
FOLDS = [  # (name, train_end, test_start, test_end)
    ("fold_2022", "2021-12-31", "2022-01-01", "2022-12-31"),
    ("fold_2023", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("holdout_2024+", "2023-12-31", "2024-01-01", "2100-01-01"),
]
REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent

RF_GRID = {"max_depth": [6, 12], "min_samples_leaf": [3, 10], "max_features": [0.5, 1.0]}
XGB_GRID = {"max_depth": [2, 4], "learning_rate": [0.03, 0.1], "min_child_weight": [3, 10]}


def load():
    """Database view if reachable, else the golden CSV (see models/data.py). Optional argv[1] = a CSV path."""
    return load_daily(csv_path=sys.argv[1] if len(sys.argv) > 1 and __name__ == "__main__" else None)


def build(df, h):
    """Feature table. h=0 -> weather only; h>=1 -> adds lagged generation from day t-h or earlier."""
    X = pd.DataFrame(index=df.index)
    if h >= 1:
        for k in range(h, h + 3):
            X[f"gen_lag{k}"] = df[TARGET].shift(k)
    doy = df.index.dayofyear
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    X["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    X["wind"] = df["wind_speed"]
    X["temp"] = df["temperature"]
    X["hum"] = df["humidity"]
    return X


def with_wind_powers(X):
    X = X.copy()
    X["wind2"], X["wind3"] = X["wind"] ** 2, X["wind"] ** 3
    return X


def metrics(y, p):
    return dict(RMSE=mean_squared_error(y, p) ** 0.5, MAE=mean_absolute_error(y, p), R2=r2_score(y, p))


def tune(model, grid, X, y):
    cv = TimeSeriesSplit(n_splits=4)  # chronological, expanding; no shuffling
    gs = GridSearchCV(model, grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
    gs.fit(X, y)
    return gs.best_estimator_, gs.best_params_


def fit_weather_only(train_df, model="xgb"):
    """Public helper for the climate module: train the weather -> generation model on rows of the golden file."""
    X = build(train_df, 0)
    d = X.join(train_df[TARGET].rename("y")).dropna()
    if model == "xgb":
        m, _ = tune(XGBRegressor(n_estimators=300, subsample=0.8, random_state=SEED, n_jobs=1, tree_method="hist"),
                    XGB_GRID, d.drop(columns="y"), d["y"])
    else:
        m, _ = tune(RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=1),
                    RF_GRID, d.drop(columns="y"), d["y"])
    return m, list(d.drop(columns="y").columns)


def main():
    global OUT
    if len(sys.argv) > 2:
        OUT = Path(sys.argv[2])
    np.random.seed(SEED)
    df = load()
    print(f"Loaded {len(df)} days, {int(df[TARGET].notna().sum())} with generation "
          f"(missing days stay as gaps, never filled).\n")
    rows, preds, importances, chosen = [], [], [], []

    for fold, tr_end, te_start, te_end in FOLDS:
        for h in HORIZONS:
            X = build(df, h)
            data = X.join(df[TARGET].rename("y"))
            valid = data.dropna()
            train = valid[valid.index <= tr_end]
            test = valid[(valid.index >= te_start) & (valid.index <= te_end)]
            if len(test) < 30:
                continue
            feats = [c for c in valid.columns if c != "y"]
            Xtr, ytr, Xte, yte = train[feats], train["y"], test[feats], test["y"]

            cand = {}
            if h >= 1:
                cand["persistence"] = test[f"gen_lag{h}"]
            else:  # monthly climatology learned from the training window only
                clim = train.groupby(train.index.month)["y"].mean()
                cand["climatology"] = pd.Series(test.index.month.map(clim).values, index=test.index)
            lin = LinearRegression().fit(with_wind_powers(Xtr), ytr)
            cand["linear"] = pd.Series(lin.predict(with_wind_powers(Xte)), index=test.index).clip(lower=0)
            rf, rf_p = tune(RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=1), RF_GRID, Xtr, ytr)
            xg, xg_p = tune(XGBRegressor(n_estimators=300, subsample=0.8, random_state=SEED, n_jobs=1,
                                         tree_method="hist"), XGB_GRID, Xtr, ytr)
            cand["random_forest"] = pd.Series(rf.predict(Xte), index=test.index).clip(lower=0)
            cand["xgboost"] = pd.Series(xg.predict(Xte), index=test.index).clip(lower=0)
            chosen.append(dict(fold=fold, horizon_days=h, rf=str(rf_p), xgb=str(xg_p)))

            ref_name = "persistence" if h >= 1 else "climatology"
            ref_rmse = metrics(yte, cand[ref_name])["RMSE"]
            ref_err = (yte - cand[ref_name]).abs()
            for name, p in cand.items():
                m = metrics(yte, p)
                m.update(fold=fold, horizon_days=h, model=name, n_train=len(train), n_test=len(test),
                         skill_vs_reference_pct=100 * (1 - m["RMSE"] / ref_rmse),
                         days_beating_reference_pct=(np.nan if name == ref_name
                                                     else 100 * float(((yte - p).abs() < ref_err).mean())))
                rows.append(m)
                if fold.startswith("holdout"):
                    preds.append(pd.DataFrame({"ts": test.index, "horizon_days": h, "model": name,
                                               "actual_mu": yte.values, "predicted_mu": p.values}))
            if fold.startswith("holdout"):
                for name, mdl in (("random_forest", rf), ("xgboost", xg)):
                    importances.append(pd.DataFrame({"horizon_days": h, "model": name, "feature": feats,
                                                     "importance": mdl.feature_importances_}))
            print(f"{fold:14s} h={h}: train {len(train)} test {len(test)} done")

    res = pd.DataFrame(rows)[["fold", "horizon_days", "model", "n_train", "n_test", "RMSE", "MAE", "R2",
                              "skill_vs_reference_pct", "days_beating_reference_pct"]]
    res.to_csv(OUT / "trees_results_all_folds.csv", index=False)
    pd.concat(preds).to_csv(OUT / "trees_holdout_predictions.csv", index=False)
    pd.concat(importances).to_csv(OUT / "trees_feature_importance.csv", index=False)
    pd.DataFrame(chosen).to_csv(OUT / "trees_chosen_params.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== Headline holdout (2024-01-01 onward), million units per day ===")
    hold = res[res.fold.str.startswith("holdout")].sort_values(["horizon_days", "RMSE"])
    print(hold.drop(columns=["fold", "n_train"]).round(3).to_string(index=False))
    print("\n=== Walk-forward stability: RMSE per fold ===")
    print(res.pivot_table(index=["horizon_days", "model"], columns="fold", values="RMSE").round(2).to_string())

    # figure: holdout RMSE by model and horizon
    order = ["persistence", "climatology", "linear", "random_forest", "xgboost"]
    colors = {"persistence": "#7a7973", "climatology": "#7a7973", "linear": "#eda100",
              "random_forest": "#1baf7a", "xgboost": "#2a78d6"}
    fig, ax = plt.subplots(figsize=(10, 4.6), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    width = 0.17
    for i, m in enumerate(order):
        xs, ys = [], []
        for j, h in enumerate(HORIZONS):
            r = hold[(hold.horizon_days == h) & (hold.model == m)]
            if len(r):
                xs.append(j + (i - 2) * width)
                ys.append(float(r.RMSE.iloc[0]))
        ax.bar(xs, ys, width - 0.02, color=colors[m], label=m)
    ax.set_xticks(range(len(HORIZONS)))
    ax.set_xticklabels(["same-day\n(weather only)", "1 day ahead", "2 days ahead", "3 days ahead"])
    ax.set_ylabel("Test RMSE (MU per day), lower is better")
    ax.set_title("Tamil Nadu wind generation: holdout error by model and horizon", loc="left", fontsize=11)
    ax.set_ylim(0, hold.RMSE.max() * 1.25)
    ax.grid(axis="y", color="#e6e5e1")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, ncol=5, loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "trees_holdout_rmse.png", dpi=150)
    print("\nSaved trees_results_all_folds.csv, trees_holdout_predictions.csv, trees_feature_importance.csv, "
          "trees_chosen_params.csv, trees_holdout_rmse.png")


if __name__ == "__main__":
    main()
