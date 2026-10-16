"""Climate-shift analysis for Tamil Nadu wind generation (delta-change method).

Run from the repo root:   python -m climate.climate_impact
Optional:                 python -m climate.climate_impact --write-db     (also upserts into table climate_impact)

What it does, in plain terms
  1. Trains the WEATHER-ONLY model (wind, temperature, humidity + day-of-year; no lagged generation) on every day
     with a generation value. Future climate has no "yesterday's generation", so this is the model that can be
     pushed into the future. Primary model = XGBoost; a linear model is the sensitivity check.
  2. Takes the real ERA5 weather record as the "today" climate and, for each of the 10 climate models, each
     scenario (ssp245, ssp585) and each window (2030 = 2021-2040, 2050 = 2041-2060), applies that model's MONTHLY
     change factors:  wind x wind_ratio,  temperature + temp_delta_c,  humidity + humidity_delta_pct (clipped 0-100).
  3. Predicts generation for the shifted weather and compares with the prediction for the unshifted weather.
  4. Driver split: the same change with ONLY wind shifted, or ONLY temperature shifted (humidity is negligible).
     Shift diagnostic: share of days where wind, temperature or humidity falls outside the 1st-99th percentile of
     the training days. The model has never seen those conditions, so its answer there is an extrapolation.
  5. Spread across the 10 climate models (p10, p90) is the uncertainty. Models are never averaged before the
     model is run.

Honest limits (state these in the paper)
  - Tree models cannot predict beyond the range of generation values they saw, so they under-react to large shifts;
    the linear model is the opposite risk. We report both.
  - Only about 5 years of generation exist; it is a short training record.
  - Capacity additions up to 2030/2050 are NOT modelled: this is the effect of weather change alone on today's fleet.
  - ERA5 wind at 10 m is low versus turbine hub height; the delta-change method relies on the change, not the level.

Outputs (in climate/): climate_impact.csv (table shape frozen at G3), climate_impact_sensitivity_linear.csv,
climate_baseline_reference.csv, climate_impact_by_model.png.
"""
import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from models.data import REPO_ROOT, load_daily
from models.trees import SEED, TARGET, build, fit_weather_only, with_wind_powers

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent
FACTORS_CSV = REPO_ROOT / "data" / "golden" / "change_factors_region_monthly.csv"
REGION = "Tamil Nadu"
BOUND_PCT = (1, 99)  # training range used by the shift diagnostic
FEATS = ["wind_speed", "temperature", "humidity"]


def load_factors():
    """Monthly change factors: database view if reachable, else the CSV Pavan committed."""
    try:
        from etl.db import connect

        with connect() as conn:
            rows = conn.execute("SELECT * FROM v_change_factors_region WHERE month IS NOT NULL").fetchall()
            cols = [d.name for d in conn.execute("SELECT * FROM v_change_factors_region LIMIT 0").description]
        f = pd.DataFrame(rows, columns=cols)
        print("[factors] source: database view v_change_factors_region")
    except Exception as e:
        print(f"[factors] database not available ({type(e).__name__}); using {FACTORS_CSV.name}")
        f = pd.read_csv(FACTORS_CSV)
    f = f[f["region"] == REGION].dropna(subset=["month"]).copy()
    f["month"] = f["month"].astype(int)
    f["target_year"] = f["target_year"].astype(int)
    return f


def shifted_weather(base, fac):
    """Apply one climate model's 12 monthly factors to the baseline daily weather."""
    f = fac.set_index("month")
    m = base.index.month
    w = base.copy()
    w["wind_speed"] = base["wind_speed"] * f["wind_ratio"].reindex(m).values
    w["temperature"] = base["temperature"] + f["temp_delta_c"].reindex(m).values
    w["humidity"] = (base["humidity"] + f["humidity_delta_pct"].reindex(m).values).clip(0, 100)
    return w


def to_features(w):
    X = build(w.assign(**{TARGET: np.nan}), 0)
    return X


def fit_models(train):
    """Primary = tuned XGBoost (weather only). Sensitivity = linear with wind^2, wind^3."""
    xgb, cols = fit_weather_only(train, "xgb")
    d = build(train, 0).join(train[TARGET].rename("y")).dropna()
    lin = LinearRegression().fit(with_wind_powers(d[cols]), d["y"])

    def p_xgb(X):
        return np.clip(xgb.predict(X[cols]), 0, None)

    def p_lin(X):
        return np.clip(lin.predict(with_wind_powers(X[cols])), 0, None)

    return {"xgboost": p_xgb, "linear": p_lin}


def run(write_db=False):
    np.random.seed(SEED)
    df = load_daily()
    weather = df[FEATS].dropna()
    train = df.dropna(subset=FEATS + [TARGET])
    print(f"[data] {len(weather)} baseline weather days, {len(train)} training days with generation "
          f"({train.index.min().date()} to {train.index.max().date()})")
    models = fit_models(train)

    lo = train[FEATS].quantile(BOUND_PCT[0] / 100)
    hi = train[FEATS].quantile(BOUND_PCT[1] / 100)

    def out_of_range(w):
        return ((w[FEATS] < lo) | (w[FEATS] > hi)).any(axis=1)

    fac_all = load_factors()
    jjas = weather.index.month.isin([6, 7, 8, 9])
    Xb = to_features(weather)

    ref, rows = [], {"xgboost": [], "linear": []}
    for name, pred in models.items():
        base_pred = pred(Xb)
        ref.append(dict(model=name, baseline_mean_generation_mu=base_pred.mean(),
                        baseline_mean_generation_mu_jun_sep=base_pred[jjas].mean(),
                        baseline_share_days_out_of_range=out_of_range(weather).mean()))
        for (scen, year), g in fac_all.groupby(["scenario", "target_year"]):
            per_model = []
            for cm, fac in g.groupby("climate_model"):
                w = shifted_weather(weather, fac)
                p = pred(to_features(w))
                oor = out_of_range(w)
                r = dict(region=REGION, scenario=scen, target_year=int(year), climate_model=cm)
                def only(col):  # change in generation if ONLY this variable shifts (which driver matters?)
                    x = weather.copy()
                    x[col] = w[col]
                    return 100 * (pred(to_features(x)).mean() / base_pred.mean() - 1)

                vals = {
                    "mean_generation_change_pct": 100 * (p.mean() / base_pred.mean() - 1),
                    "mean_generation_change_pct_wind_only": only("wind_speed"),
                    "mean_generation_change_pct_temperature_only": only("temperature"),
                    "mean_generation_change_pct_jun_sep": 100 * (p[jjas].mean() / base_pred[jjas].mean() - 1),
                    "share_days_out_of_range": float(oor.mean()),
                    "share_days_temperature_out_of_range": float(
                        ((w["temperature"] < lo["temperature"]) | (w["temperature"] > hi["temperature"])).mean()),
                }
                for k, v in vals.items():
                    rows[name].append({**r, "metric": k, "value": v, "p10": np.nan, "p90": np.nan})
                per_model.append(vals)
            pm = pd.DataFrame(per_model)
            for k in pm.columns:  # ensemble summary: mean over models, p10 / p90 across models
                rows[name].append(dict(region=REGION, scenario=scen, target_year=int(year), climate_model="ensemble",
                                       metric=k, value=float(pm[k].mean()),
                                       p10=float(np.percentile(pm[k], 10)), p90=float(np.percentile(pm[k], 90))))
            rows[name].append(dict(region=REGION, scenario=scen, target_year=int(year), climate_model="ensemble",
                                   metric="models_with_increase_pct",
                                   value=float(100 * (pm["mean_generation_change_pct"] > 0).mean()),
                                   p10=np.nan, p90=np.nan))

    cols = ["region", "scenario", "target_year", "climate_model", "metric", "value", "p10", "p90"]
    prim = pd.DataFrame(rows["xgboost"])[cols]
    prim.to_csv(OUT / "climate_impact.csv", index=False)
    pd.DataFrame(rows["linear"])[cols].to_csv(OUT / "climate_impact_sensitivity_linear.csv", index=False)
    pd.DataFrame(ref).to_csv(OUT / "climate_baseline_reference.csv", index=False)

    pd.set_option("display.width", 200)
    ens = prim[prim.climate_model == "ensemble"].pivot_table(
        index=["scenario", "target_year"], columns="metric", values="value")
    p10 = prim[prim.climate_model == "ensemble"].pivot_table(
        index=["scenario", "target_year"], columns="metric", values="p10")
    p90 = prim[prim.climate_model == "ensemble"].pivot_table(
        index=["scenario", "target_year"], columns="metric", values="p90")
    print("\nBaseline reference:\n", pd.DataFrame(ref).round(3).to_string(index=False))
    print("\nENSEMBLE (XGBoost weather-only): mean generation change %, p10..p90 across 10 climate models")
    for idx in ens.index:
        print(f"  {idx[0]} {idx[1]}: {ens.loc[idx, 'mean_generation_change_pct']:+.2f}%  "
              f"[{p10.loc[idx, 'mean_generation_change_pct']:+.2f}, {p90.loc[idx, 'mean_generation_change_pct']:+.2f}]  "
              f"Jun-Sep {ens.loc[idx, 'mean_generation_change_pct_jun_sep']:+.2f}%  "
              f"days out of range {100 * ens.loc[idx, 'share_days_out_of_range']:.1f}%  "
              f"models up {ens.loc[idx, 'models_with_increase_pct']:.0f}%")
    lin = pd.DataFrame(rows["linear"])
    le = lin[(lin.climate_model == "ensemble") & (lin.metric == "mean_generation_change_pct")]
    print("\nSENSITIVITY (linear weather-only), mean generation change %:")
    for _, r in le.iterrows():
        print(f"  {r.scenario} {r.target_year}: {r.value:+.2f}% [{r.p10:+.2f}, {r.p90:+.2f}]")

    plot(prim)
    if write_db:
        upsert(prim)
    return prim


def plot(prim):
    d = prim[prim.metric == "mean_generation_change_pct"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.2), sharey=True, facecolor="#fcfcfb")
    for ax, (scen, year) in zip(axes, [("ssp245", 2030), ("ssp245", 2050), ("ssp585", 2030), ("ssp585", 2050)]):
        s = d[(d.scenario == scen) & (d.target_year == year)]
        m = s[s.climate_model != "ensemble"].sort_values("climate_model", ascending=False)  # same row order in every panel
        e = s[s.climate_model == "ensemble"].iloc[0]
        ax.set_facecolor("#fcfcfb")
        ax.scatter(m["value"], range(len(m)), color="#2a78d6", s=28, zorder=3)
        ax.axvline(0, color="#7a7973", lw=0.8)
        ax.axvspan(e.p10, e.p90, color="#2a78d6", alpha=0.12, label="p10-p90 across models")
        ax.axvline(e.value, color="#eb6834", lw=1.6, label="ensemble mean")
        ax.set_yticks(range(len(m)))
        ax.set_yticklabels(m["climate_model"], fontsize=7)
        ax.set_title(f"{scen} {year}", loc="left", fontsize=10)
        ax.set_xlabel("change in mean generation (%)")
        ax.grid(axis="x", color="#e6e5e1")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle("Tamil Nadu wind generation: projected change by climate model (weather-only XGBoost)",
                 x=0.01, ha="left", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT / "climate_impact_by_model.png", dpi=150)


def upsert(prim):
    from etl.db import connect

    sql = ("INSERT INTO climate_impact (region, scenario, target_year, climate_model, metric, value, p10, p90) "
           "VALUES (%s,%s,%s,%s,%s,%s,%s,%s) "
           "ON CONFLICT (region, scenario, target_year, climate_model, metric) "
           "DO UPDATE SET value = EXCLUDED.value, p10 = EXCLUDED.p10, p90 = EXCLUDED.p90")
    recs = [(r.region, r.scenario, int(r.target_year), r.climate_model, r.metric, float(r.value),
             None if pd.isna(r.p10) else float(r.p10), None if pd.isna(r.p90) else float(r.p90))
            for r in prim.itertuples()]
    with connect() as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, recs)
        conn.commit()
    print(f"[db] upserted {len(recs)} rows into climate_impact")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-db", action="store_true", help="also upsert results into climate_impact")
    run(write_db=ap.parse_args().write_db)
