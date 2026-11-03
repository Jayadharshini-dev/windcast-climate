"""Sudden changes and extreme days, for the Researchers view.

Unlike the rest of the dashboard these results are read from the result files written by
`python -m models.ramps` and `python -m climate.extremes_impact` (and `python -m climate.daily_checks`),
not from the database. Re-run those commands to refresh them.
"""
from __future__ import annotations

import calendar
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
FILES = {
    "ramps": ROOT / "models" / "ramps_scores.csv",
    "ramps_month": ROOT / "models" / "ramps_by_month.csv",
    "impact": ROOT / "climate" / "extremes_impact.csv",
    "impact_linear": ROOT / "climate" / "extremes_impact_linear.csv",
    "baseline": ROOT / "climate" / "extremes_baseline_reference.csv",
    "baseline_linear": ROOT / "climate" / "extremes_baseline_reference_linear.csv",
    "daily": ROOT / "climate" / "variability_test.csv",
}
METRICS = {
    "ramp_day_share_pct": "Days with a sudden change (%)",
    "low_day_share_pct": "Very low generation days (%)",
    "high_day_share_pct": "Very high generation days (%)",
    "spells_of_5plus_low_days_per_year": "Low spells of 5 days or more, per year",
    "longest_low_spell_days": "Longest low spell (days)",
}
MODELS = {"persistence": "Repeat the last known day", "linear": "Linear regression",
          "random_forest": "Random forest", "xgboost": "XGBoost"}


def load(name: str, files: dict | None = None) -> pd.DataFrame | None:
    """One result file as a frame, or None when it has not been produced yet."""
    p = (files or FILES)[name]
    return pd.read_csv(p) if pathlib.Path(p).exists() else None


def ramp_hit_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Share of real ramps that each model flagged, by horizon (reference model 'climatology' left out)."""
    d = df[df["model"].isin(MODELS)].copy()
    d["Model"] = d["model"].map(MODELS)
    d["Horizon"] = d["horizon_days"].map(lambda h: "same day" if h == 0 else f"{h} day{'s' if h > 1 else ''} ahead")
    d["Hit rate (%)"] = 100 * d["hit_rate"]
    d["False alarms (%)"] = 100 * d["false_alarm_ratio"]
    return d


def ramp_months(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["Month"] = d["month"].map(lambda m: calendar.month_abbr[int(m)])
    return d


def impact_rows(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    d = df[df["metric"] == metric].copy()
    sc = {"ssp245": "Moderate", "ssp585": "High"}
    d["case"] = d["scenario"].map(sc) + " " + d["target_year"].astype(int).astype(str)
    return d


CASES = [f"{s} {y}" for s in ("Moderate", "High") for y in (2030, 2050)]


def reading(d: pd.DataFrame, base: pd.DataFrame) -> str:
    """Plain-language reading of the ensemble rows, written from the numbers so it cannot go stale."""
    e = d[d["climate_model"] == "ensemble"]

    def rng(metric):
        x = e[e["metric"] == metric]
        return x["value"].min(), x["value"].max(), x

    lo, hi, _ = rng("ramp_day_share_pct")
    today = float(base.iloc[0]["ramp_day_share_pct"])
    lowlo, lowhi, lx = rng("low_day_share_pct")
    low_today = float(base.iloc[0]["low_day_share_pct"])
    n_clear = int(((lx["p90"] < low_today) | (lx["p10"] > low_today)).sum())
    return (f"**Reading this.** Sudden-change days are {lo:.1f} to {hi:.1f}% in the four cases against {today:.1f}% today, so ramp frequency is not detectably changed. "
            f"Very low generation days are {lowlo:.1f} to {lowhi:.1f}% against {low_today:.1f}% today; the model range excludes today's level in {n_clear} of the four cases, "
            "and the answer depends on which weather-to-power model is used. Because the method keeps today's day-to-day weather and shifts only monthly averages, "
            "it cannot show changes in the variability of the weather itself.")


def daily_caption(daily: pd.DataFrame) -> str:
    n = len(daily)
    down = int((daily["low_day_share_pct_future"] < daily["low_day_share_pct_hist"]).sum())
    r = daily["ramp_day_share_pct_future"]
    return (f"{n} models, high-emissions scenario, 2041 to 2060, using each model's own daily output. Sudden-change days are {r.min():.1f} to {r.max():.1f}% in the future. "
            f"Very low days fall in {down} of {n} models and rise in {n - down}, so that result is not settled.")


def render(st, alt, show, empty, blue="#2a78d6", gray="#9aa3a7", files: dict | None = None) -> None:
    """Draw the section. `show` and `empty` are the dashboard's own chart and notice helpers."""
    st.markdown("## Sudden changes and extreme days")
    st.caption("Read from the result files of the ramp and extreme-day analysis, not from the database. "
               "Refresh them with python -m models.ramps and python -m climate.extremes_impact.")
    ramps, months = load("ramps", files), load("ramps_month", files)
    if ramps is None or months is None:
        empty("Ramp results are not available.", "Run python -m models.ramps.")
    else:
        st.markdown("### How many sudden changes do the forecasts catch?")
        t = ramp_hit_rates(ramps)
        order = list(MODELS.values())
        show(alt.Chart(t).mark_bar(cornerRadiusEnd=3).encode(
            x=alt.X("Horizon:N", title=None, sort=["same day", "1 day ahead", "2 days ahead", "3 days ahead"], axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("Model:N", sort=order), y=alt.Y("Hit rate (%):Q", title="Real ramps that were flagged (%)", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("Model:N", sort=order, scale=alt.Scale(domain=order, range=[gray, "#e0a21b", "#1baf7a", blue]), legend=alt.Legend(title=None, orient="top")),
            tooltip=["Model", "Horizon", alt.Tooltip("Hit rate (%):Q", format=".0f"), alt.Tooltip("False alarms (%):Q", format=".0f"),
                     alt.Tooltip("n_real_ramps:Q", title="Real ramps")]), 300)
        st.caption("A sudden change is a change in generation at least as large as the top 10% of changes in the training years. "
                   "Repeating the last known day never flags one. With only 23 to 26 real events per case, differences between the learning models are not reliable.")
        m = ramp_months(months)
        st.markdown("### When do they happen?")
        show(alt.Chart(m).mark_bar(cornerRadiusEnd=3, color=blue).encode(
            x=alt.X("Month:N", sort=[calendar.month_abbr[i] for i in range(1, 13)], title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("ramp_share_pct:Q", title="Days with a one-day sudden change (%)"),
            tooltip=["Month", alt.Tooltip("ramp_share_pct:Q", title="Share of days (%)", format=".1f")]), 240)

    imp = load("impact", files)
    if imp is None:
        empty("Extreme-day results are not available.", "Run python -m climate.extremes_impact.")
        return
    st.markdown("### Do climate models change how often extreme days occur?")
    c = st.columns(2)
    metric = c[0].selectbox("Measure", list(METRICS), format_func=METRICS.get, key="ext_metric")
    which = c[1].radio("Weather-to-power model", ["XGBoost", "Linear regression"], horizontal=True, key="ext_model")
    lin = which != "XGBoost"
    d = load("impact_linear" if lin else "impact", files)
    base = load("baseline_linear" if lin else "baseline", files)
    if d is None or base is None:
        empty("This version of the results is not available.", "Run python -m climate.extremes_impact --model linear.")
        return
    d_all = d
    d = impact_rows(d, metric)
    pm, em = d[d["climate_model"] != "ensemble"], d[d["climate_model"] == "ensemble"]
    today = float(base.iloc[0][metric])
    show(alt.layer(
        alt.Chart(pd.DataFrame({"y": [today]})).mark_rule(color="#8aa0a8", strokeDash=[4, 4]).encode(y="y:Q"),
        alt.Chart(em).mark_rule(strokeWidth=6, opacity=0.18, color=blue).encode(x=alt.X("case:N", sort=CASES, title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y("p10:Q", title=METRICS[metric], scale=alt.Scale(zero=False)), y2="p90:Q"),
        alt.Chart(pm).mark_circle(size=70, color=gray, opacity=0.95).encode(x=alt.X("case:N", sort=CASES), y="value:Q",
                                                                            tooltip=[alt.Tooltip("climate_model:N", title="Model"), alt.Tooltip("value:Q", format=".2f")]),
        alt.Chart(em).mark_tick(color=blue, thickness=3, size=34).encode(x=alt.X("case:N", sort=CASES), y="value:Q", tooltip=[alt.Tooltip("value:Q", title="Ensemble mean", format=".2f")])), 340)
    st.caption("Grey dots: the 10 climate models. Blue tick: ensemble mean. Blue band: 10th to 90th percentile across models. Dashed line: today.")
    st.markdown(reading(d_all, base))
    daily = load("daily", files)
    if daily is not None:
        st.markdown("### Check with daily climate-model output")
        t = pd.DataFrame({"Climate model": daily["model"],
                          "Sudden-change days (%) today to future": [f"{a:.1f} to {b:.1f}" for a, b in zip(daily["ramp_day_share_pct_hist"], daily["ramp_day_share_pct_future"])],
                          "Very low days (%) today to future": [f"{a:.1f} to {b:.1f}" for a, b in zip(daily["low_day_share_pct_hist"], daily["low_day_share_pct_future"])],
                          "Very high days (%) today to future": [f"{a:.1f} to {b:.1f}" for a, b in zip(daily["high_day_share_pct_hist"], daily["high_day_share_pct_future"])]})
        st.dataframe(t, hide_index=True, use_container_width=True)
        st.caption(daily_caption(daily))
    dl = st.columns(3)
    dl[0].download_button("Extremes, XGBoost", imp.to_csv(index=False), "extremes_impact.csv", "text/csv")
    if ramps is not None:
        dl[1].download_button("Ramp scores", ramps.to_csv(index=False), "ramps_scores.csv", "text/csv")
    if daily is not None:
        dl[2].download_button("Daily-output check", daily.to_csv(index=False), "variability_test.csv", "text/csv")
