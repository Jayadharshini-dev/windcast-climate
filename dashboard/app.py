"""WindCast Climate dashboard: what Tamil Nadu's wind can deliver, how well we forecast it, and how climate may change it.

    streamlit run dashboard/app.py

Five views for three kinds of reader: Overview (everyone), Public, Researchers, Planners, Data and method.
Everything shown is read from the database; nothing is typed in by hand.
"""
from __future__ import annotations

import calendar
import pathlib
import sys

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from dashboard import queries as q  # noqa: E402
from dashboard import theme  # noqa: E402
from dashboard.theme import BLUE, GRAY, ORANGE  # noqa: E402
from etl.db import connect  # noqa: E402

st.set_page_config(page_title="WindCast Climate", page_icon=None, layout="wide", initial_sidebar_state="collapsed")
theme.inject(st)

MON = [calendar.month_abbr[i] for i in range(1, 13)]
DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
SCEN = {"ssp245": "Moderate emissions (SSP2-4.5)", "ssp585": "High emissions (SSP5-8.5)"}
SCEN_SHORT = {"ssp245": "Moderate", "ssp585": "High"}
SCEN_COLOR = {"ssp245": BLUE, "ssp585": ORANGE}
MODEL_NAME = {"persistence": "Repeat the last known day", "linear": "Linear regression",
              "random_forest": "Random forest", "xgboost": "XGBoost"}
VIEWS = ["Overview", "Public", "Researchers", "Planners", "Data and method"]


# ---------------------------------------------------------------- data access
@st.cache_resource
def get_conn():
    return connect()


def run(fn, *args, **kw):
    try:
        return fn(get_conn(), *args, **kw)
    except Exception:
        get_conn.clear()
        return fn(get_conn(), *args, **kw)


@st.cache_data(ttl=300, show_spinner=False)
def cached(name: str, *args):
    return run(getattr(q, name), *args)


def chart(c, height=300):
    return c.properties(height=height).configure(
        background="transparent",
        axis=dict(gridColor="#8aa0a844", domainColor="#8aa0a8", tickColor="#8aa0a8", labelColor="#6f858d",
                  titleColor="#6f858d", labelFontSize=12, titleFontSize=12, titleFontWeight="normal"),
        view=dict(strokeWidth=0),
        legend=dict(labelColor="#6f858d", titleColor="#6f858d", labelFontSize=12),
        title=dict(color="#6f858d", fontWeight="normal", anchor="start", fontSize=13))


def show(c, height=300):
    st.altair_chart(chart(c, height), use_container_width=True, theme=None)


def fmt(x, d=1, sign=False):
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


# ---------------------------------------------------------------- derived facts (all from the database)
def season_facts():
    clim = cached("monthly_climatology")
    m = clim["generation_mu"].to_numpy()
    share = float((m[5:9] * np.array(DAYS[5:9])).sum() / (m * np.array(DAYS)).sum())
    return clim, share


def skill_table():
    sk = cached("forecast_skill")
    if sk.empty:
        return sk
    ref = sk[sk["model"] == "persistence"].set_index("horizon_days")["rmse"]
    sk = sk.copy()
    sk["skill_pct"] = [100 * (1 - r.rmse / ref[r.horizon_days]) if r.horizon_days in ref.index else np.nan
                       for r in sk.itertuples()]
    return sk


def impact_summary():
    """Ensemble rows of the headline metric, one per scenario and window."""
    ci = cached("climate_impact")
    if ci.empty:
        return None, ci
    ens = ci[(ci["climate_model"] == "ensemble") & (ci["metric"] == "mean_generation_change_pct")].copy()
    up = ci[(ci["climate_model"] == "ensemble") & (ci["metric"] == "models_with_increase_pct")].set_index(
        ["scenario", "target_year"])["value"]
    ens["models_up"] = [round(up.get((r.scenario, r.target_year), np.nan) / 10) if (r.scenario, r.target_year) in up.index
                        else np.nan for r in ens.itertuples()]
    return ens, ci


def verdict(row) -> str:
    """Plain-language reading of one ensemble row; says 'no clear direction' when the model range straddles zero."""
    lo, hi, mid = row.p10, row.p90, row.value
    if lo < 0 < hi:
        return f"No clear direction: the climate models range from {fmt(lo, 1, True)}% to {fmt(hi, 1, True)}%."
    word = "more" if mid > 0 else "less"
    return f"Most models agree on {word} wind power: {fmt(lo, 1, True)}% to {fmt(hi, 1, True)}%."


def empty(title, body):
    st.markdown(theme.note(f"<b>{theme.esc(title)}</b> {theme.esc(body)}"), unsafe_allow_html=True)


# ---------------------------------------------------------------- views
def view_overview():
    clim, share = season_facts()
    peak_m = int(clim["month"][clim["generation_mu"].idxmax()])
    ribbon = theme.ribbon_svg(list(clim["generation_mu"]), MON, 6, 9,
                              f"Jun to Sep: {share * 100:.0f}% of the year's wind power")
    st.markdown(
        f'<div class="hero"><div><h1>Tamil Nadu\'s wind arrives with the monsoon. Will it keep arriving?</h1>'
        f"<p>Five years of measured generation, forecasts we tested on days they had never seen, and what ten climate "
        f"models say about 2030 and 2050.</p></div>{ribbon}</div>", unsafe_allow_html=True)

    sk = skill_table()
    ens, _ = impact_summary()
    cards = [theme.answer(
        "When does the wind deliver?", f"{calendar.month_name[peak_m]} is the strongest month; June to September carries {share * 100:.0f}% of the year.",
        f"Average {fmt(clim['generation_mu'].mean())} MU a day across the year; the weakest months give about "
        f"{fmt(clim['generation_mu'].min())} MU.", 3, "High: measured, five years")]
    best = sk[(sk["horizon_days"] == 1) & (sk["model"] != "persistence")].sort_values("rmse").head(1) if not sk.empty else sk
    if len(best):
        b = best.iloc[0]
        cards.append(theme.answer(
            "Can we forecast it?", f"A day ahead, {MODEL_NAME[b['model']]} is off by {fmt(b['rmse'])} MU on a typical day.",
            f"That is {fmt(b['skill_pct'], 0)}% less error than repeating yesterday's value, tested on {int(b['n_days'])} days "
            f"the model never saw.", 3, "High: held-out test"))
    else:
        cards.append(theme.answer("Can we forecast it?", "Forecast results are not loaded yet.",
                                  "Run python -m etl.rebuild to load them.", 1, "Not available"))
    if ens is not None and len(ens):
        r = ens[(ens["scenario"] == "ssp585") & (ens["target_year"] == 2050)]
        r = r.iloc[0] if len(r) else ens.iloc[-1]
        cards.append(theme.answer(
            "Will climate change it?", f"By {int(r.target_year)} the models do not agree on a direction." if r.p10 < 0 < r.p90
            else f"By {int(r.target_year)} most models agree on a shift.",
            f"Central estimate {fmt(r.value, 1, True)}% with a range of {fmt(r.p10, 1, True)}% to {fmt(r.p90, 1, True)}% "
            f"({SCEN_SHORT[r.scenario].lower()} emissions). {int(r.models_up)} of 10 models show an increase.", 1,
            "Low: models disagree"))
    else:
        cards.append(theme.answer("Will climate change it?", "Climate results are not loaded yet.",
                                  "Run python -m etl.rebuild to load them.", 1, "Not available"))
    st.markdown(f'<div class="answers">{"".join(cards)}</div>', unsafe_allow_html=True)

    st.markdown("## The record")
    df = cached("daily_features")
    lo, hi = df["ts"].min().date(), df["ts"].max().date()
    start, end = st.slider("Show", lo, hi, (lo, hi), format="MMM YYYY", label_visibility="collapsed")
    d = df[(df["ts"].dt.date >= start) & (df["ts"].dt.date <= end)]
    show(alt.Chart(d, title="Daily wind generation, Tamil Nadu (MU per day)").mark_line(color=BLUE, strokeWidth=1.2).encode(
        x=alt.X("ts:T", title=None, axis=alt.Axis(format="%b %Y", tickCount=6)),
        y=alt.Y("generation_mu:Q", title=None), tooltip=[alt.Tooltip("ts:T", title="Day"), alt.Tooltip("generation_mu:Q", title="MU", format=".1f")]), 240)
    st.caption("Blank stretches are days with no reported value. We never fill them in.")


def view_public():
    st.markdown("## Which months are windy?")
    clim, share = season_facts()
    mi = st.select_slider("Month", options=MON, value="Jul", label_visibility="collapsed")
    m = MON.index(mi) + 1
    yearly = cached("monthly_by_year")
    ym = yearly[(yearly["month"] == m) & (yearly["n_days"] >= 15)]
    mean_all = float((clim["generation_mu"] * np.array(DAYS)).sum() / 365)
    v = float(clim.loc[clim["month"] == m, "generation_mu"].iloc[0])
    ratio = v / mean_all
    left, right = st.columns([1, 1.5])
    with left:
        level = "well above" if ratio > 1.3 else "above" if ratio > 1.05 else "close to" if ratio > 0.8 else "below"
        st.markdown(f'<div class="panel"><div class="big">{fmt(v)} MU a day</div>'
                    f'<p class="small">A typical {calendar.month_name[m]} is {level} the yearly average of {fmt(mean_all)} MU a day '
                    f"({ratio:.1f} times).</p>"
                    f'<p class="small">Across the years we have, {calendar.month_name[m]} ranged from {fmt(ym["generation_mu"].min())} '
                    f'to {fmt(ym["generation_mu"].max())} MU a day. Wind varies from year to year.</p></div>', unsafe_allow_html=True)
    with right:
        c = clim.assign(Month=MON, picked=[i + 1 == m for i in range(12)])
        show(alt.Chart(c).mark_bar(cornerRadiusEnd=3, size=18).encode(
            x=alt.X("Month:N", sort=MON, title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y("generation_mu:Q", title="MU per day"),
            color=alt.condition("datum.picked", alt.value(BLUE), alt.value(GRAY)),
            tooltip=["Month", alt.Tooltip("generation_mu:Q", title="MU per day", format=".1f")]), 220)

    st.markdown("## What might change by 2050?")
    ens, ci = impact_summary()
    if ens is None:
        empty("Climate results are not loaded yet.", "Run python -m etl.rebuild.")
    else:
        c1, c2 = st.columns(2)
        sc = c1.segmented_control("Emissions", list(SCEN), default="ssp585", format_func=lambda s: SCEN[s].split(" (")[0]) or "ssp585"
        yr = c2.segmented_control("Year", [2030, 2050], default=2050) or 2050
        r = ens[(ens["scenario"] == sc) & (ens["target_year"] == yr)].iloc[0]
        st.markdown(f'<div class="panel"><p class="verdict">{theme.esc(verdict(r))}</p>'
                    f'<p class="small">Central estimate {fmt(r.value, 1, True)}% change in yearly wind power. '
                    f'{int(r.models_up)} of 10 climate models show an increase, {10 - int(r.models_up)} a decrease.</p></div>', unsafe_allow_html=True)
        pm = ci[(ci["climate_model"] != "ensemble") & (ci["metric"] == "mean_generation_change_pct") &
                (ci["scenario"] == sc) & (ci["target_year"] == yr)]
        band = pd.DataFrame({"lo": [r.p10], "hi": [r.p90]})
        base = alt.Chart(pm)
        show(alt.layer(
            alt.Chart(band).mark_rect(color=SCEN_COLOR[sc], opacity=0.14).encode(x="lo:Q", x2="hi:Q"),
            alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#8aa0a8", strokeDash=[4, 4]).encode(x="x:Q"),
            base.mark_circle(size=90, color=SCEN_COLOR[sc], opacity=0.9).encode(
                x=alt.X("value:Q", title="Change in yearly wind power (%)"), y=alt.Y("climate_model:N", title=None, sort="x"),
                tooltip=[alt.Tooltip("climate_model:N", title="Model"), alt.Tooltip("value:Q", title="Change %", format="+.1f")])), 300)
        st.caption("Each dot is one climate model. The shaded band covers the middle 80% of models. Left of the dashed line means less wind power.")

    st.markdown(theme.note(
        "<b>What this does not tell you.</b> It covers the effect of weather change only. It does not include new wind farms, "
        "grid limits or electricity demand, so it is not a forecast of how much power Tamil Nadu will generate in 2050. "
        "The climate models are global and coarse, and they disagree here."), unsafe_allow_html=True)
    st.markdown("## Where the weather comes from")
    pts = cached("stations")
    pts = pts.assign(label=pts["name"].str.replace("TN_", "").str.replace("_", " / "))
    pts_ch = alt.Chart(pts).mark_circle(size=220, color=BLUE, opacity=0.85).encode(
        x=alt.X("longitude:Q", title="Longitude", scale=alt.Scale(domain=[76.6, 78.2])),
        y=alt.Y("latitude:Q", title="Latitude", scale=alt.Scale(domain=[8.0, 11.0])),
        tooltip=["label", alt.Tooltip("latitude:Q", format=".2f"), alt.Tooltip("longitude:Q", format=".2f")])
    txt = alt.Chart(pts).mark_text(align="left", dx=12, color="#6f858d", fontSize=13).encode(
        x="longitude:Q", y="latitude:Q", text="label:N")
    show(pts_ch + txt, 260)
    st.caption("Weather is the average of three wind-belt points: Muppandal, Palghat gap and Theni. "
               "Generation is the state total reported by the Central Electricity Authority.")


def view_researchers():
    st.markdown("## Climate impact by model")
    ens, ci = impact_summary()
    if ens is None:
        empty("Climate results are not loaded yet.", "Run python -m etl.rebuild.")
    else:
        metrics = sorted(ci["metric"].unique())
        pretty = {"mean_generation_change_pct": "Change in mean generation (%)",
                  "mean_generation_change_pct_wind_only": "Change if only wind shifts (%)",
                  "mean_generation_change_pct_temperature_only": "Change if only temperature shifts (%)",
                  "mean_generation_change_pct_jun_sep": "Change in Jun to Sep generation (%)",
                  "share_days_out_of_range": "Share of days outside the training range",
                  "share_days_temperature_out_of_range": "Share of days with temperature outside training range",
                  "models_with_increase_pct": "Models showing an increase (%)"}
        metric = st.selectbox("Measure", [m for m in metrics if m != "models_with_increase_pct"],
                              format_func=lambda m: pretty.get(m, m))
        d = ci[(ci["metric"] == metric)].copy()
        d["case"] = d["scenario"].map(SCEN_SHORT) + " " + d["target_year"].astype(str)
        order = [f"{s} {y}" for s in ("Moderate", "High") for y in (2030, 2050)]
        pm, em = d[d["climate_model"] != "ensemble"], d[d["climate_model"] == "ensemble"]
        show(alt.layer(
            alt.Chart(em).mark_rule(strokeWidth=6, opacity=0.18, color=BLUE).encode(x=alt.X("case:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)), y="p10:Q", y2="p90:Q"),
            alt.Chart(pm).mark_circle(size=70, color=GRAY, opacity=0.95).encode(
                x=alt.X("case:N", sort=order), y=alt.Y("value:Q", title=pretty.get(metric, metric)),
                tooltip=[alt.Tooltip("climate_model:N", title="Model"), alt.Tooltip("value:Q", format=".3f")]),
            alt.Chart(em).mark_tick(color=BLUE, thickness=3, size=34).encode(x=alt.X("case:N", sort=order), y="value:Q",
                                                                                 tooltip=[alt.Tooltip("value:Q", title="Ensemble mean", format=".3f")])), 340)
        st.caption("Grey dots: the 10 climate models. Blue tick: ensemble mean. Blue band: 10th to 90th percentile across models.")
        drv = ci[(ci["climate_model"] == "ensemble") & ci["metric"].isin(["mean_generation_change_pct_wind_only", "mean_generation_change_pct_temperature_only"])].copy()
        drv["driver"] = drv["metric"].map({"mean_generation_change_pct_wind_only": "Wind only", "mean_generation_change_pct_temperature_only": "Temperature only"})
        drv["case"] = drv["scenario"].map(SCEN_SHORT) + " " + drv["target_year"].astype(str)
        st.markdown("### What drives the change")
        show(alt.Chart(drv).mark_bar(cornerRadiusEnd=3).encode(
            x=alt.X("case:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)), xOffset="driver:N", y=alt.Y("value:Q", title="Change in mean generation (%)"),
            color=alt.Color("driver:N", scale=alt.Scale(domain=["Wind only", "Temperature only"], range=[BLUE, ORANGE]), legend=alt.Legend(title=None, orient="top")),
            tooltip=["case", "driver", alt.Tooltip("value:Q", format="+.2f")]), 260)

    st.markdown("## Monthly change factors")
    c = st.columns(3)
    scen = c[0].selectbox("Scenario", list(SCEN), format_func=SCEN.get)
    year = c[1].selectbox("Window", [2030, 2050], format_func=lambda y: f"{y} ({y - 9} to {y + 10})")
    var = c[2].selectbox("Variable", ["wind_ratio", "temp_delta_c", "humidity_delta_pct"],
                         format_func={"wind_ratio": "Wind speed (ratio)", "temp_delta_c": "Temperature (change, deg C)",
                                      "humidity_delta_pct": "Humidity (change, % points)"}.get)
    mf = cached("monthly_factors", scen, year)
    if not mf.empty:
        mf[var] = mf[var].astype(float)
        mean = mf.groupby("month", as_index=False)[var].mean()
        x = alt.X("month:O", title=None, axis=alt.Axis(labelExpr="['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][datum.value-1]", labelAngle=0))
        zero = 1.0 if var == "wind_ratio" else 0.0
        show(alt.layer(
            alt.Chart(pd.DataFrame({"y": [zero]})).mark_rule(color="#8aa0a8", strokeDash=[4, 4]).encode(y="y:Q"),
            alt.Chart(mf).mark_line(color=GRAY, strokeWidth=1, opacity=0.8).encode(x=x, y=alt.Y(f"{var}:Q", title=None, scale=alt.Scale(zero=False)), detail="climate_model:N",
                                                                                    tooltip=["climate_model", "month", alt.Tooltip(f"{var}:Q", format=".3f")]),
            alt.Chart(mean).mark_line(color=BLUE, strokeWidth=3, point=alt.OverlayMarkDef(size=45, filled=True)).encode(x=x, y=f"{var}:Q", tooltip=["month", alt.Tooltip(f"{var}:Q", title="Model average", format=".3f")])), 320)
        st.caption("Grey: each of the 10 models. Blue: their average. Dashed: no change. Humidity plus its change is clipped to 0 to 100% when applied.")

    st.markdown("## Forecast models on unseen days")
    sk = skill_table()
    if sk.empty:
        empty("Forecast results are not loaded yet.", "Run python -m etl.rebuild.")
    else:
        t = sk.assign(Model=sk["model"].map(MODEL_NAME), Horizon=sk["horizon_days"].map(lambda h: f"{h} day{'s' if h > 1 else ''} ahead"))
        show(alt.Chart(t).mark_bar(cornerRadiusEnd=3).encode(
            x=alt.X("Horizon:N", title=None, axis=alt.Axis(labelAngle=0)), xOffset=alt.XOffset("Model:N", sort=list(MODEL_NAME.values())),
            y=alt.Y("rmse:Q", title="Error (RMSE, MU per day), lower is better"),
            color=alt.Color("Model:N", sort=list(MODEL_NAME.values()), scale=alt.Scale(domain=list(MODEL_NAME.values()), range=[GRAY, "#e0a21b", "#1baf7a", BLUE]), legend=alt.Legend(title=None, orient="top")),
            tooltip=["Model", "Horizon", alt.Tooltip("rmse:Q", title="RMSE", format=".2f"), alt.Tooltip("mae:Q", title="MAE", format=".2f"), alt.Tooltip("skill_pct:Q", title="Skill vs repeat-last-day %", format=".0f")]), 300)
        st.dataframe(t[["Horizon", "Model", "n_days", "rmse", "mae", "skill_pct"]].rename(columns={"n_days": "Test days", "rmse": "RMSE", "mae": "MAE", "skill_pct": "Skill vs persistence (%)"})
                     .style.format({"RMSE": "{:.2f}", "MAE": "{:.2f}", "Skill vs persistence (%)": "{:.0f}"}), hide_index=True, use_container_width=True)
        st.caption("Test period: 2024 onward, trained on earlier years only. Skill = 1 minus (model error divided by the error of repeating the last known day).")

    st.markdown("## Download the data")
    d1, d2, d3, d4 = st.columns(4)
    d1.download_button("Daily features", cached("daily_features").to_csv(index=False), "daily_features.csv", "text/csv")
    d2.download_button("Change factors", cached("change_factors_download").to_csv(index=False), "change_factors.csv", "text/csv")
    if ens is not None:
        d3.download_button("Climate impact", ci.to_csv(index=False), "climate_impact.csv", "text/csv")
    if not sk.empty:
        d4.download_button("Forecast skill", sk.to_csv(index=False), "forecast_skill.csv", "text/csv")


def view_planners():
    st.markdown("## Forecast against what happened")
    runs = cached("runs_available")
    if runs.empty:
        empty("Forecast results are not loaded yet.", "Run python -m etl.rebuild.")
    else:
        c = st.columns(2)
        h = c[0].segmented_control("Days ahead", sorted(runs["horizon_days"].unique()), default=1, format_func=lambda x: f"{x} day{'s' if x > 1 else ''}") or 1
        models = [m for m in MODEL_NAME if m in set(runs[runs["horizon_days"] == h]["model"])]
        model = c[1].selectbox("Model", models, index=models.index("xgboost") if "xgboost" in models else 0, format_func=MODEL_NAME.get)
        rid = int(runs[(runs["horizon_days"] == h) & (runs["model"] == model)]["run_id"].iloc[0])
        d = cached("predictions_vs_actual", rid).dropna(subset=["actual"])
        err = (d["predicted_value"] - d["actual"]).abs()
        q80 = float(err.quantile(0.8))
        d = d.assign(lo=(d["predicted_value"] - q80).clip(lower=0), hi=d["predicted_value"] + q80)
        long = d.melt("ts", ["actual", "predicted_value"], var_name="series", value_name="MU")
        long["series"] = long["series"].map({"actual": "Actual", "predicted_value": "Forecast"})
        show(alt.layer(
            alt.Chart(d).mark_area(color=ORANGE, opacity=0.14).encode(x="ts:T", y="lo:Q", y2="hi:Q"),
            alt.Chart(long).mark_line(strokeWidth=1.4).encode(
                x=alt.X("ts:T", title=None, axis=alt.Axis(format="%b %Y", tickCount=6)), y=alt.Y("MU:Q", title="MU per day"),
                color=alt.Color("series:N", scale=alt.Scale(domain=["Actual", "Forecast"], range=[BLUE, ORANGE]), legend=alt.Legend(title=None, orient="top")),
                tooltip=[alt.Tooltip("ts:T", title="Day"), "series", alt.Tooltip("MU:Q", format=".1f")])), 320)
        st.markdown(f'<div class="panel"><p class="verdict">On 8 days out of 10, the actual value fell within {fmt(q80)} MU of the forecast.</p>'
                    f'<p class="small">Shaded band = forecast plus or minus {fmt(q80)} MU, the size of the error on 80% of the {len(d)} test days. '
                    f"Average generation is {fmt(d['actual'].mean())} MU a day, so that is about {q80 / d['actual'].mean() * 100:.0f}% of a typical day.</p></div>", unsafe_allow_html=True)
        st.markdown(theme.note("<b>This is a back-test, not tomorrow's forecast.</b> The models use the weather that actually occurred. "
                               "A live forecast would need a weather forecast feed, which this project does not have."), unsafe_allow_html=True)

    st.markdown("## Planning calendar")
    yearly = cached("monthly_by_year")
    yearly = yearly[yearly["n_days"] >= 15]
    cal = yearly.groupby("month")["generation_mu"].agg(["mean", "min", "max"]).reset_index()
    cal["Month"] = [MON[int(m) - 1] for m in cal["month"]]
    tert = cal["mean"].quantile([1 / 3, 2 / 3]).tolist()
    cal["level"] = ["High" if v > tert[1] else "Medium" if v > tert[0] else "Low" for v in cal["mean"]]
    show(alt.layer(
        alt.Chart(cal).mark_rule(strokeWidth=2, color=GRAY).encode(x=alt.X("Month:N", sort=MON, title=None, axis=alt.Axis(labelAngle=0)), y="min:Q", y2="max:Q"),
        alt.Chart(cal).mark_bar(cornerRadiusEnd=3, size=20, opacity=0.9).encode(
            x=alt.X("Month:N", sort=MON), y=alt.Y("mean:Q", title="MU per day"),
            color=alt.Color("level:N", scale=alt.Scale(domain=["Low", "Medium", "High"], range=["#b8c7cc", "#6aa7d8", BLUE]), legend=alt.Legend(title="Season", orient="top")),
            tooltip=["Month", alt.Tooltip("mean:Q", title="Average", format=".1f"), alt.Tooltip("min:Q", title="Lowest year", format=".1f"), alt.Tooltip("max:Q", title="Highest year", format=".1f")])), 300)
    st.caption("Bar: average over the years. Grey line: the lowest and highest year for that month. A high month can still have a weak year.")

    st.markdown("## Climate risk by scenario")
    ens, ci = impact_summary()
    if ens is None:
        empty("Climate results are not loaded yet.", "Run python -m etl.rebuild.")
        return
    e = ens.assign(Case=ens["scenario"].map(SCEN_SHORT) + " emissions, " + ens["target_year"].astype(str))
    show(alt.layer(
        alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#8aa0a8", strokeDash=[4, 4]).encode(x="x:Q"),
        alt.Chart(e).mark_rule(strokeWidth=5, opacity=0.35).encode(y=alt.Y("Case:N", sort=None, title=None), x=alt.X("p10:Q", title="Change in yearly wind power (%)"), x2="p90:Q",
                                                                    color=alt.Color("scenario:N", scale=alt.Scale(domain=list(SCEN), range=[BLUE, ORANGE]), legend=None)),
        alt.Chart(e).mark_point(filled=True, size=110, opacity=1).encode(y=alt.Y("Case:N", sort=None), x="value:Q",
                                                                          color=alt.Color("scenario:N", scale=alt.Scale(domain=list(SCEN), range=[BLUE, ORANGE]), legend=None),
                                                                          tooltip=["Case", alt.Tooltip("value:Q", title="Central", format="+.1f"), alt.Tooltip("p10:Q", format="+.1f"), alt.Tooltip("p90:Q", format="+.1f")])), 230)
    tab = e[["Case", "value", "p10", "p90", "models_up"]].rename(columns={"value": "Central (%)", "p10": "Low end (%)", "p90": "High end (%)", "models_up": "Models showing an increase (of 10)"})
    st.dataframe(tab.style.format({"Central (%)": "{:+.1f}", "Low end (%)": "{:+.1f}", "High end (%)": "{:+.1f}", "Models showing an increase (of 10)": "{:.0f}"}), hide_index=True, use_container_width=True)
    lines = ["# WindCast Climate: one-page summary", "", f"Source: database snapshot; Tamil Nadu wind generation.", ""]
    clim, share = season_facts()
    lines += [f"- Average generation {fmt(clim['generation_mu'].mean())} MU/day; June to September carries {share * 100:.0f}% of the year.", ""]
    for r in e.itertuples():
        lines.append(f"- {r.Case}: central {fmt(r.value, 1, True)}%, range {fmt(r.p10, 1, True)}% to {fmt(r.p90, 1, True)}%, {int(r.models_up)} of 10 models up. {verdict(r)}")
    lines += ["", "Limits: weather effect only (no new capacity, grid or demand); 10 climate models, coarse resolution; 5 years of generation history."]
    st.download_button("Download this summary", "\n".join(lines), "windcast_summary.md", "text/markdown")


def view_data():
    st.markdown("## Data and method")
    dq = run(q.data_quality)
    st.markdown(f'<div class="panel"><p class="verdict">{dq["days"]:,} days, {dq["first_day"]:%d %b %Y} to {dq["last_day"]:%d %b %Y}</p>'
                f'<p class="small">{dq["days_with_generation"]:,} days have a generation value. The other {dq["days"] - dq["days_with_generation"]} are blank: '
                f"22 days missing from the official file and 30 weather-only days around its edges. Blanks are never filled in. "
                f'All {dq["days_all_3_points"]:,} days have weather from all three points.</p></div>', unsafe_allow_html=True)
    st.markdown("### What is in the database")
    st.dataframe(cached("table_counts"), hide_index=True, use_container_width=True)
    st.markdown("### Sources")
    st.markdown(
        "- **Generation:** Central Electricity Authority, daily wind generation for Tamil Nadu, in million units (MU) per day.\n"
        "- **Weather:** ERA5 reanalysis through Open-Meteo. Wind at 10 m in m/s, temperature in deg C, humidity in %. Averaged over three wind-belt points.\n"
        "- **Climate:** NASA NEX-GDDP-CMIP6 through Google Earth Engine. 10 models, two emissions scenarios, windows 2021 to 2040 and 2041 to 2060, against 1995 to 2014.")
    st.markdown("### Method in four lines")
    st.markdown(
        "1. A model learns how wind, temperature, humidity and the time of year relate to generation, using only past days.\n"
        "2. We test it on 2024 onward, days it never saw, and compare with simply repeating the last known day.\n"
        "3. Each climate model gives monthly changes in wind, temperature and humidity. We apply them to today's weather and ask the model what generation would be.\n"
        "4. We keep every climate model's answer. The spread between them is the uncertainty, and we show it.")
    st.markdown("### Limits")
    st.markdown(
        "- Weather effect only. New wind farms, grid limits and demand are not modelled.\n"
        "- Five years of generation is a short record.\n"
        "- Tree models cannot go beyond the range of generation values they have seen, so they may understate large shifts. A linear model is the opposite risk. We report both.\n"
        "- Climate models and ERA5 differ in absolute wind speed, so only changes are used, never levels.\n"
        "- The wind-belt points are approximate.")


# ---------------------------------------------------------------- shell
st.markdown('<div class="brand"><b>WindCast Climate</b><span>Tamil Nadu wind power, today and under a changing climate</span></div>', unsafe_allow_html=True)
view = st.segmented_control("View", VIEWS, default="Overview", key="view", label_visibility="collapsed") or "Overview"
try:
    {"Overview": view_overview, "Public": view_public, "Researchers": view_researchers, "Planners": view_planners,
     "Data and method": view_data}[view]()
except Exception as exc:  # database down or not rebuilt
    st.error("This page could not load its data.")
    st.code(f"{type(exc).__name__}: {exc}")
    st.markdown("Start the database with `docker compose up -d`, then fill it with `python -m etl.rebuild`.")
