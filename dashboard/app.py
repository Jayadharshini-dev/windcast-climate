"""WindCast Climate dashboard (read-only view of the database).

    streamlit run dashboard/app.py

Pages: Overview, Climate factors, Forecasts, Climate impact, Data status.
Pages that depend on the model team's tables (forecasts, climate impact) show an explanatory
empty state until those tables are filled.
"""
from __future__ import annotations

import sys
import pathlib

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from dashboard import queries as q  # noqa: E402
from etl.db import connect  # noqa: E402

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#b4b4b0"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SCEN_LABEL = {"ssp245": "SSP2-4.5 (middle path)", "ssp585": "SSP5-8.5 (high emissions)"}

st.set_page_config(page_title="WindCast Climate", page_icon=None, layout="wide")


@st.cache_resource
def get_conn():
    return connect()


def run(fn, *args, **kw):
    """Run a query on a fresh-enough connection; reconnect once if it dropped."""
    try:
        return fn(get_conn(), *args, **kw)
    except Exception:
        get_conn.clear()
        return fn(get_conn(), *args, **kw)


@st.cache_data(ttl=300, show_spinner=False)
def cached(name: str, *args):
    return run(getattr(q, name), *args)


def empty_state(title: str, body: str) -> None:
    st.info(f"**{title}**\n\n{body}")


def line_chart(df, x, y, title, color=BLUE, y_title=None):
    return (
        alt.Chart(df, title=title)
        .mark_line(color=color, strokeWidth=1.5)
        .encode(
            x=alt.X(f"{x}:T", title=None, axis=alt.Axis(format="%b %Y", tickCount=6)),
            y=alt.Y(f"{y}:Q", title=y_title, scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip(f"{x}:T"), alt.Tooltip(f"{y}:Q", format=".2f")],
        )
        .properties(height=220)
    )


# ---------------------------------------------------------------- pages
def page_overview():
    st.header("Overview")
    st.caption("Tamil Nadu wind generation and weather, daily. Weather = ERA5 mean of 3 wind-belt points; "
               "generation = CEA state total (MU/day).")
    dq = run(q.data_quality)
    c = st.columns(4)
    c[0].metric("Days of data", f"{dq['days']:,}")
    c[1].metric("Days with generation", f"{dq['days_with_generation']:,}")
    c[2].metric("Generation span", f"{dq['gen_first']:%Y} to {dq['gen_last']:%Y}")
    c[3].metric("Mean generation", f"{cached('daily_features')['generation_mu'].mean():.1f} MU/day")

    df = cached("daily_features")
    lo, hi = df["ts"].min().date(), df["ts"].max().date()
    start, end = st.slider("Date range", lo, hi, (lo, hi), format="MMM YYYY")
    d = df[(df["ts"].dt.date >= start) & (df["ts"].dt.date <= end)]
    st.altair_chart(line_chart(d, "ts", "generation_mu", "Daily wind generation", BLUE, "MU/day"),
                    use_container_width=True)
    st.altair_chart(line_chart(d, "ts", "wind_speed", "Daily mean wind speed at 10 m", BLUE, "m/s"),
                    use_container_width=True)

    clim = cached("monthly_climatology")
    clim["Month"] = [MONTHS[int(m) - 1] for m in clim["month"]]
    left, right = st.columns(2)
    for col, field, title, unit in ((left, "generation_mu", "Generation by calendar month", "MU/day"),
                                    (right, "wind_speed", "Wind speed by calendar month", "m/s")):
        ch = (alt.Chart(clim, title=title).mark_bar(color=BLUE, cornerRadiusEnd=3, size=14)
              .encode(x=alt.X("Month:N", sort=MONTHS, title=None), y=alt.Y(f"{field}:Q", title=unit),
                      tooltip=["Month", alt.Tooltip(f"{field}:Q", format=".2f")]).properties(height=220))
        col.altair_chart(ch, use_container_width=True)
    st.caption("Wind season is June to August; the trough is in the cooler months. "
               "Days without a generation value are left blank, never filled.")


def page_climate():
    st.header("Climate change factors")
    st.caption("How much each climate model says Tamil Nadu's wind, temperature and humidity change "
               "against its own 1995-2014 baseline. Wind is a ratio (1.00 = no change); temperature and "
               "humidity are differences. The grey lines are the individual models, the blue line is their average.")
    c = st.columns(3)
    scen = c[0].selectbox("Scenario", list(SCEN_LABEL), format_func=SCEN_LABEL.get)
    year = c[1].selectbox("Window", [2030, 2050], format_func=lambda y: f"{y} ({y - 9}-{y + 10})")
    var = c[2].selectbox("Variable", ["wind_ratio", "temp_delta_c", "humidity_delta_pct"],
                         format_func={"wind_ratio": "Wind speed (ratio)", "temp_delta_c": "Temperature (deg C change)",
                                      "humidity_delta_pct": "Humidity (% points change)"}.get)
    m = cached("monthly_factors", scen, year)
    if m.empty:
        empty_state("No climate data", "Run `python -m etl.rebuild` to load the NEX-GDDP change factors.")
        return
    m[var] = m[var].astype(float)
    mean = m.groupby("month", as_index=False)[var].mean()
    mean["climate_model"] = "Model average"
    x = alt.X("month:O", title=None, axis=alt.Axis(labelExpr="['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][datum.value-1]"))
    base_line = 1.0 if var == "wind_ratio" else 0.0
    ylab = {"wind_ratio": "Wind ratio (future / baseline)", "temp_delta_c": "Change (deg C)",
            "humidity_delta_pct": "Change (% points)"}[var]
    models = (alt.Chart(m).mark_line(color=GRAY, strokeWidth=1)
              .encode(x=x, y=alt.Y(f"{var}:Q", title=ylab, scale=alt.Scale(zero=False)), detail="climate_model:N",
                      tooltip=["climate_model", "month", alt.Tooltip(f"{var}:Q", format=".3f")]))
    avg = (alt.Chart(mean).mark_line(color=BLUE, strokeWidth=2.5, point=alt.OverlayMarkDef(size=40, filled=True))
           .encode(x=x, y=f"{var}:Q", tooltip=[alt.Tooltip("month:O"), alt.Tooltip(f"{var}:Q", title="Model average", format=".3f")]))
    rule = alt.Chart(pd.DataFrame({"y": [base_line]})).mark_rule(color="#888", strokeDash=[4, 4]).encode(y="y:Q")
    st.altair_chart((rule + models + avg).properties(height=340), use_container_width=True)
    st.caption("Grey: one line per climate model (10). Blue: average of the 10 models. Dashed: no change.")

    a = cached("change_factors", scen, year, None)
    if not a.empty:
        st.subheader("Annual change per model")
        st.dataframe(a.rename(columns={"climate_model": "Model", "wind_ratio": "Wind ratio",
                                       "temp_delta_c": "Temp change (C)", "humidity_delta_pct": "Humidity change (pts)"})
                     .style.format({"Wind ratio": "{:.3f}", "Temp change (C)": "{:.2f}", "Humidity change (pts)": "{:.2f}"}),
                     hide_index=True, use_container_width=True)
        n_up = int((a["wind_ratio"].astype(float) > 1).sum())
        st.caption(f"{n_up} of {len(a)} models give higher annual wind than their baseline, "
                   f"{len(a) - n_up} give lower. Where the models disagree on the sign, the change is not reliable.")


def page_forecasts():
    st.header("Forecasts")
    runs = cached("forecast_runs")
    if runs.empty or int(runs["n_predictions"].sum()) == 0:
        empty_state("No forecasts stored yet",
                    "The modelling team writes each run to `forecast_runs` and `model_predictions`. "
                    "Once a run exists, this page shows predicted against actual generation with its "
                    "uncertainty band.")
        return
    runs = runs[runs["n_predictions"] > 0]
    label = {r.run_id: f"Run {r.run_id}: {r.model_version} ({r.horizon_days}-day horizon, {r.n_predictions} days)"
             for r in runs.itertuples()}
    run_id = st.selectbox("Forecast run", list(label), format_func=label.get)
    d = cached("predictions_vs_actual", int(run_id))
    long = d.melt("ts", ["actual", "predicted_value"], var_name="series", value_name="MU")
    long["series"] = long["series"].map({"actual": "Actual", "predicted_value": "Forecast"})
    band = alt.Chart(d.dropna(subset=["lower", "upper"])).mark_area(color=ORANGE, opacity=0.15).encode(
        x="ts:T", y="lower:Q", y2="upper:Q")
    lines = alt.Chart(long).mark_line(strokeWidth=1.5).encode(
        x=alt.X("ts:T", title=None), y=alt.Y("MU:Q", title="MU/day", scale=alt.Scale(zero=False)),
        color=alt.Color("series:N", scale=alt.Scale(domain=["Actual", "Forecast"], range=[BLUE, ORANGE]), title=None),
        tooltip=["ts:T", "series", alt.Tooltip("MU:Q", format=".2f")])
    st.altair_chart((band + lines).properties(height=340), use_container_width=True)
    both = d.dropna(subset=["actual"])
    if len(both):
        mae = (both["actual"] - both["predicted_value"]).abs().mean()
        st.metric("Mean absolute error on days with an actual value", f"{mae:.2f} MU/day")
    with st.expander("Run parameters"):
        st.json(runs.loc[runs["run_id"] == run_id, "params"].iloc[0] or {})


def page_impact():
    st.header("Climate impact on generation")
    ci = cached("climate_impact")
    if ci.empty:
        empty_state("Climate impact not computed yet",
                    "The modelling team writes results to `climate_impact` after applying the change factors "
                    "to the trained model. Each row is one metric for one climate model, scenario and window. "
                    "This page will show the model average with the 10th-90th percentile range across models.")
        return
    metric = st.selectbox("Metric", sorted(ci["metric"].unique()))
    d = ci[(ci["metric"] == metric)].copy()
    ens = d[d["climate_model"] == "ensemble"].copy()
    if ens.empty:
        empty_state("No ensemble row", "Rows with `climate_model = 'ensemble'` hold the summary shown here.")
        return
    short = {"ssp245": "SSP2-4.5", "ssp585": "SSP5-8.5"}
    ens["Case"] = ens["scenario"].map(short) + " " + ens["target_year"].astype(str)
    order = list(ens.sort_values(["scenario", "target_year"])["Case"])
    ens["Scenario"] = ens["scenario"].map(short)
    base = alt.Chart(ens)
    bars = base.mark_bar(cornerRadiusEnd=3, size=28).encode(
        x=alt.X("Case:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("value:Q", title=metric),
        color=alt.Color("Scenario:N", scale=alt.Scale(domain=list(short.values()), range=[BLUE, ORANGE]),
                        legend=alt.Legend(title=None)),
        tooltip=["Case", alt.Tooltip("value:Q", format=".2f"), alt.Tooltip("p10:Q", format=".2f"),
                 alt.Tooltip("p90:Q", format=".2f")])
    whisk = base.mark_rule(color="#555", strokeWidth=1.5).encode(x=alt.X("Case:N", sort=order), y="p10:Q", y2="p90:Q")
    st.altair_chart((bars + whisk).properties(height=320), use_container_width=True)
    st.caption("Bar: ensemble value. Whisker: 10th to 90th percentile across climate models.")
    st.dataframe(d, hide_index=True, use_container_width=True)


def page_status():
    st.header("Data status")
    dq = run(q.data_quality)
    st.write(f"**{dq['days']:,}** days from {dq['first_day']} to {dq['last_day']}. "
             f"**{dq['days_with_generation']:,}** have a generation value; "
             f"**{dq['days'] - dq['days_with_generation']}** are blank (22 missing CEA days plus 30 weather-only "
             f"margin days). **{dq['days_all_3_points']:,}** have all 3 weather points.")
    st.subheader("Rows per table")
    st.dataframe(cached("table_counts"), hide_index=True, use_container_width=True)
    st.subheader("Sources and units")
    st.markdown(
        "- **Generation:** CEA state-wise daily renewable generation, wind, MU/day (Tamil Nadu).\n"
        "- **Weather:** ERA5 via Open-Meteo; wind at 10 m in m/s, temperature in deg C, humidity in % RH; "
        "India calendar days.\n"
        "- **Climate:** NASA NEX-GDDP-CMIP6 via Google Earth Engine; baseline 1995-2014, windows 2021-2040 and "
        "2041-2060, SSP2-4.5 and SSP5-8.5; 10 models. Only *changes* are used, never absolute levels.\n"
        "- Pressure is not used as a model input.")


PAGES = {"Overview": page_overview, "Climate factors": page_climate, "Forecasts": page_forecasts,
         "Climate impact": page_impact, "Data status": page_status}

st.sidebar.title("WindCast Climate")
choice = st.sidebar.radio("Page", list(PAGES), label_visibility="collapsed")
try:
    PAGES[choice]()
except Exception as exc:  # database down or not rebuilt
    st.error("Could not show this page.")
    st.code(f"{type(exc).__name__}: {exc}")
    st.markdown("Start it with `docker compose up -d` and fill it with `python -m etl.rebuild`.")
