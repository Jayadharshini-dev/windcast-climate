"""Archived-forecast test: fetching helpers and scoring helpers, on synthetic data (no network, no database)."""
import numpy as np
import pandas as pd
import pytest

from etl import fetch_forecast_weather as ff
from models import forecast_weather_test as fw


def _api_json(days=3, lead=1, miss_hour=None, wind=5.0):
    times = pd.date_range("2024-01-01", periods=24 * days, freq="h")
    d = {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in times]}
    d[f"wind_speed_10m_previous_day{lead}"] = [wind] * len(times)
    d[f"temperature_2m_previous_day{lead}"] = [25.0] * len(times)
    d[f"relative_humidity_2m_previous_day{lead}"] = [60.0] * len(times)
    if miss_hour is not None:
        d[f"wind_speed_10m_previous_day{lead}"][miss_hour] = None
    return {"hourly": d, "hourly_units": {f"wind_speed_10m_previous_day{lead}": "m/s"}}


def test_date_chunks_cover_range_without_overlap():
    ch = ff.date_chunks("2024-01-01", "2025-04-30", 120)
    assert ch[0][0] == "2024-01-01" and ch[-1][1] == "2025-04-30"
    for (a, b), (c, d) in zip(ch, ch[1:]):
        assert pd.Timestamp(c) == pd.Timestamp(b) + pd.Timedelta(days=1)


def test_hourly_names_cover_three_variables_and_leads():
    assert len(ff.hourly_names()) == 9 and "wind_speed_10m_previous_day3" in ff.hourly_names()


def test_to_daily_means_and_incomplete_day_is_nan():
    d = ff.to_daily(_api_json(miss_hour=30), 1)  # hour 30 is on day 2
    assert d.loc["2024-01-01", "wind"] == 5.0 and np.isnan(d.loc["2024-01-02", "wind"]) and d.loc["2024-01-03", "temp"] == 25.0


def test_to_daily_rejects_wrong_wind_unit():
    j = _api_json()
    j["hourly_units"]["wind_speed_10m_previous_day1"] = "km/h"
    with pytest.raises(ValueError):
        ff.to_daily(j, 1)


def test_state_mean_needs_all_points():
    a, b = ff.to_daily(_api_json(wind=4.0), 1), ff.to_daily(_api_json(wind=6.0, miss_hour=30), 1)
    m = ff.state_mean([a, b])
    assert m.loc["2024-01-01", "wind"] == 5.0 and np.isnan(m.loc["2024-01-02", "wind"])


def _frames(n=120, scale=1.2, seed=0):
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"wind_speed": rng.uniform(1, 8, n), "temperature": rng.uniform(24, 32, n), "humidity": rng.uniform(50, 90, n)}, index=idx)
    fc = pd.DataFrame({f"{k}_d1": df[c] for k, c in fw.WX.items()})
    fc["wind_d1"] = df["wind_speed"] * scale
    fc["temp_d1"] = df["temperature"] - 1.5
    return df, fc


def test_calibration_recovers_the_offsets():
    df, fc = _frames()
    cal = fw.calibration(df, fc, 1, end="2024-04-30")
    assert cal["wind"] == pytest.approx(1 / 1.2) and cal["temp"] == pytest.approx(1.5) and cal["hum"] == pytest.approx(0)
    back = fw.with_forecast(df, fc, 1, cal)
    assert np.allclose(back["wind_speed"], df["wind_speed"]) and np.allclose(back["temperature"], df["temperature"])


def test_weather_error_reports_bias_and_relative_rmse():
    df, fc = _frames()
    e = fw.weather_error(df, fc, 1)
    assert e["wind_ratio"] == pytest.approx(1.2) and e["temp_bias"] == pytest.approx(-1.5) and e["wind_corr"] == pytest.approx(1.0)
    assert e["wind_rel_rmse_pct"] == pytest.approx(100 * 0.2 * np.sqrt((df["wind_speed"] ** 2).mean()) / df["wind_speed"].mean(), rel=1e-6)


def test_with_forecast_leaves_generation_and_clips():
    df, fc = _frames()
    df["generation_mu"] = 1.0
    fc["hum_d1"] = 120.0
    out = fw.with_forecast(df, fc, 1)
    assert (out["generation_mu"] == 1.0).all() and out["humidity"].max() == 100.0


def test_lead_of():
    assert [fw.lead_of(h) for h in (0, 1, 2, 3)] == [1, 1, 2, 3]
