"""Tests for the delta-change step (climate/climate_impact.py). No database needed."""
import numpy as np
import pandas as pd

from climate.climate_impact import FACTORS_CSV, shifted_weather


def _factors(wind_ratio=1.1, temp=1.0, hum=5.0):
    return pd.DataFrame({"month": range(1, 13), "wind_ratio": wind_ratio, "temp_delta_c": temp,
                         "humidity_delta_pct": hum})


def test_factors_applied_by_month_and_humidity_clipped():
    idx = pd.date_range("2022-01-01", periods=400, freq="D")
    base = pd.DataFrame({"wind_speed": 5.0, "temperature": 28.0, "humidity": 97.0}, index=idx)
    out = shifted_weather(base, _factors())
    assert np.allclose(out["wind_speed"], 5.5)
    assert np.allclose(out["temperature"], 29.0)
    assert out["humidity"].max() == 100.0          # 97 + 5 is clipped to 100


def test_month_specific_factor_hits_only_that_month():
    idx = pd.date_range("2022-01-01", periods=60, freq="D")  # Jan + Feb + a bit of Mar
    base = pd.DataFrame({"wind_speed": 4.0, "temperature": 25.0, "humidity": 60.0}, index=idx)
    f = _factors(wind_ratio=1.0)
    f.loc[f.month == 2, "wind_ratio"] = 1.5
    out = shifted_weather(base, f)
    assert (out.loc[out.index.month == 2, "wind_speed"] == 6.0).all()
    assert (out.loc[out.index.month != 2, "wind_speed"] == 4.0).all()


def test_committed_factor_file_is_complete():
    f = pd.read_csv(FACTORS_CSV).dropna(subset=["month"])
    assert f["climate_model"].nunique() == 10
    assert f.groupby(["scenario", "target_year", "climate_model"]).size().eq(12).all()
