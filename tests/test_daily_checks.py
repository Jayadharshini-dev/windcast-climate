"""Tests for the daily-output checks (climate/daily_checks.py, etl/fetch_nexgddp_daily.py). No Earth Engine, no database."""
import numpy as np
import pandas as pd
import pytest

from climate.daily_checks import compare_to_era5, load_nex_daily, qdm_month, qdm_series
from etl.fetch_nexgddp_daily import out_path, parse_spec, rows_from_daily_features


def test_parse_spec_and_paths(tmp_path):
    assert parse_spec("historical:1995-2014,ssp585:2041-2060") == [("historical", 1995, 2014), ("ssp585", 2041, 2060)]
    with pytest.raises(ValueError):
        parse_spec("ssp999:2020-2025")
    with pytest.raises(ValueError):
        parse_spec("ssp245:2025-2020")
    assert out_path("MIROC6", "ssp585", 2045, tmp_path).name == "MIROC6_ssp585_2045.csv"


def test_rows_from_features_sorted_and_loud_on_missing():
    feats = [{"properties": {"date": "2030-01-02", "station": "B", "sfcWind": 3.0, "tas": 300.0, "hurs": 70.0}},
             {"properties": {"date": "2030-01-01", "station": "A", "sfcWind": 2.0, "tas": 299.0, "hurs": 60.0}}]
    rows = rows_from_daily_features(feats)
    assert [r["date"] for r in rows] == ["2030-01-01", "2030-01-02"]
    feats[0]["properties"]["hurs"] = None
    with pytest.raises(ValueError):
        rows_from_daily_features(feats)


def test_qdm_with_identical_future_and_history_maps_to_observed_distribution():
    rng = np.random.default_rng(0)
    hist = rng.gamma(4, 1.0, 3000)            # model's own (biased) climate
    obs = rng.gamma(4, 1.5, 3000)              # observed climate: windier
    out = qdm_month(hist, hist, obs, "mult")
    assert abs(out.mean() - obs.mean()) < 0.05 * obs.mean()
    assert abs(np.quantile(out, 0.9) - np.quantile(obs, 0.9)) < 0.1 * np.quantile(obs, 0.9)


def test_qdm_keeps_the_projected_relative_change():
    rng = np.random.default_rng(1)
    hist = rng.gamma(4, 1.0, 4000)
    obs = rng.gamma(4, 1.5, 4000)
    fut = hist * 1.10                          # the model projects +10% wind everywhere
    out = qdm_month(fut, hist, obs, "mult")
    assert abs(out.mean() / obs.mean() - 1.10) < 0.02
    add = qdm_month(hist + 2.0, hist, obs + 5.0, "add")   # +2 degC projected, observed climate 5 degC warmer
    assert abs(add.mean() - (obs + 5.0).mean() - 2.0) < 0.1 or abs(add.mean() - ((obs + 5.0).mean() + 2.0)) < 0.2


def test_qdm_series_clips_humidity_and_works_month_by_month():
    idx = pd.date_range("2000-01-01", periods=1200, freq="D")
    rng = np.random.default_rng(2)
    w = pd.DataFrame({"wind_speed": rng.gamma(4, 1, 1200), "temperature": rng.normal(28, 2, 1200), "humidity": rng.normal(80, 10, 1200)}, index=idx)
    o = w.assign(humidity=w.humidity + 15, wind_speed=w.wind_speed * 1.2)
    out = qdm_series(w, w, o)
    assert out["humidity"].between(0, 100).all() and (out["wind_speed"] > 0).all()
    assert abs(out["wind_speed"].mean() / o["wind_speed"].mean() - 1) < 0.03


def test_loader_converts_kelvin_and_averages_points(tmp_path):
    for y in (2020, 2021):
        rows = []
        for d in pd.date_range(f"{y}-01-01", periods=3):
            for st, w in (("A", 2.0), ("B", 4.0)):
                rows.append({"date": d.strftime("%Y-%m-%d"), "station": st, "sfcWind": w, "tas": 300.15, "hurs": 50.0})
        pd.DataFrame(rows).to_csv(tmp_path / f"M1_ssp245_{y}.csv", index=False)
    g = load_nex_daily("M1", "ssp245", [2020, 2021], tmp_path)
    assert len(g) == 6 and abs(g["temperature"].iloc[0] - 27.0) < 1e-9 and g["wind_speed"].iloc[0] == 3.0
    assert load_nex_daily("M1", "ssp245", [2020, 2021, 2022], tmp_path) is None   # incomplete -> not used


def test_compare_to_era5_detects_a_windier_model():
    idx = pd.date_range("2020-05-01", periods=700, freq="D")
    rng = np.random.default_rng(3)
    base = pd.DataFrame({"wind_speed": rng.gamma(4, 1, 700), "temperature": rng.normal(28, 2, 700), "humidity": rng.normal(70, 8, 700)}, index=idx)
    nex = base.assign(wind_speed=base.wind_speed * 1.25, temperature=base.temperature + 1.0)
    r = compare_to_era5(nex, base)
    assert abs(r["wind_mean_ratio"] - 1.25) < 1e-9 and abs(r["temperature_mean_bias"] - 1.0) < 1e-9
    assert r["n_days"] == 700 and abs(r["wind_speed_std_ratio"] - 1.25) < 1e-9
