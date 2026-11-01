"""Tests for the robustness checks (models/robustness.py) and the capacity check (climate/capacity_check.py)."""
import numpy as np
import pandas as pd

from climate.capacity_check import REF_DATE, adjust_generation, capacity_series
from models.robustness import _scores_np, block_bootstrap_idx, dm_test, noisy_weather


def _weather(n=500):
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    return pd.DataFrame({"wind_speed": 4.0, "temperature": 28.0, "humidity": 60.0, "generation_mu": 30.0}, index=idx)


def test_noise_zero_changes_nothing_and_is_reproducible():
    df = _weather()
    same = noisy_weather(df, 0.0, np.random.default_rng(1))
    assert np.allclose(same["wind_speed"], df["wind_speed"]) and np.allclose(same["temperature"], df["temperature"])
    a = noisy_weather(df, 0.3, np.random.default_rng(7))
    b = noisy_weather(df, 0.3, np.random.default_rng(7))
    assert a.equals(b)


def test_noise_keeps_physical_limits_and_has_the_stated_size():
    df = _weather(20000)
    n = noisy_weather(df, 0.3, np.random.default_rng(0))
    assert n["wind_speed"].min() >= 0 and 0 <= n["humidity"].min() and n["humidity"].max() <= 100
    assert abs((n["wind_speed"] / 4.0).std() - 0.3) < 0.02          # about 30% relative wind error
    assert abs((n["temperature"] - 28).std() - 1.5) < 0.05          # 5 * s degC
    assert (n["generation_mu"] == 30).all()                          # the target is never touched


def test_dm_sign_and_symmetry():
    rng = np.random.default_rng(3)
    e1 = rng.normal(0, 1, 400)
    e2 = rng.normal(0, 2, 400)                                       # model 2 clearly worse
    s, p = dm_test(e1, e2, 1)
    assert s < 0 and p < 0.001                                       # negative favours model 1
    s2, p2 = dm_test(e2, e1, 1)
    assert s2 > 0 and abs(p - p2) < 1e-9


def test_block_bootstrap_indices_are_valid():
    idx = block_bootstrap_idx(483, np.random.default_rng(0), block=7)
    assert len(idx) == 483 and idx.min() >= 0 and idx.max() < 483
    assert (np.diff(idx[:7]) == 1).all()                              # first block is 7 consecutive days


def test_ramp_scores_numpy_version():
    t = np.array([1, -1, 0, 0, 1, 0])
    p = np.array([1, 1, 0, -1, 0, 0])
    pod, far, csi = _scores_np(t, p)
    assert abs(pod - 1 / 3) < 1e-9 and abs(far - 2 / 3) < 1e-9 and abs(csi - 1 / 5) < 1e-9


def test_capacity_interpolation_and_adjustment():
    cap = pd.DataFrame({"date": pd.to_datetime(["2020-03-31", "2021-03-31", "2023-03-31"]),
                        "capacity_mw": [9000.0, 9500.0, 10000.0]})
    idx = pd.DatetimeIndex(["2020-03-31", "2021-03-31", "2023-03-31", "2019-01-01", "2025-01-01"])
    c = capacity_series(idx, cap)
    assert c.iloc[0] == 9000 and c.iloc[1] == 9500 and c.iloc[2] == 10000
    assert c.iloc[3] == 9000 and c.iloc[4] == 10000                  # flat outside the dated range
    df = pd.DataFrame({"generation_mu": 30.0}, index=pd.DatetimeIndex(["2020-03-31", REF_DATE]))
    adj = adjust_generation(df, cap)
    assert abs(adj.loc[REF_DATE, "generation_mu"] - 30.0) < 1e-9     # unchanged on the reference date
    assert adj.iloc[0]["generation_mu"] > 30.0                       # an older, smaller fleet is scaled up to today's size
