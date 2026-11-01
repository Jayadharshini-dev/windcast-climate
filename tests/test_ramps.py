"""Tests for the ramp / extreme-day analysis (models/ramps.py, climate/extremes_impact.py). No database needed."""
import numpy as np
import pandas as pd

from climate.extremes_impact import extreme_metrics, thresholds
from models.ramps import k_day_change, ramp_labels, ramp_scores, ramp_threshold, spell_lengths


def _daily(values, start="2022-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="D"), dtype=float)


def test_k_day_change_uses_calendar_days_and_gaps_give_nan():
    g = _daily([10, 20, 35, 30, 30])
    g = g.drop(g.index[2])                       # a missing day must not be bridged
    ch = k_day_change(g, 1)
    assert ch.iloc[1] == 10                      # 20 - 10
    assert np.isnan(ch.iloc[2])                  # day after the gap: yesterday is missing
    assert ch.iloc[3] == 0                       # 30 - 30


def test_ramp_labels_direction_and_threshold():
    ch = pd.Series([0.0, 25.0, -30.0, 5.0, np.nan])
    lab = ramp_labels(ch, 20)
    assert list(lab.iloc[:4]) == [0.0, 1.0, -1.0, 0.0]
    assert np.isnan(lab.iloc[4])


def test_threshold_ignores_test_years():
    train = _daily([10, 12] * 30, "2021-11-01")                       # tiny changes through 2021-12-31
    test = _daily([10, 90] * 30, "2024-02-01")                        # huge swings in 2024
    g = pd.concat([train, test])
    assert ramp_threshold(g, 1, train_end="2023-12-31") < 5            # not pulled up by the 2024 swings


def test_ramp_scores_counts_and_wrong_direction_is_not_a_hit():
    idx = pd.date_range("2024-01-01", periods=6)
    true = pd.Series([1, -1, 0, 0, 1, 0], index=idx, dtype=float)
    pred = pd.Series([1, 1, 0, -1, 0, 0], index=idx, dtype=float)      # hit, wrong sign, ok, false alarm, miss, ok
    s = ramp_scores(true, pred)
    assert (s["n_real_ramps"], s["n_flagged"], s["hits"]) == (3, 3, 1)
    assert abs(s["hit_rate"] - 1 / 3) < 1e-9
    assert abs(s["false_alarm_ratio"] - 2 / 3) < 1e-9
    assert abs(s["csi"] - 1 / 5) < 1e-9


def test_persistence_style_forecast_never_flags_a_ramp():
    true = pd.Series([1, -1, 0, 1.0])
    pred = pd.Series([0, 0, 0, 0.0])
    s = ramp_scores(true, pred)
    assert s["hits"] == 0 and s["hit_rate"] == 0.0 and s["n_flagged"] == 0


def test_spell_lengths_and_gap_breaks_run():
    flag = pd.Series([True, True, False, True, True, True],
                     index=pd.to_datetime(["2022-01-01", "2022-01-02", "2022-01-03",
                                           "2022-01-04", "2022-01-05", "2022-01-06"]))
    assert spell_lengths(flag) == [2, 3]
    gap = flag.drop(pd.Timestamp("2022-01-05"))
    assert spell_lengths(gap) == [2, 1, 1]       # the missing day splits the second run


def test_baseline_extremes_are_about_ten_percent_by_construction():
    rng = np.random.default_rng(0)
    pred = _daily(30 + 15 * np.sin(np.arange(1500) / 20) + rng.normal(0, 5, 1500))
    m = extreme_metrics(pred, thresholds(pred))
    assert abs(m["ramp_day_share_pct"] - 10) < 0.5
    assert abs(m["low_day_share_pct"] - 10) < 0.5
    assert abs(m["high_day_share_pct"] - 10) < 0.5


def test_shifting_series_down_raises_low_days_and_lowers_high_days():
    rng = np.random.default_rng(1)
    pred = _daily(40 + rng.normal(0, 10, 1000))
    thr = thresholds(pred)
    lower = extreme_metrics(pred - 5, thr)
    assert lower["low_day_share_pct"] > extreme_metrics(pred, thr)["low_day_share_pct"]
    assert lower["high_day_share_pct"] < extreme_metrics(pred, thr)["high_day_share_pct"]
