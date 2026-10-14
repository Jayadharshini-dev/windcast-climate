"""Checks for etl.features.read_features against the real loaded database.

    pip install pytest
    pytest tests/test_features.py

Skipped automatically if the database is not reachable or empty (run docker compose up -d
and python -m etl.run_phase2 first).
"""
import numpy as np
import pandas as pd
import pytest

from etl.db import ROOT, connect
from etl.features import COLUMNS, read_features


@pytest.fixture(scope="module")
def df():
    try:
        return read_features()
    except Exception as exc:  # DB down, or not loaded yet
        pytest.skip(f"database not ready: {exc}")


def test_columns_dtypes_and_order(df):
    assert list(df.columns) == COLUMNS
    assert str(df["ts"].dtype) == "datetime64[ns]"
    assert df["ts"].is_monotonic_increasing
    assert df["ts"].is_unique
    for c in ["wind_speed", "temperature", "humidity", "generation_mu", "capacity_mw", "capacity_factor"]:
        assert df[c].dtype == "float64", c


def test_no_pressure_column(df):
    assert not any("pressure" in c for c in df.columns)  # rule 2


def test_missing_generation_stays_nan_and_days_are_not_filled(df):
    span = pd.date_range(df["ts"].min(), df["ts"].max(), freq="D")
    assert len(df) == len(span)  # no day silently dropped
    assert df["generation_mu"].isna().sum() == 22  # 20 absent from CEA + 2 all-zero reporting gaps


def test_require_generation_drops_only_nan_days(df):
    train = read_features(require_generation=True)
    assert len(train) == df["generation_mu"].notna().sum() == 1774
    assert train["generation_mu"].notna().all()


def test_date_bounds_are_inclusive(df):
    sub = read_features(start="2021-01-01", end="2021-12-31")
    assert sub["ts"].min() == pd.Timestamp("2021-01-01")
    assert sub["ts"].max() == pd.Timestamp("2021-12-31")
    assert len(sub) == 365


def test_wind_is_equal_weight_mean_of_the_three_points(df):
    """Recompute the daily mean straight from raw_weather_readings and compare."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT ts, avg(wind_speed), avg(temperature), avg(humidity), count(wind_speed) "
            "FROM raw_weather_readings WHERE station_id IN (2, 3, 4) GROUP BY ts ORDER BY ts"
        ).fetchall()
    raw = pd.DataFrame(rows, columns=["ts", "wind_speed", "temperature", "humidity", "n"]).set_index("ts")
    raw.index = pd.to_datetime(raw.index)
    joined = df.set_index("ts").join(raw, rsuffix="_raw")
    for c in ["wind_speed", "temperature", "humidity"]:
        np.testing.assert_allclose(joined[c], joined[f"{c}_raw"], rtol=1e-9)
    assert (joined["n_points"] == joined["n"]).all()


def test_values_are_physically_plausible(df):
    assert df["wind_speed"].between(0, 30).all()
    assert df["temperature"].between(10, 45).all()
    assert df["humidity"].between(0, 100).all()
    assert (df["generation_mu"].dropna() >= 0).all()


def test_unknown_region_gives_a_clear_error():
    with pytest.raises(ValueError, match="no rows"):
        read_features(region="Atlantis")
