"""The committed seed files must be complete and consistent (no database needed)."""
import pandas as pd

from etl.db import ROOT

SEED = ROOT / "data" / "seed"


def test_cea_seed_covers_span_with_known_gaps():
    df = pd.read_csv(SEED / "cea_tn_daily.csv", parse_dates=["ts"])
    assert len(df) == 1796 and df["ts"].is_unique
    assert df["ts"].min() == pd.Timestamp("2020-05-28") and df["ts"].max() == pd.Timestamp("2025-04-27")
    assert df["generation_mu"].isna().sum() == 22  # missing + all-zero reporting gaps, never filled


def test_era5_seed_three_points_full_years():
    df = pd.read_csv(SEED / "era5_daily.csv", parse_dates=["ts"])
    assert df["point"].nunique() == 3
    assert (df.groupby("point").size() == 1826).all()
    assert not df.duplicated(["point", "ts"]).any()
    assert df["wind_speed"].between(0, 30).all() and df["humidity"].between(0, 100).all()
