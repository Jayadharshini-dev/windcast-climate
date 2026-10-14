"""Checks for the NEX-GDDP pipeline (fetch helpers, loader validation, change-factor views).

    pytest tests/test_nexgddp.py

Everything here uses SYNTHETIC numbers with a fake model name ('TEST-A'). The database tests run
inside a transaction that is always rolled back, so nothing is ever written to your real tables.
The Earth Engine calls themselves are NOT tested here (they need your account); the first real run
prints a preflight check instead.
"""
import pandas as pd
import pytest

from etl.db import connect
from etl.fetch_nexgddp import VARIABLES, grid_cell_center, rows_from_features
from etl.load_nexgddp import prepare, upsert, with_annual

STATIONS = ["TN_Muppandal_Kayathar", "TN_Palghat_Udumalpet", "TN_Theni_Cumbum"]
BASE_WIND = {"TN_Muppandal_Kayathar": 4.0, "TN_Palghat_Udumalpet": 6.0, "TN_Theni_Cumbum": 8.0}
FUT_WIND = {"TN_Muppandal_Kayathar": 4.4, "TN_Palghat_Udumalpet": 6.6, "TN_Theni_Cumbum": 8.0}


def synthetic(model="TEST-A", drop_month=None):
    """Baseline + ssp245/2030 for one fake model; tas in Kelvin like the real data."""
    rows = []
    for scen, y0, y1, ty, wind, dtemp, dhum in [("historical", 1995, 2014, 2005, BASE_WIND, 0.0, 0.0),
                                                ("ssp245", 2021, 2040, 2030, FUT_WIND, 1.5, -2.0)]:
        for month in range(1, 13):
            if scen == "ssp245" and month == drop_month:
                continue
            for st in STATIONS:
                rows.append(dict(model=model, scenario=scen, window_start=y0, window_end=y1, target_year=ty,
                                 month=month, station=st, n_images=600, sfcWind=wind[st],
                                 tas=300.15 + dtemp, hurs=75.0 + dhum))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ pure functions
def test_rows_from_features_parses_and_fails_loudly_on_masked_cell():
    good = [{"properties": {"station": "A", "n_images": 600, "sfcWind": 5.0, "tas": 299.0, "hurs": 80.0}}]
    r = rows_from_features(good, "M", "historical", 1995, 2014, 2005, 7)[0]
    assert r["station"] == "A" and r["month"] == 7 and r["tas"] == 299.0 and r["n_images"] == 600
    masked = [{"properties": {"station": "A", "n_images": 600, "sfcWind": None, "tas": None, "hurs": None}}]
    with pytest.raises(ValueError, match="masked"):
        rows_from_features(masked, "M", "historical", 1995, 2014, 2005, 7)


def test_grid_cell_center():
    assert grid_cell_center(8.25, 77.55) == (8.375, 77.625)
    assert grid_cell_center(10.6, 77.25) == (10.625, 77.375)


def test_prepare_converts_kelvin_and_keeps_complete_blocks():
    monthly, report = prepare(synthetic(), STATIONS)
    assert len(monthly) == 2 * 12 * 3
    assert monthly.loc[monthly.scenario == "historical", "temp_c"].round(6).eq(27.0).all()  # 300.15 K
    assert report["blocks_incomplete_skipped"] == []


def test_prepare_skips_incomplete_block_and_reports_it():
    monthly, report = prepare(synthetic(drop_month=7), STATIONS)
    assert len(report["blocks_incomplete_skipped"]) == 1
    assert report["blocks_incomplete_skipped"][0]["scenario"] == "ssp245"
    assert (monthly["scenario"] == "historical").all()  # only the baseline remains


def test_prepare_excludes_future_rows_of_a_model_without_baseline():
    df = synthetic()
    df = df[df.scenario != "historical"]
    monthly, report = prepare(df, STATIONS)
    assert monthly.empty and report["models_without_complete_baseline_excluded"] == ["TEST-A"]


@pytest.mark.parametrize("col,bad", [("tas", 20.0), ("sfcWind", 40.0), ("hurs", 130.0)])
def test_prepare_stops_on_implausible_values(col, bad):
    df = synthetic()
    df.loc[0, col] = bad  # tas=20 means someone forgot it is already deg C / wrong units
    with pytest.raises(ValueError, match="outside"):
        prepare(df, STATIONS)


def test_prepare_rejects_duplicates_and_unknown_stations():
    with pytest.raises(ValueError, match="duplicate"):
        prepare(pd.concat([synthetic(), synthetic().head(1)]), STATIONS)
    with pytest.raises(ValueError, match="not in the database"):
        prepare(synthetic(), STATIONS[:2])


def test_annual_rows_are_the_mean_of_the_12_months():
    monthly, _ = prepare(synthetic(), STATIONS)
    allrows = with_annual(monthly)
    annual = allrows[allrows["month"].isna()]
    assert len(annual) == 2 * 3  # 2 windows x 3 stations
    assert len(allrows) == 2 * 12 * 3 + 6


# ------------------------------------------------------------------ database (always rolled back)
@pytest.fixture()
def conn():
    try:
        c = connect()
        c.execute("SELECT 1 FROM climate_scenarios LIMIT 1")
    except Exception as exc:
        pytest.skip(f"database not ready: {exc}")
    try:
        yield c
    finally:
        c.rollback()  # never keep synthetic rows
        c.close()


def test_views_compute_ratio_and_differences_and_region_uses_mean_of_points(conn):
    monthly, _ = prepare(synthetic(), STATIONS)
    rows = with_annual(monthly)
    assert upsert(conn, rows) == len(rows)

    pts = conn.execute("SELECT scenario, month, wind_ratio, temp_delta_c, humidity_delta_pct, station_id "
                       "FROM v_change_factors WHERE climate_model = 'TEST-A'").fetchall()
    assert len(pts) == 13 * 3  # 12 months + annual, x 3 points, future window only
    for scen, month, ratio, dt_, dh, sid in pts:
        assert scen == "ssp245"
        assert dt_ == pytest.approx(1.5) and dh == pytest.approx(-2.0)
    ratios = sorted({round(r[2], 6) for r in pts})
    assert ratios == [1.0, 1.1]  # point 8.0->8.0 and the two points x1.1

    reg = conn.execute("SELECT month, n_points, wind_ratio, temp_delta_c FROM v_change_factors_region "
                       "WHERE climate_model = 'TEST-A' ORDER BY month NULLS LAST").fetchall()
    assert len(reg) == 13 and all(r[1] == 3 for r in reg)
    # ratio of the point-means (19/3) / (18/3) = 1.05556, NOT the mean of the point ratios (1.0667)
    assert all(r[2] == pytest.approx(19 / 18) for r in reg)
    assert all(r[3] == pytest.approx(1.5) for r in reg)
    assert reg[-1][0] is None  # the annual summary row


def test_upsert_is_idempotent_including_annual_null_month_rows(conn):
    monthly, _ = prepare(synthetic(), STATIONS)
    rows = with_annual(monthly)
    upsert(conn, rows)
    first = conn.execute("SELECT count(*) FROM climate_scenarios WHERE climate_model = 'TEST-A'").fetchone()[0]
    upsert(conn, rows)
    second = conn.execute("SELECT count(*) FROM climate_scenarios WHERE climate_model = 'TEST-A'").fetchone()[0]
    assert first == second == len(rows)
