"""Dashboard tests: queries return sane frames; every page renders without an exception.
Skipped when the database is empty (run `python -m etl.rebuild` first)."""
import pytest

from dashboard import queries as q
from etl.db import connect


@pytest.fixture(scope="module")
def conn():
    try:
        c = connect()
        n = c.execute("SELECT count(*) FROM mv_daily_features").fetchone()[0]
    except Exception:
        pytest.skip("database not available")
    if n == 0:
        pytest.skip("database is empty")
    yield c
    c.close()


def test_daily_features(conn):
    df = q.daily_features(conn)
    assert len(df) == 1826 and df["generation_mu"].notna().sum() == 1774


def test_data_quality(conn):
    d = q.data_quality(conn)
    assert d["days"] == 1826 and d["days_all_3_points"] == 1826


def test_monthly_factors_shape(conn):
    m = q.monthly_factors(conn, "ssp245", 2050)
    assert m["climate_model"].nunique() == 10 and len(m) == 120
    a = q.change_factors(conn, "ssp585", 2030, None)
    assert len(a) == 10


def test_forecast_and_impact_queries_run_when_empty(conn):
    q.forecast_runs(conn)
    q.climate_impact(conn)
    assert len(q.table_counts(conn)) == 7


HEADINGS = {"Overview": "The record", "Public": "Which months are windy?", "Researchers": "Climate impact by model",
            "Planners": "Planning calendar", "Data and method": "What is in the database"}


@pytest.mark.parametrize("page", list(HEADINGS))
def test_pages_render(conn, page):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("dashboard/app.py", default_timeout=90)
    at.session_state["view"] = page
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    text = " ".join(m.value for m in at.markdown)
    assert HEADINGS[page] in text, f"page {page} did not show its content"
