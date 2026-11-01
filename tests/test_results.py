"""Committed model/climate results load into the database cleanly and re-loading does not duplicate (needs the database)."""
import pytest

from etl import load_results
from etl.db import connect


@pytest.fixture(scope="module")
def conn():
    try:
        c = connect()
        c.execute("SELECT 1 FROM forecast_runs LIMIT 1")
    except Exception:
        pytest.skip("database not available")
    yield c
    c.close()


def test_load_is_idempotent(conn):
    first = load_results.main()
    second = load_results.main()
    assert first == second
    assert first["forecast_runs"] == 12  # 4 models x horizons 1-3
    assert first["model_predictions"] == 12 * 483
    assert first["climate_impact"] == 268
    n = conn.execute("SELECT count(*) FROM forecast_runs WHERE params->>'source' = %s", (load_results.SOURCE,)).fetchone()[0]
    assert n == 12


def test_predictions_cover_holdout_days(conn):
    load_results.main()
    lo, hi, n = conn.execute(
        "SELECT min(forecast_date), max(forecast_date), count(DISTINCT forecast_date) FROM model_predictions").fetchone()
    assert str(lo) == "2024-01-01" and str(hi) == "2025-04-27" and n == 483
