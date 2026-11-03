"""The ramp and extreme-day section of the Researchers view: loaders, plain-language text, and the page itself.
Needs only the committed result files, not the database."""
import pandas as pd
import pytest

from dashboard import extremes as ex


def test_result_files_load_with_expected_shape():
    imp = ex.load("impact")
    assert imp is not None
    e = imp[imp["climate_model"] == "ensemble"]
    assert imp["climate_model"].nunique() == 11 and len(e) == 4 * len(ex.METRICS) + 4 * 3  # metrics incl. up/down ramp shares
    assert set(ex.CASES) == set(ex.impact_rows(imp, "ramp_day_share_pct")["case"])


def test_ramp_hit_rates_exclude_reference_model():
    t = ex.ramp_hit_rates(ex.load("ramps"))
    assert "climatology" not in set(t["model"])
    assert t.loc[t["model"] == "persistence", "Hit rate (%)"].eq(0).all()
    assert t["Hit rate (%)"].between(0, 100).all()


def test_reading_and_caption_follow_the_numbers():
    txt = ex.reading(ex.load("impact"), ex.load("baseline"))
    assert "ramp frequency is not detectably changed" in txt and "%" in txt
    daily = pd.DataFrame({"low_day_share_pct_hist": [10, 10], "low_day_share_pct_future": [8, 12],
                          "ramp_day_share_pct_future": [9.5, 10.5]})
    cap = ex.daily_caption(daily)
    assert "fall in 1 of 2" in cap and "rise in 1" in cap


def _app():
    import altair as alt
    import streamlit as st
    from dashboard import extremes

    def show(c, h=300):
        st.altair_chart(c.properties(height=h))

    extremes.render(st, alt, show, lambda a, b: st.warning(a + b))


def test_section_renders_and_responds_to_choices():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_app, default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    at.radio[0].set_value("Linear regression").run()
    at.selectbox[0].set_value("longest_low_spell_days").run()
    assert not at.exception, [e.value for e in at.exception]
    assert "Sudden changes and extreme days" in " ".join(m.value for m in at.markdown)


def test_missing_files_give_a_notice_not_a_crash(tmp_path):
    from streamlit.testing.v1 import AppTest

    def app():
        import altair as alt
        import streamlit as st
        from dashboard import extremes
        files = {k: "/nonexistent/" + k for k in extremes.FILES}
        extremes.render(st, alt, lambda c, h=300: None, lambda a, b: st.warning(a + b), files=files)

    at = AppTest.from_function(app, default_timeout=60)
    at.run()
    assert not at.exception and len(at.warning) >= 1
