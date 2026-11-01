"""Streamlit integration smoke checks for the real app, without a browser server."""
from pathlib import Path
import pytest

APP = Path(__file__).resolve().parents[1] / "app.py"

def test_default_streamlit_app_runs_without_exception():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(APP), default_timeout=90).run()
    assert not app.exception, [str(error.message) for error in app.exception]
    assert app.title and app.title[0].value == "GammaSpec"
    assert any(widget.label == "Source" for widget in app.selectbox)

def test_switch_to_overlapping_demo():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(APP), default_timeout=90).run()
    if app.exception:
        pytest.fail(str([e.message for e in app.exception]))
    source = next(widget for widget in app.selectbox if widget.label == "Source")
    source.set_value("Overlapping demo").run()
    assert not app.exception, [str(error.message) for error in app.exception]
