"""Static navigation regression checks; no Streamlit server required."""
import ast
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "app.py"

def test_app_parses():
    ast.parse(APP.read_text(encoding="utf-8"))

def test_analysis_sections_present_once():
    source = APP.read_text(encoding="utf-8")
    for heading in (
        "Fit residual diagnostics",
        "AIC / BIC model comparison",
        "Poisson likelihood model comparison",
        "Monte Carlo peak recovery",
        "Peak resolution study",
        "Detector resolution and FWHM",
        "Detector efficiency and counting statistics",
        "Gamma attenuation and shielding explorer",
        "Compton scattering explorer",
        "Klein–Nishina scattering analysis",
    ):
        assert source.count('with st.expander("' + heading + '"') == 1

def test_navigation_sections():
    source = APP.read_text(encoding="utf-8")
    for tab in ("advanced_tab", "detector_tab", "interactions_tab"):
        assert f"with {tab}:" in source
