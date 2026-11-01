"""Regression checks for the reorganized Streamlit workspace."""
import ast
from pathlib import Path

SOURCE = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")

def test_app_is_valid_python():
    ast.parse(SOURCE)

def test_sidebar_sections():
    for title in ("Acquisition", "Import spectrum", "Peak analysis settings",
                  "Overlapping peaks (experimental)", "Energy calibration"):
        assert SOURCE.count('with st.expander("' + title + '"') == 1

def test_core_controls_and_exports_preserved():
    for snippet in ('isotope = st.selectbox', 'upload = st.file_uploader',
                    'overlap_enabled = st.toggle', 'enable_calibration = st.toggle',
                    'Download HTML analysis report', 'Export spectrum CSV'):
        assert snippet in SOURCE

def test_mobile_style_and_capability_overview():
    assert "@media (max-width: 640px)" in SOURCE
    assert "Explore GammaSpec capabilities" in SOURCE
