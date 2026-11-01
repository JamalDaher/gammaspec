import pandas as pd
from gammaspec.core import simulate_spectrum
from gammaspec.reporting import render_report

def test_report_contains_summary():
    report = render_report(simulate_spectrum(), dataset="Cs-137", generated_at="2026-01-01")
    assert "Cs-137" in report
    assert "2026-01-01" in report
    assert "Total counts" in report

def test_report_escapes_dataset():
    report = render_report(simulate_spectrum(), dataset="<script>alert(1)</script>")
    assert "<script>" not in report
    assert "&lt;script&gt;" in report

def test_report_includes_fit_table():
    fits = pd.DataFrame([{"centroid_kev": 661.7, "status": "ok"}])
    report = render_report(simulate_spectrum(), fits=fits)
    assert "661.7" in report
    assert "Gaussian peak fits" in report

def test_report_empty_results():
    report = render_report(simulate_spectrum(), peaks=pd.DataFrame())
    assert "No results available." in report
