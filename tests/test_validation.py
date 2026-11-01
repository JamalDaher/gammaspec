import pandas as pd
import pytest
from gammaspec.validation import benchmark, summary

def test_reproducible_benchmark():
    pd.testing.assert_frame_equal(benchmark(seed=42), benchmark(seed=42))

def test_all_reference_lines_present():
    results = benchmark()
    assert len(results) == 3
    assert set(results.source) == {"Cs-137", "Co-60"}

def test_centroid_regression():
    results = benchmark()
    assert (results.status == "ok").all()
    assert (results.absolute_error_kev < 5).all()

def test_fwhm_regression():
    results = benchmark()
    assert (results.fwhm_error_kev.abs() < 12).all()

def test_summary():
    result = summary(benchmark())
    assert result["total_cases"] == 3
    assert result["successful_fits"] == 3
    assert result["centroid_mae_kev"] >= 0

def test_summary_rejects_empty():
    with pytest.raises(ValueError):
        summary(pd.DataFrame())
