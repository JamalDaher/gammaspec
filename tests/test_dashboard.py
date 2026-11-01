import json
import pandas as pd
import pytest
from gammaspec.dashboard import build_results_summary, summary_csv, summary_json

def test_simulated_summary_and_exports():
    spectrum = pd.DataFrame({"energy_kev": [1., 2., 3.], "counts": [3, 4, 5]})
    peaks = pd.DataFrame({"energy_kev": [2.]})
    fits = pd.DataFrame({"status": ["ok", "failed"]})
    result = build_results_summary(spectrum, peaks, fits, source_label="Synthetic Cs-137",
        is_imported=False, overlap_result={"components": [{}, {}]})
    assert result["total_counts"] == 12
    assert result["successful_single_peak_fits"] == 1
    assert result["overlap_fit_components"] == 2
    assert result["data_origin"] == "Synthetic / simulated"
    assert json.loads(summary_json(result))["channels"] == 3
    assert "candidate_maxima" in summary_csv(result)

def test_imported_data_is_explicitly_labeled():
    spectrum = pd.DataFrame({"energy_kev": [10.], "counts": [0.]})
    result = build_results_summary(spectrum, pd.DataFrame(), source_label="sample.csv",
        is_imported=True, calibration_applied=True)
    assert result["data_origin"] == "Imported / user-supplied"
    assert result["energy_calibration"] == "Applied in app"

def test_reject_invalid_data():
    with pytest.raises(ValueError):
        build_results_summary(pd.DataFrame({"energy_kev": [1], "counts": [-1]}),
            pd.DataFrame(), source_label="bad", is_imported=True)
