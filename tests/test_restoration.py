import pandas as pd
import pytest
from gammaspec.restoration import restoration_settings, compare_reanalysis

SPECTRUM = pd.DataFrame({"channel": [0, 1, 2], "energy_kev": [0., 1., 2.],
                         "counts": [1, 2, 3]})
def record():
    return {"acquisition": {"peak_prominence": 150, "fit_half_window_kev": 55,
        "minimum_peak_separation_kev": 35, "fit_enabled": True},
        "calibration": {"applied_in_app": True}}

def test_restore_settings_and_preserve_calibrated_axis():
    settings = restoration_settings(record(), SPECTRUM)
    assert settings["peak_prominence"] == 150
    assert settings["calibration_applied_in_archive"] is True

def test_invalid_saved_controls_are_rejected():
    item = record()
    item["acquisition"]["peak_prominence"] = 99999
    with pytest.raises(ValueError, match="peak_prominence"):
        restoration_settings(item, SPECTRUM)

def test_comparison_detects_drift():
    result = compare_reanalysis({"channels": 3, "total_counts": 10},
        {"channels": 3, "total_counts": 11})
    assert result[0]["matches"] is True
    assert result[1]["matches"] is False
