"""Validate saved experiment settings before restoring an archive to a workspace."""
import numpy as np
from .core import _validate

def restoration_settings(record, spectrum):
    _validate(spectrum)
    acquisition = record["acquisition"]
    calibration = record["calibration"]
    result = {}
    bounded = {"peak_prominence": (10, 300), "fit_half_window_kev": (25, 90),
               "minimum_peak_separation_kev": (0, 100)}
    for key, (low, high) in bounded.items():
        value = acquisition.get(key)
        if type(value) not in (int, float) or not np.isfinite(value) or not low <= value <= high:
            raise ValueError(f"Unsupported saved setting: {key}")
        result[key] = int(value)
    if type(acquisition.get("fit_enabled")) is not bool:
        raise ValueError("Invalid saved fit setting")
    result["fit_enabled"] = acquisition["fit_enabled"]
    # Archived spectrum already contains the saved energy axis. Never
    # reapply polynomial calibration to the archived channel values.
    result["calibration_applied_in_archive"] = calibration.get("applied_in_app") is True
    return result

def compare_reanalysis(saved, recalculated):
    keys = ("channels", "total_counts", "candidate_maxima",
            "successful_single_peak_fits", "overlap_fit_components")
    rows = []
    for key in keys:
        previous = saved.get(key)
        current = recalculated.get(key)
        rows.append({"metric": key, "saved": previous, "recalculated": current,
                     "matches": (previous == current)})
    return rows
