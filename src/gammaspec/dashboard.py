"""Pure scientific summary and export helpers for the GammaSpec dashboard."""
import json
import math
import pandas as pd

def build_results_summary(spectrum, peaks, fits=None, *, source_label, is_imported,
                          calibration_applied=False, overlap_result=None):
    """Consolidate observable results without claiming isotope identification."""
    if spectrum.empty or not {"energy_kev", "counts"}.issubset(spectrum.columns):
        raise ValueError("Spectrum requires energy_kev and counts")
    energies = pd.to_numeric(spectrum["energy_kev"], errors="raise")
    counts = pd.to_numeric(spectrum["counts"], errors="raise")
    if not energies.map(math.isfinite).all() or not counts.map(math.isfinite).all():
        raise ValueError("Spectrum must contain finite numeric values")
    if (counts < 0).any():
        raise ValueError("Counts must be nonnegative")
    successful_fits = (fits.loc[fits["status"] == "ok"] if fits is not None
                       and not fits.empty and "status" in fits.columns else pd.DataFrame())
    overlap_components = overlap_result.get("components", []) if overlap_result else []
    return {
        "dataset": str(source_label),
        "data_origin": "Imported / user-supplied" if is_imported else "Synthetic / simulated",
        "energy_calibration": "Applied in app" if calibration_applied else "Not applied in app",
        "channels": int(len(spectrum)),
        "total_counts": float(counts.sum()),
        "energy_min_kev": float(energies.min()),
        "energy_max_kev": float(energies.max()),
        "candidate_maxima": int(len(peaks)),
        "successful_single_peak_fits": int(len(successful_fits)),
        "overlap_fit_components": int(len(overlap_components)),
        "limitations": "Educational results; not validated for dosimetry, safety, or isotope identification.",
    }

def summary_csv(summary):
    return pd.DataFrame([{"Metric": k, "Value": v} for k, v in summary.items()]).to_csv(index=False)

def summary_json(summary):
    return json.dumps(summary, indent=2, allow_nan=False)
