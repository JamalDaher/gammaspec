"""Reproducible synthetic regression benchmarks.

These tests compare fitted results with the simulator's injected parameters.
They do NOT validate the simulator against physical detector measurements.
"""
import numpy as np
import pandas as pd
from .core import simulate_spectrum
from .fitting import fit_photopeak, FACTOR
from .identification import match_candidates

CASES = (
    ("Cs-137", 661.657, 55.0),
    ("Co-60", 1173.228, 55.0),
    ("Co-60", 1332.492, 55.0),
)

def benchmark(seed=42, live_time_s=60, tolerance_kev=12.0):
    """Evaluate centroid, FWHM, and candidate matching for seeded toy spectra."""
    rows = []
    for source, reference, half_window in CASES:
        spectrum = simulate_spectrum(source, seed=seed, live_time_s=live_time_s)
        try:
            result = fit_photopeak(spectrum, reference, half_window_kev=half_window)
            predicted = result["centroid_kev"]
            sigma = max(4.0, 0.025 * reference)
            true_fwhm = float(FACTOR * sigma)
            candidates = match_candidates([predicted], tolerance_kev=tolerance_kev)
            rows.append({
                "source": source,
                "reference_kev": reference,
                "fitted_kev": predicted,
                "centroid_error_kev": predicted - reference,
                "absolute_error_kev": abs(predicted - reference),
                "fitted_fwhm_kev": result["fwhm_kev"],
                "injected_fwhm_kev": true_fwhm,
                "fwhm_error_kev": result["fwhm_kev"] - true_fwhm,
                "candidate_match": source in candidates.candidate.tolist(),
                "status": "ok",
            })
        except ValueError as exc:
            rows.append({"source": source, "reference_kev": reference,
                         "candidate_match": False, "status": str(exc)})
    return pd.DataFrame(rows)

def summary(results):
    """Summarize successful rows and report failures explicitly."""
    if results.empty or "status" not in results:
        raise ValueError("No benchmark results")
    valid = results[results.status == "ok"]
    return {
        "total_cases": len(results),
        "successful_fits": len(valid),
        "failed_fits": len(results) - len(valid),
        "centroid_mae_kev": float(valid.absolute_error_kev.mean()) if len(valid) else np.nan,
        "fwhm_mae_kev": float(valid.fwhm_error_kev.abs().mean()) if len(valid) else np.nan,
        "candidate_matches": int(valid.candidate_match.sum()) if len(valid) else 0,
    }
