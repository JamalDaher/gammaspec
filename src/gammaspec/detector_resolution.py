"""Illustrative detector energy-resolution models, not calibrated specifications."""
import numpy as np

FWHM_FACTOR = 2 * np.sqrt(2 * np.log(2))
# Representative illustrative FWHM at 662 keV; individual detectors vary.
REFERENCE_FWHM_662 = {"HPGe (illustrative)": 1.8, "NaI(Tl) (illustrative)": 46.0}

def fwhm_kev(energy_kev, detector, reference_fwhm_kev=None):
    """Simplified square-root energy scaling anchored at 662 keV.

    FWHM(E) = FWHM(662) * sqrt(E/662). Not a calibrated detector model.
    """
    if detector not in REFERENCE_FWHM_662:
        raise ValueError("Unknown detector")
    energy = np.asarray(energy_kev, dtype=float)
    ref = REFERENCE_FWHM_662[detector] if reference_fwhm_kev is None else float(reference_fwhm_kev)
    if not np.isfinite(energy).all() or np.any(energy <= 0) or not np.isfinite(ref) or ref <= 0:
        raise ValueError("Energy and reference FWHM must be positive and finite")
    result = ref * np.sqrt(energy / 662.0)
    return float(result) if result.ndim == 0 else result

def resolution_percent(energy_kev, detector, reference_fwhm_kev=None):
    return 100 * fwhm_kev(energy_kev, detector, reference_fwhm_kev) / np.asarray(energy_kev, dtype=float)

def illustrative_doublet(energy_kev, center_kev=495, separation_kev=20,
                         detector="HPGe (illustrative)", reference_fwhm_kev=None):
    """Two equal-area Gaussian peaks with detector-dependent widths."""
    energy = np.asarray(energy_kev, dtype=float)
    if not np.isfinite(separation_kev) or separation_kev <= 0 or not np.isfinite(center_kev) or center_kev <= 0:
        raise ValueError("Invalid peak centers")
    centers = (center_kev - separation_kev / 2, center_kev + separation_kev / 2)
    if centers[0] <= 0:
        raise ValueError("Peak energy must be positive")
    curves = []
    for center in centers:
        sigma = fwhm_kev(center, detector, reference_fwhm_kev) / FWHM_FACTOR
        curves.append(np.exp(-0.5 * ((energy - center) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi)))
    return curves[0], curves[1], curves[0] + curves[1]
