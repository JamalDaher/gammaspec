"""Gaussian photopeak fitting with a local linear background.

Reported standard errors are covariance-based approximations under the
weighted least-squares model, not a full counting-statistics analysis.
"""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from .core import _validate, find_photopeaks

FACTOR = 2 * np.sqrt(2 * np.log(2))

def gaussian_background(x, amplitude, centroid, sigma, intercept, slope):
    return amplitude * np.exp(-0.5 * ((x - centroid) / sigma) ** 2) + intercept + slope * (x - centroid)

def fit_photopeak(spectrum, center_kev, half_window_kev=65):
    """Fit Gaussian plus local linear background and estimate 1-sigma errors."""
    _validate(spectrum)
    if not np.isfinite(center_kev) or not np.isfinite(half_window_kev) or half_window_kev <= 0:
        raise ValueError("Invalid fit window")
    local = spectrum.loc[(spectrum.energy_kev >= center_kev - half_window_kev) &
                         (spectrum.energy_kev <= center_kev + half_window_kev)]
    if len(local) < 12:
        raise ValueError("Insufficient samples in fit window")
    x = local.energy_kev.to_numpy(dtype=float)
    y = local.counts.to_numpy(dtype=float)
    step = float(np.median(np.diff(x)))
    baseline = float(np.percentile(y, 20))
    amplitude = max(float(y.max() - baseline), 1.0)
    center = float(x[np.argmax(y)])
    sigma0 = max(3 * step, half_window_kev / 5)
    bounds = ([0, x.min(), step / 2, 0, -np.inf],
              [np.inf, x.max(), half_window_kev * 2, np.inf, np.inf])
    try:
        parameters, covariance = curve_fit(
            gaussian_background, x, y,
            p0=[amplitude, center, sigma0, baseline, 0],
            sigma=np.sqrt(np.maximum(y, 1)), absolute_sigma=True,
            bounds=bounds, maxfev=20000)
    except (RuntimeError, ValueError) as exc:
        raise ValueError("Peak fit did not converge") from exc
    if not np.isfinite(covariance).all():
        raise ValueError("Peak covariance could not be estimated")
    errors = np.sqrt(np.maximum(np.diag(covariance), 0))
    a, mu, width, background, slope = parameters
    a_err, mu_err, width_err, _, _ = errors
    # Integrated peak area in count*keV; divide by channel spacing for counts.
    area = a * width * np.sqrt(2 * np.pi) / step
    grad = np.array([width, a]) * np.sqrt(2 * np.pi) / step
    area_var = float(grad @ covariance[np.ix_([0, 2], [0, 2])] @ grad)
    return {
        "centroid_kev": float(mu),
        "centroid_error_kev": float(mu_err),
        "fwhm_kev": float(FACTOR * width),
        "fwhm_error_kev": float(FACTOR * width_err),
        "resolution_percent": float(100 * FACTOR * width / mu) if mu > 0 else np.nan,
        "net_peak_counts": float(area),
        "net_peak_counts_error": float(np.sqrt(max(area_var, 0))),
        "background_at_centroid": float(background),
        "amplitude_counts": float(a),
        "slope": float(slope),
        "sigma_kev": float(width),
    }

def fit_detected_peaks(spectrum, prominence=150, half_window_kev=65, max_peaks=10):
    """Fit strongest detected peaks; report failures without hiding them."""
    candidates = find_photopeaks(spectrum, prominence=prominence)
    candidates = candidates.nlargest(max_peaks, "prominence")
    results = []
    for candidate in candidates.itertuples():
        try:
            result = fit_photopeak(spectrum, candidate.energy_kev, half_window_kev)
            result["status"] = "ok"
        except ValueError as exc:
            result = {"centroid_kev": float(candidate.energy_kev), "status": str(exc)}
        results.append(result)
    return pd.DataFrame(results)
