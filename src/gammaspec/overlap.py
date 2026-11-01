"""Exploratory simultaneous fitting of two overlapping Gaussian photopeaks.

This is an illustrative weighted least-squares model, not a validated detector
response or evidence of two physically distinct emissions.
"""
import numpy as np
from scipy.optimize import curve_fit
from .core import _validate
from .fitting import FACTOR


def double_gaussian_background(x, a1, mu1, sigma1, a2, mu2, sigma2, intercept, slope):
    """Two Gaussian components with one shared local linear background."""
    midpoint = (mu1 + mu2) / 2
    return (a1 * np.exp(-0.5 * ((x - mu1) / sigma1) ** 2)
            + a2 * np.exp(-0.5 * ((x - mu2) / sigma2) ** 2)
            + intercept + slope * (x - midpoint))


def fit_overlapping_peaks(spectrum, centers_kev, half_window_kev=65.0):
    """Fit two seeded peaks simultaneously; return components ordered by centroid.

    Centers are required to reduce ambiguity. Covariance-derived uncertainties
    assume the specified model and weights; overlapping components may be
    non-identifiable even when optimization converges.
    """
    _validate(spectrum)
    centers = np.asarray(centers_kev, dtype=float)
    if centers.shape != (2,) or not np.isfinite(centers).all():
        raise ValueError("Provide exactly two finite initial centroids")
    centers = np.sort(centers)
    if (not np.isfinite(half_window_kev) or half_window_kev <= 0
            or centers[0] == centers[1]):
        raise ValueError("Invalid centers or fit window")
    local = spectrum.loc[
        (spectrum.energy_kev >= centers[0] - half_window_kev)
        & (spectrum.energy_kev <= centers[1] + half_window_kev)
    ]
    if len(local) <= 8:
        raise ValueError("Insufficient samples for two-peak fit")
    x = local.energy_kev.to_numpy(dtype=float)
    y = local.counts.to_numpy(dtype=float)
    step = float(np.median(np.diff(x)))
    baseline = float(np.percentile(y, 20))
    heights = [max(float(y[np.argmin(abs(x - c))] - baseline), 1.0) for c in centers]
    width = max(3 * step, half_window_kev / 5)
    low = [0, centers[0] - half_window_kev, step / 2,
           0, centers[1] - half_window_kev, step / 2, 0, -np.inf]
    high = [np.inf, centers[0] + half_window_kev, 2 * half_window_kev,
            np.inf, centers[1] + half_window_kev, 2 * half_window_kev,
            np.inf, np.inf]
    try:
        params, covariance = curve_fit(
            double_gaussian_background, x, y,
            p0=[heights[0], centers[0], width,
                heights[1], centers[1], width, baseline, 0],
            bounds=(low, high), sigma=np.sqrt(np.maximum(y, 1)),
            absolute_sigma=True, maxfev=30000,
        )
    except (ValueError, RuntimeError) as exc:
        raise ValueError("Two-peak fit did not converge") from exc
    if not np.isfinite(covariance).all():
        raise ValueError("Two-peak covariance could not be estimated")
    if params[1] >= params[4]:
        raise ValueError("Peak centroids crossed; two-peak fit is ambiguous")
    errors = np.sqrt(np.maximum(np.diag(covariance), 0))
    residual = (y - double_gaussian_background(x, *params)) / np.sqrt(np.maximum(y, 1))
    reduced_chi_square = float(np.sum(residual ** 2) / (len(y) - 8))
    components = []
    for i in (0, 3):
        amp, mu, sigma = params[i:i + 3]
        components.append({
            "centroid_kev": float(mu),
            "centroid_error_kev": float(errors[i + 1]),
            "fwhm_kev": float(FACTOR * sigma),
            "fwhm_error_kev": float(FACTOR * errors[i + 2]),
            "amplitude_counts": float(amp),
            "net_peak_counts": float(amp * sigma * np.sqrt(2 * np.pi) / step),
        })
    return {
        "components": components,
        "background_intercept": float(params[6]),
        "background_slope": float(params[7]),
        "reduced_chi_square": reduced_chi_square,
        "quality_flag": "review_fit" if reduced_chi_square > 3 else "nominal",
    }
