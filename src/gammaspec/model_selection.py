"""Approximate information-criterion comparison on a shared fit window."""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from .fitting import gaussian_background
from .overlap import double_gaussian_background


def information_criteria(weighted_rss, n, k):
    """Gaussian weighted-residual AIC/BIC up to a common additive constant."""
    if not np.isfinite(weighted_rss) or weighted_rss <= 0 or n <= k or k < 1:
        raise ValueError("Invalid sample count, parameters or weighted RSS")
    return {
        "AIC": float(n * np.log(weighted_rss / n) + 2 * k),
        "BIC": float(n * np.log(weighted_rss / n) + k * np.log(n)),
    }


def compare_information_criteria(spectrum, overlap_result, centers, half_window_kev):
    """Refit a single Gaussian in the same window used for the two-peak fit.

    The two-peak fit must have been computed with these centers and window.
    This uses a Gaussian weighted-residual approximation, not a Poisson
    likelihood or a calibrated significance test.
    """
    c = np.sort(np.asarray(centers, dtype=float))
    if c.shape != (2,) or not np.isfinite(c).all() or c[0] == c[1]:
        raise ValueError("Two distinct finite centers required")
    if not np.isfinite(half_window_kev) or half_window_kev <= 0:
        raise ValueError("Invalid fit window")
    local = spectrum.loc[
        (spectrum.energy_kev >= c[0] - half_window_kev)
        & (spectrum.energy_kev <= c[1] + half_window_kev)
    ]
    x = local.energy_kev.to_numpy(dtype=float)
    y = local.counts.to_numpy(dtype=float)
    n = len(y)
    if n <= 8 or np.any(np.diff(x) <= 0):
        raise ValueError("Insufficient or unsorted fit samples")
    weights = np.sqrt(np.maximum(y, 1))
    baseline = float(np.percentile(y, 20))
    p0 = [max(float(y.max() - baseline), 1), float(x[np.argmax(y)]),
          max(float(np.median(np.diff(x))) * 3, (x[-1] - x[0]) / 10),
          max(baseline, 0), 0]
    low = [0, float(x.min()), float(np.median(np.diff(x))) / 2, 0, -np.inf]
    high = [np.inf, float(x.max()), float(x[-1] - x[0]), np.inf, np.inf]
    try:
        params, _ = curve_fit(
            gaussian_background, x, y, p0=p0, bounds=(low, high),
            sigma=weights, absolute_sigma=True, maxfev=30000,
        )
    except (ValueError, RuntimeError) as exc:
        raise ValueError("Single-Gaussian refit failed") from exc
    single_rss = float(np.sum(((y - gaussian_background(x, *params)) / weights) ** 2))
    a, b = overlap_result["components"]
    two_pred = double_gaussian_background(
        x, a["amplitude_counts"], a["centroid_kev"], a["fwhm_kev"] / 2.354820045,
        b["amplitude_counts"], b["centroid_kev"], b["fwhm_kev"] / 2.354820045,
        overlap_result["background_intercept"], overlap_result["background_slope"],
    )
    double_rss = float(np.sum(((y - two_pred) / weights) ** 2))
    rows = []
    for name, k, rss in (("Single Gaussian", 5, single_rss),
                         ("Two components", 8, double_rss)):
        scores = information_criteria(rss, n, k)
        rows.append({"Model": name, "Parameters": k, "Channels": n,
                     "Weighted RSS": rss, **scores})
    result = pd.DataFrame(rows)
    result["Delta AIC"] = result.AIC - result.AIC.min()
    result["Delta BIC"] = result.BIC - result.BIC.min()
    return result
