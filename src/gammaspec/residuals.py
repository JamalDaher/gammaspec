"""Residual diagnostics for comparing local Gaussian fit models."""
import numpy as np
import pandas as pd
from .fitting import gaussian_background
from .overlap import double_gaussian_background


def residual_comparison(spectrum, single_fit, overlap_result, centers, half_window_kev):
    """Evaluate both fitted models on identical channels in the two-peak window.

    Models may have been optimized on different windows. Their residual metrics
    are descriptive, not a formal model-selection test.
    """
    if single_fit is None or overlap_result is None:
        raise ValueError("Both successful fits are required")
    c = sorted(float(v) for v in centers)
    local = spectrum.loc[
        (spectrum.energy_kev >= c[0] - half_window_kev)
        & (spectrum.energy_kev <= c[1] + half_window_kev)
    ]
    if local.empty:
        raise ValueError("No channels in comparison window")
    x = local.energy_kev.to_numpy(dtype=float)
    observed = local.counts.to_numpy(dtype=float)
    single = gaussian_background(
        x, single_fit["amplitude_counts"], single_fit["centroid_kev"],
        single_fit["sigma_kev"], single_fit["background_at_centroid"], single_fit["slope"],
    )
    a, b = overlap_result["components"]
    two = double_gaussian_background(
        x, a["amplitude_counts"], a["centroid_kev"], a["fwhm_kev"] / 2.354820045,
        b["amplitude_counts"], b["centroid_kev"], b["fwhm_kev"] / 2.354820045,
        overlap_result["background_intercept"], overlap_result["background_slope"],
    )
    scale = np.sqrt(np.maximum(observed, 1))
    return pd.DataFrame({
        "energy_kev": x,
        "observed_counts": observed,
        "single_model_counts": single,
        "two_model_counts": two,
        "single_residual_counts": observed - single,
        "two_residual_counts": observed - two,
        "single_standardized_residual": (observed - single) / scale,
        "two_standardized_residual": (observed - two) / scale,
    })


def residual_rms(values):
    """Root mean square of residual values."""
    a = np.asarray(values, dtype=float)
    if a.size == 0 or not np.isfinite(a).all():
        raise ValueError("Residuals must be finite and nonempty")
    return float(np.sqrt(np.mean(a * a)))
