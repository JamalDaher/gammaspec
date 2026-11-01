"""Poisson maximum-likelihood fits and information criteria for binned counts."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from .fitting import gaussian_background
from .overlap import double_gaussian_background

def poisson_nll(y, expected):
    """Full Poisson negative log likelihood, including log factorials."""
    y = np.asarray(y, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if y.shape != expected.shape or y.size == 0 or not np.isfinite(y).all() or np.any(y < 0):
        raise ValueError("Invalid observed counts")
    if not np.isfinite(expected).all() or np.any(expected <= 0):
        return np.inf
    return float(np.sum(expected - y * np.log(expected) + gammaln(y + 1)))

def compare_poisson_models(spectrum, centers, half_window_kev):
    """Fit 1 and 2 Gaussian peaks plus constant background on identical channels.

    The constant background is a deliberate constraint ensuring positive expected
    counts; fitted parameter counts are 4 and 7, respectively.
    """
    centers = np.sort(np.asarray(centers, dtype=float))
    if centers.shape != (2,) or not np.isfinite(centers).all() or centers[0] == centers[1]:
        raise ValueError("Two distinct finite centers required")
    if not np.isfinite(half_window_kev) or half_window_kev <= 0:
        raise ValueError("Invalid half-window")
    local = spectrum.loc[(spectrum.energy_kev >= centers[0] - half_window_kev)
                         & (spectrum.energy_kev <= centers[1] + half_window_kev)]
    x = local.energy_kev.to_numpy(dtype=float)
    y = local.counts.to_numpy(dtype=float)
    n = len(x)
    if n <= 8 or not np.isfinite(y).all() or np.any(y < 0) or np.any(np.diff(x) <= 0):
        raise ValueError("Insufficient or invalid channels")
    step = float(np.median(np.diff(x)))
    width = max(3 * step, half_window_kev / 4)
    bg = max(float(np.percentile(y, 15)), 0.1)
    amplitude = max(float(y.max() - bg), 1.0)
    lo, hi = float(x.min()), float(x.max())
    # Parameters: amplitude, center, width, [second peak...], constant background.
    models = [
        ("Single Gaussian", [amplitude, float(x[np.argmax(y)]), width, bg],
         [(0, None), (lo, hi), (step / 2, hi - lo), (1e-8, None)]),
        ("Two components", [amplitude * 0.65, centers[0], width,
                            amplitude * 0.45, centers[1], width, bg],
         [(0, None), (lo, hi), (step / 2, hi - lo),
          (0, None), (lo, hi), (step / 2, hi - lo), (1e-8, None)]),
    ]
    rows = []
    for name, initial, bounds in models:
        def objective(p):
            if len(p) == 4:
                expected = gaussian_background(x, p[0], p[1], p[2], p[3], 0)
            else:
                expected = double_gaussian_background(
                    x, p[0], p[1], p[2], p[3], p[4], p[5], p[6], 0)
            return poisson_nll(y, expected)
        best = None
        starts = [initial]
        if len(initial) == 7:
            starts.append([amplitude * 0.45, centers[0], width,
                           amplitude * 0.65, centers[1], width, bg])
        for start in starts:
            fit = minimize(objective, start, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 3000})
            if np.isfinite(fit.fun) and (best is None or fit.fun < best.fun):
                best = fit
        if best is None or not best.success:
            raise ValueError(f"{name} Poisson optimization did not converge")
        k = len(initial)
        nll = float(best.fun)
        rows.append({"Model": name, "Parameters": k, "Channels": n,
                     "Poisson NLL": nll, "AIC": 2 * k + 2 * nll,
                     "BIC": k * np.log(n) + 2 * nll})
    result = pd.DataFrame(rows)
    result["Delta AIC"] = result["AIC"] - result["AIC"].min()
    result["Delta BIC"] = result["BIC"] - result["BIC"].min()
    return result
