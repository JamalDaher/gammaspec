"""Comparisons on a shared energy grid, with explicit normalization semantics."""
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from .core import _validate

def compare_spectra(primary, secondary, *, normalization="area", points=500):
    """Interpolate count densities on the common energy interval.

    Count density is counts/channel divided by local channel width in keV.
    Area normalization compares spectral shape, not absolute rates. Interpolated
    values are descriptive and not independent Poisson observations.
    """
    _validate(primary)
    _validate(secondary)
    if normalization not in ("area", "raw"):
        raise ValueError("Normalization must be area or raw")
    if not isinstance(points, int) or points < 16 or points > 10000:
        raise ValueError("Grid points must be an integer from 16 to 10000")
    a_energy = primary.energy_kev.to_numpy(dtype=float)
    b_energy = secondary.energy_kev.to_numpy(dtype=float)
    lower = max(a_energy[0], b_energy[0])
    upper = min(a_energy[-1], b_energy[-1])
    if not upper > lower:
        raise ValueError("Spectra have no overlapping energy interval")
    grid = np.linspace(lower, upper, points)
    def density(frame):
        energy = frame.energy_kev.to_numpy(dtype=float)
        counts = frame.counts.to_numpy(dtype=float)
        if len(energy) < 2:
            raise ValueError("At least two channels required for comparison")
        widths = np.gradient(energy)
        if np.any(widths <= 0):
            raise ValueError("Energy bins must have positive width")
        return np.interp(grid, energy, counts / widths)
    a, b = density(primary), density(secondary)
    if normalization == "area":
        a_area = trapezoid(a, grid)
        b_area = trapezoid(b, grid)
        if a_area <= 0 or b_area <= 0:
            raise ValueError("Area normalization requires nonzero counts in both spectra")
        a, b = a / a_area, b / b_area
    return pd.DataFrame({"energy_kev": grid, "primary": a, "secondary": b,
                         "difference_primary_minus_secondary": a - b})

def comparison_metrics(comparison):
    a = comparison.primary.to_numpy(dtype=float)
    b = comparison.secondary.to_numpy(dtype=float)
    return {"mean_absolute_difference": float(np.mean(np.abs(a - b))),
            "rms_difference": float(np.sqrt(np.mean((a - b)**2))),
            "pearson_correlation": (float(np.corrcoef(a, b)[0, 1])
                                    if np.std(a) > 0 and np.std(b) > 0 else None)}
