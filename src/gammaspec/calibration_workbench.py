"""Validation helpers for energy calibration and measured detector resolution."""
import numpy as np
import pandas as pd
from .calibration import reference_table

def calibration_diagnostics(channels, energies_kev, calibration, spectrum_channels=None):
    table = reference_table(channels, energies_kev, calibration)
    x = np.asarray(channels, dtype=float)
    residuals = table.residual_kev.to_numpy(dtype=float)
    if not np.isfinite(residuals).all():
        raise ValueError("Calibration residuals must be finite")
    if spectrum_channels is not None:
        grid = np.asarray(spectrum_channels, dtype=float)
        if grid.ndim != 1 or not np.isfinite(grid).all() or len(grid) < 2:
            raise ValueError("Spectrum channels must be a finite 1D array")
        predicted = calibration.predict(grid)
        if np.any(np.diff(predicted) <= 0):
            raise ValueError("Calibration is not increasing over the spectrum")
        extrapolated = bool(np.min(grid) < np.min(x) or np.max(grid) > np.max(x))
    else:
        extrapolated = False
    stats = {"rmse_kev": float(np.sqrt(np.mean(residuals**2))),
        "max_absolute_residual_kev": float(np.max(np.abs(residuals))),
        "mean_residual_kev": float(np.mean(residuals)),
        "reference_count": int(len(x)), "polynomial_degree": int(calibration.degree),
        "extrapolates_beyond_references": extrapolated}
    return table, stats

def measured_resolution_table(energies_kev, fwhm_values_kev):
    energies = np.asarray(energies_kev, dtype=float)
    widths = np.asarray(fwhm_values_kev, dtype=float)
    if (energies.ndim != 1 or widths.ndim != 1 or len(energies) != len(widths)
            or len(energies) == 0 or not np.isfinite(energies).all()
            or not np.isfinite(widths).all() or np.any(energies <= 0)
            or np.any(widths <= 0)):
        raise ValueError("Energy and FWHM must be paired positive finite arrays")
    return pd.DataFrame({"energy_kev": energies, "fwhm_kev": widths,
                         "resolution_percent": 100 * widths / energies})
