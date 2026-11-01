"""Polynomial channel-to-energy calibration with fit diagnostics."""
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class EnergyCalibration:
    coefficients: np.ndarray
    degree: int
    rmse_kev: float
    residuals_kev: np.ndarray
    covariance: np.ndarray

    def predict(self, channels):
        values = np.asarray(channels, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Channels must be finite")
        return np.polyval(self.coefficients, values)

    def uncertainty(self, channels):
        """Approximate 1σ uncertainty in fitted mean energy, not prediction intervals."""
        values = np.asarray(channels, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Channels must be finite")
        design = np.vander(values.reshape(-1), N=self.degree + 1)
        variance = np.einsum("ij,jk,ik->i", design, self.covariance, design)
        result = np.sqrt(np.maximum(variance, 0))
        return result.reshape(values.shape)

def fit_calibration(channels, energies_kev, degree=1):
    """Ordinary least squares polynomial fit of reference energy versus channel.

    Requires more distinct reference points than coefficients to estimate
    coefficient covariance. Residual-based uncertainty is conditional on
    reference data and the model; it excludes reference-line uncertainties.
    """
    if degree not in (1, 2):
        raise ValueError("Only linear and quadratic calibration are supported")
    x = np.asarray(channels, dtype=float)
    y = np.asarray(energies_kev, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < degree + 2:
        raise ValueError("Insufficient matching calibration reference points")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError("Reference values must be finite and energies positive")
    if len(np.unique(x)) != len(x):
        raise ValueError("Reference channels must be unique")
    if np.ptp(x) <= 0:
        raise ValueError("Reference channels must span a range")
    design = np.vander(x, N=degree + 1)
    if np.linalg.matrix_rank(design) < degree + 1:
        raise ValueError("Calibration matrix is rank deficient")
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ coeffs
    dof = len(x) - (degree + 1)
    covariance = (residuals @ residuals / dof) * np.linalg.inv(design.T @ design)
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    return EnergyCalibration(coeffs, degree, rmse, residuals, covariance)

def calibrate_spectrum(spectrum, calibration):
    """Return a copy with energies derived from its channel column."""
    if not {"channel", "counts"}.issubset(spectrum.columns):
        raise ValueError("Spectrum needs channel and counts")
    result = spectrum.copy()
    result["energy_kev"] = calibration.predict(result["channel"].to_numpy())
    if not np.isfinite(result.energy_kev.to_numpy()).all() or (np.diff(result.energy_kev) <= 0).any():
        raise ValueError("Calibration does not yield increasing finite energies")
    return result

def reference_table(channels, energies_kev, calibration):
    return pd.DataFrame({
        "channel": channels,
        "reference_kev": energies_kev,
        "fitted_kev": calibration.predict(channels),
        "residual_kev": calibration.residuals_kev,
        "fit_uncertainty_kev": calibration.uncertainty(channels),
    })
