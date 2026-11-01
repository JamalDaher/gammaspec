import numpy as np
import pytest
from gammaspec.calibration import fit_calibration
from gammaspec.calibration_workbench import calibration_diagnostics, measured_resolution_table

def test_linear_calibration_diagnostics():
    channels = np.array([100., 200., 300., 400.])
    energies = 0.8 * channels + 2
    fit = fit_calibration(channels, energies, degree=1)
    table, stats = calibration_diagnostics(channels, energies, fit,
        spectrum_channels=np.arange(0, 500))
    assert stats["rmse_kev"] < 1e-9
    assert stats["extrapolates_beyond_references"] is True
    assert len(table) == 4

def test_quadratic_calibration_diagnostics():
    x = np.array([100., 200., 300., 400., 500.])
    y = 0.0001*x*x + 0.6*x + 2
    fit = fit_calibration(x, y, degree=2)
    _, stats = calibration_diagnostics(x, y, fit, spectrum_channels=x)
    assert stats["rmse_kev"] < 1e-8
    assert stats["extrapolates_beyond_references"] is False

def test_measured_resolution_table():
    result = measured_resolution_table([100., 1000.], [5., 10.])
    assert result.resolution_percent.tolist() == pytest.approx([5., 1.])

def test_invalid_resolution():
    with pytest.raises(ValueError):
        measured_resolution_table([100], [-1])
    with pytest.raises(ValueError):
        measured_resolution_table([100, 200], [1])
