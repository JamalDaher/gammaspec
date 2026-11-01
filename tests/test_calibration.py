import numpy as np
import pandas as pd
import pytest
from gammaspec.calibration import fit_calibration, calibrate_spectrum, reference_table

def test_linear_exact():
    x = [100, 500, 1000, 1500]
    y = [80, 400, 800, 1200]
    fit = fit_calibration(x, y)
    assert fit.predict(750) == pytest.approx(600)
    assert fit.rmse_kev < 1e-8

def test_quadratic_exact():
    x = np.array([100, 300, 500, 700, 900], dtype=float)
    y = 0.001 * x*x + 0.5*x + 5
    fit = fit_calibration(x, y, degree=2)
    assert fit.predict(400) == pytest.approx(365)
    assert fit.rmse_kev < 1e-8

def test_uncertainty_nonnegative():
    fit = fit_calibration([100, 300, 500, 700], [81, 241, 399, 561])
    assert (fit.uncertainty([200, 400]) >= 0).all()

def test_reference_table():
    x = [100, 300, 500, 700]
    y = [80, 240, 400, 560]
    result = reference_table(x, y, fit_calibration(x, y))
    assert len(result) == 4
    assert "residual_kev" in result

def test_calibrate_spectrum():
    fit = fit_calibration([100, 500, 1000], [80, 400, 800])
    data = pd.DataFrame({"channel": [0, 1, 2], "counts": [10, 20, 30]})
    result = calibrate_spectrum(data, fit)
    assert result.energy_kev.iloc[2] == pytest.approx(1.6)

def test_reject_duplicate_channels():
    with pytest.raises(ValueError):
        fit_calibration([100, 100, 500], [80, 81, 400])

def test_reject_insufficient_points():
    with pytest.raises(ValueError):
        fit_calibration([100, 500], [80, 400])

def test_reject_nonmonotonic_calibration():
    fit = fit_calibration([100, 300, 500], [500, 300, 100])
    with pytest.raises(ValueError):
        calibrate_spectrum(pd.DataFrame({"channel": [0, 1, 2], "counts": [1, 2, 3]}), fit)
