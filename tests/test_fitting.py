import numpy as np
import pytest
from gammaspec.core import simulate_spectrum
from gammaspec.fitting import FACTOR, gaussian_background, fit_photopeak, fit_detected_peaks

def test_gaussian_fwhm_definition():
    assert FACTOR == pytest.approx(2.35482, rel=1e-5)

def test_gaussian_center_and_background():
    x = np.array([0., 2., 4.])
    y = gaussian_background(x, 10, 2, 1, 3, 0)
    assert y[1] == pytest.approx(13)
    assert y[0] == pytest.approx(y[2])

def test_cs137_fit():
    s = simulate_spectrum("Cs-137", seed=42)
    r = fit_photopeak(s, 661.657)
    assert abs(r["centroid_kev"] - 661.657) < 3
    assert 20 < r["fwhm_kev"] < 60
    assert r["net_peak_counts"] > 0
    assert r["centroid_error_kev"] > 0

def test_co60_fit():
    s = simulate_spectrum("Co-60", seed=42)
    for energy in (1173.228, 1332.492):
        r = fit_photopeak(s, energy, half_window_kev=55)
        assert abs(r["centroid_kev"] - energy) < 5

def test_invalid_window():
    with pytest.raises(ValueError):
        fit_photopeak(simulate_spectrum(), 661, half_window_kev=-1)

def test_automatic_fit_table():
    result = fit_detected_peaks(simulate_spectrum(), prominence=150)
    assert not result.empty
    assert "status" in result.columns
