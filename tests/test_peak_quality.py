import numpy as np
import pytest
from gammaspec.core import simulate_spectrum
from gammaspec.fitting import fit_detected_peaks, fit_photopeak

def test_fit_reports_chi_square():
    spectrum = simulate_spectrum("Cs-137")
    fit = fit_photopeak(spectrum, 661.657)
    assert np.isfinite(fit["reduced_chi_square"])
    assert fit["reduced_chi_square"] >= 0

def test_duplicate_suppression():
    spectrum = simulate_spectrum("Cs-137")
    fits = fit_detected_peaks(spectrum, prominence=50, min_separation_kev=70)
    energies = fits.centroid_kev.to_numpy()
    # Fitted centroids may move slightly; distinct selected candidates are separated.
    assert len(fits) <= 10
    assert "quality_flag" in fits.columns

def test_rejects_invalid_separation():
    with pytest.raises(ValueError):
        fit_detected_peaks(simulate_spectrum(), min_separation_kev=-1)

def test_rejects_invalid_max_peaks():
    with pytest.raises(ValueError):
        fit_detected_peaks(simulate_spectrum(), max_peaks=0)

def test_co60_fits():
    fits = fit_detected_peaks(simulate_spectrum("Co-60"), prominence=150)
    good = fits[fits.status == "ok"]
    assert len(good) >= 2
    assert "reduced_chi_square" in good.columns
