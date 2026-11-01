import numpy as np
import pytest
from gammaspec.detector_resolution import fwhm_kev, resolution_percent, illustrative_doublet

def test_reference_fwhm_and_scaling():
    assert fwhm_kev(662, "HPGe (illustrative)") == pytest.approx(1.8)
    assert fwhm_kev(662, "NaI(Tl) (illustrative)") == pytest.approx(46)
    assert fwhm_kev(2648, "HPGe (illustrative)") == pytest.approx(3.6)
    assert resolution_percent(662, "HPGe (illustrative)") == pytest.approx(100 * 1.8 / 662)

def test_doublet_sum():
    x = np.linspace(400, 600, 1001)
    a, b, total = illustrative_doublet(x)
    np.testing.assert_allclose(total, a + b)
    assert np.all(total >= 0)

def test_invalid_detector_or_energy():
    with pytest.raises(ValueError):
        fwhm_kev(662, "Unknown")
    with pytest.raises(ValueError):
        fwhm_kev(-1, "HPGe (illustrative)")
