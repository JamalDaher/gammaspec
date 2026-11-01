import numpy as np
import pandas as pd
import pytest
from gammaspec.background_subtraction import subtract_background, integrate_net_region

def spectrum(counts):
    return pd.DataFrame({"channel": [0, 1, 2], "energy_kev": [0., 1., 2.], "counts": counts})

def test_live_time_scaling_and_poisson_uncertainty():
    result = subtract_background(spectrum([10, 4, 0]), spectrum([2, 6, 0]), 20, 40)
    np.testing.assert_allclose(result.net_counts, [9, 1, 0])
    np.testing.assert_allclose(result.net_sigma_counts, np.sqrt([10.5, 5.5, 0]))
    np.testing.assert_allclose(result.net_rate_cps, [0.45, 0.05, 0])

def test_negative_net_counts_are_preserved():
    result = subtract_background(spectrum([0, 1, 0]), spectrum([0, 4, 0]), 20, 20)
    assert result.net_counts.iloc[1] == -3

def test_region_uncertainty_quadrature():
    result = subtract_background(spectrum([10, 4, 0]), spectrum([2, 6, 0]), 20, 40)
    roi = integrate_net_region(result, 0, 1)
    assert roi["net_counts"] == 10
    assert roi["uncertainty_counts"] == pytest.approx(4)

def test_reject_mismatched_grids_and_invalid_live_time():
    with pytest.raises(ValueError, match="Live times"):
        subtract_background(spectrum([1, 2, 3]), spectrum([1, 2, 3]), 0, 30)
    other = spectrum([1, 2, 3])
    other.loc[1, "energy_kev"] = 1.1
    with pytest.raises(ValueError, match="energy grids"):
        subtract_background(spectrum([1, 2, 3]), other, 30, 30)
