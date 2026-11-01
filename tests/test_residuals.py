import numpy as np
import pandas as pd
import pytest
from gammaspec.residuals import residual_comparison, residual_rms


def test_residual_comparison_uses_same_channels():
    x = np.arange(450.0, 541.0)
    spectrum = pd.DataFrame({"energy_kev": x, "counts": np.full(len(x), 10)})
    single = {"amplitude_counts": 0, "centroid_kev": 495,
              "sigma_kev": 10, "background_at_centroid": 10, "slope": 0}
    component = lambda mu: {"amplitude_counts": 0, "centroid_kev": mu,
                            "fwhm_kev": 20}
    overlap = {"components": [component(485), component(505)],
               "background_intercept": 10, "background_slope": 0}
    result = residual_comparison(spectrum, single, overlap, (485, 505), 20)
    assert len(result) == 61
    assert np.allclose(result.single_standardized_residual, 0)
    assert np.allclose(result.two_standardized_residual, 0)


def test_residual_rms():
    assert residual_rms([3, 4]) == pytest.approx(np.sqrt(12.5))
    with pytest.raises(ValueError):
        residual_rms([])
