import numpy as np
import pytest
from gammaspec.core import simulate_spectrum
from gammaspec.poisson import poisson_nll, compare_poisson_models

def test_poisson_nll_known_values():
    assert poisson_nll([0, 1], [1, 1]) == pytest.approx(2.0)
    assert poisson_nll([0], [0]) == np.inf

def test_poisson_model_comparison_demo():
    spectrum = simulate_spectrum("Overlapping demo")
    result = compare_poisson_models(spectrum, (485, 505), 35)
    assert result.Model.tolist() == ["Single Gaussian", "Two components"]
    assert result.Parameters.tolist() == [4, 7]
    assert result.Channels.nunique() == 1
    assert np.isfinite(result[["AIC", "BIC"]].to_numpy()).all()
    assert result["Delta AIC"].min() == pytest.approx(0)
    assert result["Delta BIC"].min() == pytest.approx(0)
