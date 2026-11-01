import numpy as np
import pandas as pd
import pytest
from gammaspec.comparison import compare_spectra, comparison_metrics

def spec(energies, counts):
    return pd.DataFrame({"channel": np.arange(len(energies)),
                         "energy_kev": energies, "counts": counts})

def test_identical_spectra_have_zero_difference():
    a = spec(np.linspace(0, 100, 101), np.linspace(1, 30, 101))
    result = compare_spectra(a, a)
    assert np.max(np.abs(result.difference_primary_minus_secondary)) < 1e-12
    assert comparison_metrics(result)["pearson_correlation"] == pytest.approx(1)

def test_area_normalization_removes_scale():
    a = spec(np.linspace(0, 100, 101), np.linspace(1, 30, 101))
    b = spec(np.linspace(0, 100, 101), np.linspace(1, 30, 101) * 3)
    result = compare_spectra(a, b)
    assert np.max(np.abs(result.difference_primary_minus_secondary)) < 1e-12
    raw = compare_spectra(a, b, normalization="raw")
    assert raw.secondary.iloc[50] == pytest.approx(3 * raw.primary.iloc[50])

def test_partial_overlap_and_invalid_inputs():
    a = spec(np.linspace(0, 100, 101), np.ones(101))
    b = spec(np.linspace(50, 150, 101), np.ones(101))
    result = compare_spectra(a, b)
    assert result.energy_kev.iloc[0] == 50
    assert result.energy_kev.iloc[-1] == 100
    with pytest.raises(ValueError, match="no overlapping"):
        compare_spectra(a, spec(np.linspace(200, 300, 101), np.ones(101)))
    with pytest.raises(ValueError, match="nonzero"):
        compare_spectra(a, spec(np.linspace(0, 100, 101), np.zeros(101)))
