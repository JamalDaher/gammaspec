import numpy as np
import pandas as pd
import pytest
from gammaspec.model_selection import information_criteria, compare_information_criteria
from gammaspec.core import simulate_spectrum
from gammaspec.overlap import fit_overlapping_peaks


def test_criteria_penalize_extra_parameters_at_fixed_rss():
    a = information_criteria(100, 80, 5)
    b = information_criteria(100, 80, 8)
    assert a["AIC"] < b["AIC"]
    assert a["BIC"] < b["BIC"]


def test_invalid_criteria():
    with pytest.raises(ValueError):
        information_criteria(0, 10, 5)


def test_simulated_overlap_returns_comparable_rows():
    spectrum = simulate_spectrum("Overlapping demo")
    fit = fit_overlapping_peaks(spectrum, (485, 505), half_window_kev=35)
    table = compare_information_criteria(spectrum, fit, (485, 505), 35)
    assert table.Model.tolist() == ["Single Gaussian", "Two components"]
    assert table.Channels.nunique() == 1
    assert table["Delta AIC"].min() == pytest.approx(0)
    assert table["Delta BIC"].min() == pytest.approx(0)
    assert np.isfinite(table[["AIC", "BIC"]].to_numpy()).all()
