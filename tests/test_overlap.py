import numpy as np
import pandas as pd
import pytest
from gammaspec.overlap import double_gaussian_background, fit_overlapping_peaks


def synthetic_double():
    x = np.linspace(400, 600, 801)
    expected = double_gaussian_background(x, 500, 485, 8, 350, 505, 10, 15, 0.01)
    counts = np.random.default_rng(7).poisson(expected)
    return pd.DataFrame({"channel": np.arange(len(x)), "energy_kev": x, "counts": counts})


def test_simultaneous_fit_recovers_two_centroids():
    result = fit_overlapping_peaks(synthetic_double(), (485, 505), half_window_kev=30)
    a, b = result["components"]
    assert abs(a["centroid_kev"] - 485) < 2
    assert abs(b["centroid_kev"] - 505) < 2
    assert a["fwhm_kev"] > 0 and b["fwhm_kev"] > 0
    assert a["net_peak_counts"] > 0 and b["net_peak_counts"] > 0
    assert np.isfinite(result["reduced_chi_square"])


@pytest.mark.parametrize("centers,window", [
    ((500,), 30), ((500, 500), 30), ((np.nan, 505), 30),
    ((485, 505), 0), ((485, 505), -5),
])
def test_invalid_inputs(centers, window):
    with pytest.raises(ValueError):
        fit_overlapping_peaks(synthetic_double(), centers, half_window_kev=window)


def test_rejects_insufficient_samples():
    frame = synthetic_double().iloc[::100].reset_index(drop=True)
    with pytest.raises(ValueError):
        fit_overlapping_peaks(frame, (485, 505), half_window_kev=10)
