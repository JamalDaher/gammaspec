import numpy as np
import pandas as pd
from gammaspec.core import simulate_spectrum
from gammaspec.overlap import fit_overlapping_peaks


def test_overlap_preset_reproducible_and_nonnegative():
    a = simulate_spectrum("Overlapping demo", seed=11)
    b = simulate_spectrum("Overlapping demo", seed=11)
    pd.testing.assert_frame_equal(a, b)
    assert (a.counts >= 0).all()
    assert np.issubdtype(a.counts.dtype, np.integer)


def test_overlap_preset_recovers_two_centers():
    spectrum = simulate_spectrum("Overlapping demo", seed=42)
    result = fit_overlapping_peaks(spectrum, (485, 505), half_window_kev=35)
    centers = [component["centroid_kev"] for component in result["components"]]
    assert abs(centers[0] - 485) < 3
    assert abs(centers[1] - 505) < 3
