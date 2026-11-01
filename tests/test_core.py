import numpy as np
import pandas as pd
import pytest
from gammaspec.core import simulate_spectrum, find_photopeaks, read_spectrum, write_spectrum

def test_reproducible():
    a = simulate_spectrum(seed=7)
    b = simulate_spectrum(seed=7)
    pd.testing.assert_frame_equal(a, b)

def test_nonnegative_integer_counts():
    s = simulate_spectrum("Co-60")
    assert (s.counts >= 0).all()
    assert np.issubdtype(s.counts.dtype, np.integer)

def test_cs137_peak():
    s = simulate_spectrum("Cs-137")
    p = find_photopeaks(s, prominence=150)
    assert any(abs(p.energy_kev - 661.657) < 20)

def test_co60_peaks():
    s = simulate_spectrum("Co-60")
    p = find_photopeaks(s, prominence=150)
    assert any(abs(p.energy_kev - 1173.228) < 30)
    assert any(abs(p.energy_kev - 1332.492) < 30)

def test_csv_roundtrip(tmp_path):
    s = simulate_spectrum()
    path = tmp_path / "test.csv"
    write_spectrum(s, path)
    pd.testing.assert_frame_equal(s, read_spectrum(path))

def test_invalid_isotope():
    with pytest.raises(ValueError):
        simulate_spectrum("unknown")

def test_invalid_csv(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("channel,energy_kev,counts\n0,1,-2\n")
    with pytest.raises(ValueError):
        read_spectrum(path)
