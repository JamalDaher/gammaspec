import numpy as np
import pytest
from gammaspec.decay import (decay_constant, remaining_activity,
                            decay_fraction, parent_daughter_atoms)

def test_half_lives_and_fraction():
    assert remaining_activity(100, 0, 5) == pytest.approx(100)
    assert remaining_activity(100, 5, 5) == pytest.approx(50)
    assert remaining_activity(100, 10, 5) == pytest.approx(25)
    assert decay_fraction(5, 5) == pytest.approx(0.5)
    assert decay_constant(5) == pytest.approx(np.log(2) / 5)

def test_parent_daughter_initial_and_conservation_when_daughter_stable_like():
    p, d = parent_daughter_atoms(1000, np.array([0, 1, 5]), 2, 1e12)
    assert p[0] == pytest.approx(1000)
    assert d[0] == pytest.approx(0)
    assert np.all(p >= 0) and np.all(d >= 0)
    assert p[1] + d[1] == pytest.approx(1000, rel=1e-10)

def test_equal_half_lives():
    p, d = parent_daughter_atoms(100, 2, 2, 2)
    lam = np.log(2) / 2
    assert d == pytest.approx(100 * lam * 2 * np.exp(-lam * 2))

def test_invalid_values():
    with pytest.raises(ValueError):
        remaining_activity(-1, 2, 3)
    with pytest.raises(ValueError):
        remaining_activity(1, -2, 3)
    with pytest.raises(ValueError):
        decay_constant(0)
    with pytest.raises(ValueError):
        parent_daughter_atoms(100, -1, 2, 3)
