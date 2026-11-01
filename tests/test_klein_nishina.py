import numpy as np
import pytest
from gammaspec.klein_nishina import (CLASSICAL_ELECTRON_RADIUS_CM,
    differential_cross_section, angular_distribution, total_cross_section)

def test_forward_cross_section():
    for energy in (100, 662, 1332):
        assert differential_cross_section(energy, 0) == pytest.approx(CLASSICAL_ELECTRON_RADIUS_CM**2)

def test_normalization_and_positive():
    angles, density, cross = angular_distribution(662)
    assert np.all(density >= 0)
    assert np.all(cross > 0)
    assert np.trapezoid(density, angles) == pytest.approx(1, rel=1e-9)
    assert density[0] == pytest.approx(0, abs=1e-12)

def test_low_energy_thomson_limit():
    thomson = (8 * np.pi / 3) * CLASSICAL_ELECTRON_RADIUS_CM**2
    assert total_cross_section(0.01) == pytest.approx(thomson, rel=1e-4)

def test_total_cross_section_decreases_with_energy():
    assert total_cross_section(100) > total_cross_section(662) > total_cross_section(1332)

def test_bad_inputs():
    with pytest.raises(ValueError):
        differential_cross_section(-1, 90)
    with pytest.raises(ValueError):
        angular_distribution(662, [90, 0])
