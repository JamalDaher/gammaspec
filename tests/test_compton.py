import numpy as np
import pytest
from gammaspec.compton import scattered_energy, recoil_energy, compton_edge, scattered_wavelength_shift_pm

def test_zero_angle():
    assert scattered_energy(661.657, 0) == pytest.approx(661.657)
    assert recoil_energy(661.657, 0) == pytest.approx(0)
    assert scattered_wavelength_shift_pm(0) == pytest.approx(0)

def test_compton_edge_and_conservation():
    energy = 661.657
    edge = compton_edge(energy)
    assert edge == pytest.approx(477.33, abs=0.2)
    angles = np.array([0, 45, 90, 180])
    assert np.allclose(scattered_energy(energy, angles) + recoil_energy(energy, angles), energy)
    assert recoil_energy(energy, 180) == pytest.approx(edge)
    assert np.all(np.diff(scattered_energy(energy, angles)) < 0)

def test_invalid_parameters():
    with pytest.raises(ValueError):
        scattered_energy(-1, 90)
    with pytest.raises(ValueError):
        scattered_energy(662, 181)
