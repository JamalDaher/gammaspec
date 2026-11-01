"""Numerical and physical invariants across the educational simulation modules."""
import numpy as np
import pytest
from gammaspec.decay import remaining_activity, parent_daughter_atoms
from gammaspec.decay_chains import chain_populations
from gammaspec.compton import scattered_energy, compton_edge

@pytest.mark.parametrize("half_life", [0.01, 1.0, 1e6])
def test_exponential_decay_monotone_and_finite(half_life):
    times = np.linspace(0, 10 * half_life, 101)
    values = remaining_activity(1000, times, half_life)
    assert np.isfinite(values).all()
    assert np.all(np.diff(values) <= 1e-10)
    assert values[0] == pytest.approx(1000)
    assert values[-1] == pytest.approx(1000 / 1024)

@pytest.mark.parametrize("ratio", [0.01, 1.0, 100.0])
def test_three_generation_chain_is_nonnegative_and_bounded(ratio):
    times = np.linspace(0, 100, 51)
    atoms, activity = chain_populations([10 * ratio, 10, 1], times, 1000, [0.7, 0.4])
    assert np.isfinite(atoms).all() and np.isfinite(activity).all()
    assert np.all(atoms >= 0) and np.all(activity >= 0)
    assert np.all(atoms.sum(axis=1) <= 1000 + 1e-6)

def test_two_member_near_equal_half_lives():
    t = np.linspace(0, 10, 31)
    parent, daughter = parent_daughter_atoms(1000, t, 2, 2 * (1 + 1e-12))
    atoms, _ = chain_populations([2, 2 * (1 + 1e-12)], t, 1000)
    assert np.allclose(parent, atoms[:, 0], rtol=1e-8)
    assert np.allclose(daughter, atoms[:, 1], rtol=1e-8, atol=1e-7)

def test_compton_energy_bounds():
    incident = 661.657
    for angle in [0, 30, 90, 150, 180]:
        scattered = scattered_energy(incident, angle)
        assert 0 < scattered <= incident
    assert compton_edge(incident) == pytest.approx(
        incident - scattered_energy(incident, 180))
