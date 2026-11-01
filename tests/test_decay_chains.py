import numpy as np
import pytest
from gammaspec.decay_chains import chain_populations, equilibrium_class
from gammaspec.decay import parent_daughter_atoms

def test_two_generation_matches_bateman():
    times = np.array([0., 0.5, 2., 8.])
    populations, activities = chain_populations([4., 1.], times, 1000)
    parent, daughter = parent_daughter_atoms(1000, times, 4., 1.)
    assert np.allclose(populations[:, 0], parent)
    assert np.allclose(populations[:, 1], daughter)
    assert np.allclose(activities, populations * (np.log(2) / [4., 1.]))

def test_equal_rates_and_branching():
    times = np.array([0., 1., 5.])
    populations, _ = chain_populations([2., 2., 2.], times, 1000, [1., 1.])
    assert np.all(populations >= 0)
    assert populations[0].tolist() == pytest.approx([1000, 0, 0])
    unbranched, _ = chain_populations([2., 2., 2.], times, 1000, [0., 1.])
    assert np.allclose(unbranched[:, 1:], 0)

def test_classification_and_invalid_inputs():
    assert "Secular" in equilibrium_class(1000, 1)
    assert "Transient" in equilibrium_class(10, 1)
    assert "No conventional" in equilibrium_class(1, 10)
    with pytest.raises(ValueError):
        chain_populations([1, 2], [0, -1])
    with pytest.raises(ValueError):
        chain_populations([1, 2], [0, 1], branches=[1.5])
    with pytest.raises(ValueError):
        chain_populations([1], [0, 1])
