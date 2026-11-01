"""Idealized radioactive chains with branching and numerical Bateman solutions.

Uses matrix exponentials to remain stable for equal or nearly equal decay constants.
All times and half-lives must share units. Populations are atom counts, not Bq.
"""
import numpy as np
from scipy.linalg import expm

def chain_populations(half_lives, times, initial_atoms=1000.0, branches=None):
    """Sequential chain with optional branching to a stable sink.

    Each radioactive generation i -> i+1 with fraction branches[i], otherwise
    to an untracked stable sink. Last generation decays to a stable sink.
    """
    half_lives = np.asarray(half_lives, dtype=float)
    times = np.asarray(times, dtype=float)
    initial_atoms = float(initial_atoms)
    n = len(half_lives)
    if n < 2 or n > 8 or not np.isfinite(half_lives).all() or np.any(half_lives <= 0):
        raise ValueError("Provide 2 to 8 positive finite half-lives")
    if not np.isfinite(times).all() or np.any(times < 0) or not np.isfinite(initial_atoms) or initial_atoms < 0:
        raise ValueError("Times and initial atoms must be finite and nonnegative")
    if branches is None:
        branches = np.ones(n - 1)
    branches = np.asarray(branches, dtype=float)
    if branches.shape != (n - 1,) or not np.isfinite(branches).all() or np.any((branches < 0) | (branches > 1)):
        raise ValueError("Each branch fraction must be between zero and one")
    rates = np.log(2.0) / half_lives
    matrix = np.diag(-rates)
    for i, fraction in enumerate(branches):
        matrix[i + 1, i] = fraction * rates[i]
    initial = np.zeros(n)
    initial[0] = initial_atoms
    flat_times = np.atleast_1d(times).reshape(-1)
    # A nearly repeated eigenvalue can cause cancellation in generic matrix
    # exponential implementations. Use the stable two-member Bateman limit.
    if n == 2 and abs(rates[1] - rates[0]) <= 1e-10 * max(rates):
        parent = initial_atoms * np.exp(-rates[0] * flat_times)
        daughter = (initial_atoms * branches[0] * rates[0] * flat_times
                    * np.exp(-rates[0] * flat_times))
        populations = np.column_stack((parent, daughter))
    else:
        populations = np.stack([expm(matrix * t) @ initial for t in flat_times])
    populations = np.maximum(populations, 0.0)
    activities = populations * rates
    if times.ndim == 0:
        return populations[0], activities[0]
    return populations.reshape(times.shape + (n,)), activities.reshape(times.shape + (n,))

def equilibrium_class(parent_half_life, daughter_half_life):
    """Qualitative classification, not proof that equilibrium has been reached."""
    p, d = float(parent_half_life), float(daughter_half_life)
    if not np.isfinite([p, d]).all() or min(p, d) <= 0:
        raise ValueError("Half-lives must be positive")
    if p / d >= 100:
        return "Secular-equilibrium regime (parent much longer-lived)"
    if p > d:
        return "Transient-equilibrium regime (parent longer-lived)"
    return "No conventional secular/transient equilibrium (daughter as long- or longer-lived)"
