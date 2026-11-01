"""Educational exponential radioactive decay and two-member decay chains."""
import numpy as np

# Approximate half-lives in years for teaching; not nuclear data evaluation.
HALF_LIVES_YEARS = {
    "Cs-137": 30.05,
    "Co-60": 5.2714,
    "I-131": 8.02 / 365.25,
    "Tc-99m": 6.01 / (24 * 365.25),
    "C-14": 5730.0,
}

def decay_constant(half_life):
    half_life = np.asarray(half_life, dtype=float)
    if not np.isfinite(half_life).all() or np.any(half_life <= 0):
        raise ValueError("Half-life must be positive and finite")
    result = np.log(2) / half_life
    return float(result) if result.ndim == 0 else result

def remaining_activity(initial_activity, elapsed_time, half_life):
    """A(t) = A0 * 2^(-t/T_half), using consistent time units."""
    initial = np.asarray(initial_activity, dtype=float)
    time = np.asarray(elapsed_time, dtype=float)
    if (not np.isfinite(initial).all() or np.any(initial < 0)
            or not np.isfinite(time).all() or np.any(time < 0)):
        raise ValueError("Activity and elapsed time must be finite and nonnegative")
    result = initial * np.exp(-decay_constant(half_life) * time)
    return float(result) if result.ndim == 0 else result

def decay_fraction(elapsed_time, half_life):
    """Fraction of initial parent nuclei that have decayed."""
    return 1 - remaining_activity(1.0, elapsed_time, half_life)

def parent_daughter_atoms(initial_parent_atoms, elapsed_time,
                          parent_half_life, daughter_half_life):
    """Two-member Bateman chain P -> D -> stable, with D(0)=0, 100% branching.

    Returns (N_parent, N_daughter) in atom counts, not activities.
    """
    initial = float(initial_parent_atoms)
    time = np.asarray(elapsed_time, dtype=float)
    if not np.isfinite(initial) or initial < 0 or not np.isfinite(time).all() or np.any(time < 0):
        raise ValueError("Initial atom count and time must be finite and nonnegative")
    lp = decay_constant(parent_half_life)
    ld = decay_constant(daughter_half_life)
    parent = initial * np.exp(-lp * time)
    # Stable near-equal lambda limit to avoid catastrophic cancellation.
    delta = ld - lp
    daughter = initial * lp * np.exp(-lp * time) * (-np.expm1(-delta * time)) / delta if abs(delta) > 1e-10 * max(lp, ld) else initial * lp * time * np.exp(-lp * time)
    if np.ndim(time) == 0:
        return float(parent), float(daughter)
    return parent, daughter
