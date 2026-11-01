"""Explainable educational reference-line ranking, not isotope identification."""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from .identification import REFERENCE_LINES_KEV

def rank_reference_candidates(observed_kev, tolerance_kev=12.0, library=None):
    """Maximum-cardinality one-to-one matches, then minimum total energy error.

    The ranking score is a heuristic *not* a probability or confidence interval.
    """
    library = REFERENCE_LINES_KEV if library is None else library
    obs = np.asarray(observed_kev, dtype=float)
    if obs.ndim != 1 or not np.isfinite(obs).all() or np.any(obs < 0):
        raise ValueError("Observed energies must be finite, nonnegative and one-dimensional")
    if not np.isfinite(tolerance_kev) or tolerance_kev <= 0:
        raise ValueError("Tolerance must be positive and finite")
    rows, matches = [], []
    for isotope, reference in library.items():
        ref = np.asarray(reference, dtype=float)
        if ref.ndim != 1 or not len(ref) or not np.isfinite(ref).all() or np.any(ref <= 0):
            raise ValueError("Reference energies must be positive and finite")
        matched = []
        if len(obs):
            errors = np.abs(obs[:, None] - ref[None, :])
            # Dummy columns allow each observation to remain unmatched.
            # A valid match always costs less than an unmatched assignment;
            # invalid matches cost more than leaving both sides unmatched.
            penalty = 2 * tolerance_kev + 1
            costs = np.full((len(obs), len(ref) + len(obs)), penalty * 3.0)
            costs[:, :len(ref)] = np.where(errors <= tolerance_kev, errors, penalty * 3.0)
            costs[:, len(ref):] = penalty
            row_ids, col_ids = linear_sum_assignment(costs)
            for i, j in zip(row_ids, col_ids):
                if j < len(ref) and errors[i, j] <= tolerance_kev:
                    delta = float(obs[i] - ref[j])
                    matched.append((i, j, delta))
                    matches.append({"candidate": isotope, "observed_kev": float(obs[i]),
                        "reference_kev": float(ref[j]), "residual_kev": delta,
                        "absolute_error_kev": abs(delta)})
        count = len(matched)
        coverage = count / len(ref)
        mean_error = float(np.mean([abs(item[2]) for item in matched])) if count else None
        # Explicit heuristic: favors coverage, rewards multiple lines and
        # downweights poor energy agreement. Never interpreted as probability.
        score = (coverage * (1 - mean_error / tolerance_kev) * np.log1p(count)
                 if count else 0.0)
        rows.append({"candidate": isotope, "matched_lines": count,
            "reference_lines": len(ref), "coverage": coverage,
            "mean_abs_error_kev": mean_error, "ranking_score": float(score),
            "unmatched_reference_lines": len(ref) - count,
            "unmatched_observed_peaks": len(obs) - count})
    ranking = pd.DataFrame(rows).sort_values(
        ["ranking_score", "matched_lines", "mean_abs_error_kev"],
        ascending=[False, False, True], na_position="last", ignore_index=True)
    detail = pd.DataFrame(matches, columns=["candidate", "observed_kev",
        "reference_kev", "residual_kev", "absolute_error_kev"])
    return ranking, detail
