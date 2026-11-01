"""Educational gamma-line reference matching, not isotope identification.

Energies are approximate reference values; line intensities, branching ratios,
detector efficiency, escape peaks, and coincidence summing are not modeled.
"""
import numpy as np
import pandas as pd

REFERENCE_LINES_KEV = {
    "Cs-137": (661.657,),
    "Co-60": (1173.228, 1332.492),
    "Na-22": (511.0, 1274.537),
    "K-40": (1460.822,),
    "Am-241": (59.5409,),
}

def match_candidates(peak_energies_kev, tolerance_kev=12.0, library=None):
    """Match observed peaks to reference lines, one-to-one per isotope.

    Ranked by fraction of expected lines matched then mean absolute deviation.
    Does not compute isotope probabilities or exclude mixed-source spectra.
    """
    library = REFERENCE_LINES_KEV if library is None else library
    energies = np.asarray(peak_energies_kev, dtype=float)
    if energies.ndim != 1 or not np.isfinite(energies).all() or (energies < 0).any():
        raise ValueError("Peak energies must be a finite nonnegative 1D sequence")
    if not np.isfinite(tolerance_kev) or tolerance_kev <= 0:
        raise ValueError("Tolerance must be positive and finite")
    results = []
    for name, lines in library.items():
        refs = np.asarray(lines, dtype=float)
        if refs.ndim != 1 or len(refs) == 0 or not np.isfinite(refs).all() or (refs <= 0).any():
            raise ValueError("Invalid reference line library")
        possible = sorted(
            (abs(float(obs - ref)), i, j, float(obs), float(ref))
            for i, obs in enumerate(energies) for j, ref in enumerate(refs)
            if abs(obs - ref) <= tolerance_kev
        )
        used_observed, used_reference, matched = set(), set(), []
        for delta, i, j, observed, reference in possible:
            if i not in used_observed and j not in used_reference:
                used_observed.add(i)
                used_reference.add(j)
                matched.append((observed, reference, delta))
        if not matched:
            continue
        results.append({
            "candidate": name,
            "matched_lines": len(matched),
            "reference_lines": len(refs),
            "coverage": len(matched) / len(refs),
            "mean_abs_error_kev": float(np.mean([item[2] for item in matched])),
            "matches": "; ".join(f"{obs:.2f}→{ref:.2f} keV" for obs, ref, _ in matched),
        })
    columns = ["candidate", "matched_lines", "reference_lines", "coverage",
               "mean_abs_error_kev", "matches"]
    return pd.DataFrame(results, columns=columns).sort_values(
        ["coverage", "matched_lines", "mean_abs_error_kev"],
        ascending=[False, False, True], ignore_index=True
    ) if results else pd.DataFrame(columns=columns)
