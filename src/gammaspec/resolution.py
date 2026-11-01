"""Synthetic peak-separation sweep with repeated Poisson noise draws."""
import numpy as np
import pandas as pd
from .overlap import fit_overlapping_peaks

def resolution_study(separations=(10, 15, 20, 25, 30, 40), trials=5,
                     live_time_s=60, seed=42, tolerance_kev=5):
    """Recovery fraction is a predefined accuracy criterion, not proof of resolvability."""
    separations = np.asarray(separations, dtype=float)
    if (separations.ndim != 1 or not len(separations) or
            not np.isfinite(separations).all() or np.any(separations <= 0) or
            np.any(separations > 100)):
        raise ValueError("Separations must be positive finite values <= 100 keV")
    if isinstance(trials, bool) or not isinstance(trials, int) or not 2 <= trials <= 30:
        raise ValueError("Trials must be between 2 and 30")
    if not np.isfinite(live_time_s) or live_time_s <= 0 or not np.isfinite(tolerance_kev) or tolerance_kev <= 0:
        raise ValueError("Invalid live time or tolerance")
    rng = np.random.default_rng(seed)
    energy = np.linspace(0, 1600, 2048)
    step = energy[1] - energy[0]
    rows = []
    for separation in separations:
        centers = (495 - separation / 2, 495 + separation / 2)
        expected = (0.1 * live_time_s + 0.3 * live_time_s * np.exp(-energy / 450)
                    + 12 * live_time_s * np.exp(-0.5 * ((energy - centers[0]) / 8) ** 2)
                    + 8.4 * live_time_s * np.exp(-0.5 * ((energy - centers[1]) / 10) ** 2))
        for trial in range(1, trials + 1):
            counts = rng.poisson(expected)
            spectrum = pd.DataFrame({"channel": np.arange(len(energy)),
                                     "energy_kev": energy, "counts": counts})
            row = {"separation_kev": float(separation), "trial": trial,
                   "status": "failed", "recovered": False,
                   "error_1_kev": np.nan, "error_2_kev": np.nan}
            try:
                fit = fit_overlapping_peaks(spectrum, centers, half_window_kev=35)
                estimates = [v["centroid_kev"] for v in fit["components"]]
                errors = [estimates[i] - centers[i] for i in (0, 1)]
                row.update(status="ok", recovered=all(abs(v) <= tolerance_kev for v in errors),
                           error_1_kev=errors[0], error_2_kev=errors[1])
            except ValueError:
                pass
            rows.append(row)
    return pd.DataFrame(rows)

def summarize_resolution(trials):
    return (trials.groupby("separation_kev", as_index=False)
            .agg(trials=("trial", "count"), fit_successes=("status", lambda s: int((s == "ok").sum())),
                 accurate_recoveries=("recovered", "sum"))
            .assign(success_rate=lambda d: d.fit_successes / d.trials,
                    recovery_rate=lambda d: d.accurate_recoveries / d.trials))
