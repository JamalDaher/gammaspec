"""Monte Carlo recovery study for the synthetic overlapping-peak example."""
import numpy as np
import pandas as pd
from .core import simulate_spectrum
from .overlap import fit_overlapping_peaks

TRUTH = (485.0, 505.0)


def monte_carlo_recovery(trials=20, live_time_s=60, seed=42, half_window_kev=35):
    """Repeat independently seeded Poisson simulations and weighted two-peak fits.

    This is a simulation-based recovery experiment, not a posterior distribution
    or a calibrated confidence interval for an uploaded measurement.
    """
    if isinstance(trials, bool) or not isinstance(trials, int) or not 2 <= trials <= 100:
        raise ValueError("trials must be an integer between 2 and 100")
    if not np.isfinite(live_time_s) or live_time_s <= 0:
        raise ValueError("live_time_s must be positive")
    if not np.isfinite(half_window_kev) or half_window_kev <= 0:
        raise ValueError("half_window_kev must be positive")
    rng = np.random.default_rng(seed)
    rows = []
    for trial, trial_seed in enumerate(rng.integers(0, 2**32 - 1, size=trials), 1):
        spectrum = simulate_spectrum("Overlapping demo", live_time_s=live_time_s,
                                     seed=int(trial_seed))
        row = {"trial": trial, "seed": int(trial_seed), "status": "failed",
               "center_1_kev": np.nan, "center_2_kev": np.nan}
        try:
            result = fit_overlapping_peaks(spectrum, TRUTH,
                                           half_window_kev=half_window_kev)
            row.update(status="ok",
                       center_1_kev=result["components"][0]["centroid_kev"],
                       center_2_kev=result["components"][1]["centroid_kev"])
        except ValueError:
            pass
        rows.append(row)
    return pd.DataFrame(rows)


def recovery_summary(results):
    """Summarize success, signed bias and empirical spread of recovered centers."""
    successful = results.loc[results.status == "ok"]
    output = {"trials": len(results), "successes": len(successful),
              "success_rate": len(successful) / len(results)}
    for i, truth in enumerate(TRUTH, 1):
        values = successful[f"center_{i}_kev"].to_numpy(dtype=float)
        output[f"bias_{i}_kev"] = float(np.mean(values - truth)) if len(values) else np.nan
        output[f"sd_{i}_kev"] = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
    return output
