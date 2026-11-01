"""Live-time-normalized background subtraction with counting uncertainty."""
import numpy as np
import pandas as pd
from .core import _validate

def subtract_background(sample, background, sample_live_s, background_live_s):
    """Return counts, rates and independent-Poisson propagated uncertainties.

    Spectra must have identical energy grids; no implicit rebinning.
    """
    _validate(sample)
    _validate(background)
    for value in (sample_live_s, background_live_s):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Live times must be finite and positive")
    if len(sample) != len(background) or not np.allclose(
        sample["energy_kev"].to_numpy(), background["energy_kev"].to_numpy(),
        atol=1e-8, rtol=0):
        raise ValueError("Sample and background must have matching energy grids")
    if not np.array_equal(sample["channel"].to_numpy(), background["channel"].to_numpy()):
        raise ValueError("Sample and background channel indices must match")
    s = sample["counts"].to_numpy(dtype=float)
    b = background["counts"].to_numpy(dtype=float)
    factor = float(sample_live_s) / float(background_live_s)
    net = s - factor * b
    sigma = np.sqrt(s + factor**2 * b)
    return pd.DataFrame({
        "channel": sample["channel"].to_numpy(),
        "energy_kev": sample["energy_kev"].to_numpy(),
        "sample_counts": s,
        "background_counts": b,
        "scaled_background_counts": factor * b,
        "net_counts": net,
        "net_sigma_counts": sigma,
        "sample_rate_cps": s / sample_live_s,
        "background_rate_cps": b / background_live_s,
        "net_rate_cps": net / sample_live_s,
        "net_sigma_rate_cps": sigma / sample_live_s,
    })

def integrate_net_region(corrected, energy_low_kev, energy_high_kev):
    if not np.isfinite([energy_low_kev, energy_high_kev]).all() or energy_low_kev >= energy_high_kev:
        raise ValueError("Invalid energy window")
    region = corrected.loc[
        corrected["energy_kev"].between(energy_low_kev, energy_high_kev)]
    if region.empty:
        raise ValueError("No channels inside energy window")
    total = float(region["net_counts"].sum())
    uncertainty = float(np.sqrt(np.square(region["net_sigma_counts"].to_numpy()).sum()))
    return {"net_counts": total, "uncertainty_counts": uncertainty,
            "net_rate_cps": float(region["net_rate_cps"].sum()),
            "uncertainty_rate_cps": float(np.sqrt(np.square(region["net_sigma_rate_cps"].to_numpy()).sum())),
            "channels": len(region)}
