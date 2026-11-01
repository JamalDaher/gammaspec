"""Simulation and basic peak finding; not a detector response model."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

LINES = {"Cs-137": [(661.657, 1.0)], "Co-60": [(1173.228, 0.95), (1332.492, 1.0)]}

def simulate_spectrum(isotope="Cs-137", channels=2048, max_energy_kev=1600.0,
                      live_time_s=60.0, seed=42, background_rate=0.1):
    """Generate reproducible Poisson counts from Gaussian peaks plus a toy background.

    Peak intensities are illustrative, not calibrated emission probabilities.
    """
    if isotope not in (*LINES, "Background"):
        raise ValueError("Unknown isotope")
    if channels < 16 or max_energy_kev <= 0 or live_time_s <= 0 or background_rate < 0:
        raise ValueError("Invalid simulation parameters")
    energy = np.linspace(0, max_energy_kev, channels)
    expected = np.full(channels, background_rate * live_time_s)
    expected += 0.3 * live_time_s * np.exp(-energy / 450)
    for line, strength in LINES.get(isotope, []):
        sigma = max(4.0, 0.025 * line)
        expected += 12.0 * live_time_s * strength * np.exp(-0.5 * ((energy-line)/sigma)**2)
    counts = np.random.default_rng(seed).poisson(expected)
    return pd.DataFrame({"channel": np.arange(channels), "energy_kev": energy, "counts": counts})

def find_photopeaks(spectrum, prominence=40, min_distance_channels=12):
    """Find candidate peaks; no isotope identification or background correction."""
    if prominence <= 0 or min_distance_channels < 1:
        raise ValueError("Invalid peak detection settings")
    _validate(spectrum)
    idx, properties = find_peaks(spectrum["counts"].to_numpy(), prominence=prominence,
                                 distance=min_distance_channels)
    return pd.DataFrame({"channel": spectrum.iloc[idx]["channel"].to_numpy(),
                         "energy_kev": spectrum.iloc[idx]["energy_kev"].to_numpy(),
                         "counts": spectrum.iloc[idx]["counts"].to_numpy(),
                         "prominence": properties["prominences"]})

def _validate(frame):
    required = {"channel", "energy_kev", "counts"}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("Spectrum must have channel, energy_kev and counts columns")
    vals = frame[["channel", "energy_kev", "counts"]].to_numpy(dtype=float)
    if not np.isfinite(vals).all() or (vals[:, 2] < 0).any() or (np.diff(vals[:, 1]) <= 0).any():
        raise ValueError("Invalid spectrum values or energy order")

def read_spectrum(path):
    frame = pd.read_csv(path)
    _validate(frame)
    return frame

def write_spectrum(frame, path):
    _validate(frame)
    frame.to_csv(Path(path), index=False)
