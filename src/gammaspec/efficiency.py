"""Educational full-energy peak efficiency and Poisson counting models."""
import numpy as np

# Arbitrary illustrative efficiencies at 662 keV, not detector calibrations.
REFERENCE_EFFICIENCY = {"HPGe (illustrative)": 0.08, "NaI(Tl) (illustrative)": 0.15}

def peak_efficiency(energy_kev, detector, efficiency_at_662=None):
    """Bounded illustrative power-law peak efficiency vs energy."""
    if detector not in REFERENCE_EFFICIENCY:
        raise ValueError("Unknown detector")
    energy = np.asarray(energy_kev, dtype=float)
    ref = REFERENCE_EFFICIENCY[detector] if efficiency_at_662 is None else float(efficiency_at_662)
    if not np.isfinite(energy).all() or np.any(energy <= 0) or not np.isfinite(ref) or not 0 < ref <= 1:
        raise ValueError("Energy must be positive and reference efficiency in (0, 1]")
    exponent = -0.65 if detector.startswith("HPGe") else -0.85
    efficiency = np.minimum(1.0, ref * (energy / 662.0) ** exponent)
    return float(efficiency) if efficiency.ndim == 0 else efficiency

def counting_statistics(energy_kev, detector, emission_rate_s, live_time_s,
                        background_rate_s=0, efficiency_at_662=None):
    """Expected signal/background counts, uncertainty and approximate net-count SNR.

    Background rate is counts per second in the selected peak region. Background
    is assumed known; sqrt(S+B) is not a background-estimation uncertainty.
    """
    if not all(np.isfinite(v) for v in (emission_rate_s, live_time_s, background_rate_s)):
        raise ValueError("Rates and time must be finite")
    if emission_rate_s < 0 or live_time_s <= 0 or background_rate_s < 0:
        raise ValueError("Invalid rates or live time")
    eff = peak_efficiency(energy_kev, detector, efficiency_at_662)
    signal = emission_rate_s * live_time_s * eff
    background = background_rate_s * live_time_s
    total = signal + background
    sigma = np.sqrt(total)
    snr = np.divide(signal, sigma, out=np.zeros_like(np.asarray(signal, dtype=float)),
                    where=np.asarray(sigma) > 0)
    return {"efficiency": eff, "signal_counts": signal, "background_counts": background,
            "total_counts": total, "poisson_sigma_counts": sigma, "approx_snr": snr}
