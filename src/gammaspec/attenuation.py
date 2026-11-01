"""Illustrative narrow-beam gamma attenuation, not shielding design data."""
import numpy as np

# Example mass attenuation coefficients in cm^2/g at 662 keV, deliberately
# approximate for education. NOT NIST tables or validated design coefficients.
MATERIALS = {
    "Lead": (11.34, 0.11),
    "Aluminum": (2.70, 0.075),
    "Steel": (7.85, 0.075),
    "Concrete": (2.30, 0.080),
    "Water": (1.00, 0.085),
}

def linear_attenuation(material, energy_kev):
    """Approximate mu in cm^-1 with a pedagogical energy dependence."""
    if material not in MATERIALS:
        raise ValueError("Unknown material")
    energy = np.asarray(energy_kev, dtype=float)
    if not np.isfinite(energy).all() or np.any(energy <= 0):
        raise ValueError("Gamma energy must be positive and finite")
    density, mass_mu_662 = MATERIALS[material]
    # Illustrative photoelectric-like contribution plus flatter Compton-like term.
    relative = 0.7 * (energy / 662.0) ** (-0.35) + 0.3 * (energy / 662.0) ** (-2.0)
    result = density * mass_mu_662 * relative
    return float(result) if result.ndim == 0 else result

def transmission(material, energy_kev, thickness_cm):
    """Uncollided narrow-beam transmission exp(-mu*x), excluding buildup."""
    thickness = np.asarray(thickness_cm, dtype=float)
    if not np.isfinite(thickness).all() or np.any(thickness < 0):
        raise ValueError("Thickness must be nonnegative and finite")
    result = np.exp(-linear_attenuation(material, energy_kev) * thickness)
    return float(result) if np.ndim(result) == 0 else result

def attenuation_lengths(material, energy_kev):
    mu = linear_attenuation(material, energy_kev)
    return {"hvl_cm": np.log(2) / mu, "tvl_cm": np.log(10) / mu}
