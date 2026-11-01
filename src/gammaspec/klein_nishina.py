"""Unpolarized Klein–Nishina scattering for free electrons at rest."""
import numpy as np
from gammaspec.compton import ELECTRON_REST_KEV, scattered_energy

CLASSICAL_ELECTRON_RADIUS_CM = 2.8179403262e-13

def differential_cross_section(incident_kev, angle_deg):
    """Unpolarized dσ/dΩ in cm²/sr per free electron."""
    energy = np.asarray(incident_kev, dtype=float)
    theta = np.asarray(angle_deg, dtype=float)
    ratio = scattered_energy(energy, theta) / energy
    cosine = np.cos(np.deg2rad(theta))
    result = 0.5 * CLASSICAL_ELECTRON_RADIUS_CM**2 * ratio**2 * (
        ratio + 1 / ratio - (1 - cosine**2))
    return float(result) if np.ndim(result) == 0 else result

def angular_distribution(incident_kev, angles_deg=None):
    """Return angle grid, normalized density per degree, and dσ/dΩ.

    Probability in a polar angle interval includes solid-angle sin(theta).
    """
    angles = np.linspace(0, 180, 721) if angles_deg is None else np.asarray(angles_deg, dtype=float)
    if angles.ndim != 1 or len(angles) < 2 or not np.all(np.diff(angles) > 0):
        raise ValueError("Angles must be an increasing 1D grid")
    cross = differential_cross_section(incident_kev, angles)
    density = 2 * np.pi * cross * np.sin(np.deg2rad(angles)) * np.pi / 180
    integral = np.trapezoid(density, angles)
    if not np.isfinite(integral) or integral <= 0:
        raise ValueError("Invalid angular distribution")
    return angles, density / integral, cross

def total_cross_section(incident_kev):
    """Numerically integrate Klein–Nishina dσ/dΩ over solid angle, cm²."""
    angles = np.linspace(0, np.pi, 2001)
    differential = differential_cross_section(incident_kev, np.rad2deg(angles))
    return float(2 * np.pi * np.trapezoid(differential * np.sin(angles), angles))
