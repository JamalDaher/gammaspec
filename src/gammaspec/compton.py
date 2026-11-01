"""Ideal free-electron Compton scattering kinematics."""
import numpy as np

ELECTRON_REST_KEV = 510.99895

def scattered_energy(incident_kev, angle_deg):
    """E' = E / [1 + (E/m_ec^2)(1-cos theta)]."""
    energy = np.asarray(incident_kev, dtype=float)
    angle = np.asarray(angle_deg, dtype=float)
    if (not np.isfinite(energy).all() or np.any(energy <= 0)
            or not np.isfinite(angle).all() or np.any(angle < 0) or np.any(angle > 180)):
        raise ValueError("Energy must be positive and angle in [0, 180] degrees")
    return energy / (1 + (energy / ELECTRON_REST_KEV) * (1 - np.cos(np.deg2rad(angle))))

def recoil_energy(incident_kev, angle_deg):
    """Kinetic energy transferred to the recoil electron."""
    return np.asarray(incident_kev, dtype=float) - scattered_energy(incident_kev, angle_deg)

def compton_edge(incident_kev):
    """Maximum recoil energy, at 180 degree photon backscatter."""
    return recoil_energy(incident_kev, 180)

def scattered_wavelength_shift_pm(angle_deg):
    """Free-electron wavelength shift in picometers (Compton wavelength 2.42631 pm)."""
    angle = np.asarray(angle_deg, dtype=float)
    if not np.isfinite(angle).all() or np.any(angle < 0) or np.any(angle > 180):
        raise ValueError("Angle must be in [0, 180] degrees")
    return 2.42631023867 * (1 - np.cos(np.deg2rad(angle)))
