"""Deterministic educational gamma spectroscopy exercises and answer checks."""
import math

from gammaspec.attenuation import attenuation_lengths
from gammaspec.compton import compton_edge
from gammaspec.detector_resolution import fwhm_kev

EXERCISES = {
    "Photopeak identification": {
        "prompt": "Which source emits a prominent gamma line near 661.7 keV?",
        "choices": ("Cs-137", "Co-60", "Neither"),
        "answer": "Cs-137",
        "explanation": "Cs-137's daughter Ba-137m emits a gamma ray at about 661.7 keV. Real identification requires more than a single peak.",
    },
    "Counting statistics": {
        "prompt": "If a Poisson count measurement lasts four times as long at constant rate, how does its relative statistical uncertainty change?",
        "choices": ("It halves", "It doubles", "It stays the same"),
        "answer": "It halves",
        "explanation": "Counts scale with time; Poisson standard deviation scales with the square root of counts, so relative uncertainty falls as 1/sqrt(time).",
    },
    "Detector resolution": {
        "prompt": "Which illustrative detector has the narrower FWHM at 662 keV?",
        "choices": ("HPGe", "NaI(Tl)", "They are equal"),
        "answer": "HPGe",
        "explanation": "The illustrative HPGe FWHM is smaller, allowing closer lines to be resolved more readily.",
    },
    "Compton edge": {
        "prompt": "For a 661.657 keV photon scattering from a free electron, at what photon scattering angle is electron recoil energy maximal?",
        "choices": ("0 degrees", "90 degrees", "180 degrees"),
        "answer": "180 degrees",
        "explanation": "Maximum transfer occurs for photon backscattering (180 degrees).",
    },
    "Attenuation and HVL": {
        "prompt": "In a narrow-beam model, what fraction of uncollided photons remains after two half-value layers?",
        "choices": ("25%", "50%", "75%"),
        "answer": "25%",
        "explanation": "Each HVL halves the uncollided beam: (1/2)^2 = 1/4.",
    },
}

def check_answer(exercise, selected):
    if exercise not in EXERCISES:
        raise ValueError("Unknown exercise")
    if selected not in EXERCISES[exercise]["choices"]:
        raise ValueError("Invalid answer choice")
    return selected == EXERCISES[exercise]["answer"]

def lab_example(exercise):
    """Return a computed physical example from the same modules used in the app."""
    if exercise == "Photopeak identification":
        return "Reference gamma line: Cs-137 / Ba-137m, 661.657 keV"
    if exercise == "Counting statistics":
        return "100 counts: 10% relative Poisson uncertainty; 400 counts: 5%"
    if exercise == "Detector resolution":
        return f"At 662 keV: HPGe FWHM {fwhm_kev(662, 'HPGe (illustrative)'):.1f} keV; NaI(Tl) {fwhm_kev(662, 'NaI(Tl) (illustrative)'):.1f} keV"
    if exercise == "Compton edge":
        return f"Cs-137 free-electron Compton edge: {compton_edge(661.657):.1f} keV"
    if exercise == "Attenuation and HVL":
        return f"Illustrative lead HVL at 662 keV: {attenuation_lengths('Lead', 662)['hvl_cm']:.2f} cm"
    raise ValueError("Unknown exercise")
