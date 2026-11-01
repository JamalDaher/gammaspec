# GammaSpec

**Interactive gamma-ray spectroscopy and radiation physics workbench**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://gammaspec-jdaher.streamlit.app/)
[![Latest Release](https://img.shields.io/github/v/release/JamalDaher/gammaspec?label=release)](https://github.com/JamalDaher/gammaspec/releases)
[![Tests](https://github.com/JamalDaher/gammaspec/actions/workflows/tests.yml/badge.svg)](https://github.com/JamalDaher/gammaspec/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[**Launch the interactive demo**](https://gammaspec-jdaher.streamlit.app/) · [**View releases**](https://github.com/JamalDaher/gammaspec/releases) · [**Browse source**](src/gammaspec)

GammaSpec is an open-source Python and Streamlit application for exploring gamma-ray spectra, statistical peak analysis, detector response, and foundational radiation physics. It combines interactive visualizations with inspectable numerical methods and reproducible exports.

> **Scientific scope:** GammaSpec is an educational and exploratory research prototype—not a validated detector system. Do not use it for radiation safety, dosimetry, or certified isotope identification.

## What's new in v1.1.0

- **Background subtraction:** compare sample and background spectra using separate acquisition live times, with independent-Poisson uncertainty propagation and region-of-interest (ROI) net counts.
- **Reproducible experiments:** save experiment notebooks, export integrity-checked ZIP archives, restore spectra, and compare saved results against reanalysis.
- **Spectrum workbenches:** compare spectra, inspect peak matches, and explore calibration residuals and measured resolution.
- **Expanded physics:** investigate attenuation, Compton scattering, Klein–Nishina distributions, detector efficiency, radioactive decay, and decay chains.
- **Refreshed interface:** responsive scientific workspace, streamlined sidebar controls, and organized analysis tabs.

See the [v1.1.0 release notes](https://github.com/JamalDaher/gammaspec/releases/tag/v1.1.0) for the release summary.

## Capabilities

| Area | Available tools |
| --- | --- |
| Spectra | Seeded Poisson simulation for Cs-137, Co-60, background and an illustrative overlapping-peak example; CSV/text import; interactive plots |
| Peak analysis | Candidate detection, Gaussian fitting, two-peak overlap fitting, residuals, model comparison and Monte Carlo studies |
| Background analysis | Live-time-scaled subtraction, corrected rates, Poisson uncertainty propagation, ROI integration and CSV export |
| Calibration | Linear/quadratic channel-to-energy calibration, reference residuals, calibration diagnostics and resolution summaries |
| Detector physics | FWHM, energy resolution, efficiency concepts and counting statistics |
| Radiation interactions | Attenuation, Compton kinematics and Klein–Nishina calculations |
| Decay | Activity/half-life relationships and illustrative decay-chain models |
| Reporting | Scientific results dashboard, spectrum comparison, reference matching and downloadable HTML reports |
| Reproducibility | JSON experiment notebooks, ZIP archives with integrity metadata, restore/reanalysis and tabular exports |
| Learning | Interactive educational examples and progress exercises |

**Important distinctions:** Reference-energy matching is not definitive nuclide identification. ROI net counts from background subtraction are not automatically continuum-corrected photopeak areas. Simulated spectra are illustrative, not a complete physical detector-response model. Background subtraction requires matching channel and energy grids.

## Try it online

Open **[gammaspec-jdaher.streamlit.app](https://gammaspec-jdaher.streamlit.app/)**.

1. Choose a simulated source or import a spectrum in **Workspace controls**.
2. Inspect the **Dashboard** and **Spectrum** tabs.
3. Use focused analysis tabs for fitting, background subtraction, calibration, detector physics, or radiation interactions.
4. Export a spectrum, report, or experiment archive for later review.

For restored experiments, use **Restore experiment** at the bottom of the sidebar.

## Install locally

**Requirements:** Python 3.10 or newer.

```bash
git clone https://github.com/JamalDaher/gammaspec.git
cd gammaspec
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
streamlit run app.py
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1` instead of the `source` command.

Run the automated test suite:

```bash
python -m pytest -q
```

## Data and reproducibility

Spectrum tables use `channel`, `energy_kev`, and `counts` fields. Import validation rejects invalid numerical values and non-monotonic energy axes. Experiment archives contain a notebook record, spectrum CSV, and an integrity manifest. Archived spectra can be restored for reanalysis without applying saved energy calibration twice.

Exported analyses depend on the input spectrum, acquisition metadata, analysis parameters, and the assumptions of each numerical model. Verify live times, calibration, energy alignment, and background measurements independently for experimental data.

## Scientific assumptions and validation

Read [Model assumptions](docs/MODEL_ASSUMPTIONS.md), [Validation status](docs/VALIDATION.md), and the [Release checklist](docs/RELEASE_CHECKLIST.md). The test suite covers core numerical behavior and app smoke tests, but passing tests does not establish metrological accuracy or regulatory suitability.

## Roadmap

- Continuum-corrected net photopeak area and uncertainty analysis
- Detection significance and sensitivity concepts with explicit assumptions
- Improved detector data adapters and real-measurement validation
- More rigorous efficiency calibration and uncertainty budgets
- Additional reference spectra and reproducible benchmarks

## Contributing

Bug reports and reproducible test cases are welcome via [GitHub Issues](https://github.com/JamalDaher/gammaspec/issues). For changes to scientific calculations, include assumptions, units, and tests where practical.

## License

MIT — see [LICENSE](LICENSE).
