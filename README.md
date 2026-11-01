# GammaSpec

Educational, open-source gamma-ray spectroscopy simulator and spectrum viewer.

## Features (v0.8)
- Seeded Poisson simulation of illustrative Cs-137, Co-60 and background spectra
- Interactive Plotly chart with logarithmic scale option
- Candidate peak detection using SciPy prominence
- CSV import/export (`channel,energy_kev,counts`)
- Weighted Gaussian peak fitting and FWHM estimates
- Linear/quadratic energy calibration, reference residuals and approximate fit uncertainty
- Educational reference-line candidate matching for Cs-137, Co-60, Na-22, K-40 and Am-241
- Reproducible synthetic regression benchmarks for fitted centroids, FWHM and reference matching
- Text-based measured-spectrum import (CSV or one-count-per-line) with validation and explicit energy scale
- Five-tab analysis workspace with acquisition metrics, grouped controls and export actions
- Downloadable, printable self-contained HTML reports with spectrum chart, fits and reference matches
- Automated pytest suite and GitHub Actions CI

## Quick start
Requires Python 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e '.[dev]'
streamlit run app.py
```

Run tests with `python -m pytest -q`.

## Scientific scope
Synthetic peaks use Gaussian profiles with illustrative amplitudes and widths plus a toy continuum and Poisson counting statistics. These are **not validated detector response models**. Candidate peak detection is not isotope identification. The app must not be used for radiation safety decisions, dosimetry, or field identification.

## Roadmap
- v0.2: robust peak fitting, uncertainties and background estimation
- v0.3: channel-to-energy calibration and approximate fit uncertainty (implemented)
- v0.4: small reference line library and candidate matching (implemented)
- v0.5: reproducible synthetic regression benchmark (implemented)
- v0.6: text-based spectrum import (implemented); independent measured-spectrum validation remains outstanding
- v0.7: organized five-tab analysis workspace (implemented)
- v0.8: downloadable HTML analysis reports (implemented)
- Later: detector data adapters, measured-spectrum validation and efficiency calibration

## License
MIT — see LICENSE.
