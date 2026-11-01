# Model assumptions and validation boundaries

GammaSpec is an **educational simulator**, not a calibrated spectrometer, radiation safety calculator, dose estimator, or authoritative isotope identification tool.

## Numerical models

| Module | Simplifying assumptions | What the output cannot establish |
| --- | --- | --- |
| Synthetic spectra | Seeded Poisson noise, illustrative Gaussian peaks and smooth background | Actual detector response, source purity or measured nuclide identity |
| Peak fitting | Gaussian peaks and simplified local backgrounds | Validated uncertainty under pileup, non-Gaussian response or unknown systematic error |
| Model selection | Candidate models, fitted parameter counts and likelihood/weighted-residual assumptions | Proof that a physical number of gamma lines exists |
| Detector resolution/efficiency | Illustrative curves, not measured instrument calibration | Manufacturer performance or absolute detection efficiency |
| Attenuation | Narrow-beam uncollided exponential attenuation with approximate coefficients | Safe shielding thickness, dose rate or broad-beam buildup |
| Compton and Klein–Nishina | Free electrons at rest; unpolarized scattering in Klein–Nishina | Bound-electron, material or full detector response |
| Radioactive decay | Exponential decay with approximate half-lives | Radiation dose or nuclide assay |
| Decay chains | Sequential chain, specified branch fraction to next generation, remainder to untracked stable sink; no external feeding | Complete real isotope branching networks or equilibrium already reached |
| Learning lab | Deterministic educational questions and session-only progress | Formal assessment or persistent educational records |

## Validation strategy

- Unit tests check analytical identities, numerical limits, invalid inputs and reproducibility.
- Streamlit AppTest smoke checks exercise the default app and a source switch without a browser.
- A passing test suite demonstrates regression coverage, **not** experimental validation.
- Real detector validation would require traceable reference spectra, calibrated efficiency/energy response, independent uncertainty budgets and documented acceptance criteria.

Run locally:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
streamlit run app.py
```

The Streamlit smoke tests use `streamlit.testing.v1.AppTest` and can catch application exceptions, but not every browser rendering or interaction issue.
