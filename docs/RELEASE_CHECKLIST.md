# GammaSpec v1.0 release checklist

## Automated gates

- [ ] GitHub Actions passes on Python 3.10, 3.12 and 3.13
- [ ] `python -m pytest -q` passes locally
- [ ] `python -m compileall -q src app.py` passes
- [ ] Install in a clean environment with `pip install -e '.[dev]'`

## Manual smoke test

- [ ] Start `streamlit run app.py` without errors
- [ ] Simulate Cs-137 and Co-60 and inspect peaks and fit quality
- [ ] Toggle off reference matching and download an HTML report (no NameError)
- [ ] Toggle off peak fitting and verify candidate matching remains usable
- [ ] Import CSV with channel,energy_kev,counts
- [ ] Import counts-only text and supply an explicit keV/channel scale
- [ ] Enable linear and quadratic calibration with default reference pairs
- [ ] Run synthetic benchmark and export CSV
- [ ] Download HTML report, open offline and print to PDF
- [ ] Check mobile and desktop layouts

## Publication

- [ ] Capture real app screenshots and add them to README (do not fabricate screenshots)
- [ ] Confirm scientific limitations and validation gaps are prominent
- [ ] Verify license and repository visibility
- [ ] Create GitHub release tag `v1.0.0` only after checks pass

**Scope:** Version 1.0 is an educational software milestone, not a validated physical measurement instrument. No independent measured-spectrum validation or authoritative isotope identification is claimed.
