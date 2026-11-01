# GammaSpec Validation Status

## v0.5: Synthetic regression benchmark

The benchmark generates seeded Poisson spectra from GammaSpec's own toy simulator and fits the known injected peaks: Cs-137 at 661.657 keV, Co-60 at 1173.228 keV and 1332.492 keV. It reports centroid error, FWHM error against the injected Gaussian width, and whether the reference-line matcher returns the source. Results are deterministic for a fixed seed.

Run `python -m pytest -q` or launch the Streamlit app and open **Synthetic benchmark**. No measured detector data are bundled.

## Limitations

**This is regression testing, not independent physical validation.** The benchmark and fit use related Gaussian assumptions; agreement does not demonstrate real detector accuracy. Synthetic widths, backgrounds, intensities and count rates are illustrative. Neither uncertainty coverage nor false identification rates have been established using measured spectra. No efficiency, dead-time, escape-peak, pile-up, Compton scattering or coincidence summing model is validated.

## Next independent validation steps

1. Select publicly licensed, documented gamma spectra with detector model, acquisition time, channel count, calibration, and provenance.
2. Preserve the source URL, license, checksum, original data and preprocessing steps.
3. Fit calibration and peaks without tuning thresholds against the evaluation spectra.
4. Report centroid bias, FWHM error, missing/false peak rates, and uncertainties over multiple sources and count levels.
5. Publish reproducible scripts and limitations, including failures.

Never use GammaSpec for radiation protection, dose assessment or authoritative isotope identification.
