"""Self-contained, printable HTML analysis report with escaped user data."""
from datetime import datetime, timezone
from html import escape
import numpy as np
import pandas as pd
from .core import _validate

def _table(frame, columns=None):
    if frame is None or frame.empty:
        return "<p>No results available.</p>"
    if columns is not None:
        frame = frame[[c for c in columns if c in frame.columns]]
    return frame.to_html(index=False, escape=True, border=0, classes="data")

def render_report(spectrum, *, dataset="Spectrum", peaks=None, fits=None,
                  candidates=None, calibration=None, plot_html=None,
                  generated_at=None):
    """Build a report; Plotly HTML should be produced by trusted app code."""
    _validate(spectrum)
    timestamp = generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    total = int(spectrum.counts.sum())
    details = (
        f"<p><strong>Dataset:</strong> {escape(str(dataset))}</p>"
        f"<p><strong>Generated:</strong> {escape(str(timestamp))}</p>"
        f"<p><strong>Channels:</strong> {len(spectrum):,} &nbsp; "
        f"<strong>Total counts:</strong> {total:,} &nbsp; "
        f"<strong>Energy range:</strong> {spectrum.energy_kev.min():.2f}–"
        f"{spectrum.energy_kev.max():.2f} keV</p>"
    )
    cal = "<p>No user calibration applied.</p>"
    if calibration is not None:
        cal = (f"<p>Polynomial degree: {calibration.degree}; "
               f"reference residual RMSE: {calibration.rmse_kev:.4f} keV</p>")
    chart = plot_html if plot_html is not None else "<p>Chart not included.</p>"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>GammaSpec Analysis Report</title>
<style>body{{font:15px system-ui,sans-serif;max-width:1100px;margin:36px auto;padding:0 18px;color:#182536}}
h1,h2{{color:#173a57}}section{{margin:24px 0}}.data{{border-collapse:collapse;width:100%;font-size:13px}}
.data th,.data td{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}
.notice{{background:#fff4da;padding:14px;border-left:4px solid #c77d00}}
@media print{{body{{margin:10mm}}}}</style></head><body>
<h1>GammaSpec · Analysis Report</h1>{details}
<p class="notice"><strong>Educational research prototype.</strong> Candidate matches are not
confirmed isotope identifications. The model is not independently validated for measured
spectra. Do not use for radiation safety, dosimetry or regulatory decisions.</p>
<section><h2>Spectrum</h2>{chart}</section>
<section><h2>Candidate peaks</h2>{_table(peaks)}</section>
<section><h2>Gaussian peak fits</h2>{_table(fits, ["centroid_kev","centroid_error_kev","fwhm_kev","fwhm_error_kev","net_peak_counts","status"])}</section>
<section><h2>Reference-line candidates</h2>{_table(candidates)}</section>
<section><h2>Energy calibration</h2>{cal}</section>
<footer><p>Fitted uncertainties are approximate. Data provenance, calibration and
measurement conditions must be independently documented.</p></footer>
</body></html>"""
