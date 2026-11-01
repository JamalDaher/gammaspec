"""Launch with: streamlit run app.py"""
import plotly.graph_objects as go
import streamlit as st
from gammaspec.core import simulate_spectrum, find_photopeaks, read_spectrum
from gammaspec.fitting import fit_detected_peaks, gaussian_background
import numpy as np
import pandas as pd
from gammaspec.calibration import fit_calibration, calibrate_spectrum, reference_table
from gammaspec.identification import match_candidates
from gammaspec.validation import benchmark, summary

st.set_page_config(page_title="GammaSpec", page_icon="☢️", layout="wide")
st.title("GammaSpec · Gamma-Ray Spectroscopy")
st.caption("Educational synthetic spectra only. Not for radiation safety or real isotope identification.")
with st.sidebar:
    st.header("Spectrum")
    isotope = st.selectbox("Source", ["Cs-137", "Co-60", "Background"])
    live_time = st.slider("Live time (s)", 10, 300, 60)
    seed = st.number_input("Random seed", min_value=0, value=42)
    prominence = st.slider("Peak prominence", 10, 300, 150)
    log_scale = st.toggle("Logarithmic counts")
    st.subheader("Energy calibration")
    enable_calibration = st.toggle("Apply channel calibration")
    calibration_degree = st.selectbox("Polynomial degree", [1, 2])
    default_refs = "100,80\\n500,400\\n1000,800\\n1500,1200" if calibration_degree == 1 else "100,80\\n400,320\\n700,560\\n1000,800"
    calibration_csv = st.text_area("Reference pairs: channel,energy_keV", default_refs, height=110)
    fit_enabled = st.toggle("Fit candidate photopeaks", value=True)
    match_enabled = st.toggle("Compare reference gamma lines", value=True)
    match_tolerance = st.slider("Line matching tolerance (keV)", 1, 30, 12)
    fit_window = st.slider("Fit half-window (keV)", 25, 90, 55)
    upload = st.file_uploader("Import CSV", type=["csv"])
try:
    spectrum = read_spectrum(upload) if upload else simulate_spectrum(isotope, live_time_s=live_time, seed=int(seed))
    if enable_calibration:
        try:
            pairs = pd.read_csv(__import__("io").StringIO(calibration_csv), header=None, names=["channel", "energy_kev"])
            calibration = fit_calibration(pairs.channel.to_numpy(), pairs.energy_kev.to_numpy(), degree=calibration_degree)
            spectrum = calibrate_spectrum(spectrum, calibration)
            st.sidebar.metric("Calibration RMSE (keV)", f"{calibration.rmse_kev:.4f}")
            st.sidebar.download_button("Download calibration residuals", reference_table(pairs.channel, pairs.energy_kev, calibration).to_csv(index=False), "calibration.csv", "text/csv")
        except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
            st.error(f"Calibration failed: {exc}")
            st.stop()
    peaks = find_photopeaks(spectrum, prominence=prominence)
    fig = go.Figure(go.Scatter(x=spectrum.energy_kev, y=spectrum.counts, mode="lines", name="Counts"))
    fig.add_trace(go.Scatter(x=peaks.energy_kev, y=peaks.counts, mode="markers",
                             marker={"size": 9, "symbol": "x"}, name="Candidate peaks"))
    fits = fit_detected_peaks(spectrum, prominence=prominence, half_window_kev=fit_window) if fit_enabled else None
    if fits is not None:
        for fit in fits.to_dict("records"):
            if fit.get("status") != "ok":
                continue
            center = fit["centroid_kev"]
            x = np.linspace(center - fit_window, center + fit_window, 180)
            y = gaussian_background(x, fit["amplitude_counts"], center, fit["sigma_kev"],
                                    fit["background_at_centroid"], fit["slope"])
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=f"Fit {center:.1f} keV"))
    fig.update_layout(xaxis_title="Energy (keV)", yaxis_title="Counts per channel",
                      yaxis_type="log" if log_scale else "linear", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Candidate peaks")
    st.dataframe(peaks, use_container_width=True)
    if fits is not None:
        st.subheader("Gaussian photopeak fits")
        st.dataframe(fits, use_container_width=True)
        st.caption("Uncertainties are approximate 1σ covariance estimates from weighted least squares; overlapping peaks may bias fits.")
        st.download_button("Download fit table CSV", fits.to_csv(index=False), "peak_fits.csv", "text/csv")
    if match_enabled:
        st.subheader("Reference line candidates")
        observed = (fits.loc[fits.status == "ok", "centroid_kev"].to_numpy()
                    if fits is not None and not fits.empty else peaks.energy_kev.to_numpy())
        candidates = match_candidates(observed, tolerance_kev=match_tolerance)
        if candidates.empty:
            st.info("No reference-line matches within the selected tolerance.")
        else:
            st.dataframe(candidates, use_container_width=True)
            st.download_button("Download candidates CSV", candidates.to_csv(index=False), "candidates.csv", "text/csv")
        st.caption("Candidate matching is not isotope identification. Missing lines, mixtures, calibration error and detector response can change results.")
    with st.expander("Synthetic benchmark (v0.5)"):
        st.caption("Regression against injected toy-model parameters, NOT validation against measured spectra.")
        if st.button("Run benchmark"):
            benchmark_results = benchmark(seed=int(seed), live_time_s=live_time)
            metrics = summary(benchmark_results)
            st.json(metrics)
            st.dataframe(benchmark_results, use_container_width=True)
            st.download_button("Download benchmark CSV", benchmark_results.to_csv(index=False), "benchmark.csv", "text/csv")
    st.download_button("Download spectrum CSV", spectrum.to_csv(index=False), "spectrum.csv", "text/csv")
except (ValueError, KeyError) as exc:
    st.error(f"Could not process spectrum: {exc}")
