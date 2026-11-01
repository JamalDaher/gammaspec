"""Launch with: streamlit run app.py"""
import plotly.graph_objects as go
import streamlit as st
from gammaspec.core import simulate_spectrum, find_photopeaks, read_spectrum
from gammaspec.fitting import fit_detected_peaks, gaussian_background
import numpy as np

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
    fit_enabled = st.toggle("Fit candidate photopeaks", value=True)
    fit_window = st.slider("Fit half-window (keV)", 25, 90, 55)
    upload = st.file_uploader("Import CSV", type=["csv"])
try:
    spectrum = read_spectrum(upload) if upload else simulate_spectrum(isotope, live_time_s=live_time, seed=int(seed))
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
    st.download_button("Download spectrum CSV", spectrum.to_csv(index=False), "spectrum.csv", "text/csv")
except (ValueError, KeyError) as exc:
    st.error(f"Could not process spectrum: {exc}")
