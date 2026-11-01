"""Launch with: streamlit run app.py"""
import plotly.graph_objects as go
import streamlit as st
from gammaspec.core import simulate_spectrum, find_photopeaks, read_spectrum

st.set_page_config(page_title="GammaSpec", page_icon="☢️", layout="wide")
st.title("GammaSpec · Gamma-Ray Spectroscopy")
st.caption("Educational synthetic spectra only. Not for radiation safety or real isotope identification.")
with st.sidebar:
    st.header("Spectrum")
    isotope = st.selectbox("Source", ["Cs-137", "Co-60", "Background"])
    live_time = st.slider("Live time (s)", 10, 300, 60)
    seed = st.number_input("Random seed", min_value=0, value=42)
    prominence = st.slider("Peak prominence", 10, 300, 80)
    log_scale = st.toggle("Logarithmic counts")
    upload = st.file_uploader("Import CSV", type=["csv"])
try:
    spectrum = read_spectrum(upload) if upload else simulate_spectrum(isotope, live_time_s=live_time, seed=int(seed))
    peaks = find_photopeaks(spectrum, prominence=prominence)
    fig = go.Figure(go.Scatter(x=spectrum.energy_kev, y=spectrum.counts, mode="lines", name="Counts"))
    fig.add_trace(go.Scatter(x=peaks.energy_kev, y=peaks.counts, mode="markers",
                             marker={"size": 9, "symbol": "x"}, name="Candidate peaks"))
    fig.update_layout(xaxis_title="Energy (keV)", yaxis_title="Counts per channel",
                      yaxis_type="log" if log_scale else "linear", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Candidate peaks")
    st.dataframe(peaks, use_container_width=True)
    st.download_button("Download spectrum CSV", spectrum.to_csv(index=False), "spectrum.csv", "text/csv")
except (ValueError, KeyError) as exc:
    st.error(f"Could not process spectrum: {exc}")
