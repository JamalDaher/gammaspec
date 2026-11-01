"""Launch with: streamlit run app.py"""
from pathlib import Path
import sys

# Use this checkout rather than a potentially stale installed GammaSpec package.
# This matters on hosted Streamlit deployments after source-only updates.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import plotly.graph_objects as go
import streamlit as st
from gammaspec import core as gammaspec_core
from gammaspec.core import simulate_spectrum, find_photopeaks, read_spectrum
from gammaspec.fitting import fit_detected_peaks, gaussian_background
from gammaspec.overlap import fit_overlapping_peaks, double_gaussian_background
from gammaspec.selection import selected_peak_energies
import numpy as np
import pandas as pd
from gammaspec.calibration import fit_calibration, calibrate_spectrum, reference_table
from gammaspec.identification import match_candidates
from gammaspec.validation import benchmark, summary
from gammaspec.importers import import_spectrum_text
from gammaspec.reporting import render_report

def handle_peak_selection():
    event = st.session_state.get("spectrum_peak_selector", {})
    points = event.get("selection", {}).get("points", [])
    energies = selected_peak_energies(points, st.session_state.get("candidate_peak_energies", []))
    if len(energies) == 2:
        st.session_state["overlap_center_1"] = energies[0]
        st.session_state["overlap_center_2"] = energies[1]
        st.session_state["overlap_fit_enabled"] = True


st.set_page_config(page_title="GammaSpec", page_icon="☢️", layout="wide")
st.title("GammaSpec")
st.markdown("**Gamma-ray spectroscopy workbench** · Simulation, calibration and peak analysis")
st.caption("Research and education prototype · Not for radiation safety, dosimetry, or authoritative isotope identification.")
with st.sidebar:
    st.header("Acquisition")
    isotope = st.selectbox("Source", ["Cs-137", "Co-60", "Background", "Overlapping demo"])
    with st.expander("Deployment diagnostics"):
        st.code(f"Loaded core: {gammaspec_core.__file__}\\nAvailable sources: {list(gammaspec_core.LINES)}")
        if "Overlapping demo" not in gammaspec_core.LINES:
            st.error("Running GammaSpec core is outdated. Confirm deployment branch is main and reboot/redeploy.")
    live_time = st.slider("Live time (s)", 10, 300, 60)
    seed = st.number_input("Random seed", min_value=0, value=42)
    prominence = st.slider("Peak prominence", 10, 300, 150)
    log_scale = st.toggle("Logarithmic counts")
    st.divider()
    st.subheader("Calibration settings")
    enable_calibration = st.toggle("Apply channel calibration")
    calibration_degree = st.selectbox("Polynomial degree", [1, 2])
    default_refs = "100,80\n500,400\n1000,800\n1500,1200" if calibration_degree == 1 else "100,80\n400,320\n700,560\n1000,800"
    calibration_csv = st.text_area("Reference pairs: channel,energy_keV", default_refs, height=110)
    st.divider()
    st.subheader("Analysis settings")
    fit_enabled = st.toggle("Fit candidate photopeaks", value=True)
    match_enabled = st.toggle("Compare reference gamma lines", value=True)
    match_tolerance = st.slider("Line matching tolerance (keV)", 1, 30, 12)
    fit_window = st.slider("Fit half-window (keV)", 25, 90, 55)
    peak_separation = st.slider("Minimum peak separation (keV)", 0, 100, 35)
    st.divider()
    st.subheader("Overlapping peaks (experimental)")
    if isotope == "Overlapping demo":
        st.caption("Synthetic example: peaks at 485 and 505 keV; not an isotope.")
    overlap_enabled = st.toggle("Fit two overlapping peaks", value=isotope == "Overlapping demo", key="overlap_fit_enabled")
    if overlap_enabled:
        overlap_center_1 = st.number_input("First peak center (keV)", min_value=0.0, value=485.0, key="overlap_center_1")
        overlap_center_2 = st.number_input("Second peak center (keV)", min_value=0.0, value=505.0, key="overlap_center_2")
        overlap_window = st.slider("Two-peak half-window (keV)", 10, 100, 35)
    st.divider()
    upload = st.file_uploader("Import spectrum CSV or counts text", type=["csv", "txt"])
    if upload is not None:
        import_format = st.selectbox("Import format", ["auto", "csv", "counts"])
        channel_width = st.number_input("Energy scale (keV/channel) for uncalibrated files", min_value=0.001, value=0.8)
        energy_offset = st.number_input("Energy offset (keV)", value=0.0)
try:
    if not upload and isotope not in (*gammaspec_core.LINES, "Background"):
        st.error(f"Selected source {isotope!r} is unavailable in the loaded GammaSpec core. See Deployment diagnostics in the sidebar.")
        st.stop()
    spectrum = (import_spectrum_text(upload.getvalue(), format=import_format,
                energy_per_channel_kev=channel_width, energy_offset_kev=energy_offset)
                if upload else simulate_spectrum(isotope, live_time_s=live_time, seed=int(seed)))
    if upload:
        st.info("Imported spectrum: energy scale must be independently calibrated before interpreting peak energies.")
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
    st.divider()
    source_label = upload.name if upload else f"Synthetic {isotope}"
    st.subheader("Workspace")
    metrics = st.columns(4)
    metrics[0].metric("Dataset", source_label)
    metrics[1].metric("Channels", f"{len(spectrum):,}")
    metrics[2].metric("Total counts", f"{int(spectrum.counts.sum()):,}")
    metrics[3].metric("Energy span", f"{spectrum.energy_kev.max():.0f} keV")
    peaks = find_photopeaks(spectrum, prominence=prominence)
    st.session_state["candidate_peak_energies"] = peaks.energy_kev.tolist()
    fig = go.Figure(go.Scatter(x=spectrum.energy_kev, y=spectrum.counts, mode="lines", name="Counts"))
    fig.add_trace(go.Scatter(x=peaks.energy_kev, y=peaks.counts, mode="markers",
                             marker={"size": 9, "symbol": "x"}, name="Candidate peaks"))
    fits = fit_detected_peaks(spectrum, prominence=prominence, half_window_kev=fit_window,
                              min_separation_kev=peak_separation) if fit_enabled else None
    overlap_result = None
    if overlap_enabled:
        try:
            overlap_result = fit_overlapping_peaks(
                spectrum, (overlap_center_1, overlap_center_2), half_window_kev=overlap_window
            )
            components = overlap_result["components"]
            x_overlap = np.linspace(
                min(overlap_center_1, overlap_center_2) - overlap_window,
                max(overlap_center_1, overlap_center_2) + overlap_window, 400,
            )
            c1, c2 = components
            combined = double_gaussian_background(
                x_overlap,
                c1["amplitude_counts"], c1["centroid_kev"], c1["fwhm_kev"] / 2.354820045,
                c2["amplitude_counts"], c2["centroid_kev"], c2["fwhm_kev"] / 2.354820045,
                overlap_result["background_intercept"], overlap_result["background_slope"],
            )
            fig.add_trace(go.Scatter(x=x_overlap, y=combined, mode="lines",
                                     name="Two-peak combined fit", line={"dash": "dash"}))
            for i, component in enumerate(components, 1):
                mu = component["centroid_kev"]
                sigma = component["fwhm_kev"] / 2.354820045
                profile = component["amplitude_counts"] * np.exp(-0.5 * ((x_overlap - mu) / sigma) ** 2)
                fig.add_trace(go.Scatter(x=x_overlap, y=profile, mode="lines",
                                         name=f"Gaussian component {i}", line={"dash": "dot"}))
        except ValueError as exc:
            st.warning(f"Two-peak fit unavailable: {exc}")
    if fits is not None:
        for fit in fits.to_dict("records"):
            if fit.get("status") != "ok":
                continue
            center = fit["centroid_kev"]
            x = np.linspace(center - fit_window, center + fit_window, 180)
            y = gaussian_background(x, fit["amplitude_counts"], center, fit["sigma_kev"],
                                    fit["background_at_centroid"], fit["slope"])
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=f"Fit {center:.1f} keV"))
    if isotope == "Overlapping demo" and upload is None and not enable_calibration:
        demo_x = spectrum.energy_kev.to_numpy(dtype=float)
        for center, width, strength in ((485.0, 8.0, 1.0), (505.0, 10.0, 0.7)):
            component = 12.0 * live_time * strength * np.exp(-0.5 * ((demo_x - center) / width) ** 2)
            fig.add_trace(go.Scatter(x=demo_x, y=component, mode="lines",
                                     name=f"Injected component {center:g} keV",
                                     line={"dash": "dot", "width": 3}))
        st.caption("Demo component curves represent known injected model contributions, not independent measured peaks.")
    if isotope == "Overlapping demo" and upload is None and not enable_calibration:
        for energy in (485.0, 505.0):
            fig.add_vline(x=energy, line_dash="dot", line_color="orange",
                          annotation_text=f"Injected {energy:g} keV",
                          annotation_position="top")
        fig.update_xaxes(range=[450, 540])
    fig.update_layout(xaxis_title="Energy (keV)", yaxis_title="Counts per channel",
                      yaxis_type="log" if log_scale else "linear", template="plotly_dark")
    overview_tab, peaks_tab, match_tab, calibration_tab, validation_tab, report_tab = st.tabs(
        ["Spectrum", "Peak analysis", "Reference matching", "Calibration", "Validation", "Report"])
    with overview_tab:
        if isotope == "Overlapping demo" and upload is None and not enable_calibration:
            st.info("Zoomed to 450–540 keV. Orange vertical lines mark injected centers; dotted curves show each injected Gaussian component. The measured total can still have only one maximum. × markers show noisy local maxima.")
        st.caption("Select two candidate × markers using Shift-click or box selection to seed the two-peak fit.")
        st.plotly_chart(fig, use_container_width=True, key="spectrum_peak_selector",
                        on_select=handle_peak_selection, selection_mode=("points", "box"))
        st.download_button("Export spectrum CSV", spectrum.to_csv(index=False), "spectrum.csv", "text/csv")
    with peaks_tab:
        st.caption("Detection finds local maxima; fitting estimates model parameters. One detected maximum may contain multiple overlapping components.")
        comparison_tab, detected_tab, single_tab, double_tab = st.tabs(
            ["Compare", "Detected peaks", "Single-peak fits", "Two-component fit"]
        )
        with comparison_tab:
            st.subheader("Peak energy comparison")
            rows = []
            for energy in peaks.energy_kev:
                rows.append({"Method": "Detected maximum", "Energy (keV)": float(energy)})
            if fits is not None and not fits.empty:
                for fit in fits.to_dict("records"):
                    if fit.get("status") == "ok":
                        rows.append({"Method": "Single-Gaussian centroid", "Energy (keV)": float(fit["centroid_kev"])})
            if overlap_result is not None:
                for component in overlap_result["components"]:
                    rows.append({"Method": "Two-component centroid", "Energy (keV)": float(component["centroid_kev"])})
            if isotope == "Overlapping demo" and upload is None and not enable_calibration:
                for energy in (485.0, 505.0):
                    rows.append({"Method": "Injected demo center", "Energy (keV)": energy})
            if rows:
                comparison = pd.DataFrame(rows)
                st.dataframe(comparison, use_container_width=True, hide_index=True)
                comparison_fig = go.Figure()
                for method, group in comparison.groupby("Method", sort=False):
                    comparison_fig.add_trace(go.Scatter(
                        x=group["Energy (keV)"], y=[method] * len(group),
                        mode="markers", marker={"size": 12}, name=method))
                comparison_fig.update_layout(xaxis_title="Energy (keV)",
                                             template="plotly_dark", height=330)
                st.plotly_chart(comparison_fig, use_container_width=True)
                st.download_button("Download comparison CSV", comparison.to_csv(index=False),
                                   "peak_comparison.csv", "text/csv")
            else:
                st.info("No detected or fitted peak energies to compare.")
            st.caption("Detected maxima and fitted centroids are different quantities. Two fitted components do not establish two physical emissions.")
        with detected_tab:
            st.subheader("Detected local maxima")
            st.caption("These are noisy candidate maxima, not necessarily separate gamma emissions.")
            st.metric("Detected maxima", len(peaks))
            st.dataframe(peaks, use_container_width=True, hide_index=True)
        with single_tab:
            st.subheader("Single-Gaussian fits")
            if fits is None:
                st.info("Enable Fit candidate photopeaks in the sidebar to calculate single-peak fits.")
            elif fits.empty:
                st.info("No candidates passed the current prominence threshold.")
            else:
                st.dataframe(fits, use_container_width=True, hide_index=True)
                st.caption("A single Gaussian fitted to overlapping peaks can return a centroid between their true centers.")
                st.caption("Quality flags are heuristic; uncertainties assume the chosen fit model.")
                st.download_button("Download single-peak fits CSV", fits.to_csv(index=False),
                                   "peak_fits.csv", "text/csv")
        with double_tab:
            st.subheader("Simultaneous two-peak fit")
            if not overlap_enabled:
                st.info("Enable Fit two overlapping peaks in the sidebar to estimate two components.")
            elif overlap_result is None:
                st.warning("Two-component fitting is unavailable for the current settings; see the warning above.")
            else:
                components_table = pd.DataFrame(overlap_result["components"])
                st.dataframe(components_table, use_container_width=True, hide_index=True)
                st.metric("Reduced chi-square (descriptive)",
                          f'{overlap_result["reduced_chi_square"]:.2f}')
                st.caption(f'Quality flag: {overlap_result["quality_flag"]}. The two-component model does not establish two physical gamma emissions.')
                st.download_button("Download two-component fits CSV",
                                   components_table.to_csv(index=False),
                                   "overlap_fits.csv", "text/csv")
    candidates = None
    if match_enabled:
      with match_tab:
        st.subheader("Reference line candidates")
        successful_fits = (fits.loc[fits.status == "ok", "centroid_kev"].to_numpy()
                           if fits is not None and not fits.empty else np.array([]))
        observed = successful_fits if len(successful_fits) else peaks.energy_kev.to_numpy()
        candidates = match_candidates(observed, tolerance_kev=match_tolerance)
        if candidates.empty:
            st.info("No reference-line matches within the selected tolerance.")
        else:
            st.dataframe(candidates, use_container_width=True)
            st.download_button("Download candidates CSV", candidates.to_csv(index=False), "candidates.csv", "text/csv")
        st.caption("Candidate matching is not isotope identification. Missing lines, mixtures, calibration error and detector response can change results.")
    with calibration_tab:
        st.subheader("Energy calibration")
        if enable_calibration:
            st.dataframe(reference_table(pairs.channel, pairs.energy_kev, calibration), use_container_width=True)
            st.caption(f"Polynomial degree {calibration_degree}; RMSE {calibration.rmse_kev:.4f} keV")
        else:
            st.info("Enable channel calibration in the sidebar to inspect residuals.")
    with validation_tab:
      with st.expander("Synthetic regression benchmark", expanded=True):
        st.caption("Regression against injected toy-model parameters, NOT validation against measured spectra.")
        if st.button("Run benchmark"):
            benchmark_results = benchmark(seed=int(seed), live_time_s=live_time)
            metrics = summary(benchmark_results)
            st.json(metrics)
            st.dataframe(benchmark_results, use_container_width=True)
            st.download_button("Download benchmark CSV", benchmark_results.to_csv(index=False), "benchmark.csv", "text/csv")

    with report_tab:
        st.subheader("Export scientific analysis")
        st.caption("Self-contained HTML report with embedded interactive spectrum chart. Open in a browser to print or save as PDF.")
        report = render_report(
            spectrum, dataset=source_label, peaks=peaks, fits=fits,
            candidates=candidates if match_enabled else None,
            calibration=calibration if enable_calibration else None,
            plot_html=fig.to_html(full_html=False, include_plotlyjs=True),
        )
        st.download_button("Download HTML analysis report", report, "gammaspec_report.html", "text/html")

except (ValueError, KeyError) as exc:
    st.error(f"Could not process spectrum: {exc}")
