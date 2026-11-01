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
from gammaspec.residuals import residual_comparison, residual_rms
from gammaspec.model_selection import compare_information_criteria
from gammaspec.poisson import compare_poisson_models
from gammaspec.monte_carlo import monte_carlo_recovery, recovery_summary
from gammaspec.resolution import resolution_study, summarize_resolution
from gammaspec.detector_resolution import fwhm_kev, resolution_percent, illustrative_doublet
from gammaspec.efficiency import peak_efficiency, counting_statistics
from gammaspec.attenuation import MATERIALS, linear_attenuation, transmission, attenuation_lengths
from gammaspec.compton import scattered_energy, recoil_energy, compton_edge, scattered_wavelength_shift_pm
from gammaspec.klein_nishina import differential_cross_section, angular_distribution, total_cross_section
from gammaspec.learning_lab import EXERCISES, check_answer, lab_example
from gammaspec.learning_progress import CHALLENGES, grade_numeric, record_attempt, progress_summary, progress_csv
from gammaspec.decay import HALF_LIVES_YEARS, remaining_activity, decay_fraction, parent_daughter_atoms
from gammaspec.decay_chains import chain_populations, equilibrium_class
from gammaspec.dashboard import build_results_summary, summary_csv, summary_json
from gammaspec.comparison import compare_spectra, comparison_metrics
from gammaspec.matching_workbench import rank_reference_candidates
from gammaspec.calibration_workbench import calibration_diagnostics, measured_resolution_table
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
st.markdown("""
<style>
/* Compact scientific workspace: avoid overflowing charts and long labels. */
.block-container {padding-top: 1.5rem; padding-bottom: 2.5rem; max-width: 1500px;}
[data-testid="stMetricValue"] {font-size: clamp(1.05rem, 2.3vw, 1.7rem);}
[data-testid="stHorizontalBlock"] {flex-wrap: wrap;}
[data-testid="stHorizontalBlock"] > div {min-width: min(100%, 170px);}
[data-testid="stDataFrame"] {max-width: 100%;}
@media (max-width: 640px) {
    .block-container {padding: 1rem .7rem 2rem;}
    h1 {font-size: 1.8rem !important;}
    [data-testid="stPlotlyChart"] {max-width: 100%; overflow-x: auto;}
}
</style>
""", unsafe_allow_html=True)
st.title("GammaSpec")
st.markdown("**Gamma-ray spectroscopy workbench** · Spectra, detector response and radiation interactions")
st.caption("Use the tabs below to explore spectra, advanced fitting, detector physics and radiation interactions.")
with st.expander("Explore GammaSpec capabilities", expanded=False):
    st.markdown("""
    **Spectroscopy:** simulate or import spectra, detect peaks, fit Gaussian models and export results.

    **Advanced fitting:** compare statistical models, inspect residuals and run Monte Carlo studies.

    **Detector physics:** explore illustrative resolution, efficiency and counting statistics.

    **Radiation interactions:** study attenuation, Compton scattering and Klein–Nishina distributions.

    **Reproducibility:** download data and generate a self-contained analysis report.
    """)
st.caption("Research and education prototype · Not for radiation safety, dosimetry, or authoritative isotope identification.")
with st.sidebar:
    st.header("Workspace controls")
    st.caption("Choose a source, then expand the settings you need.")
    isotope = st.selectbox("Source", ["Cs-137", "Co-60", "Background", "Overlapping demo"])
    with st.expander("Acquisition", expanded=True):
        live_time = st.slider("Live time (s)", 10, 300, 60)
        seed = st.number_input("Random seed", min_value=0, value=42)
        prominence = st.slider("Peak prominence", 10, 300, 150)
        log_scale = st.toggle("Logarithmic counts")
    with st.expander("Import spectrum", expanded=False):
        upload = st.file_uploader("Import spectrum CSV or counts text", type=["csv", "txt"])
        if upload is not None:
            import_format = st.selectbox("Import format", ["auto", "csv", "counts"])
            channel_width = st.number_input("Energy scale (keV/channel) for uncalibrated files", min_value=0.001, value=0.8)
            energy_offset = st.number_input("Energy offset (keV)", value=0.0)
    with st.expander("Peak analysis settings", expanded=False):
        fit_enabled = st.toggle("Fit candidate photopeaks", value=True)
        match_enabled = st.toggle("Compare reference gamma lines", value=True)
        match_tolerance = st.slider("Line matching tolerance (keV)", 1, 30, 12)
        fit_window = st.slider("Fit half-window (keV)", 25, 90, 55)
        peak_separation = st.slider("Minimum peak separation (keV)", 0, 100, 35)
    with st.expander("Overlapping peaks (experimental)", expanded=False):
        if isotope == "Overlapping demo":
            st.caption("Synthetic example: peaks at 485 and 505 keV; not an isotope.")
        overlap_enabled = st.toggle("Fit two overlapping peaks", value=isotope == "Overlapping demo", key="overlap_fit_enabled")
        if overlap_enabled:
            overlap_center_1 = st.number_input("First peak center (keV)", min_value=0.0, value=485.0, key="overlap_center_1")
            overlap_center_2 = st.number_input("Second peak center (keV)", min_value=0.0, value=505.0, key="overlap_center_2")
            overlap_window = st.slider("Two-peak half-window (keV)", 10, 100, 35)
    with st.expander("Energy calibration", expanded=False):
        enable_calibration = st.toggle("Apply channel calibration")
        calibration_degree = st.selectbox("Polynomial degree", [1, 2])
        default_refs = "100,80\n500,400\n1000,800\n1500,1200" if calibration_degree == 1 else "100,80\n400,320\n700,560\n1000,800"
        calibration_csv = st.text_area("Reference pairs: channel,energy_keV", default_refs, height=110)
    with st.expander("Deployment diagnostics"):
        st.code(f"Loaded core: {gammaspec_core.__file__}\\nAvailable sources: {list(gammaspec_core.LINES)}")
        if "Overlapping demo" not in gammaspec_core.LINES:
            st.error("Running GammaSpec core is outdated. Confirm deployment branch is main and reboot/redeploy.")

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
    dashboard_tab, overview_tab, comparison_tab_main, peaks_tab, advanced_tab, detector_tab, interactions_tab, decay_tab, learning_tab, match_tab, calibration_tab, validation_tab, report_tab = st.tabs(
        ["Dashboard", "Spectrum", "Compare spectra", "Peak analysis", "Advanced fitting", "Detector physics", "Radiation interactions", "Radioactive decay", "Learning lab", "Reference matching", "Calibration", "Validation", "Report"])
    with dashboard_tab:
        st.subheader("Scientific results dashboard")
        dashboard_summary = build_results_summary(
            spectrum, peaks, fits, source_label=source_label,
            is_imported=upload is not None, calibration_applied=enable_calibration,
            overlap_result=overlap_result)
        st.markdown(f"**Data origin:** {dashboard_summary['data_origin']}")
        st.caption(f"Dataset: {dashboard_summary['dataset']} · Calibration: {dashboard_summary['energy_calibration']}")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Total counts", f"{dashboard_summary['total_counts']:,.0f}")
        d2.metric("Candidate maxima", dashboard_summary["candidate_maxima"])
        d3.metric("Successful Gaussian fits", dashboard_summary["successful_single_peak_fits"])
        d4.metric("Overlapping components", dashboard_summary["overlap_fit_components"])
        st.caption(f"Energy range: {dashboard_summary['energy_min_kev']:.2f}–{dashboard_summary['energy_max_kev']:.2f} keV across {dashboard_summary['channels']:,} channels.")
        st.plotly_chart(fig, use_container_width=True, key="dashboard_spectrum")
        st.markdown("**Analysis summary**")
        st.dataframe(pd.DataFrame([{"Metric": key.replace("_", " ").title(), "Value": value}
            for key, value in dashboard_summary.items()]), use_container_width=True, hide_index=True)
        export_left, export_right = st.columns(2)
        with export_left:
            st.download_button("Download summary CSV", summary_csv(dashboard_summary),
                "gammaspec_results_summary.csv", "text/csv", key="dashboard_csv")
        with export_right:
            st.download_button("Download summary JSON", summary_json(dashboard_summary),
                "gammaspec_results_summary.json", "application/json", key="dashboard_json")
        st.warning("Candidate maxima and fitted peaks are model-dependent. Imported data may require independent energy calibration. No result establishes isotope identity or radiation safety.")
    with overview_tab:
        if isotope == "Overlapping demo" and upload is None and not enable_calibration:
            st.info("Zoomed to 450–540 keV. Orange vertical lines mark injected centers; dotted curves show each injected Gaussian component. The measured total can still have only one maximum. × markers show noisy local maxima.")
        st.caption("Select two candidate × markers using Shift-click or box selection to seed the two-peak fit.")
        st.plotly_chart(fig, use_container_width=True, key="spectrum_peak_selector",
                        on_select=handle_peak_selection, selection_mode=("points", "box"))
        st.download_button("Export spectrum CSV", spectrum.to_csv(index=False), "spectrum.csv", "text/csv")
    with comparison_tab_main:
        st.subheader("Gamma-ray spectrum comparison")
        st.caption("Compare the current spectrum against a second simulated source or uploaded spectrum. Energy axes must overlap; imported energies need independent calibration.")
        compare_mode = st.radio("Second spectrum source", ["Simulated", "Upload CSV or counts text"],
            horizontal=True, key="comparison_mode")
        secondary = None
        secondary_label = None
        if compare_mode == "Simulated":
            compare_isotope = st.selectbox("Comparison source",
                ["Cs-137", "Co-60", "Background", "Overlapping demo"], key="comparison_source")
            compare_time = st.slider("Comparison live time (s)", 10, 300, 60,
                key="comparison_live_time")
            secondary = simulate_spectrum(compare_isotope, live_time_s=compare_time,
                seed=int(seed) + 1)
            secondary_label = f"Simulated {compare_isotope}"
        else:
            compare_upload = st.file_uploader("Second spectrum file", type=["csv", "txt"],
                key="comparison_upload")
            if compare_upload is not None:
                compare_format = st.selectbox("Second spectrum format", ["auto", "csv", "counts"],
                    key="comparison_format")
                compare_width = st.number_input("Second spectrum keV/channel (if needed)",
                    min_value=0.001, value=0.8, key="comparison_width")
                compare_offset = st.number_input("Second spectrum energy offset (keV)",
                    value=0.0, key="comparison_offset")
                secondary_label = compare_upload.name
                try:
                    secondary = import_spectrum_text(compare_upload.getvalue(),
                        format=compare_format, energy_per_channel_kev=compare_width,
                        energy_offset_kev=compare_offset)
                except ValueError as exc:
                    st.error(f"Second spectrum could not be imported: {exc}")
        normalization = st.radio("Comparison scaling",
            ["Area normalized (shape)", "Raw count density (counts/keV)"],
            horizontal=True, key="comparison_scaling")
        if secondary is not None:
            try:
                mode = "area" if normalization.startswith("Area") else "raw"
                compared = compare_spectra(spectrum, secondary, normalization=mode)
                comparison_stats = comparison_metrics(compared)
                left, right = st.columns(2)
                left.metric("Mean absolute difference",
                    f"{comparison_stats['mean_absolute_difference']:.4g}")
                corr = comparison_stats["pearson_correlation"]
                right.metric("Shape correlation", f"{corr:.3f}" if corr is not None else "N/A")
                overlay = go.Figure()
                overlay.add_trace(go.Scatter(x=compared.energy_kev, y=compared.primary,
                    mode="lines", name=source_label))
                overlay.add_trace(go.Scatter(x=compared.energy_kev, y=compared.secondary,
                    mode="lines", name=secondary_label))
                overlay.update_layout(xaxis_title="Energy (keV)",
                    yaxis_title="Normalized density (1/keV)" if mode == "area" else "Count density (counts/keV)",
                    height=350)
                st.plotly_chart(overlay, use_container_width=True)
                difference_fig = go.Figure(go.Scatter(x=compared.energy_kev,
                    y=compared.difference_primary_minus_secondary, mode="lines",
                    name="Primary − secondary"))
                difference_fig.add_hline(y=0, line_dash="dash")
                difference_fig.update_layout(xaxis_title="Energy (keV)",
                    yaxis_title="Primary minus secondary", height=280)
                st.plotly_chart(difference_fig, use_container_width=True)
                st.download_button("Download comparison CSV",
                    compared.to_csv(index=False), "spectrum_comparison.csv",
                    "text/csv", key="comparison_export")
                st.caption("Curves are interpolated count densities on a common energy grid. Area normalization removes overall intensity; raw count density is not a live-time-corrected rate. Difference values are descriptive, not Poisson significance or detector calibration.")
            except ValueError as exc:
                st.warning(f"Comparison unavailable: {exc}")
        else:
            st.info("Upload a second spectrum to compare.")
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
                                             template="plotly_dark", height=320)
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
    with advanced_tab:
        st.subheader("Advanced peak fitting and statistical diagnostics")
        st.caption("Model comparisons, fit residuals and Monte Carlo studies are grouped here to keep routine peak analysis focused.")
        with st.expander("Fit residual diagnostics", expanded=False):
            st.caption("Compare observed-minus-model residuals on the same energy channels. Standardized residuals divide by √max(observed counts, 1).")
            successful_fits = ([] if fits is None else
                               [row for row in fits.to_dict("records") if row.get("status") == "ok"])
            if overlap_result is None or not successful_fits:
                st.info("Enable both single-peak and two-component fitting to compare residuals.")
            else:
                try:
                    midpoint = (overlap_center_1 + overlap_center_2) / 2
                    closest_fit = min(successful_fits,
                                      key=lambda row: abs(row["centroid_kev"] - midpoint))
                    residuals = residual_comparison(
                        spectrum, closest_fit, overlap_result,
                        (overlap_center_1, overlap_center_2), overlap_window
                    )
                    residual_fig = go.Figure()
                    for field, label in (
                        ("single_standardized_residual", "Single Gaussian"),
                        ("two_standardized_residual", "Two components"),
                    ):
                        residual_fig.add_trace(go.Scatter(
                            x=residuals.energy_kev, y=residuals[field],
                            mode="lines+markers", name=label,
                            marker={"size": 4},
                        ))
                    residual_fig.add_hline(y=0, line_dash="dash", line_color="gray")
                    residual_fig.update_layout(
                        xaxis_title="Energy (keV)",
                        yaxis_title="Standardized residual",
                        template="plotly_dark", height=320,
                    )
                    st.plotly_chart(residual_fig, use_container_width=True)
                    col1, col2 = st.columns(2)
                    col1.metric("Single Gaussian · RMS",
                                f'{residual_rms(residuals["single_standardized_residual"]):.2f}')
                    col2.metric("Two components · RMS",
                                f'{residual_rms(residuals["two_standardized_residual"]):.2f}')
                    st.download_button("Download residual data CSV",
                                       residuals.to_csv(index=False),
                                       "fit_residuals.csv", "text/csv")
                    st.caption("Lower RMS indicates smaller residuals on this window, but the models have different numbers of parameters and may have been fitted on different windows. This is not a formal statistical model-selection test.")
                except (ValueError, KeyError) as exc:
                    st.warning(f"Residual comparison unavailable: {exc}")
        with st.expander("AIC / BIC model comparison", expanded=False):
            st.caption("Both models are evaluated on the same channels. The single Gaussian is refitted in the two-component fit window.")
            if overlap_result is None:
                st.info("Enable the two-component fit to compare the one- and two-Gaussian models.")
            else:
                try:
                    scores = compare_information_criteria(
                        spectrum, overlap_result,
                        (overlap_center_1, overlap_center_2), overlap_window
                    )
                    st.dataframe(scores.round(3), use_container_width=True, hide_index=True)
                    st.caption("Lower AIC/BIC is preferred under this approximate Gaussian weighted-residual model. Δ values are relative to the lowest score. These are not p-values, and do not prove two physical gamma emissions.")
                    st.caption("AIC/BIC assume the fitted likelihood and effective sample count are appropriate; the current weighting and Gaussian approximation limit scientific interpretation.")
                    st.download_button("Download AIC/BIC CSV", scores.to_csv(index=False),
                                       "model_comparison.csv", "text/csv")
                except (ValueError, KeyError) as exc:
                    st.warning(f"Model comparison unavailable: {exc}")
        with st.expander("Poisson likelihood model comparison", expanded=False):
            st.caption("Refits both models using a Poisson count likelihood and a nonnegative constant background on the same channels.")
            if not overlap_enabled:
                st.info("Enable Fit two overlapping peaks to choose the centers and analysis window.")
            else:
                try:
                    poisson_scores = compare_poisson_models(
                        spectrum, (overlap_center_1, overlap_center_2), overlap_window
                    )
                    st.dataframe(poisson_scores.round(3), use_container_width=True, hide_index=True)
                    st.download_button("Download Poisson comparison CSV",
                                       poisson_scores.to_csv(index=False),
                                       "poisson_model_comparison.csv", "text/csv")
                    st.caption("Lower AIC/BIC favors a model under the assumed independent Poisson counts and constant-background models. These models use 4 and 7 fitted parameters, unlike the linear-background weighted Gaussian comparison above.")
                    st.caption("Results depend on initialization, model assumptions and fit convergence. This does not establish two physical emissions or constitute a calibrated significance test.")
                except (ValueError, KeyError) as exc:
                    st.warning(f"Poisson comparison unavailable: {exc}")
        with st.expander("Monte Carlo peak recovery", expanded=False):
            st.caption("Repeat independent Poisson simulations of the known 485/505 keV overlap example. Fit each simulation to estimate recovery bias and variability.")
            mc_trials = st.slider("Monte Carlo trials", 2, 50, 10, key="mc_trials")
            mc_time = st.slider("Simulated live time (s)", 10, 300, 60, key="mc_live_time")
            mc_seed = st.number_input("Monte Carlo seed", min_value=0, value=42, key="mc_seed")
            if st.button("Run recovery study", key="run_mc"):
                with st.spinner("Simulating and fitting independent spectra..."):
                    try:
                        st.session_state["mc_results"] = monte_carlo_recovery(
                            trials=mc_trials, live_time_s=mc_time, seed=int(mc_seed)
                        )
                        st.session_state["mc_settings"] = (mc_trials, mc_time, int(mc_seed))
                    except ValueError as exc:
                        st.error(f"Monte Carlo study unavailable: {exc}")
            mc_results = st.session_state.get("mc_results")
            if mc_results is not None:
                settings = st.session_state.get("mc_settings")
                if settings != (mc_trials, mc_time, int(mc_seed)):
                    st.info("Displayed results use previous settings. Run the study again to update.")
                summary_mc = recovery_summary(mc_results)
                st.metric("Successful fits", f'{summary_mc["successes"]}/{summary_mc["trials"]}')
                st.dataframe(pd.DataFrame([
                    {"Injected center (keV)": 485.0, "Bias (keV)": summary_mc["bias_1_kev"],
                     "Empirical SD (keV)": summary_mc["sd_1_kev"]},
                    {"Injected center (keV)": 505.0, "Bias (keV)": summary_mc["bias_2_kev"],
                     "Empirical SD (keV)": summary_mc["sd_2_kev"]},
                ]), use_container_width=True, hide_index=True)
                good = mc_results.loc[mc_results.status == "ok"]
                if not good.empty:
                    mc_fig = go.Figure()
                    for column, label in (("center_1_kev", "First recovered center"),
                                          ("center_2_kev", "Second recovered center")):
                        mc_fig.add_trace(go.Histogram(x=good[column], name=label, opacity=0.65))
                    mc_fig.update_layout(barmode="overlay", xaxis_title="Recovered centroid (keV)",
                                         yaxis_title="Trials", template="plotly_dark", height=320)
                    st.plotly_chart(mc_fig, use_container_width=True)
                st.download_button("Download Monte Carlo trials CSV", mc_results.to_csv(index=False),
                                   "monte_carlo_recovery.csv", "text/csv")
                st.caption("Simulation study only. Failed fits are excluded from bias/SD, so estimates may be selection-biased. Empirical SD is not a confidence interval or measurement uncertainty for uploaded data.")
        with st.expander("Peak resolution study", expanded=False):
            st.caption("Vary injected peak separation around 495 keV, resimulate Poisson counts, and measure centroid recovery.")
            rs_trials = st.slider("Trials per separation", 2, 15, 4, key="rs_trials")
            rs_time = st.slider("Resolution study live time (s)", 10, 300, 60, key="rs_time")
            rs_tol = st.slider("Recovery tolerance (keV)", 1, 15, 5, key="rs_tol")
            if st.button("Run separation sweep", key="run_resolution"):
                with st.spinner("Running independent simulations and fits..."):
                    try:
                        st.session_state["resolution_trials"] = resolution_study(
                            trials=rs_trials, live_time_s=rs_time, tolerance_kev=rs_tol
                        )
                        st.session_state["resolution_settings"] = (rs_trials, rs_time, rs_tol)
                    except ValueError as exc:
                        st.warning(f"Resolution study unavailable: {exc}")
            trial_results = st.session_state.get("resolution_trials")
            if trial_results is not None:
                if st.session_state.get("resolution_settings") != (rs_trials, rs_time, rs_tol):
                    st.info("Results use earlier settings; rerun to refresh.")
                resolution_table = summarize_resolution(trial_results)
                st.dataframe(resolution_table, use_container_width=True, hide_index=True)
                resolution_fig = go.Figure()
                resolution_fig.add_trace(go.Scatter(
                    x=resolution_table.separation_kev, y=resolution_table.recovery_rate,
                    mode="lines+markers", name="Accurate recovery fraction"))
                resolution_fig.add_trace(go.Scatter(
                    x=resolution_table.separation_kev, y=resolution_table.success_rate,
                    mode="lines+markers", name="Fit convergence fraction"))
                resolution_fig.update_layout(xaxis_title="Injected separation (keV)",
                                             yaxis_title="Fraction of trials",
                                             yaxis_range=[0, 1.05], template="plotly_dark",
                                             height=340)
                st.plotly_chart(resolution_fig, use_container_width=True)
                st.download_button("Download separation summary CSV",
                                   resolution_table.to_csv(index=False),
                                   "resolution_summary.csv", "text/csv")
                st.download_button("Download all separation trials CSV",
                                   trial_results.to_csv(index=False),
                                   "resolution_trials.csv", "text/csv")
                st.caption("Accurate recovery requires both fitted centroids within the selected tolerance of known injected centers. Seeded fits and this success criterion do not establish that peaks are independently detectable; small trial counts give noisy estimates.")
    with detector_tab:
        st.subheader("Detector physics")
        st.caption("Explore illustrative detector energy resolution, efficiency and counting uncertainty.")
        with st.expander("Detector resolution and FWHM", expanded=False):
            st.caption("Illustrative detector response models: representative reference widths at 662 keV with square-root energy scaling, not measured detector specifications.")
            detector_choice = st.selectbox("Detector response", ["HPGe (illustrative)", "NaI(Tl) (illustrative)"], key="detector_choice")
            detector_separation = st.slider("Peak separation (keV)", 2, 100, 20, key="detector_separation")
            detector_energy = st.slider("Doublet midpoint (keV)", 200, 1400, 495, key="detector_energy")
            detector_ref = st.number_input("Reference FWHM at 662 keV (keV)",
                min_value=0.1, max_value=150.0,
                value=1.8 if detector_choice.startswith("HPGe") else 46.0,
                step=0.5, key="detector_reference_fwhm")
            curve_energy = np.linspace(100, 1600, 300)
            widths = fwhm_kev(curve_energy, detector_choice, detector_ref)
            resolution_fig = go.Figure()
            resolution_fig.add_trace(go.Scatter(x=curve_energy, y=widths,
                name="FWHM (keV)", mode="lines"))
            resolution_fig.update_layout(xaxis_title="Energy (keV)",
                yaxis_title="FWHM (keV)", template="plotly_dark", height=300)
            st.plotly_chart(resolution_fig, use_container_width=True)
            local_energy = np.linspace(detector_energy - 120, detector_energy + 120, 800)
            first, second, combined = illustrative_doublet(
                local_energy, detector_energy, detector_separation,
                detector_choice, detector_ref)
            doublet_fig = go.Figure()
            for values, label in ((first, "First component"), (second, "Second component"),
                                  (combined, "Combined response")):
                doublet_fig.add_trace(go.Scatter(x=local_energy, y=values,
                    mode="lines", name=label))
            doublet_fig.update_layout(xaxis_title="Energy (keV)",
                yaxis_title="Normalized density (1/keV)", template="plotly_dark",
                height=340)
            st.plotly_chart(doublet_fig, use_container_width=True)
            st.metric("FWHM at midpoint", f"{fwhm_kev(detector_energy, detector_choice, detector_ref):.2f} keV")
            st.metric("Resolution at midpoint",
                f"{resolution_percent(detector_energy, detector_choice, detector_ref):.2f}%")
            st.caption("The two equal-area Gaussian components are idealized and noiseless. The plotted density is not a count spectrum. Real HPGe and NaI(Tl) resolution varies with detector design, electronics, and energy.")
        with st.expander("Detector efficiency and counting statistics", expanded=False):
            st.caption("Illustrative full-energy peak efficiencies, not absolute calibrations or manufacturer specifications.")
            eff_detector = st.selectbox("Efficiency model", ["HPGe (illustrative)", "NaI(Tl) (illustrative)"], key="eff_detector")
            eff_ref = st.slider("Peak efficiency at 662 keV (%)", 1, 60,
                                8 if eff_detector.startswith("HPGe") else 15, key="eff_ref")
            eff_energy = st.slider("Gamma energy (keV)", 100, 1600, 662, key="eff_energy")
            eff_rate = st.number_input("Incident gamma rate (photons/s)", min_value=0.0,
                                       max_value=100000.0, value=100.0, key="eff_rate")
            eff_time = st.slider("Counting time (s)", 1, 600, 60, key="eff_time")
            eff_background = st.number_input("Background rate in peak region (counts/s)",
                                             min_value=0.0, value=2.0, key="eff_background")
            eff_energy_axis = np.linspace(100, 1600, 300)
            eff_curve = 100 * peak_efficiency(eff_energy_axis, eff_detector, eff_ref / 100)
            eff_fig = go.Figure()
            eff_fig.add_trace(go.Scatter(x=eff_energy_axis, y=eff_curve, mode="lines",
                                         name="Full-energy peak efficiency"))
            eff_fig.update_layout(xaxis_title="Gamma energy (keV)", yaxis_title="Efficiency (%)",
                                  template="plotly_dark", height=300)
            st.plotly_chart(eff_fig, use_container_width=True)
            stats = counting_statistics(eff_energy, eff_detector, eff_rate, eff_time,
                                        eff_background, eff_ref / 100)
            c1, c2, c3 = st.columns(3)
            c1.metric("Expected peak counts", f'{stats["signal_counts"]:,.1f}')
            c2.metric("Poisson σ (total counts)", f'{stats["poisson_sigma_counts"]:,.1f}')
            c3.metric("Approx. SNR", f'{stats["approx_snr"]:.2f}')
            times = np.linspace(1, 600, 120)
            snr_values = [counting_statistics(eff_energy, eff_detector, eff_rate, t,
                          eff_background, eff_ref / 100)["approx_snr"] for t in times]
            snr_fig = go.Figure()
            snr_fig.add_trace(go.Scatter(x=times, y=snr_values, mode="lines", name="SNR"))
            snr_fig.update_layout(xaxis_title="Counting time (s)", yaxis_title="Approximate SNR",
                                  template="plotly_dark", height=280)
            st.plotly_chart(snr_fig, use_container_width=True)
            st.caption("Expected counts use incident rate × time × full-energy peak efficiency. SNR = expected signal / sqrt(signal + background), assuming known background; background estimation, dead time, pile-up, geometry and other detector effects are excluded.")
    with interactions_tab:
        st.subheader("Radiation interactions")
        st.caption("Explore educational gamma-ray attenuation, Compton kinematics and angular scattering distributions.")
        with st.expander("Gamma attenuation and shielding explorer", expanded=False):
            st.caption("Educational narrow-beam model using approximate coefficients, not validated shielding data. Do not use for radiation safety or shield design.")
            shield_material = st.selectbox("Shield material", list(MATERIALS), key="shield_material")
            shield_energy = st.slider("Gamma energy (keV)", 100, 1600, 662, key="shield_energy")
            shield_thickness = st.slider("Shield thickness (cm)", 0.0, 30.0, 5.0, 0.5, key="shield_thickness")
            shield_mu = linear_attenuation(shield_material, shield_energy)
            shield_lengths = attenuation_lengths(shield_material, shield_energy)
            shield_transmission = transmission(shield_material, shield_energy, shield_thickness)
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Uncollided transmission", f"{100 * shield_transmission:.3f}%")
            sc2.metric("HVL", f'{shield_lengths["hvl_cm"]:.2f} cm')
            sc3.metric("TVL", f'{shield_lengths["tvl_cm"]:.2f} cm')
            thickness_axis = np.linspace(0, 30, 200)
            attenuation_fig = go.Figure()
            for mat in MATERIALS:
                attenuation_fig.add_trace(go.Scatter(
                    x=thickness_axis, y=100 * transmission(mat, shield_energy, thickness_axis),
                    mode="lines", name=mat))
            attenuation_fig.update_layout(xaxis_title="Thickness (cm)",
                yaxis_title="Uncollided transmission (%)", yaxis_type="log",
                template="plotly_dark", height=320)
            st.plotly_chart(attenuation_fig, use_container_width=True)
            shield_table = pd.DataFrame([
                {"Material": mat, "Density (g/cm³)": MATERIALS[mat][0],
                 "Linear attenuation (cm⁻¹)": linear_attenuation(mat, shield_energy),
                 "HVL (cm)": attenuation_lengths(mat, shield_energy)["hvl_cm"],
                 "TVL (cm)": attenuation_lengths(mat, shield_energy)["tvl_cm"],
                 "Transmission (%)": 100 * transmission(mat, shield_energy, shield_thickness)}
                for mat in MATERIALS])
            st.dataframe(shield_table.round(3), use_container_width=True, hide_index=True)
            st.download_button("Download attenuation comparison CSV",
                shield_table.to_csv(index=False), "attenuation_comparison.csv", "text/csv")
            st.caption("Beer–Lambert uncollided photon transmission only; scattered radiation, buildup, geometry, secondary radiation, mixtures and detector response are omitted. Coefficients are intentionally illustrative and must not be used for exposure or shielding decisions.")
        with st.expander("Compton scattering explorer", expanded=False):
            st.caption("Ideal scattering from a free electron initially at rest; excludes binding effects and detector response.")
            compton_source = st.selectbox("Gamma-ray line", ["Cs-137 (661.657 keV)",
                "Co-60 (1173.228 keV)", "Co-60 (1332.492 keV)", "Custom"],
                key="compton_source")
            compton_default = {"Cs-137 (661.657 keV)": 661.657,
                "Co-60 (1173.228 keV)": 1173.228,
                "Co-60 (1332.492 keV)": 1332.492}
            compton_energy = (st.number_input("Incident energy (keV)", min_value=1.0,
                max_value=10000.0, value=662.0, key="compton_custom_energy")
                if compton_source == "Custom" else compton_default[compton_source])
            compton_angle = st.slider("Photon scattering angle (degrees)", 0, 180, 90,
                                      key="compton_angle")
            scattered = scattered_energy(compton_energy, compton_angle)
            recoil = recoil_energy(compton_energy, compton_angle)
            edge = compton_edge(compton_energy)
            c1, c2, c3 = st.columns(3)
            c1.metric("Scattered photon", f"{scattered:.2f} keV")
            c2.metric("Recoil electron", f"{recoil:.2f} keV")
            c3.metric("Compton edge", f"{edge:.2f} keV")
            compton_angles = np.linspace(0, 180, 181)
            compton_fig = go.Figure()
            compton_fig.add_trace(go.Scatter(x=compton_angles,
                y=scattered_energy(compton_energy, compton_angles), name="Scattered photon"))
            compton_fig.add_trace(go.Scatter(x=compton_angles,
                y=recoil_energy(compton_energy, compton_angles), name="Recoil electron"))
            compton_fig.add_vline(x=compton_angle, line_dash="dash")
            compton_fig.update_layout(xaxis_title="Photon scattering angle (degrees)",
                yaxis_title="Energy (keV)", template="plotly_dark", height=320)
            st.plotly_chart(compton_fig, use_container_width=True)
            st.caption(f"Wavelength shift at selected angle: {scattered_wavelength_shift_pm(compton_angle):.3f} pm. The Compton edge is the maximum energy transfer to one electron, not the full measured continuum shape.")
            compton_table = pd.DataFrame({"Angle (degrees)": compton_angles,
                "Scattered photon (keV)": scattered_energy(compton_energy, compton_angles),
                "Recoil electron (keV)": recoil_energy(compton_energy, compton_angles)})
            st.download_button("Download Compton kinematics CSV",
                compton_table.to_csv(index=False), "compton_kinematics.csv", "text/csv")
        with st.expander("Klein–Nishina scattering analysis", expanded=False):
            st.caption("Unpolarized photons scattered by free electrons initially at rest. Differential cross sections are per electron, not a material attenuation model.")
            kn_energy = st.selectbox("Incident photon energy (keV)",
                [100.0, 511.0, 661.657, 1173.228, 1332.492], index=2, key="kn_energy")
            kn_angles, kn_density, kn_cross = angular_distribution(kn_energy)
            kn_total = total_cross_section(kn_energy)
            st.metric("Integrated cross section", f"{kn_total / 1e-24:.4f} barns/electron")
            kn_fig = go.Figure()
            for energy in (100.0, 661.657, 1332.492):
                angles, density, _ = angular_distribution(energy)
                kn_fig.add_trace(go.Scatter(x=angles, y=density,
                    name=f"{energy:g} keV", mode="lines"))
            kn_fig.update_layout(xaxis_title="Photon scattering angle (degrees)",
                yaxis_title="Probability density (per degree)", template="plotly_dark", height=320)
            st.plotly_chart(kn_fig, use_container_width=True)
            st.caption("Angular probability includes the sin(θ) solid-angle factor. A zero probability density at 0° does not mean the forward differential cross section vanishes.")
            kn_fig2 = go.Figure()
            kn_fig2.add_trace(go.Scatter(x=kn_angles, y=kn_cross / 1e-24,
                name="Differential cross section", mode="lines"))
            kn_fig2.update_layout(xaxis_title="Photon scattering angle (degrees)",
                yaxis_title="dσ/dΩ (barns/sr/electron)", template="plotly_dark", height=320)
            st.plotly_chart(kn_fig2, use_container_width=True)
            kn_table = pd.DataFrame({"Angle (degrees)": kn_angles,
                "Probability density (per degree)": kn_density,
                "dσ/dΩ (barns/sr/electron)": kn_cross / 1e-24})
            st.download_button("Download Klein–Nishina CSV", kn_table.to_csv(index=False),
                "klein_nishina.csv", "text/csv")
    with decay_tab:
        st.subheader("Radioactive decay and half-life simulator")
        st.caption("Ideal exponential decay using approximate educational half-lives. Activity is a rate; it is not a radiation dose.")
        decay_isotope = st.selectbox("Radionuclide", list(HALF_LIVES_YEARS), key="decay_isotope")
        decay_half_life = HALF_LIVES_YEARS[decay_isotope]
        decay_activity = st.number_input("Initial activity (Bq)", min_value=0.0,
            value=1000.0, step=100.0, key="decay_activity")
        decay_half_lives = st.slider("Elapsed time (half-lives)", 0.0, 10.0, 2.0,
            step=0.25, key="decay_elapsed")
        elapsed_years = decay_half_lives * decay_half_life
        activity_now = remaining_activity(decay_activity, elapsed_years, decay_half_life)
        d1, d2, d3 = st.columns(3)
        d1.metric("Remaining activity", f"{activity_now:,.3g} Bq")
        d2.metric("Remaining fraction", f"{100 * (1 - decay_fraction(elapsed_years, decay_half_life)):.2f}%")
        d3.metric("Elapsed time", f"{elapsed_years:.4g} years")
        decay_axis = np.linspace(0, 10, 201)
        decay_curve = remaining_activity(decay_activity, decay_axis * decay_half_life, decay_half_life)
        decay_fig = go.Figure()
        decay_fig.add_trace(go.Scatter(x=decay_axis, y=decay_curve, mode="lines", name="Parent activity"))
        decay_fig.add_vline(x=decay_half_lives, line_dash="dash")
        decay_fig.update_layout(xaxis_title="Elapsed time (half-lives)",
            yaxis_title="Activity (Bq)", template="plotly_dark", height=320)
        st.plotly_chart(decay_fig, use_container_width=True)
        decay_table = pd.DataFrame({"Elapsed half-lives": decay_axis,
            "Elapsed years": decay_axis * decay_half_life, "Activity (Bq)": decay_curve})
        st.download_button("Download decay curve CSV", decay_table.to_csv(index=False),
            "radioactive_decay.csv", "text/csv")
        with st.expander("Parent–daughter decay chain (idealized)", expanded=False):
            st.caption("Two-step chain P → D → stable; daughter initially absent, 100% branching, no other pathways. Both curves show numbers of atoms, not activity.")
            parent_hl = st.number_input("Parent half-life (arbitrary time units)",
                min_value=0.01, value=4.0, key="chain_parent_hl")
            daughter_hl = st.number_input("Daughter half-life (same units)",
                min_value=0.01, value=1.0, key="chain_daughter_hl")
            chain_time = np.linspace(0, 10 * max(parent_hl, daughter_hl), 301)
            chain_parent, chain_daughter = parent_daughter_atoms(
                1000, chain_time, parent_hl, daughter_hl)
            chain_fig = go.Figure()
            chain_fig.add_trace(go.Scatter(x=chain_time, y=chain_parent,
                name="Parent atoms", mode="lines"))
            chain_fig.add_trace(go.Scatter(x=chain_time, y=chain_daughter,
                name="Daughter atoms", mode="lines"))
            chain_fig.update_layout(xaxis_title="Time (arbitrary units)",
                yaxis_title="Number of atoms", template="plotly_dark", height=320)
            st.plotly_chart(chain_fig, use_container_width=True)
            chain_table = pd.DataFrame({"Time": chain_time, "Parent atoms": chain_parent,
                "Daughter atoms": chain_daughter})
            st.download_button("Download decay chain CSV", chain_table.to_csv(index=False),
                "parent_daughter_decay.csv", "text/csv")
    with decay_tab:
        with st.expander("Multi-generation decay chain explorer", expanded=False):
            st.caption("Idealized sequential chain with branching to untracked stable products. Times and half-lives use the same arbitrary units.")
            chain_generations = st.slider("Radioactive generations", 2, 5, 3, key="multi_chain_generations")
            chain_half_lives = []
            chain_branches = []
            for i in range(chain_generations):
                chain_half_lives.append(st.number_input(
                    f"Generation {i + 1} half-life", min_value=0.01,
                    value=float([100.0, 10.0, 1.0, 0.5, 0.25][i]),
                    key=f"multi_chain_hl_{i}"))
                if i < chain_generations - 1:
                    chain_branches.append(st.slider(
                        f"Fraction of generation {i + 1} decays feeding generation {i + 2}",
                        0.0, 1.0, 1.0, 0.05, key=f"multi_chain_branch_{i}"))
            chain_initial = st.number_input("Initial parent atoms", min_value=0.0,
                value=1000.0, step=100.0, key="multi_chain_atoms")
            chain_duration = st.slider("Time span (parent half-lives)", 0.5, 20.0,
                8.0, 0.5, key="multi_chain_span")
            chain_times = np.linspace(0, chain_duration * chain_half_lives[0], 201)
            chain_pop, chain_activities = chain_populations(
                chain_half_lives, chain_times, chain_initial, chain_branches)
            chain_view = st.radio("Display", ["Atom populations", "Activities (decays per time unit)"],
                horizontal=True, key="multi_chain_view")
            chain_values = chain_pop if chain_view == "Atom populations" else chain_activities
            multi_fig = go.Figure()
            for i in range(chain_generations):
                multi_fig.add_trace(go.Scatter(x=chain_times, y=chain_values[:, i],
                    mode="lines", name=f"Generation {i + 1}"))
            multi_fig.update_layout(xaxis_title="Time (arbitrary units)",
                yaxis_title=chain_view, height=360)
            st.plotly_chart(multi_fig, use_container_width=True)
            st.info(equilibrium_class(chain_half_lives[0], chain_half_lives[1]))
            st.caption("Equilibrium classification indicates a possible regime, not whether equilibrium has been reached. A branch below 100% reduces daughter feeding.")
            multi_table = pd.DataFrame({"Time": chain_times})
            for i in range(chain_generations):
                multi_table[f"Generation {i + 1} atoms"] = chain_pop[:, i]
                multi_table[f"Generation {i + 1} decays/time"] = chain_activities[:, i]
            st.download_button("Download multi-generation chain CSV",
                multi_table.to_csv(index=False), "multigeneration_decay_chain.csv",
                "text/csv", key="multi_chain_csv")
    with learning_tab:
        st.subheader("Gamma-ray spectrum learning lab")
        st.caption("Session-only progress: it resets when the app session ends. Export your progress before leaving.")
        topics = list(EXERCISES) + list(CHALLENGES)
        if "learning_progress" not in st.session_state:
            st.session_state["learning_progress"] = {}
        progress = st.session_state["learning_progress"]
        stats = progress_summary(progress, topics)
        p1, p2, p3 = st.columns(3)
        p1.metric("Topics completed", f'{stats["completed"]}/{stats["total"]}')
        p2.metric("Attempts", stats["attempts"])
        p3.metric("Correct attempts", f'{stats["accuracy_percent"]:.0f}%')
        st.progress(stats["completed"] / stats["total"] if stats["total"] else 0)
        st.dataframe(pd.DataFrame([
            {"Topic": topic, "Completed": "Yes" if progress.get(topic, {}).get("completed") else "No",
             "Attempts": progress.get(topic, {}).get("attempts", 0),
             "Correct": progress.get(topic, {}).get("correct", 0)}
            for topic in topics]), use_container_width=True, hide_index=True)
        st.download_button("Download learning progress CSV",
            progress_csv(progress, topics), "gammaspec_learning_progress.csv", "text/csv",
            key="learning_progress_download")
        lab_topic = st.selectbox("Choose an exercise", topics, key="lab_topic")
        if lab_topic in EXERCISES:
            lab_exercise = EXERCISES[lab_topic]
            st.markdown(f"**Question:** {lab_exercise['prompt']}")
            st.info(lab_example(lab_topic))
            lab_choice = st.radio("Your answer", lab_exercise["choices"], index=None,
                                  key=f"lab_answer_{lab_topic}")
            if st.button("Check answer", key="lab_check_answer", disabled=lab_choice is None):
                correct = check_answer(lab_topic, lab_choice)
                st.session_state["learning_progress"] = record_attempt(
                    st.session_state["learning_progress"], lab_topic, correct)
                st.session_state[f"lab_feedback_{lab_topic}"] = (correct, lab_choice)
                st.rerun()
            feedback = st.session_state.get(f"lab_feedback_{lab_topic}")
            if feedback is not None:
                if feedback[0]:
                    st.success("Correct!")
                else:
                    st.warning(f"Not quite. The correct answer is: {lab_exercise['answer']}.")
                st.markdown(lab_exercise["explanation"])
            if st.button("Retry this question", key="lab_retry_choice"):
                st.session_state.pop(f"lab_feedback_{lab_topic}", None)
                st.session_state.pop(f"lab_answer_{lab_topic}", None)
                st.rerun()
        else:
            challenge = CHALLENGES[lab_topic]
            st.markdown(f"**Calculation:** {challenge['prompt']}")
            numeric_answer = st.number_input(
                f"Answer ({challenge['unit']})", value=0.0, step=0.1,
                key=f"lab_numeric_{lab_topic}")
            if st.button("Check calculation", key="lab_check_numeric"):
                correct = grade_numeric(lab_topic, numeric_answer)
                st.session_state["learning_progress"] = record_attempt(
                    st.session_state["learning_progress"], lab_topic, correct)
                st.session_state[f"lab_feedback_{lab_topic}"] = (correct, numeric_answer)
                st.rerun()
            feedback = st.session_state.get(f"lab_feedback_{lab_topic}")
            if feedback is not None:
                if feedback[0]:
                    st.success("Correct!")
                else:
                    st.warning(f"Not quite. Expected approximately {challenge['answer']:g} {challenge['unit']}.")
                st.markdown(challenge["explanation"])
            if st.button("Retry this calculation", key="lab_retry_numeric"):
                st.session_state.pop(f"lab_feedback_{lab_topic}", None)
                st.session_state.pop(f"lab_numeric_{lab_topic}", None)
                st.rerun()
        st.caption("Educational examples only; not for radiation safety, dosimetry, or isotope identification.")
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
        with st.expander("Peak matching and candidate ranking workbench", expanded=True):
            st.caption("Rank reference-line energy agreements using one-to-one assignments. Scores are heuristics, NOT probabilities or validated confidence levels.")
            matching_input = st.radio("Observed peak energies",
                ["Fitted centroids (fallback to maxima)", "Detected maxima"],
                horizontal=True, key="matching_input")
            workbench_observed = (observed if matching_input.startswith("Fitted")
                                  else peaks.energy_kev.to_numpy())
            ranking, matched_details = rank_reference_candidates(
                workbench_observed, tolerance_kev=match_tolerance)
            if ranking.matched_lines.max() == 0:
                st.info("No reference-line agreements within the selected tolerance.")
            else:
                st.dataframe(ranking, use_container_width=True, hide_index=True)
                st.markdown("**Matched line residuals**")
                st.dataframe(matched_details, use_container_width=True, hide_index=True)
                ranked_fig = go.Figure(go.Bar(
                    x=ranking.candidate, y=ranking.ranking_score,
                    name="Heuristic ranking score"))
                ranked_fig.update_layout(yaxis_title="Heuristic score (not probability)",
                    xaxis_title="Reference candidate", height=300)
                st.plotly_chart(ranked_fig, use_container_width=True)
                st.download_button("Download candidate ranking CSV",
                    ranking.to_csv(index=False), "gamma_candidate_ranking.csv",
                    "text/csv", key="ranking_csv")
                st.download_button("Download line matching details CSV",
                    matched_details.to_csv(index=False), "gamma_line_matches.csv",
                    "text/csv", key="matching_details_csv")
            st.caption("The reference library is small and incomplete. Missing lines, mixtures, escape peaks, efficiency, intensity, coincidence effects, calibration and statistical uncertainty are not accounted for.")
        st.caption("Candidate matching is not isotope identification. Missing lines, mixtures, calibration error and detector response can change results.")
    with calibration_tab:
        st.subheader("Detector calibration and energy resolution")
        st.caption("Calibration references must be independently established. Fitted mean-energy uncertainty does not include reference-line uncertainties.")
        if enable_calibration:
            calibration_table, calibration_stats = calibration_diagnostics(
                pairs.channel, pairs.energy_kev, calibration,
                spectrum_channels=spectrum.channel.to_numpy())
            c1, c2, c3 = st.columns(3)
            c1.metric("Calibration RMSE", f"{calibration_stats['rmse_kev']:.4f} keV")
            c2.metric("Maximum absolute residual", f"{calibration_stats['max_absolute_residual_kev']:.4f} keV")
            c3.metric("Reference points", calibration_stats["reference_count"])
            if calibration_stats["extrapolates_beyond_references"]:
                st.warning("The calibrated spectrum extends beyond the reference channel range. Extrapolated energies may be unreliable.")
            st.dataframe(calibration_table, use_container_width=True, hide_index=True)
            residual_fig = go.Figure(go.Scatter(x=calibration_table.channel,
                y=calibration_table.residual_kev, mode="markers+lines",
                name="Reference minus fitted energy"))
            residual_fig.add_hline(y=0, line_dash="dash")
            residual_fig.update_layout(xaxis_title="Reference channel",
                yaxis_title="Calibration residual (keV)", height=300)
            st.plotly_chart(residual_fig, use_container_width=True)
            st.download_button("Download calibration diagnostics CSV",
                calibration_table.to_csv(index=False), "calibration_diagnostics.csv",
                "text/csv", key="calibration_diagnostics_export")
        else:
            st.info("Enable channel calibration in the sidebar to inspect fitted residuals.")
        st.divider()
        st.subheader("Measured peak energy resolution")
        st.caption("FWHM values must come from independently fitted or measured peaks. The table does not automatically infer a detector response model.")
        resolution_text = st.text_area("Energy (keV), FWHM (keV) pairs",
            value="59.54,1.0\\n661.657,1.8\\n1332.492,2.1",
            height=100, key="resolution_measurements")
        try:
            resolution_pairs = pd.read_csv(__import__("io").StringIO(resolution_text),
                header=None, names=["energy_kev", "fwhm_kev"])
            resolution_data = measured_resolution_table(
                resolution_pairs.energy_kev, resolution_pairs.fwhm_kev)
            st.dataframe(resolution_data, use_container_width=True, hide_index=True)
            resolution_fig = go.Figure(go.Scatter(x=resolution_data.energy_kev,
                y=resolution_data.resolution_percent, mode="markers+lines",
                name="Measured resolution"))
            resolution_fig.update_layout(xaxis_title="Peak energy (keV)",
                yaxis_title="FWHM / energy (%)", height=300)
            st.plotly_chart(resolution_fig, use_container_width=True)
            st.download_button("Download energy resolution CSV",
                resolution_data.to_csv(index=False), "measured_energy_resolution.csv",
                "text/csv", key="resolution_export")
        except (ValueError, TypeError) as exc:
            st.warning(f"Enter valid positive energy and FWHM pairs: {exc}")
        st.caption("Example FWHM values are illustrative. Energy calibration and resolution are distinct measurements; neither validates isotope identity or detector safety.")
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
