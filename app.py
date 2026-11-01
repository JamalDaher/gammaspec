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
                        template="plotly_dark", height=340,
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
                template="plotly_dark", height=340)
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
                yaxis_title="Energy (keV)", template="plotly_dark", height=340)
            st.plotly_chart(compton_fig, use_container_width=True)
            st.caption(f"Wavelength shift at selected angle: {scattered_wavelength_shift_pm(compton_angle):.3f} pm. The Compton edge is the maximum energy transfer to one electron, not the full measured continuum shape.")
            compton_table = pd.DataFrame({"Angle (degrees)": compton_angles,
                "Scattered photon (keV)": scattered_energy(compton_energy, compton_angles),
                "Recoil electron (keV)": recoil_energy(compton_energy, compton_angles)})
            st.download_button("Download Compton kinematics CSV",
                compton_table.to_csv(index=False), "compton_kinematics.csv", "text/csv")
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
