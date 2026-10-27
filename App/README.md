# Radar Moisture Tomography — A121 Console

A desktop app replacing the Acconeer Exploration Tool for this project specifically:
connect to the A121 sensor, adjust every configurable sensor parameter, record
Sparse IQ .h5 files for ML training data, and run a live scan with graphs and a
moisture readout.

## Setup (one-time)

```
pip install PySide6 pyqtgraph numpy scipy acconeer-exptool --break-system-packages
```
(drop `--break-system-packages` on Windows/macOS if not using Debian/Raspberry Pi OS)

## Running

```
python main.py
```

This opens the Settings, Moisture Level, Amplitude, Phase, Energy, and Data Logging tabs (in that order, left to right). On small windows (under 900x560, e.g. a 5-inch 800x480 display) the app switches to a compact look automatically; press F11 or run `python main.py --fullscreen` for fullscreen. The Phase and Energy tabs subscribe to the same shared live acquisition stream; they do not open separate sensor sessions.

## Files

- `main.py` — entry point, window, and dark theme styling. Run this one.
- `sensor_manager.py` — shared sensor connection + config, used by all three tabs.
  This is what makes settings changes in the Settings tab instantly visible in
  the Record and Live tabs — they all read/write the same `SensorManager` object.
- `settings_tab.py` — every configurable A121 parameter (per Acconeer's SensorConfig
  API), organized into Range / Timing / RF / Advanced sections, plus the IP-address
  connection control.
- `record_tab.py` — mimics the Acconeer GUI's Sparse IQ Start/Stop/Record workflow.
  Use this tab to build your `.h5` training recordings, exactly like before.
- `live_tab.py` — Moisture Level dashboard: shows the live features and a placeholder
  moisture readout. Feature math is NOT here any more; it calls `features.py`.
- `features.py` — the single source of truth for every feature (boundary peaks, amplitude
  ratio, phase delta, SNR, ...). Live scan and Data Logging both call it, so live values and
  the rows in `data_log.csv` are computed identically.
- `ui_theme.py` — explicit styling for scroll bars and +/- spin boxes (finger-sized, both
  directions clickable). Set `RADAR_TOUCH_PX=30` to force a different control size.
- `health.py` — sensor-health header message (temperature drift, saturation, ...).
- `repeatability.py` — repeatability and standoff-comparison report behind the Data Logging "Analyze" button.
- `close_range.py` — Acconeer's close-range corrections (pure numpy/scipy): zero-point
  distance offset from the loopback pulse, and phase-stable leakage cancellation using a
  loopback phase reference. See "Close-range corrections" below.

- `plot_utils.py` — `CompactPlotWidget`, a pyqtgraph plot that can shrink to fit small displays.
- `diagnostic_tabs.py` — Phase and Energy live diagnostic tabs. The Energy tab plots mean squared-IQ energy versus distance in millions of arbitrary units, with separate peak-energy, peak-distance, and integrated-energy readouts. These values are signal diagnostics, not calibrated moisture measurements.

## Project default settings

Defaults follow Acconeer's own close-range design (Distance Detector close-range session):
start point 0, 80 points, step length 1 (0–0.2 m, i.e. 0–6 in from the sensor),
sweeps per frame 5 (averaged coherently), HWAAS 32, Profile 1, receiver gain 12, PRF 15.6 MHz,
phase enhancement + IQ imbalance compensation on, loopback phase reference on.
"Reset defaults" restores these (keeping the IP address). Settings saved by an older version
of the app are replaced by these defaults once (only the IP is kept).

Step length only accepts the A121's valid values 1, 2, 3, 4, 6, 8, 12, 24 (the +/- buttons
jump 4 → 6 → 8 → 12 …; typed values snap to the nearest). Start point may be negative
(down to −275). The old "Enable loopback" switch was removed: it turned the whole sweep into
an on-chip measurement with no wall data.

## Close-range corrections (from the Acconeer SDK / Exploration Tool)

1. **Zero-point offset.** Raw Sparse IQ distances contain no zero point (SDK:
   `acc_processing_points_to_meter` "does not include any zero-point offset"). At the first
   Start after connecting, the app measures the on-chip loopback pulse with Acconeer's
   offset configuration (start −30, 50 points, Profile 1, HWAAS 64) and computes the offset
   the same way the Distance Detector does. Every distance shown or logged is
   `raw − zero-point offset`, i.e. measured from the sensor. There is deliberately no user lens
   offset: the device is used flush (or on fixed case standoffs), and the physical
   sensor-to-wall gap is recorded per trial as "Sensor standoff" in Data Logging.
2. **One sensor calibration, reused.** The sensor calibration from the first session is
   passed to every later `setup_session()` (`calibrations=`), so Stop/Start and recordings
   no longer re-calibrate the sensor and shift the leakage phase under a stored background.
   **Recalibrate** (header) drops it, the offset and the background; use it when the header
   reports temperature drift or `calibration_needed`.
3. **Loopback phase reference.** Each frame has a 1-point loopback subsweep (subframe 0) in
   front of the wall range (subframe 1). The background is stored relative to the loopback
   phase and rotated to every frame's loopback phase before it is subtracted (Acconeer's
   phase-jitter compensation), then the sweeps are averaged coherently.
4. **Distance filter.** Every frame (raw and corrected) goes through Acconeer's matched
   distance filter (first-order Butterworth along distance, zero phase, "narrow" below 0.1 m)
   before amplitudes and features. Toggle: Settings → Advanced → Distance filter.
5. **TX-off noise floor.** At Start (once per settings combination) the app records 10
   frames with the transmitter off and uses their RMS as the noise floor for peak detection
   and SNR (`noise_source = tx_off` in the CSV). The old 10th-percentile estimate is only a
   fallback, because at 0–150 mm every bin contains leakage or wall.

## Features (features.py, schema 5)

- **Peak detection (Acconeer definition), ignoring the first 15 mm (leakage residual):** a point larger than both neighbours, all three
  points above `noise × 10^(10 dB/20)` and ≥ 5 % of the maximum. Peaks are ordered by
  distance; the first is the front (wall) surface.
- **Window features (primary model inputs),** anchored on the front surface so they do not
  move with standoff: front window = front ± 10 mm, rear window = 15–60 mm behind it.
  Computed per frame on the complex corrected sweep, then averaged:
  `front_distance_m, front_amplitude, rear_energy_ratio_db (+ _std), rear_centroid_offset_m,
  rear_peak_offset_m, rear_phase_rad (+ _std_rad)` (rear phase is relative to the front
  surface, fed to models as sin/cos). They exist whenever the wall is found, unlike the
  two-peak features, which need a separate second peak.
- The two-peak features (`amplitude_ratio`, `peak_separation_m`, …) are still logged.

## Repeatability & standoff comparison (Data Logging tab)

Record every condition 3+ times, moving the material away and back between recordings,
at each standoff ("Sensor standoff" field). **Analyze data_log.csv** then shows:
- per (material, moisture, sample, standoff): mean ± spread of the key features across
  placements;
- per standoff: worst-pair separation d' between Dry/Damp/Saturated, and between materials
  (stud detection). d' ≥ 2 is usable, ≥ 3 comfortable. Pick the standoff/settings with the
  highest d'. Logic is in `repeatability.py`.

## Connecting to the sensor

Enter the Pi's IP address (or `127.0.0.1` if running directly on the Pi with the
server running locally) in the Settings tab and click Connect. The `acc_exploration_server_a121`
binary must already be running on the Pi (as a systemd service, or manually via
`sudo ./acc_exploration_server_a121`) before you click Connect here.

## Experimental background/leakage correction

The **Amplitude Plot** tab includes an experimental background calibration workflow.
Start live measurements, place the sensor in a stationary reference setup with no
target wall/material in front of it, and click **Calibrate Background (30 frames)**.
After calibration, enable **Use background correction for live features** to apply
complex-IQ subtraction before amplitude is calculated. The plot compares the raw
and corrected amplitude traces.

With the loopback phase reference on, this is the same leakage-cancellation method as
Acconeer's Distance Detector close-range mode. Calibration is held in memory only and
is lost when the app closes. Recalibrate if sensor settings, mounting, lens,
temperature, or the reference environment changes. Do not assume a smaller spike
means better moisture detection: compare measurements with and without a target
and verify that genuine wall responses are not being removed. Once a background exists it is used
by the live features AND by Data Logging (see below).

## Data Logging workflow (background first)

Data Logging refuses to record without a background calibration that matches the current
sensor settings, so every CSV row is comparable.

1. Settings: connect, apply settings. 2. **Start Measurement**, point the sensor at open
air (lens and standoff attached), Amplitude tab: **Calibrate Background**. 3. **Stop**
(only one sensor session can run at a time). 4. Data Logging: fill in Materials / Moisture /
Experiment conditions (sample ID, standoff, water added, time since wet, meter reading),
press **Start recording**. The calibration is applied to the recording and, with "Save raw
.h5", saved next to it as `*_background.npz`.

Changing any sensor setting, or re-calibrating the sensor, invalidates the background (the
tab warns and refuses). Rows are written with `schema_version` 4; **old-format CSV files and the new format must not be
mixed in one file**. "Add H5 recording" uses the background saved with that recording, or the
current one only if the settings match.

Stop/Start reuses the cached sensor calibration, so the background stays valid between the
live view and recordings. Recalibrate if the sensor temperature has changed (header shows this).

## Amplitude vs energy, raw vs corrected
- **Amplitude** = |IQ| per distance point, averaged over frames. **Energy** = |IQ|² per point, averaged over frames. Energy is amplitude squared, so strong echoes stand out much more and weak ones almost vanish (use the Energy tab's log scale to see them).
- **Corrected** = computed on (raw IQ - background IQ). Corrected energy is the energy of that difference; it is NOT raw energy minus background energy (|a-b|² is not |a|²-|b|²).
- The Amplitude and Energy tabs show the raw trace (orange) and the corrected trace (blue, once the background is calibrated and correction is on).

## data_log.csv columns (schema_version 5, 63 columns)
Labels / conditions (never model inputs): `schema_version, recorded_at, material, moisture_level, trial_number, sample_id, h5_file, water_amount_ml, time_since_wet_min, measured_distance_mm, reference_moisture_pct, background_corrected`.
Window features: `front_distance_m, front_amplitude, rear_energy_ratio_db, rear_energy_ratio_db_std, rear_centroid_offset_m, rear_peak_offset_m, rear_phase_rad, rear_phase_std_rad`.
Processing: `distance_filtered, noise_source`.
Calibration / sensor state: `leakage_phase_compensated, zero_offset_m, sensor_temperature_c, calibration_temperature_c, saturated_frames`.
All `*_distance_m` columns are offset-corrected.
Quality flags: `peaks_found, two_peaks_found, strongest_at_edge`.
Raw vs corrected (sensor units; analysis only): `raw_|corrected_` + `peak_amplitude, peak_distance_m, mean_amplitude, peak_energy, integrated_energy`.
Boundary-peak features: `peak1_distance_m, peak2_distance_m, peak_separation_m, peak1_amplitude, peak2_amplitude, amplitude_ratio, amplitude_ratio_std, peak1_phase_rad, peak2_phase_rad, phase_delta_rad, peak2_phase_std_rad, tail_attenuation_slope, peak1_snr_db, total_energy`.
Legacy / diagnostic: `peak_distance_m, peak_amplitude_mean, peak_amplitude_std, peak_phase_mean_rad, peak_phase_std_rad, global_attenuation_slope, noise_floor, frame_to_frame_variance, num_frames`.
The boundary-peak and legacy columns are computed on the corrected signal when `background_corrected` = 1 (so `total_energy` equals `corrected_integrated_energy` in that case). A file with an older header is archived automatically and a new one started.

## The ML model is currently a placeholder

`live_tab.py` has a function `run_placeholder_model(features)` that fakes a moisture
percentage from amplitude alone — it has no real predictive meaning yet. Once
`train_model.py` produces a real `moisture_model.joblib`, replace that function's body
with:

```python
import joblib
model = joblib.load("moisture_model.joblib")  # load once, outside the function

def run_placeholder_model(features):
    feature_vector = [[
        # use the SAME names as the CSV columns / features.py, e.g.:
        features["amplitude_ratio"],
        features["peak_separation_m"],
        features["phase_delta_rad"],
        features["tail_attenuation_slope"],
        features["peak1_snr_db"],
    ]]
    label = model.predict(feature_vector)[0]
    # if the model supports predict_proba, use the winning class's probability
    # as a genuine confidence percentage instead of the current placeholder math
    return label, confidence_percent
```

## Known limitations / next steps

- No GPIO/button trigger — intentionally, per the project's decision to use this app's
  Start/Stop buttons instead.
- No X/Y position tracking for heatmap mode yet — the Live tab is single-point only.
- The Live tab's rolling window size (10 frames) and refresh interval (100ms) are tuned
  for responsiveness; if live predictions end up noisier than training-data predictions,
  revisit these two constants first (see `live_tab.py`, top of file).


## Phase and Energy diagnostics

The Phase tab plots circular-mean raw phase versus distance and shows a phase-at-peak history, phase variability across the rolling frame window, current peak range, background-correction state, and whether loopback phase-jitter compensation is on. Phase is angular and wraps at ±π, so the variability calculation uses circular statistics. The history is a diagnostic trend, not a moisture measurement.

The Energy tab plots the rolling mean of `abs(IQ) ** 2` at each distance point and reports peak distance and integrated energy. These are relative signal-energy readings (arbitrary units), not calibrated moisture percentages.

## Background calibration (Amplitude tab)
**Calibrate Background** - air calibration with the lens and any standoff attached and nothing within ~30 cm. Removes leakage and lens echoes. It lives only in the running app, is reused while the sensor settings are unchanged, and is discarded automatically when they change.

The header shows a sensor-health message (temperature drift since calibration, saturation, `calibration_needed`, delayed frames). Red means: press Recalibrate, Start Measurement, and redo the background calibration. Logic is in `health.py`.

## Offline scoring
`tools/contrast_score.py` compares saved .h5 recordings across geometries/settings; see its docstring for the recording protocol.


## Reference difference view (Amplitude tab)
"Set reference (current scene)" stores the current corrected COMPLEX profile (e.g. drywall
alone). The green trace then shows |current − reference|. In the v8 recordings, adding the
stud behind drywall often made the blue amplitude curve *smaller* (the stud echo is partly
out of phase with the drywall echo), yet the complex difference was 3,000–10,000, i.e.
10–30× the trial-to-trial spread. Use this view to see changes; amplitude alone hides them.
