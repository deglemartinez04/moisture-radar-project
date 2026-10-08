# Radar Moisture Tomography \u2014 A121 Console

A desktop app replacing the Acconeer Exploration Tool for this project specifically:
connect to the A121 sensor, adjust every configurable sensor parameter, record
Sparse IQ .h5 files for ML training data, and run a live scan with graphs and a
moisture readout.

## Setup (one-time)

```
pip install PySide6 pyqtgraph numpy acconeer-exptool --break-system-packages
```//
(drop `--break-system-packages` on Windows/macOS if not using Debian/Raspberry Pi OS)

## Running

```
python main.py
```

This opens a window with three tabs.

## Files

- `main.py` \u2014 entry point, window, and dark theme styling. Run this one.
- `sensor_manager.py` \u2014 shared sensor connection + config, used by all three tabs.
  This is what makes settings changes in the Settings tab instantly visible in
  the Record and Live tabs \u2014 they all read/write the same `SensorManager` object.
- `settings_tab.py` \u2014 every configurable A121 parameter (per Acconeer's SensorConfig
  API), organized into Range / Timing / RF / Advanced sections, plus the IP-address
  connection control.
- `record_tab.py` \u2014 mimics the Acconeer GUI's Sparse IQ Start/Stop/Record workflow.
  Use this tab to build your `.h5` training recordings, exactly like before.
- `live_tab.py` \u2014 continuous live scan: rolling-window feature extraction (same math
  as `extract_features.py`), live amplitude/phase/history graphs, a feature readout
  grid, and a moisture percentage/label.

## Project default settings

The Settings tab opens pre-loaded with this project's validated flush-contact config:
start point 24, 40 points, step length 2 (\u2248 0.060m\u20130.260m range), sweeps per frame 1,
HWAAS 32, Profile 1, receiver gain 16, PRF 15.6 MHz, TX enabled. "Reset to project
defaults" restores these at any time (keeping whatever IP address is currently set).

## Connecting to the sensor

Enter the Pi's IP address (or `127.0.0.1` if running directly on the Pi with the
server running locally) in the Settings tab and click Connect. The `acc_exploration_server_a121`
binary must already be running on the Pi (as a systemd service, or manually via
`sudo ./acc_exploration_server_a121`) before you click Connect here.

## The ML model is currently a placeholder

`live_tab.py` has a function `run_placeholder_model(features)` that fakes a moisture
percentage from amplitude alone \u2014 it has no real predictive meaning yet. Once
`train_model.py` produces a real `moisture_model.joblib`, replace that function's body
with:

```python
import joblib
model = joblib.load("moisture_model.joblib")  # load once, outside the function

def run_placeholder_model(features):
    feature_vector = [[
        features["peak_amplitude_mean"],
        features["peak_amplitude_std"],
        features["peak_phase_mean_rad"],
        features["peak_phase_std_rad"],
        features["attenuation_slope"],
        features["total_energy"],
        features["noise_floor"],
        features["frame_to_frame_variability"],
    ]]
    label = model.predict(feature_vector)[0]
    # if the model supports predict_proba, use the winning class's probability
    # as a genuine confidence percentage instead of the current placeholder math
    return label, confidence_percent
```

## Known limitations / next steps

- No GPIO/button trigger \u2014 intentionally, per the project's decision to use this app's
  Start/Stop buttons instead.
- No X/Y position tracking for heatmap mode yet \u2014 the Live tab is single-point only.
- The Live tab's rolling window size (10 frames) and refresh interval (100ms) are tuned
  for responsiveness; if live predictions end up noisier than training-data predictions,
  revisit these two constants first (see `live_tab.py`, top of file).
