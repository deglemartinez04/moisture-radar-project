# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Handheld 60 GHz radar (Acconeer A121) moisture tomography for non-destructive wall inspection (FIU ECE QTM Lab senior design). There is no build system, linter, or test suite; everything is plain Python scripts. Windows is the primary platform.

## Commands

```
cd App && python main.py                       # desktop console (needs PySide6, pyqtgraph, numpy, scipy, acconeer-exptool)
python main.py --fullscreen                    # fullscreen (also F11)

python tools/h5_to_features.py capture.h5 out.csv --material Wood --wall-stack-id bench_test \
    --trial-number 1 --scan-position-m 0.0 --moisture-content-pct 24.5   # appends one row
python tools/feature_utils.py raw.csv augmented.csv
python tools/contrast_score.py ...             # see its docstring for the recording protocol

# ML pipeline (ml/ml/): build_dataset.py <dir_of_h5> -o dataset.csv ; train_model.py dataset.csv
```

`ml/*.bat` are double-click wrappers: run `1_SETUP_once.bat` once (creates a venv in `%LOCALAPPDATA%\MoistureRadarML`), then `2_RUN_CHECK.bat`, `3_RETRAIN_SAVED_DATASET.bat`, `4_FIVE_RUNS.bat`, `5_SHUFFLE_TEST.bat`. Set `NOPAUSE=1` to skip the final pause.

## Architecture

Data flows: **sensor → shared acquisition stream → features → data log / live readout.**

- **`App/sensor_manager.py`** owns the single sensor connection and config. Every tab (Settings, Moisture Level, Amplitude, Phase, Energy, Data Logging) reads and writes the same `SensorManager`, and the Phase and Energy tabs subscribe to the same live stream rather than opening their own sessions. `App/main.py` wires the tabs together and runs acquisition.
- **`App/features.py` is the single source of truth for features.** Both the Data Logging tab (rows in `data_log.csv`) and the live tab call `extract_features()`, so training and live numbers can't drift apart. Distance-independent features (amplitude ratio, peak separation, phase delta) are the default model inputs; absolute values are diagnostics only.
- **`App/close_range.py`** applies Acconeer's close-range corrections (zero-point offset from the loopback pulse, phase-stable leakage cancellation). Distances shown or logged are `raw − zero-point offset`. Sensor calibration from the first session is reused for later sessions.
- The two boundaries are Boundary 1 = drywall front surface and Boundary 2 = wood stud face. `App/README.md` lists default sensor settings and the sensor-server setup (the app connects by IP to `acc_exploration_server_a121` on a Raspberry Pi).
- `live_tab.py` still shows a placeholder moisture readout until a trained model is wired in.
- The older root-level extractor (`tools/h5_to_features.py` + `tools/feature_utils.py`) is a separate, earlier whole-recording feature implementation. It buckets moisture as Low/Medium/High (15%/19%), which doesn't match the team's Dry/Damp/Saturated classes (12%/20%) defined only in `ml/ml/labels.py`.

## Repo state and gotchas

- Do not edit app code as part of repo cleanup or reorganization; move files only.
- `ml/simulated/` and `ml/saved_dataset/` are **synthetic** data. They show that the code works, not that the sensor can measure moisture; don't use their models in the app.
- `data/legacy/Wood_Dry_4.h5` has no wall reflection; don't train on it.
- `*.h5` is gitignored except under `data/recordings/` and `data/legacy/`.
- Git branches: `main`, `App-development`, and the local `app-dev-work` (a reorganization of `App-development` around app v9).
