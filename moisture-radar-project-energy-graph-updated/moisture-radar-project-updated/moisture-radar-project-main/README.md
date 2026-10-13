# Moisture Radar Project

Handheld 60 GHz radar (Acconeer A121) moisture tomography for non-destructive
wall inspection. Senior Design project, FIU ECE QTM Lab, Spring 2026.

## What's here

- `training_data_template.csv` / `live_scan_template.csv` — CSV schemas for
  training data (with ground truth) and live scans (with model predictions).
- `feature_utils.py` — computes the G2/G1 amplitude ratio, phase difference,
  signal-quality score, and Low/Medium/High moisture bucketing.
- `h5_to_features.py` — extracts one training-schema row from a raw Acconeer
  `.h5` capture: finds the two reflector gates (G1/G2), computes their stats,
  and runs them through `feature_utils`.

## Not in this repo

Raw `.h5` radar captures are excluded (see `.gitignore`) — they're large
binary files and belong on a shared drive, not in git history.

## Usage

```
python h5_to_features.py capture.h5 output.csv --material Wood \
    --wall-stack-id bench_test --trial 1 --scan-position 0.0 \
    --moisture-content-pct 24.5
```
