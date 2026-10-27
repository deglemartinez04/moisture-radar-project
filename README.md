# Moisture Radar Project

Handheld 60 GHz radar (Acconeer A121) moisture tomography for non-destructive
wall inspection. Senior Design project, FIU ECE QTM Lab, Spring 2026.

## Layout

| Folder | What's in it |
|---|---|
| `App/` | The desktop console (PySide6): connect to the A121, adjust sensor settings, record `.h5` training data, run live scans. See `App/README.md`. |
| `tools/` | Offline scripts: `h5_to_features.py` (one training-schema row from a raw `.h5`), `feature_utils.py` (engineered columns and moisture bucketing), `contrast_score.py` (score how well a geometry/settings combination separates wall conditions). |
| `data/recordings/` | Real recordings from the app (`.h5`, background `.npz`, `data_log.csv`). |
| `data/legacy/` | An older-format log and `Wood_Dry_4.h5`, which has no wall reflection. Kept for reference. |
| `data/templates/` | CSV schemas for training data and live scans. |
| `ml/` | Machine-learning pipeline and the double-click checks (`ml/README.txt`). `ml/simulated/` holds **simulated** data only. |
| `docs/` | Progress and presentation scripts. |

## Usage

```
python App/main.py

python tools/h5_to_features.py capture.h5 output.csv --material Wood \
    --wall-stack-id bench_test --trial-number 1 --scan-position-m 0.0 \
    --moisture-content-pct 24.5
```

Run the app from inside `App/` if imports fail (`cd App && python main.py`).

## Data in git

Raw `.h5` captures are ignored by default (see `.gitignore`). The small real
recordings under `data/recordings/` and `data/legacy/` are an exception.
Larger captures belong on a shared drive.
