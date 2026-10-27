"""Turn a folder of .h5 recordings into a windowed training CSV.

Each recording is cut into non-overlapping windows of --window frames (default
10, the App's live ROLLING_WINDOW_SIZE), and each window becomes one row from
ml.features.window_features(). Training on live-sized windows means the model
sees the same noise level it will see live.

Labels come from the filename the App's Data Logging tab writes:
    <Material>_<Moisture>_<Trial>.h5     e.g. Wood_Damp_3.h5, Wood_+_drywall_Dry_1.h5

Usage:
    python ml/build_dataset.py Data_Log/ -o ml/dataset.csv
    python ml/build_dataset.py a.h5 b.h5 -o ml/dataset.csv --allow-any-label
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.features import window_features  # noqa: E402
from ml.h5io import load_recording  # noqa: E402
from ml.labels import describe, level_for_mc  # noqa: E402

LABELS = ["Dry", "Damp", "Saturated"]  # App/record_tab.py MOISTURE_LEVELS
WINDOW_FRAMES = 10                      # App/live_tab.py ROLLING_WINDOW_SIZE

ID_COLUMNS = ["recording_id", "window_index", "material", "trial_number", "moisture_level"]


def parse_filename(path):
    """'Wood_+_drywall_Damp_3.h5' -> ('Wood + drywall', 'Damp', 3)."""
    parts = Path(path).stem.split("_")
    if len(parts) < 3 or not parts[-1].isdigit():
        raise ValueError(f"{Path(path).name}: expected <Material>_<Moisture>_<Trial>.h5")
    return " ".join(parts[:-2]), parts[-2], int(parts[-1])


def recording_rows(h5_path, window=WINDOW_FRAMES):
    material, moisture, trial = parse_filename(h5_path)
    iq, distances_m, config = load_recording(h5_path)
    rows = []
    for w, start in enumerate(range(0, iq.shape[0] - window + 1, window)):
        row = {
            "recording_id": Path(h5_path).name,
            "window_index": w,
            "material": material,
            "trial_number": trial,
            "moisture_level": moisture,
        }
        row.update(window_features(iq[start:start + window], distances_m))
        rows.append(row)
    return rows, config


def check_against_pin_meter(paths):
    """Warn when a filename label disagrees with the pin-meter reading.

    Looks for mc_readings.csv (columns h5_file, pin_meter_mc_pct) next to the
    recordings. Cutoffs come from ml/labels.py.
    """
    seen = set()
    for folder in {p.parent for p in paths}:
        csv_path = folder / "mc_readings.csv"
        if not csv_path.exists():
            continue
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                name = row.get("h5_file", "")
                if not any(p.name == name for p in paths) or not row.get("pin_meter_mc_pct"):
                    continue
                seen.add(name)
                mc = float(row["pin_meter_mc_pct"])
                labeled = parse_filename(name)[1]
                expected = level_for_mc(mc)
                if labeled != expected:
                    print(f"LABEL CHECK {name}: filename says {labeled}, pin meter {mc:g}% is {expected} ({describe()})")
    missing = [p.name for p in paths if p.name not in seen]
    if missing:
        print(f"Note: no pin-meter reading for {len(missing)} recording(s); labels not cross-checked.")


def collect_paths(inputs):
    paths = []
    for item in inputs:
        p = Path(item)
        paths.extend(sorted(p.glob("*.h5")) if p.is_dir() else [p])
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help=".h5 files and/or folders containing them")
    parser.add_argument("-o", "--output", default="ml/dataset.csv")
    parser.add_argument("--window", type=int, default=WINDOW_FRAMES, help="frames per row (default %(default)s)")
    parser.add_argument("--allow-any-label", action="store_true",
                        help=f"accept moisture labels outside {LABELS} (e.g. old Low/Medium/High files)")
    args = parser.parse_args()

    all_rows, configs = [], {}
    paths = collect_paths(args.inputs)
    print(f"Moisture classes: {describe()}")
    check_against_pin_meter(paths)
    for path in paths:
        try:
            _, moisture, _ = parse_filename(path)
            if moisture not in LABELS and not args.allow_any_label:
                print(f"SKIP {path.name}: label '{moisture}' not in {LABELS} (use --allow-any-label)")
                continue
            rows, config = recording_rows(path, args.window)
        except Exception as exc:
            print(f"SKIP {path.name}: {exc}")
            continue
        if not rows:
            print(f"SKIP {path.name}: fewer than {args.window} frames")
            continue
        configs[path.name] = config
        detected = sum(r["target_detected"] for r in rows)
        both = sum(r["g2_detected"] for r in rows)
        print(f"{path.name}: {len(rows)} windows | G1 found in {detected}, G1+G2 in {both}")
        if detected < len(rows) / 2:
            print(f"  WARNING: no wall reflection in most windows -- check sensor contact/aim for {path.name}")
        all_rows.extend(rows)

    if not all_rows:
        sys.exit("No usable recordings.")

    distinct = {tuple(sorted(c.items())) for c in configs.values()}
    if len(distinct) > 1:
        print("WARNING: recordings use different sensor settings; features are not comparable across them:")
        for name, c in configs.items():
            print(f"  {name}: {c}")

    feature_cols = [k for k in all_rows[0] if k not in ID_COLUMNS]
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ID_COLUMNS + feature_cols)
        writer.writeheader()
        writer.writerows(all_rows)

    meta = {
        "window_frames": args.window,
        "sensor_config": next(iter(configs.values())) if len(distinct) == 1 else None,
        "recordings": sorted(configs),
    }
    Path(args.output).with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))

    labels = sorted({r["moisture_level"] for r in all_rows})
    print(f"\nWrote {len(all_rows)} rows from {len(configs)} recordings to {args.output}")
    for label in labels:
        recs = {r["recording_id"] for r in all_rows if r["moisture_level"] == label}
        print(f"  {label}: {len(recs)} recording(s)")


if __name__ == "__main__":
    main()
