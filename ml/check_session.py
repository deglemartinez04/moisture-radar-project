"""Check one or more data_log.csv files against docs/recording_protocol.md.

Reports problems that would limit the Dry/Damp/Saturated model: missing pin-meter readings, labels that
disagree with the pin meter, wrong standoff, a day without all three classes, too few studs or days,
and wet studs logged with a single label for over an hour while drying.

Usage:  python ml/check_session.py data/recordings/2026-11-10/data_log.csv [more.csv ...]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "ml"))
from labels import LEVELS, level_for_mc  # noqa: E402

STANDOFF_MM = 50.0
TARGET_PER_CLASS, TARGET_STUDS, TARGET_DAYS = 30, 3, 3
BOUNDARY_PT = 1.0
CUTOFFS = (12.0, 20.0)


def load(paths):
    frames = []
    for p in paths:
        d = pd.read_csv(p)
        d["source"] = Path(p).name
        if "measured_distance_mm" not in d and "measured_distance_cm" in d:
            d["measured_distance_mm"] = d["measured_distance_cm"] * 10
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["day"] = d["recorded_at"].astype(str).str[:10]
    return d


def check(d):
    problems, notes = [], []
    w = d[d["material"] == "Wood + drywall"]
    if w.empty:
        return ["No 'Wood + drywall' rows: the protocol is for that material."], notes
    other = len(d) - len(w)
    if other:
        notes.append(f"{other} rows are other materials (open air, wood alone, ...): ignored below.")

    ref = pd.to_numeric(w.get("reference_moisture_pct"), errors="coerce")
    miss = int(ref.isna().sum())
    if miss:
        problems.append(f"{miss} of {len(w)} rows have no pin-meter reading (reference_moisture_pct is empty).")

    have = w[ref.notna()].assign(ref=ref[ref.notna()])
    bad = have[have["ref"].apply(level_for_mc) != have["moisture_level"]]
    if len(bad):
        problems.append(f"{len(bad)} rows have a label that disagrees with the pin meter (cutoffs {CUTOFFS[0]:g}/{CUTOFFS[1]:g}), "
                        f"e.g. {bad['h5_file'].iloc[0]}: label {bad['moisture_level'].iloc[0]}, {bad['ref'].iloc[0]:g}%.")
    near = have[have["ref"].apply(lambda v: min(abs(v - c) for c in CUTOFFS) <= BOUNDARY_PT)]
    if len(near):
        notes.append(f"{len(near)} rows are within {BOUNDARY_PT:g} point of a cutoff: mark them 'boundary', test data only.")

    dist = pd.to_numeric(w["measured_distance_mm"], errors="coerce")
    off = w[dist != STANDOFF_MM]
    if len(off):
        problems.append(f"{len(off)} rows are not at {STANDOFF_MM:g} mm standoff: " +
                        ", ".join(f"{s:g} mm x{n}" for s, n in dist[dist != STANDOFF_MM].value_counts().sort_index().items()) + ".")

    if "h5_file" in w and w["h5_file"].isna().any():
        problems.append(f"{int(w['h5_file'].isna().sum())} rows have no .h5 file (raw data not saved), so features cannot be recomputed.")

    for day, g in w.groupby("day"):
        missing = [c for c in LEVELS if c not in set(g["moisture_level"])]
        if missing:
            problems.append(f"{day}: no {', '.join(missing)} recordings (record all three classes every day).")

    n_stud = w.groupby("moisture_level")["sample_id"].nunique()
    n_day = w.groupby("moisture_level")["day"].nunique()
    n_rec = w.groupby("moisture_level").size()
    for c in LEVELS:
        r, b, dy = int(n_rec.get(c, 0)), int(n_stud.get(c, 0)), int(n_day.get(c, 0))
        short = [t for t, ok in ((f"recordings {r}/{TARGET_PER_CLASS}", r >= TARGET_PER_CLASS),
                                 (f"studs {b}/{TARGET_STUDS}", b >= TARGET_STUDS),
                                 (f"days {dy}/{TARGET_DAYS}", dy >= TARGET_DAYS)) if not ok]
        if short:
            problems.append(f"{c}: below target ({', '.join(short)}).")

    if {"time_since_wet_min", "sample_id"} <= set(w.columns):
        for (sid, day), g in w.groupby(["sample_id", "day"]):
            span = pd.to_numeric(g["time_since_wet_min"], errors="coerce")
            if g["moisture_level"].nunique() == 1 and span.max() - span.min() >= 60 and g["moisture_level"].iloc[0] != "Dry":
                has_ref = pd.to_numeric(g["reference_moisture_pct"], errors="coerce").notna().all()
                if not has_ref:
                    problems.append(f"{sid} on {day}: one label ({g['moisture_level'].iloc[0]}) over {span.max() - span.min():g} min "
                                    f"with no pin readings; a wet stud dries in that time, so the label is probably wrong later on.")
    return problems, notes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+")
    a = ap.parse_args()
    d = load(a.csv)
    problems, notes = check(d)
    w = d[d["material"] == "Wood + drywall"]
    print(f"{len(d)} rows from {len(a.csv)} file(s); {len(w)} Wood + drywall rows on {w['day'].nunique()} day(s)")
    for n in notes:
        print("  note:", n)
    if not problems:
        print("OK: no protocol problems found.")
        return 0
    print(f"{len(problems)} problem(s):")
    for p in problems:
        print("  -", p)
    return 1


if __name__ == "__main__":
    sys.exit(main())
