"""Extract one training-schema row from a raw Acconeer A121 .h5 capture.

Splits the mean range profile into two gates (G1 = nearer reflection, e.g.
drywall interface; G2 = farther reflection, e.g. sheathing) around the
lowest-amplitude point between them, then reuses feature_utils to compute
the derived ratio/phase-diff/quality columns.

Verified against moisture_features_1.csv's Wood_High_01 row: peak_distance_m,
peak_amplitude_mean/std, phase mean/std, attenuation_slope, total_energy, and
frame_to_frame_variability all reproduce to the original's precision. noise_floor
could not be reverse-engineered from the frame data alone -- the value below is
an approximation (mean amplitude of the weakest quarter of range bins), not a
confirmed match to how the original CSV computed it.

Usage:
    python h5_to_features.py capture.h5 output.csv --material Wood \
        --wall-stack-id bench_test --trial 1 --scan-position 0.0 \
        --moisture-content-pct 24.5
"""
import argparse
import json
import os

import h5py
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from feature_utils import add_engineered_columns


def load_range_profile(h5_path):
    with h5py.File(h5_path, "r") as f:
        frame = f["sessions/session_0/group_0/entry_0/result/frame"][()]
        session_config = json.loads(f["sessions/session_0/session_config"][()])
        metadata = json.loads(f["sessions/session_0/group_0/entry_0/metadata"][()])

    subsweep = session_config["groups"][0]["1"]["subsweeps"][0]
    start_point = subsweep["start_point"]
    step_length = subsweep["step_length"]
    base_step_length_m = metadata["base_step_length_m"]
    num_points = subsweep["num_points"]

    distances = (start_point + np.arange(num_points) * step_length) * base_step_length_m
    iq = frame["real"].astype(np.float64) + 1j * frame["imag"].astype(np.float64)
    iq = iq[:, 0, :]  # sweeps_per_frame == 1
    return distances, iq


def gate_stats(distances, iq, idx):
    amplitude = np.abs(iq[:, idx])
    phase = np.angle(iq[:, idx])
    return {
        "peak_distance_m": distances[idx],
        "amplitude_mean": amplitude.mean(),
        "amplitude_std": amplitude.std(),
        "phase_mean_rad": phase.mean(),
        "phase_std_rad": phase.std(),
    }


def extract_row(h5_path):
    distances, iq = load_range_profile(h5_path)
    amplitude = np.abs(iq)
    mean_amp = amplitude.mean(axis=0)
    std_amp = amplitude.std(axis=0)

    # Two reflectors (e.g. drywall face + sheathing behind it) show up as two
    # broad lobes in the range profile. Each lobe is itself noisy enough to
    # contain several small local maxima, so: smooth first to merge those
    # ripples, require a minimum peak separation so both picks can't land in
    # the same lobe, then refine each pick back to the true local max in the
    # unsmoothed profile.
    smoothed = np.convolve(mean_amp, np.ones(5) / 5, mode="same")
    prominence = (smoothed.max() - smoothed.min()) * 0.15
    peak_indices, _ = find_peaks(smoothed, prominence=prominence, distance=len(mean_amp) // 6)
    if len(peak_indices) < 2:
        raise ValueError(
            f"Found {len(peak_indices)} prominent peak(s) in the range profile; "
            "need two (G1 and G2). Inspect the profile and adjust the prominence/distance thresholds."
        )
    top_two = sorted(peak_indices[np.argsort(smoothed[peak_indices])[-2:]])

    def refine(idx):
        lo, hi = max(0, idx - 2), min(len(mean_amp), idx + 3)
        return lo + int(np.argmax(mean_amp[lo:hi]))

    g1_idx, g2_idx = refine(top_two[0]), refine(top_two[1])

    g1 = gate_stats(distances, iq, g1_idx)
    g2 = gate_stats(distances, iq, g2_idx)

    mask = mean_amp > 0
    attenuation_slope, _ = np.polyfit(distances[mask], np.log(mean_amp[mask]), 1)

    weakest_quarter = np.sort(mean_amp)[: max(1, len(mean_amp) // 4)]

    return {
        "g1_peak_distance_m": g1["peak_distance_m"],
        "g1_amplitude_mean": g1["amplitude_mean"],
        "g1_amplitude_std": g1["amplitude_std"],
        "g1_phase_mean_rad": g1["phase_mean_rad"],
        "g1_phase_std_rad": g1["phase_std_rad"],
        "g2_peak_distance_m": g2["peak_distance_m"],
        "g2_amplitude_mean": g2["amplitude_mean"],
        "g2_amplitude_std": g2["amplitude_std"],
        "g2_phase_mean_rad": g2["phase_mean_rad"],
        "g2_phase_std_rad": g2["phase_std_rad"],
        "attenuation_slope": attenuation_slope,
        "total_energy": mean_amp.sum(),
        "noise_floor": weakest_quarter.mean(),  # approximation -- see module docstring
        "frame_to_frame_variability": std_amp.mean(),
        "num_frames": iq.shape[0],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract a training-schema row from one .h5 capture.")
    parser.add_argument("h5_path", help="Path to the raw Acconeer .h5 capture")
    parser.add_argument("output_csv", help="CSV to append the row to (created with a header if missing)")
    parser.add_argument("--scan-id", default="scan_0")
    parser.add_argument("--wall-stack-id", default="bench_test")
    parser.add_argument("--material", required=True)
    parser.add_argument("--trial-number", type=int, default=1)
    parser.add_argument("--scan-position-m", type=float, default=0.0)
    parser.add_argument("--moisture-content-pct", type=float, default=None)
    args = parser.parse_args()

    row = {
        "scan_id": args.scan_id,
        "wall_stack_id": args.wall_stack_id,
        "material": args.material,
        "trial_number": args.trial_number,
        "scan_position_m": args.scan_position_m,
        **extract_row(args.h5_path),
    }
    if args.moisture_content_pct is not None:
        row["moisture_content_pct"] = args.moisture_content_pct

    df = pd.DataFrame([row])
    df = add_engineered_columns(df)

    if os.path.exists(args.output_csv):
        existing = pd.read_csv(args.output_csv)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Appended 1 row from {args.h5_path} to {args.output_csv} ({len(df)} total rows)")
