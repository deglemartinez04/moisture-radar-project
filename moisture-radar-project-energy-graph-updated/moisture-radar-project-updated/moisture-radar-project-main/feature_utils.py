"""Feature engineering helpers for the 60 GHz radar moisture project.

Computes gate-ratio features and a combined confidence score from the
per-position radar measurements described in training_data_template.csv
and live_scan_template.csv.

Usage:
    python feature_utils.py raw_capture.csv augmented_output.csv
"""
import argparse

import numpy as np
import pandas as pd


def amplitude_ratio(g1_amplitude, g2_amplitude):
    """Sheathing-to-drywall amplitude ratio (self-calibrating moisture feature)."""
    return g2_amplitude / g1_amplitude


def phase_diff_rad(g1_phase, g2_phase):
    """Phase difference between the two gates, wrapped to [-pi, pi]."""
    diff = g2_phase - g1_phase
    return (diff + np.pi) % (2 * np.pi) - np.pi


def signal_quality_score(amplitude_mean, amplitude_std):
    """1.0 = perfectly stable reading, 0.0 = as noisy as the signal itself.

    Based on the coefficient of variation of the peak amplitude across frames.
    """
    cv = amplitude_std / np.abs(amplitude_mean)
    return np.clip(1.0 - cv, 0.0, 1.0)


def combined_confidence(model_confidence, quality_score):
    """Confidence shown on the heatmap: model certainty x signal quality."""
    return model_confidence * quality_score


def bucket_moisture(moisture_content_pct, low_max=15.0, high_min=19.0):
    """Map a measured moisture content percentage to Low/Medium/High.

    Defaults follow the common wood-decay-risk convention (~15% / ~19% MC);
    override low_max/high_min once you've validated cutoffs for your materials.
    """
    if moisture_content_pct < low_max:
        return "Low"
    if moisture_content_pct < high_min:
        return "Medium"
    return "High"


def add_engineered_columns(df, low_max=15.0, high_min=19.0):
    """Fill in the derived columns from a dataframe's raw g1_/g2_ features.

    Also derives moisture_level from moisture_content_pct (training data) and
    combined_confidence from model_confidence/signal_quality_score (live scans)
    when those columns are present.
    """
    df = df.copy()

    df["g2_g1_amplitude_ratio"] = amplitude_ratio(df["g1_amplitude_mean"], df["g2_amplitude_mean"])
    df["g2_g1_phase_diff_rad"] = phase_diff_rad(df["g1_phase_mean_rad"], df["g2_phase_mean_rad"])
    df["signal_quality_score"] = signal_quality_score(df["g2_amplitude_mean"], df["g2_amplitude_std"])

    if "moisture_content_pct" in df.columns:
        df["moisture_level"] = df["moisture_content_pct"].apply(
            lambda pct: bucket_moisture(pct, low_max, high_min)
        )

    if "model_confidence" in df.columns:
        df["combined_confidence"] = combined_confidence(df["model_confidence"], df["signal_quality_score"])

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add engineered features to a raw radar feature CSV.")
    parser.add_argument("input_csv", help="CSV with raw g1_/g2_ columns (see the template CSVs)")
    parser.add_argument("output_csv", help="Where to write the augmented CSV")
    parser.add_argument("--low-max", type=float, default=15.0, help="Upper bound (%%) for the Low bucket")
    parser.add_argument("--high-min", type=float, default=19.0, help="Lower bound (%%) for the High bucket")
    args = parser.parse_args()

    data = pd.read_csv(args.input_csv)
    data = add_engineered_columns(data, args.low_max, args.high_min)
    data.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(data)} rows with engineered features to {args.output_csv}")
