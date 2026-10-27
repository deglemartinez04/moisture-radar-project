"""Shared radar feature extraction for training AND live prediction.

Every path that produces model input -- build_dataset.py (training rows from
.h5 recordings) and the App's live tab -- calls window_features() on a window
of complex IQ sweeps. Keeping one function is what guarantees the model sees
the same numbers live that it was trained on.

A window is (n_frames, n_points) complex IQ, one sweep per frame (frames with
several sweeps are averaged to one sweep before they get here).

Gating:
- The first few range bins hold the sensor's own TX->RX leakage, which is
  always the strongest thing in the profile at flush contact. Bins closer than
  MIN_GATE_M are excluded so the leakage is never picked as G1.
- G1/G2 are the two most prominent lobes beyond that (same smoothed,
  minimum-separation peak search as h5_to_features.py). If fewer than two
  lobes stand out from the noise, the missing gate's features are NaN and
  target_detected / g2_detected say so. scikit-learn's RandomForest accepts NaN.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

MIN_GATE_M = 0.075          # ignore leakage bins closer than this
SMOOTH_BINS = 5             # moving-average width before peak search
MIN_PROMINENCE_FRAC = 0.15  # of the profile's (max - min) beyond MIN_GATE_M
MIN_TARGET_SNR = 5.0        # gate amplitude / temporal IQ noise needed to count as a reflection

# Columns the model trains on, in order. Absolute distances and absolute
# phases are left out on purpose: they depend on exactly where the sensor sat,
# not on moisture.
MODEL_FEATURES = [
    "g1_amplitude_mean",
    "g1_amplitude_std",
    "g1_phase_std_rad",
    "g2_amplitude_mean",
    "g2_amplitude_std",
    "g2_phase_std_rad",
    "g2_g1_amplitude_ratio",
    "g2_g1_phase_diff_rad",
    "gate_separation_m",
    "attenuation_slope",
    "total_energy",
    "noise_floor",
    "frame_to_frame_variability",
    "signal_quality_score",
]


def circular_mean(phases, axis=0):
    return np.angle(np.mean(np.exp(1j * np.asarray(phases)), axis=axis))


def circular_std(phases, axis=0):
    resultant = np.abs(np.mean(np.exp(1j * np.asarray(phases)), axis=axis))
    return np.sqrt(-2.0 * np.log(np.clip(resultant, 1e-12, 1.0)))


def wrap_phase(rad):
    return (rad + np.pi) % (2 * np.pi) - np.pi


def find_gates(mean_amplitude, distances_m, min_gate_m=MIN_GATE_M):
    """Return (g1_idx, g2_idx); either may be None when no lobe stands out."""
    mean_amplitude = np.asarray(mean_amplitude, dtype=float)
    distances_m = np.asarray(distances_m, dtype=float)
    first = int(np.searchsorted(distances_m, min_gate_m))
    region = mean_amplitude[first:]
    if len(region) < 3:
        return None, None

    # Edge-padded moving average: zero padding would make the region's first
    # bin (the tail of the leakage) look like a peak.
    half = SMOOTH_BINS // 2
    padded = np.pad(region, half, mode="edge")
    smoothed = np.convolve(padded, np.ones(SMOOTH_BINS) / SMOOTH_BINS, mode="valid")
    span = smoothed.max() - smoothed.min()
    if span <= 0:
        return None, None
    # Treat the end of the range as low ground: a lobe cut off by the last bin
    # (e.g. G2 pushed farther out by wet wood) still counts as prominent.
    smoothed = np.append(smoothed, smoothed.min())
    peaks, _ = find_peaks(
        smoothed,
        prominence=span * MIN_PROMINENCE_FRAC,
        distance=max(1, len(mean_amplitude) // 6),
    )
    if len(peaks) == 0:
        return None, None

    top = sorted(peaks[np.argsort(smoothed[peaks])[-2:]])

    def refine(i):
        lo, hi = max(0, i - 2), min(len(region), i + 3)
        j = lo + int(np.argmax(region[lo:hi]))
        # A real reflection is a local maximum, not the edge of the gated region.
        return None if j in (0, len(region) - 1) else first + j

    gates = [g for g in (refine(i) for i in top) if g is not None]
    if not gates:
        return None, None
    return (gates[0], gates[1]) if len(gates) == 2 else (gates[0], None)


def temporal_noise(iq_window, distances_m, min_gate_m=MIN_GATE_M):
    """Noise level from frame-to-frame IQ jitter (median over bins past the leakage).

    For a static scene each bin's complex IQ is constant plus noise, so its
    spread across frames measures noise no matter how strong the reflection is.
    """
    beyond = iq_window[:, np.asarray(distances_m) >= min_gate_m]
    if beyond.shape[1] == 0:
        return np.nan
    jitter = np.abs(beyond - beyond.mean(axis=0)).mean(axis=0)
    return float(np.median(jitter))


def _gate_stats(prefix, idx, amplitude, phase, distances_m):
    if idx is None:
        return {f"{prefix}_{k}": np.nan for k in (
            "peak_distance_m", "amplitude_mean", "amplitude_std", "phase_mean_rad", "phase_std_rad")}
    return {
        f"{prefix}_peak_distance_m": float(distances_m[idx]),
        f"{prefix}_amplitude_mean": float(amplitude[:, idx].mean()),
        f"{prefix}_amplitude_std": float(amplitude[:, idx].std()),
        f"{prefix}_phase_mean_rad": float(circular_mean(phase[:, idx])),
        f"{prefix}_phase_std_rad": float(circular_std(phase[:, idx])),
    }


def window_features(iq_window, distances_m, min_gate_m=MIN_GATE_M):
    """Feature dict for one window of complex IQ, shape (n_frames, n_points)."""
    iq_window = np.asarray(iq_window)
    if iq_window.ndim != 2 or iq_window.shape[0] < 2:
        raise ValueError(f"Expected (n_frames>=2, n_points) IQ window, got {iq_window.shape}")
    distances_m = np.asarray(distances_m, dtype=float)

    amplitude = np.abs(iq_window)
    phase = np.angle(iq_window)
    mean_amp = amplitude.mean(axis=0)

    beyond = mean_amp[distances_m >= min_gate_m]
    noise_floor = float(np.mean(np.sort(beyond)[: max(1, len(beyond) // 4)])) if len(beyond) else np.nan

    g1_idx, g2_idx = find_gates(mean_amp, distances_m, min_gate_m)
    # A "lobe" that is barely above the noise is not a reflection.
    floor = max(temporal_noise(iq_window, distances_m, min_gate_m), 1e-9)
    if g1_idx is not None and mean_amp[g1_idx] / floor < MIN_TARGET_SNR:
        g1_idx, g2_idx = None, None
    if g2_idx is not None and mean_amp[g2_idx] / floor < MIN_TARGET_SNR:
        g2_idx = None

    row = {}
    row.update(_gate_stats("g1", g1_idx, amplitude, phase, distances_m))
    row.update(_gate_stats("g2", g2_idx, amplitude, phase, distances_m))

    row["target_detected"] = int(g1_idx is not None)
    row["g2_detected"] = int(g2_idx is not None)
    row["target_snr"] = float(mean_amp[g1_idx] / floor) if g1_idx is not None else np.nan

    if g2_idx is not None:
        row["g2_g1_amplitude_ratio"] = row["g2_amplitude_mean"] / row["g1_amplitude_mean"]
        row["g2_g1_phase_diff_rad"] = float(wrap_phase(row["g2_phase_mean_rad"] - row["g1_phase_mean_rad"]))
        row["gate_separation_m"] = row["g2_peak_distance_m"] - row["g1_peak_distance_m"]
        cv = row["g2_amplitude_std"] / max(abs(row["g2_amplitude_mean"]), 1e-9)
        row["signal_quality_score"] = float(np.clip(1.0 - cv, 0.0, 1.0))
    else:
        row["g2_g1_amplitude_ratio"] = np.nan
        row["g2_g1_phase_diff_rad"] = np.nan
        row["gate_separation_m"] = np.nan
        if g1_idx is not None:
            cv = row["g1_amplitude_std"] / max(abs(row["g1_amplitude_mean"]), 1e-9)
            row["signal_quality_score"] = float(np.clip(1.0 - cv, 0.0, 1.0))
        else:
            row["signal_quality_score"] = 0.0

    # Whole-profile features, kept from the team's original pipeline
    # (full-profile log-amplitude fit, sum of mean amplitude, mean per-bin std).
    safe = np.clip(mean_amp, 1e-6, None)
    row["attenuation_slope"] = float(np.polyfit(distances_m, np.log(safe), 1)[0])
    row["total_energy"] = float(mean_amp.sum())
    row["noise_floor"] = noise_floor
    row["frame_to_frame_variability"] = float(amplitude.std(axis=0).mean())
    row["num_frames"] = int(iq_window.shape[0])
    return row


def feature_vector(row, feature_names=MODEL_FEATURES):
    """Order a feature dict into the 2-D array scikit-learn expects."""
    return np.array([[float(row.get(name, np.nan)) for name in feature_names]], dtype=float)
