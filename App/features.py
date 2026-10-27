"""
features.py

Single source of truth for turning a window of radar frames into features.

BOTH the Data Logging tab (rows written to data_log.csv for training) and the
live scan (inputs to the ML model) call extract_features(), so training data
and live data are computed by the exact same code and cannot drift apart.

Pipeline (the caller supplies background-corrected amplitude/phase windows):

  1. Average the window to one amplitude profile (amplitude vs distance).
  2. Find the two boundary peaks: Boundary 1 = drywall front surface,
     Boundary 2 = wood stud face.
  3. Build features that do NOT depend on how far the sensor is from the wall:
       amplitude_ratio       peak2 / peak1  (both shrink together with distance)
       peak_separation_m     apparent drywall thickness (moisture slows the wave)
       phase_delta_rad       phase(peak2) - phase(peak1)  (standoff cancels out)
  4. Also record tail_attenuation_slope (decay of log-amplitude behind the stud) plus
     absolute/legacy values (peak amplitudes, SNR, energy, ...) for diagnostics and
     quality gating. These change with distance, so they stay OUT of the default model
     inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    from scipy.signal import find_peaks
except ImportError:  # scipy is installed with acconeer-exptool; fail clearly if not
    find_peaks = None

SCHEMA_VERSION = 5   # 5: distance filter, TX-off noise floor, Acconeer peak detection, window features


@dataclass(frozen=True)
class PeakParams:
    """Tunable peak-detection settings. Validate these against REAL drywall+stud data."""

    # Closest two boundary peaks may be. A 12.7 mm drywall sheet looks ~20 mm thick to
    # the radar (the wave is slower inside it), i.e. ~4 points at 5 mm spacing, so this
    # must stay well below that. (An 8-point / 40 mm minimum would merge the two peaks.)
    min_separation_m: float = 0.012
    # A peak must stand out from its surroundings by at least this fraction of
    # (strongest value - noise floor). Low on purpose so a weak, wet stud peak survives.
    rel_prominence: float = 0.05
    # ...AND by at least this many multiples of the noise floor (rejects noise bumps).
    # Relative to the noise floor instead of a fixed number (e.g. 50) so it keeps working
    # when amplitudes change after background subtraction or a gain change.
    noise_prominence_mult: float = 3.0
    # Tail fit: stop when the profile falls to this multiple of the noise floor.
    tail_noise_mult: float = 2.0
    tail_min_points: int = 4
    # A peak (and both its neighbours) must exceed noise_floor * 10**(min_peak_snr_db/20)
    # (Acconeer find_peaks definition: middle point larger than its two neighbours, all three
    # above the threshold).
    min_peak_snr_db: float = 10.0
    # Ignore everything closer than this (displayed distance). In the v8 recordings the
    # leakage residual and the trial-to-trial spread were largest below ~15 mm.
    min_distance_m: float = 0.015
    # Window features, anchored on the FRONT surface (first peak), so they do not move with
    # standoff. Distances are relative to the front peak.
    front_half_width_m: float = 0.010   # front window: front +/- 10 mm
    rear_start_m: float = 0.015         # rear window starts 15 mm behind the front surface...
    rear_end_m: float = 0.060           # ...and ends 60 mm behind it


DEFAULT_PEAK_PARAMS = PeakParams()

# Quality gate used by the live predictor / training filter. PLACEHOLDER value: tune
# once real data shows what SNR a trustworthy reading has.
MIN_PEAK1_SNR_DB = 10.0

# Model input sets. "Invariant" features were measured to be unaffected by how far the
# sensor sits from the wall (see the distance check in the project notes). "Extended" adds
# features that DO change with distance, for side-by-side comparison (train_model.py --compare).
#
# tail_attenuation_slope is deliberately NOT in the invariant set: on simulated data its drift
# with distance was ~5x larger than its difference between moisture classes, because its fit
# window ends where the signal reaches the noise, so a weaker signal gets a different window.
# Promote it with `train_model.py --add tail_attenuation_slope` once real data shows it is stable.
# Window features (front surface vs. the region behind it). They are defined whenever the
# front surface is found, unlike the two-peak features, which need a separate second peak.
INVARIANT_FEATURES = [
    "rear_energy_ratio_db",
    "rear_centroid_offset_m",
    "rear_peak_offset_m",
    "rear_phase_sin",
    "rear_phase_cos",
]
EXTENDED_FEATURES = INVARIANT_FEATURES + [
    "amplitude_ratio",
    "peak_separation_m",
    "phase_delta_sin",
    "phase_delta_cos",
    "front_amplitude",
    "tail_attenuation_slope",
    "peak1_amplitude",
    "peak2_amplitude",
    "peak1_snr_db",
    "peak1_distance_m",
    "total_energy",
]

NAN = float("nan")


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _refine_distance(profile: np.ndarray, distances: np.ndarray, i: int, spacing: float) -> float:
    """Sub-bin peak position via parabolic interpolation on the 3 points around the peak."""
    if 0 < i < len(profile) - 1:
        a, b, c = profile[i - 1], profile[i], profile[i + 1]
        denom = a - 2.0 * b + c
        if denom < 0:
            offset = float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5))
            return float(distances[i] + offset * spacing)
    return float(distances[i])


def _wrap(angle: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return float(np.angle(np.exp(1j * angle)))


def _find_boundary_peaks(mean_amp, noise_floor, spacing, params, first=1):
    """Acconeer-style peaks: a point larger than both neighbours, with all three points above
    the threshold (noise_floor * 10**(min_peak_snr_db/20), and >= rel_prominence * max).
    Peaks closer than min_separation_m are merged (larger one kept).
    Returns (indices of the first two peaks ordered by distance, number of candidates)."""
    amp = np.asarray(mean_amp, dtype=float)
    threshold = max(noise_floor * 10 ** (params.min_peak_snr_db / 20.0),
                    params.rel_prominence * float(np.max(amp)), 1e-12)
    above = amp > threshold
    cand = [i for i in range(max(1, first), len(amp) - 1)
            if above[i - 1] and above[i] and above[i + 1]
            and amp[i] > amp[i - 1] and amp[i] >= amp[i + 1]]
    min_pts = max(1, int(round(params.min_separation_m / spacing))) if spacing > 0 else 1
    merged = []
    for i in cand:
        if merged and i - merged[-1] < min_pts:
            if amp[i] > amp[merged[-1]]:
                merged[-1] = i
        else:
            merged.append(i)
    return np.array(merged[:2], dtype=int), int(len(merged))


WINDOW_KEYS = ["front_distance_m", "front_amplitude", "rear_energy_ratio_db",
               "rear_energy_ratio_db_std", "rear_centroid_offset_m", "rear_peak_offset_m",
               "rear_phase_rad", "rear_phase_std_rad"]


def _window_features(z, d, front_idx, front_m, params) -> dict:
    """Front-surface vs. behind-surface features from the complex per-frame sweeps z
    (frames, points). Computed per frame, then averaged; *_std = spread across frames."""
    out = {k: NAN for k in WINDOW_KEYS if k != "front_distance_m"}
    rel = d - front_m
    front = np.abs(rel) <= params.front_half_width_m
    rear = (rel >= params.rear_start_m) & (rel <= params.rear_end_m)
    if not front.any() or rear.sum() < 2 or d[-1] < front_m + params.rear_end_m - 1e-9:
        return out   # rear window not fully inside the scan range
    p = np.abs(z) ** 2
    e_front = p[:, front].sum(axis=1)
    e_rear = p[:, rear].sum(axis=1)
    ratio = 10.0 * np.log10(np.maximum(e_rear, 1e-12) / np.maximum(e_front, 1e-12))
    mean_p = p.mean(axis=0)
    rear_idx = np.flatnonzero(rear)
    k_rear = int(rear_idx[np.argmax(mean_p[rear_idx])])
    rel_phase = np.angle(z[:, k_rear] * np.conj(z[:, front_idx]))
    resultant = np.mean(np.exp(1j * rel_phase))
    out.update({
        "front_amplitude": float(np.abs(z[:, front_idx]).mean()),
        "rear_energy_ratio_db": float(ratio.mean()),
        "rear_energy_ratio_db_std": float(ratio.std()),
        "rear_centroid_offset_m": float((mean_p[rear] * rel[rear]).sum() / max(mean_p[rear].sum(), 1e-12)),
        "rear_peak_offset_m": float(rel[k_rear]),
        "rear_phase_rad": float(np.angle(resultant)),
        "rear_phase_std_rad": float(np.sqrt(max(0.0, -2.0 * math.log(max(abs(resultant), 1e-12))))),
    })
    return out


# ----------------------------------------------------------------------------
# main entry point
# ----------------------------------------------------------------------------
def extract_features(
    amplitude_window,
    phase_window,
    distances_m,
    background_corrected: bool = False,
    params: PeakParams = DEFAULT_PEAK_PARAMS,
    noise_level: float | None = None,
) -> dict:
    """
    amplitude_window, phase_window: shape (n_frames, n_points). Pass background-corrected
    values when a background is applied, and set background_corrected=True accordingly.
    Features that cannot be computed are NaN (never 0.0, which would look like a real value).
    """
    amp = np.asarray(amplitude_window, dtype=float)
    ph = np.asarray(phase_window, dtype=float)
    d = np.asarray(distances_m, dtype=float)
    if amp.ndim != 2 or amp.shape[0] == 0 or amp.shape != ph.shape or amp.shape[1] != len(d):
        raise ValueError("amplitude/phase windows must be (frames, points) and match distances.")

    n_frames, n_pts = amp.shape
    spacing = float(np.median(np.diff(d))) if n_pts > 1 else NAN
    mean_amp = amp.mean(axis=0)                              # non-coherent average
    mean_energy = (amp ** 2).mean(axis=0)                    # mean |IQ|^2 (same as Energy tab)
    mean_phase = np.angle(np.mean(np.exp(1j * ph), axis=0))  # circular mean
    # Noise floor: measured with the transmitter off (Acconeer noise calibration) when
    # available. At 0-150 mm every bin contains leakage or wall, so the old 10th-percentile
    # estimate is only a fallback.
    if noise_level is not None and np.isfinite(noise_level) and noise_level > 0:
        noise_floor, noise_source = float(noise_level), "tx_off"
    else:
        noise_floor, noise_source = float(np.percentile(mean_amp, 10)), "percentile"

    # ---- legacy statistics at the single strongest bin (kept for continuity) ----
    s_idx = int(np.argmax(mean_amp))
    s_phase = ph[:, s_idx]
    s_phase_mean = float(np.angle(np.mean(np.exp(1j * s_phase))))
    safe = np.clip(mean_amp, 1e-6, None)
    global_slope = float(np.polyfit(d, np.log(safe), 1)[0]) if n_pts >= 2 else NAN

    out = {
        "peak_distance_m": float(d[s_idx]),
        "peak_amplitude_mean": float(mean_amp[s_idx]),
        "peak_amplitude_std": float(np.std(amp[:, s_idx])),
        "peak_phase_mean_rad": s_phase_mean,
        "peak_phase_std_rad": float(np.std(np.angle(np.exp(1j * (s_phase - s_phase_mean))))),
        "global_attenuation_slope": global_slope,
        "noise_floor": noise_floor,
        "noise_source": noise_source,
        "frame_to_frame_variance": float(np.mean(np.var(amp, axis=0))),
        "num_frames": int(n_frames),
        "total_energy": float(np.sum(mean_energy)),
        # QA flag: strongest return in the first/last bin means the window is cutting off
        # the target (or leakage dominates) - find_peaks cannot report edge samples.
        "strongest_at_edge": int(s_idx in (0, n_pts - 1)),
        "background_corrected": int(bool(background_corrected)),
        # for live graphs
        "mean_amplitude_per_point": mean_amp,
        "mean_phase_per_point": mean_phase,
    }

    # ---- boundary-peak features ----
    first = int(np.searchsorted(d, params.min_distance_m))   # skip the leakage-residual zone
    peaks, n_candidates = _find_boundary_peaks(mean_amp, noise_floor, spacing, params, first)
    out["peaks_found"] = n_candidates
    out["two_peaks_found"] = int(len(peaks) == 2)

    keys = [
        "peak1_distance_m", "peak2_distance_m", "peak_separation_m",
        "peak1_amplitude", "peak2_amplitude", "amplitude_ratio", "amplitude_ratio_std",
        "peak1_phase_rad", "peak2_phase_rad", "phase_delta_rad", "peak2_phase_std_rad",
        "tail_attenuation_slope", "peak1_snr_db",
    ]
    out.update({k: NAN for k in keys})

    # Complex per-frame sweeps (the caller passes |z| and angle(z) of the corrected,
    # distance-filtered sweeps), used by the window features.
    z = amp * np.exp(1j * ph)
    out.update({k: NAN for k in WINDOW_KEYS})

    if len(peaks) >= 1:
        i1 = int(peaks[0])
        out["peak1_distance_m"] = _refine_distance(mean_amp, d, i1, spacing)
        out["front_distance_m"] = out["peak1_distance_m"]
        out.update(_window_features(z, d, i1, out["front_distance_m"], params))
        out["peak1_amplitude"] = float(mean_amp[i1])
        out["peak1_phase_rad"] = float(mean_phase[i1])
        if noise_floor > 0 and mean_amp[i1] > 0:
            out["peak1_snr_db"] = float(20.0 * math.log10(mean_amp[i1] / noise_floor))

    if len(peaks) == 2:
        i1, i2 = int(peaks[0]), int(peaks[1])
        out["peak2_distance_m"] = _refine_distance(mean_amp, d, i2, spacing)
        out["peak_separation_m"] = out["peak2_distance_m"] - out["peak1_distance_m"]
        out["peak2_amplitude"] = float(mean_amp[i2])
        out["peak2_phase_rad"] = float(mean_phase[i2])
        out["phase_delta_rad"] = _wrap(out["peak2_phase_rad"] - out["peak1_phase_rad"])

        if mean_amp[i1] > 0:
            out["amplitude_ratio"] = float(mean_amp[i2] / mean_amp[i1])
            per_frame = amp[:, i2] / np.maximum(amp[:, i1], 1e-9)
            out["amplitude_ratio_std"] = float(np.std(per_frame))

        resultant = float(abs(np.mean(np.exp(1j * ph[:, i2]))))
        out["peak2_phase_std_rad"] = float(np.sqrt(max(0.0, -2.0 * math.log(max(resultant, 1e-12)))))

        # Decay of log-amplitude behind the stud face, until it falls into the noise.
        # log-slope is unchanged by a uniform gain/distance scale factor.
        cutoff = params.tail_noise_mult * noise_floor
        tail = []
        for j in range(i2 + 1, n_pts):
            if mean_amp[j] <= cutoff:
                break
            tail.append(j)
        if len(tail) >= params.tail_min_points:
            out["tail_attenuation_slope"] = float(
                np.polyfit(d[tail], np.log(np.clip(mean_amp[tail], 1e-6, None)), 1)[0]
            )

    return out


# ----------------------------------------------------------------------------
# model-input helpers (used by BOTH train_model.py and the live predictor)
# ----------------------------------------------------------------------------
def add_derived(features: dict) -> dict:
    """Add circular-safe encodings. Raw phase jumps from +pi to -pi, which tree models
    would treat as far apart, so phase is fed to the model as sin/cos."""
    f = dict(features)
    delta = f.get("phase_delta_rad", NAN)
    try:
        delta = float(delta)
    except (TypeError, ValueError):
        delta = NAN
    f["phase_delta_sin"] = math.sin(delta) if not math.isnan(delta) else NAN
    f["phase_delta_cos"] = math.cos(delta) if not math.isnan(delta) else NAN
    try:
        rear = float(f.get("rear_phase_rad", NAN))
    except (TypeError, ValueError):
        rear = NAN
    f["rear_phase_sin"] = math.sin(rear) if not math.isnan(rear) else NAN
    f["rear_phase_cos"] = math.cos(rear) if not math.isnan(rear) else NAN
    return f


def model_vector(features: dict, names: list[str]) -> list[float]:
    f = add_derived(features)
    return [float(f.get(n, NAN)) for n in names]


def quality_check(features: dict, min_snr_db: float = MIN_PEAK1_SNR_DB, require_background: bool = True):
    """Return (ok, reason). A reading failing this must not be turned into a moisture value."""
    if require_background and not int(features.get("background_corrected", 0)):
        return False, "Background not applied. Calibrate it, then tick 'Use background correction'."
    if int(features.get("peaks_found", 0)) < 1:
        if features.get("strongest_at_edge"):
            return False, "Strongest return is at the edge of the scan window - wall not visible."
        return False, "No wall surface found above the noise threshold."
    try:
        rear_ok = not math.isnan(float(features.get("rear_energy_ratio_db", NAN)))
    except (TypeError, ValueError):
        rear_ok = False
    if not rear_ok:
        return False, "The region behind the wall surface is outside the scan range. Increase the number of points."
    snr = features.get("peak1_snr_db", NAN)
    try:
        snr = float(snr)
    except (TypeError, ValueError):
        snr = NAN
    if math.isnan(snr):
        return False, "Signal too weak to measure."
    if snr < min_snr_db:
        return False, f"Signal too weak (SNR {snr:.1f} dB < {min_snr_db:.0f} dB). Move closer."
    return True, ""


RAW_CORRECTED_KEYS = [
    "peak_amplitude", "peak_distance_m", "mean_amplitude", "peak_energy", "integrated_energy",
]


def raw_vs_corrected_summary(raw_iq, corrected_iq, distances_m) -> dict:
    """
    Amplitude and energy summaries of the raw signal and of the background-corrected signal,
    for the CSV. Inputs are complex IQ of shape (n_frames, n_points); corrected_iq may be
    None (then the corrected_* values are NaN). Definitions match the live tabs:

      amplitude  = |IQ|            averaged over frames, per distance point
      energy     = |IQ|^2          averaged over frames, per distance point
      peak_*     = maximum over distance points (peak_distance_m = where it occurs)
      mean_amplitude    = average of the amplitude profile over distance points
      integrated_energy = sum of the energy profile over distance points

    corrected energy is the energy of (raw - background); it is NOT raw energy minus
    background energy, because |a - b|^2 != |a|^2 - |b|^2.
    """
    d = np.asarray(distances_m, dtype=float)
    out = {}
    for prefix, iq in (("raw", raw_iq), ("corrected", corrected_iq)):
        if iq is None:
            for key in RAW_CORRECTED_KEYS:
                out[f"{prefix}_{key}"] = NAN
            continue
        z = np.asarray(iq, dtype=complex)
        if z.ndim != 2 or z.shape[1] != len(d):
            raise ValueError("IQ array must be (frames, points) and match distances.")
        mean_amp = np.abs(z).mean(axis=0)
        mean_energy = (np.abs(z) ** 2).mean(axis=0)
        out[f"{prefix}_peak_amplitude"] = float(mean_amp.max())
        out[f"{prefix}_peak_distance_m"] = float(d[int(np.argmax(mean_amp))])
        out[f"{prefix}_mean_amplitude"] = float(mean_amp.mean())
        out[f"{prefix}_peak_energy"] = float(mean_energy.max())
        out[f"{prefix}_integrated_energy"] = float(mean_energy.sum())
    return out
