"""
close_range.py

Close-range corrections taken from Acconeer's own Distance Detector (Exploration Tool
a121/algo/_utils.py and a121/algo/distance/*, SDK a121-v1.13.0). Pure numpy/scipy, no Qt.

1. Zero-point distance offset
   Raw Sparse IQ distances (start_point * 2.5 mm) contain no zero-point offset
   (acc_processing_points_to_meter: "Does not include any zero-point offset").
   Acconeer measures the on-chip loopback pulse, finds its peak, and converts that to an
   offset that is subtracted from every distance. With Profile 1 it is about 18 mm.

2. Phase-stable direct-leakage cancellation
   Every frame carries a 1-point loopback subsweep. Its phase follows the sensor's phase
   jitter, which the wall echo does not cause. The stored leakage is kept relative to that
   loopback phase and rotated to the current loopback phase before it is subtracted
   (Acconeer CloseRangeProcessor.apply_phase_jitter_compensation).
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.signal import butter, filtfilt
except ImportError:  # scipy ships with acconeer-exptool
    butter = filtfilt = None

APPROX_BASE_STEP_LENGTH_M = 2.5e-3

# Envelope full width at half maximum per profile (Acconeer ENVELOPE_FWHM_M).
ENVELOPE_FWHM_M = {1: 0.04, 2: 0.07, 3: 0.14, 4: 0.19, 5: 0.32}

# Loopback measurement used for the offset calibration (Acconeer
# get_calibrate_offset_sensor_config()).
OFFSET_CAL_START_POINT = -30
OFFSET_CAL_NUM_POINTS = 50
OFFSET_CAL_STEP_LENGTH = 1
OFFSET_CAL_PROFILE = 1
OFFSET_CAL_HWAAS = 64

# Loopback reference subsweep added in front of the range subsweep (Acconeer close-range
# session in distance/_translation.py). Profile 4 / 15.6 MHz / gain 15.
LOOPBACK_SUBSWEEP = {
    "start_point": 0,
    "num_points": 1,
    "step_length": 1,
    "profile": 4,
    "receiver_gain": 15,
    "prf": "PRF_15_6_MHz",
}

# Acconeer get_distance_offset(): offset = p0 * loopback_peak_m + p1, per profile and
# temperature band. Only the profile that the range subsweep uses matters.
_OFFSET_COEFFS = {
    "cold": {1: (0.81796372, 0.01561084), 2: (0.60363902, 0.01639924), 3: (0.76755397, 0.0119841),
             4: (0.91310206, 0.01822777), 5: (1.05453651, 0.0045255)},
    "normal": {1: (0.91880272, 0.01782158), 2: (0.59457893, 0.0192985), 3: (0.85039166, 0.01478081),
               4: (1.00348084, 0.02306631), 5: (0.98183729, 0.00794888)},
    "hot": {1: (0.54980311, 0.00836143), 2: (0.49726231, 0.01961523), 3: (0.53138948, 0.00784231),
            4: (0.44605765, 0.01097863), 5: (0.51716572, 0.00060455)},
}

# Amplitude temperature model (Acconeer SIGNAL_TEMPERATURE_MODEL_PARAMETER):
# amplitude scales by 2 ** (-(T - T_ref) / k).
SIGNAL_TEMPERATURE_MODEL_PARAMETER = {1: 67.0, 2: 85.0, 3: 86.0, 4: 99.0, 5: 104.0}


# ----------------------------------------------------------------------------
# distance filter + peak interpolation (same as Acconeer's)
# ----------------------------------------------------------------------------
def distance_filter_coeffs(profile: int, step_length: int, narrow: bool = False):
    """First-order Butterworth low-pass along distance with a matched-filter cutoff."""
    if butter is None:
        raise RuntimeError("scipy is required (pip install scipy).")
    wn = APPROX_BASE_STEP_LENGTH_M * step_length / ENVELOPE_FWHM_M[int(profile)]
    if narrow:
        wn *= 2.0
    return butter(N=1, Wn=min(wn, 0.99))


def distance_filter(sweep: np.ndarray, profile: int, step_length: int, narrow: bool = False) -> np.ndarray:
    """Zero-phase distance filter on a complex sweep (same length out)."""
    b, a = distance_filter_coeffs(profile, step_length, narrow)
    sweep = np.asarray(sweep)
    if sweep.shape[-1] <= 3 * max(len(a), len(b)):
        return sweep
    return filtfilt(b, a, sweep, axis=-1)


def interpolate_peak_index(y: np.ndarray, i: int) -> float:
    """Sub-bin peak position (in index units) from a parabola through 3 points."""
    if 0 < i < len(y) - 1:
        a, b, c = float(y[i - 1]), float(y[i]), float(y[i + 1])
        denom = a - 2.0 * b + c
        if denom < 0:
            return i + float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5))
    return float(i)


# ----------------------------------------------------------------------------
# 1. zero-point offset
# ----------------------------------------------------------------------------
def loopback_peak_location_m(loopback_sweep: np.ndarray,
                             start_point: int = OFFSET_CAL_START_POINT,
                             step_length: int = OFFSET_CAL_STEP_LENGTH,
                             profile: int = OFFSET_CAL_PROFILE,
                             base_step_length_m: float = APPROX_BASE_STEP_LENGTH_M) -> float:
    """Peak of the loopback envelope (Acconeer calculate_loopback_peak_location)."""
    sweep = np.asarray(loopback_sweep, dtype=complex)
    if sweep.ndim == 2:
        sweep = sweep.mean(axis=0)
    env = np.abs(distance_filter(sweep, profile, step_length, narrow=True))
    i = int(np.argmax(env))
    idx = interpolate_peak_index(env, i)
    return (start_point + idx * step_length) * base_step_length_m


def distance_offset_m(loopback_peak_m: float, profile: int = 1, temperature_c: float | None = None) -> float:
    """Acconeer get_distance_offset(): subtract the result from every raw distance."""
    if temperature_c is None or -25 <= temperature_c < 45:
        band = "normal"
    elif temperature_c < -25:
        band = "cold"
    else:
        band = "hot"
    p0, p1 = _OFFSET_COEFFS[band][int(profile)]
    return p0 * float(loopback_peak_m) + p1


# ----------------------------------------------------------------------------
# 2. phase-stable leakage cancellation
# ----------------------------------------------------------------------------
def loopback_angles(loopback_subframe: np.ndarray) -> np.ndarray:
    """Per-sweep loopback phase, shape (sweeps, 1), from a (sweeps, 1) loopback subframe."""
    lb = np.asarray(loopback_subframe, dtype=complex)
    if lb.ndim == 1:
        lb = lb[:, np.newaxis]
    return np.angle(lb[:, :1])


def phase_referenced(range_subframe: np.ndarray, lb_angle: np.ndarray) -> np.ndarray:
    """Remove the loopback phase from each sweep: range * exp(-j*lb). Shape (sweeps, points)."""
    return np.asarray(range_subframe, dtype=complex) * np.exp(-1j * lb_angle)


def leakage_reference(range_subframes, loopback_subframes) -> np.ndarray:
    """Average of several frames of phase-referenced leakage (air, sensor still).

    range_subframes: iterable of (sweeps, points); loopback_subframes: iterable of (sweeps, 1).
    Returns the stored leakage L0, shape (points,).
    """
    acc = [phase_referenced(r, loopback_angles(lb)).mean(axis=0)
           for r, lb in zip(range_subframes, loopback_subframes)]
    return np.mean(np.stack(acc, axis=0), axis=0)


def cancel_leakage(range_subframe: np.ndarray, loopback_subframe, leakage_ref: np.ndarray) -> np.ndarray:
    """Subtract the stored leakage rotated to this frame's loopback phase. (sweeps, points) out.

    If no loopback subframe is available, falls back to plain complex subtraction.
    """
    frame = np.asarray(range_subframe, dtype=complex)
    if loopback_subframe is None:
        return frame - leakage_ref[np.newaxis, :]
    lb = loopback_angles(loopback_subframe)
    return frame - leakage_ref[np.newaxis, :] * np.exp(1j * lb)


def temperature_amplitude_factor(reference_temp_c: float, current_temp_c: float, profile: int = 1) -> float:
    """Factor by which a reference amplitude should be multiplied at the current temperature."""
    dt = float(current_temp_c) - float(reference_temp_c)
    return 2.0 ** (-dt / SIGNAL_TEMPERATURE_MODEL_PARAMETER[int(profile)])


# ----------------------------------------------------------------------------
# 3. per-frame processing shared by live view, Data Logging and H5 import
# ----------------------------------------------------------------------------
NARROW_FILTER_START_POINT = 40   # Acconeer DistanceFilter: narrow filter below ~0.1 m


def process_frame(range_frame, loopback_frame, background, profile: int, step_length: int,
                  start_point: int, use_filter: bool = True):
    """One frame -> (raw_sweep, corrected_sweep), both complex (points,).

    raw       = coherent mean of the sweeps
    corrected = coherent mean after phase-stable leakage cancellation (= raw if no background)
    Both are then distance-filtered (Acconeer's matched first-order Butterworth, zero phase)
    when use_filter is True. The filter is linear, so filtering after subtraction is the same
    as subtracting a filtered background.
    """
    frame = np.asarray(range_frame, dtype=complex)
    raw = frame.mean(axis=0)
    if background is not None and np.shape(background) == raw.shape:
        corrected = cancel_leakage(frame, loopback_frame, background).mean(axis=0)
    else:
        corrected = raw
    if use_filter:
        narrow = int(start_point) < NARROW_FILTER_START_POINT
        raw = distance_filter(raw, profile, step_length, narrow)
        corrected = raw if corrected is raw else distance_filter(corrected, profile, step_length, narrow)
    return raw, corrected


def noise_level(range_frames, profile: int, step_length: int, start_point: int,
                use_filter: bool = True) -> float:
    """RMS amplitude of TX-off frames processed exactly like real frames (Acconeer measures
    the noise with the transmitter off). Same units as the processed sweep amplitudes."""
    sweeps = [process_frame(f, None, None, profile, step_length, start_point, use_filter)[0]
              for f in range_frames]
    return float(np.sqrt(np.mean(np.abs(np.stack(sweeps)) ** 2)))
