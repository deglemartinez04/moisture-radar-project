"""
health.py

Turns the per-frame status flags that the Acconeer server attaches to every
result (temperature, data_saturated, calibration_needed, frame_delayed) into one
short status for the header bar. Pure Python (no Qt) so it is easy to test.

Why this exists: Acconeer's close-range leakage calibration is only valid within
about +/-15 C of the temperature it was recorded at (Distance Detector User
Guide), and the sensor itself reports `calibration_needed` when that happens.
Without a visible warning, a stale background calibration looks like real data.
"""

from __future__ import annotations

from dataclasses import dataclass

TEMP_WARN_C = 10.0    # warn before the limit is reached
TEMP_LIMIT_C = 15.0   # Acconeer: calibration valid within +/-15 C


@dataclass
class Health:
    state: str    # "ok", "warn" or "bad"
    text: str     # short text for the header label
    detail: str   # longer explanation for the tooltip


def assess_health(temperature_c, reference_temps_c, saturated_recent, needs_cal_recent, delayed_recent):
    """
    temperature_c       latest sensor temperature (deg C) or None
    reference_temps_c   temperatures at which calibrations were taken (session
                        start, background calibration); None entries are ignored
    saturated_recent    number of recent frames flagged data_saturated
    needs_cal_recent    number of recent frames flagged calibration_needed
    delayed_recent      number of recent frames flagged frame_delayed
    """
    refs = [t for t in (reference_temps_c or []) if t is not None]
    worst_delta = None
    if temperature_c is not None and refs:
        worst_delta = max(abs(temperature_c - t) for t in refs)
    temp_txt = "" if temperature_c is None else f" · {temperature_c:.0f} °C"

    if needs_cal_recent > 0:
        return Health(
            "bad", "⚠ Recalibrate sensor" + temp_txt,
            "The sensor reports calibration_needed (its temperature drifted). "
            "Press Recalibrate, then Start Measurement, then redo the background calibration.",
        )
    if worst_delta is not None and worst_delta >= TEMP_LIMIT_C:
        return Health(
            "bad", f"⚠ Temp drift {worst_delta:.0f} °C",
            f"Sensor temperature moved {worst_delta:.0f} °C since calibration (limit about "
            f"{TEMP_LIMIT_C:.0f} °C). Press Recalibrate, Start, and redo the background calibration.",
        )
    if saturated_recent > 0:
        return Health(
            "bad", "⚠ Signal saturated" + temp_txt,
            "The receiver saturated (data corrupted). Lower the receiver gain in Settings "
            "(Acconeer Sparse IQ guide 6.1), then recalibrate.",
        )
    if worst_delta is not None and worst_delta >= TEMP_WARN_C:
        return Health(
            "warn", f"Temp drift {worst_delta:.0f} °C",
            f"Sensor temperature moved {worst_delta:.0f} °C since calibration; "
            f"recalibrate before it reaches about {TEMP_LIMIT_C:.0f} °C.",
        )
    if delayed_recent > 0:
        return Health(
            "warn", "Frames delayed" + temp_txt,
            "The host did not read a frame within the fixed frame period, so it was delayed. "
            "Lower the frame rate (or set it to Automatic), or enable double buffering.",
        )
    detail = "No saturation, calibration or timing problems reported by the sensor."
    if worst_delta is not None:
        detail += f" Temperature change since calibration: {worst_delta:.0f} °C."
    return Health("ok", "Sensor OK" + temp_txt, detail)
