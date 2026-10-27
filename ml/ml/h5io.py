"""Read Acconeer A121 .h5 recordings with plain h5py (no acconeer-exptool needed).

Layout matches what the App's H5Recorder writes and what h5_to_features.py reads:
    sessions/session_0/session_config                     JSON
    sessions/session_0/group_0/entry_0/metadata           JSON (base_step_length_m)
    sessions/session_0/group_0/entry_0/result/frame       (frames, sweeps, points) {real, imag} int16
"""

from __future__ import annotations

import json

import h5py
import numpy as np

FRAME_PATH = "sessions/session_0/group_0/entry_0/result/frame"
CONFIG_PATH = "sessions/session_0/session_config"
METADATA_PATH = "sessions/session_0/group_0/entry_0/metadata"

# Sensor settings that must match between training data and live use.
CONFIG_KEYS = ("start_point", "num_points", "step_length", "profile", "hwaas", "receiver_gain", "prf")


def load_recording(h5_path):
    """Return (iq, distances_m, sensor_config).

    iq is (n_frames, n_points) complex, with multiple sweeps per frame averaged
    into one -- the same reduction the App's live thread does.
    """
    with h5py.File(h5_path, "r") as f:
        frame = f[FRAME_PATH][()]
        session_config = json.loads(f[CONFIG_PATH][()])
        metadata = json.loads(f[METADATA_PATH][()])

    sensor = session_config["groups"][0]["1"]
    subsweep = sensor["subsweeps"][0]
    distances_m = (
        subsweep["start_point"] + np.arange(subsweep["num_points"]) * subsweep["step_length"]
    ) * metadata["base_step_length_m"]

    iq = frame["real"].astype(np.float64) + 1j * frame["imag"].astype(np.float64)
    if iq.ndim == 2:  # single sweep stored without the sweep axis
        iq = iq[:, np.newaxis, :]
    iq = iq.mean(axis=1)

    config = {k: subsweep.get(k) for k in CONFIG_KEYS}
    config["sweeps_per_frame"] = sensor.get("sweeps_per_frame")
    return iq, distances_m, config


def write_recording(h5_path, iq_frames, start_point, num_points, step_length,
                    base_step_length_m=0.00250227400101721, **subsweep_extra):
    """Write a minimal recording in the same layout (used for synthetic test data)."""
    iq_frames = np.asarray(iq_frames)
    frame = np.empty(iq_frames.shape[:1] + (1,) + iq_frames.shape[1:], dtype=[("real", "<i2"), ("imag", "<i2")])
    frame["real"][:, 0, :] = np.clip(np.round(iq_frames.real), -32768, 32767)
    frame["imag"][:, 0, :] = np.clip(np.round(iq_frames.imag), -32768, 32767)
    subsweep = {"start_point": start_point, "num_points": num_points, "step_length": step_length, **subsweep_extra}
    session_config = {"groups": [{"1": {"sweeps_per_frame": 1, "subsweeps": [subsweep]}}]}
    with h5py.File(h5_path, "w") as f:
        f[CONFIG_PATH] = json.dumps(session_config)
        f[METADATA_PATH] = json.dumps({"base_step_length_m": base_step_length_m})
        f[FRAME_PATH] = frame
