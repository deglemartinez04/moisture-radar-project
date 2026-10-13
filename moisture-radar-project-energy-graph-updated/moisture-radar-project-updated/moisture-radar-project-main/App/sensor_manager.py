"""
sensor_manager.py

Shared state for the whole app: the current SensorConfig (edited in the Settings
tab, used by both Record and Live tabs) and the Client connection lifecycle.

Keeping this in one place is what makes "settings shared between tabs" work --
every tab reads/writes the same SensorManager instance instead of keeping its
own copy of the config.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from PySide6.QtCore import QObject, Signal

try:
    from acconeer.exptool import a121
    ACCONEER_AVAILABLE = True
except ImportError:
    ACCONEER_AVAILABLE = False


# ============================================================
# Default sensor configuration for this project.
#
# These defaults reflect the project's validated flush-contact, static-material
# measurement setup:
#   - Profile 1: shortest pulse, smallest close-in distance -- needed since the
#     sensor lens sits flush against the wall (very short range).
#   - sweeps_per_frame = 1: no motion/velocity tracking needed for static material
#     scans, per Acconeer's own configuration guidance.
#   - hwaas raised (32) to compensate for sweeps_per_frame=1 and get good SNR.
#   - Short range (start_point/num_points/step_length) tuned for flush contact,
#     covering just past a typical drywall/wood thickness.
# ============================================================

DEFAULT_CONFIG = {
    "start_point": 24,
    "num_points": 40,
    "step_length": 2,
    "sweeps_per_frame": 1,
    "hwaas": 32,
    "profile": 1,
    "receiver_gain": 16,
    "prf": "PRF_15_6_MHz",
    "enable_tx": True,
    "enable_loopback": False,
    "phase_enhancement": False,
    "iq_imbalance_compensation": False,
    "frame_rate": None,          # None = unlimited
    "sweep_rate": None,          # None = max
    "continuous_sweep_mode": False,
    "double_buffering": False,
    "inter_frame_idle_state": "DEEP_SLEEP",
    "inter_sweep_idle_state": "READY",
}

BASE_STEP_LENGTH_M = 0.00250227400101721  # meters per "point" step, from sensor metadata

PRF_OPTIONS = [
    "PRF_19_5_MHz",
    "PRF_15_6_MHz",
    "PRF_13_0_MHz",
    "PRF_8_7_MHz",
    "PRF_6_5_MHz",
    "PRF_5_2_MHz",
]

IDLE_STATE_OPTIONS = ["DEEP_SLEEP", "SLEEP", "READY"]


@dataclass
class SensorParams:
    """Plain-data mirror of everything configurable on the A121, for easy UI binding."""

    start_point: int = DEFAULT_CONFIG["start_point"]
    num_points: int = DEFAULT_CONFIG["num_points"]
    step_length: int = DEFAULT_CONFIG["step_length"]
    sweeps_per_frame: int = DEFAULT_CONFIG["sweeps_per_frame"]
    hwaas: int = DEFAULT_CONFIG["hwaas"]
    profile: int = DEFAULT_CONFIG["profile"]
    receiver_gain: int = DEFAULT_CONFIG["receiver_gain"]
    prf: str = DEFAULT_CONFIG["prf"]
    enable_tx: bool = DEFAULT_CONFIG["enable_tx"]
    enable_loopback: bool = DEFAULT_CONFIG["enable_loopback"]
    phase_enhancement: bool = DEFAULT_CONFIG["phase_enhancement"]
    iq_imbalance_compensation: bool = DEFAULT_CONFIG["iq_imbalance_compensation"]
    frame_rate: float | None = DEFAULT_CONFIG["frame_rate"]
    sweep_rate: float | None = DEFAULT_CONFIG["sweep_rate"]
    continuous_sweep_mode: bool = DEFAULT_CONFIG["continuous_sweep_mode"]
    double_buffering: bool = DEFAULT_CONFIG["double_buffering"]
    inter_frame_idle_state: str = DEFAULT_CONFIG["inter_frame_idle_state"]
    inter_sweep_idle_state: str = DEFAULT_CONFIG["inter_sweep_idle_state"]

    # Connection settings (not part of SensorConfig, but shared/configurable too)
    ip_address: str = "127.0.0.1"

    def distances_m(self):
        import numpy as np

        points = self.start_point + np.arange(self.num_points) * self.step_length
        return points * BASE_STEP_LENGTH_M

    def to_sensor_config(self):
        """Build a real acconeer.exptool.a121.SensorConfig from these params."""
        if not ACCONEER_AVAILABLE:
            raise RuntimeError("acconeer.exptool is not installed in this environment")

        cfg = a121.SensorConfig()
        cfg.start_point = self.start_point
        cfg.num_points = self.num_points
        cfg.step_length = self.step_length
        cfg.sweeps_per_frame = self.sweeps_per_frame
        cfg.hwaas = self.hwaas
        cfg.profile = a121.Profile(self.profile)
        cfg.receiver_gain = self.receiver_gain
        cfg.prf = getattr(a121.PRF, self.prf)
        cfg.enable_tx = self.enable_tx
        cfg.enable_loopback = self.enable_loopback
        cfg.phase_enhancement = self.phase_enhancement
        cfg.iq_imbalance_compensation = self.iq_imbalance_compensation
        cfg.frame_rate = self.frame_rate
        cfg.sweep_rate = self.sweep_rate
        cfg.continuous_sweep_mode = self.continuous_sweep_mode
        cfg.double_buffering = self.double_buffering
        cfg.inter_frame_idle_state = getattr(a121.IdleState, self.inter_frame_idle_state)
        cfg.inter_sweep_idle_state = getattr(a121.IdleState, self.inter_sweep_idle_state)
        return cfg


class SensorManager(QObject):
    """
    Single shared instance owning the current SensorParams and the live Client
    connection. Both the Record tab and Live tab hold a reference to the same
    SensorManager, so changing a setting in the Settings tab is instantly visible
    to both.
    """

    connection_changed = Signal(bool, str)  # (connected, message)
    params_changed = Signal()

    def __init__(self):
        super().__init__()
        self.params = SensorParams()
        self.client = None

    def is_connected(self) -> bool:
        return self.client is not None

    def connect_sensor(self) -> tuple[bool, str]:
        if not ACCONEER_AVAILABLE:
            msg = "acconeer.exptool is not installed in this environment"
            self.connection_changed.emit(False, msg)
            return False, msg
        try:
            self.client = a121.Client.open(ip_address=self.params.ip_address)
            msg = f"Connected to {self.params.ip_address}"
            self.connection_changed.emit(True, msg)
            return True, msg
        except Exception as e:
            self.client = None
            msg = f"Connection failed: {e}"
            self.connection_changed.emit(False, msg)
            return False, msg

    def disconnect_sensor(self):
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass
            self.client = None
        self.connection_changed.emit(False, "Disconnected")

    def notify_params_changed(self):
        self.params_changed.emit()
