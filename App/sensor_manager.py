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

import numpy as np

from PySide6.QtCore import QObject, Signal

try:
    from acconeer.exptool import a121
    ACCONEER_AVAILABLE = True
except ImportError:
    ACCONEER_AVAILABLE = False


# ============================================================
# Default sensor configuration for this project.
#
# Based on Acconeer's own close-range design (Distance Detector close-range session,
# SDK a121-v1.13.0 / Exploration Tool):
#   - Profile 1: the only profile whose direct leakage does not swamp 0-120 mm.
#   - receiver_gain 12: between Acconeer's close-range gain (5) and our old 16. Gain 5
#     made the amplitudes ~2x smaller on the FZP-lens setup; 16 put leakage + wall near the
#     int16 full scale. The header turns red on data_saturated -> lower it then.
#   - sweeps_per_frame 5, coherently averaged inside the frame. 10 (Acconeer's value)
#     halved the frame rate (~20 Hz) and made the live view feel slow.
#   - phase enhancement + IQ imbalance compensation on (Acconeer close range uses both).
#   - PRF 15.6 MHz: Acconeer's close-range PRF. 19.5 MHz is also allowed with Profile 1.
#   - start_point 0 / 80 points / step 1: 0-0.2 m, covering 0-6 in from the sensor.
#   - use_loopback_reference: a 1-point loopback subsweep in front of the range subsweep,
#     used to keep the background (leakage) subtraction phase-stable.
# ============================================================

DEFAULT_CONFIG = {
    "start_point": 0,
    "num_points": 80,
    "step_length": 1,
    "sweeps_per_frame": 5,
    "hwaas": 32,
    "profile": 1,
    "receiver_gain": 12,
    "prf": "PRF_15_6_MHz",
    "enable_tx": True,
    "phase_enhancement": True,
    "iq_imbalance_compensation": True,
    "use_loopback_reference": True,
    "distance_filter": True,
    "frame_rate": None,          # None = unlimited
    "sweep_rate": None,          # None = max
    "continuous_sweep_mode": False,
    "double_buffering": False,
    "inter_frame_idle_state": "DEEP_SLEEP",
    "inter_sweep_idle_state": "READY",
}

# Fallback only: the real value comes from the session metadata (base_step_length_m).
BASE_STEP_LENGTH_M = 0.00250227400101721

# A121 constraint: step length must divide 24 or be a multiple of 24. The UI offers the
# divisors only (2.5 mm ... 60 mm), which covers this project's range.
STEP_LENGTH_OPTIONS = [1, 2, 3, 4, 6, 8, 12, 24]

START_POINT_MIN = -275       # A121 SubsweepConfig validation limit

PRF_OPTIONS = [
    "PRF_19_5_MHz",
    "PRF_15_6_MHz",
    "PRF_13_0_MHz",
    "PRF_8_7_MHz",
    "PRF_6_5_MHz",
    "PRF_5_2_MHz",
]

IDLE_STATE_OPTIONS = ["DEEP_SLEEP", "SLEEP", "READY"]


def snap_step_length(value: int) -> int:
    """Nearest valid step length (ties go to the smaller value)."""
    return min(STEP_LENGTH_OPTIONS, key=lambda s: (abs(s - int(value)), s))


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
    phase_enhancement: bool = DEFAULT_CONFIG["phase_enhancement"]
    iq_imbalance_compensation: bool = DEFAULT_CONFIG["iq_imbalance_compensation"]
    use_loopback_reference: bool = DEFAULT_CONFIG["use_loopback_reference"]
    # Processing only (not a sensor setting): Acconeer's matched distance filter.
    distance_filter: bool = DEFAULT_CONFIG["distance_filter"]
    frame_rate: float | None = DEFAULT_CONFIG["frame_rate"]
    sweep_rate: float | None = DEFAULT_CONFIG["sweep_rate"]
    continuous_sweep_mode: bool = DEFAULT_CONFIG["continuous_sweep_mode"]
    double_buffering: bool = DEFAULT_CONFIG["double_buffering"]
    inter_frame_idle_state: str = DEFAULT_CONFIG["inter_frame_idle_state"]
    inter_sweep_idle_state: str = DEFAULT_CONFIG["inter_sweep_idle_state"]

    # Connection settings (not part of SensorConfig, but shared/configurable too)
    ip_address: str = "127.0.0.1"

    def signature(self) -> tuple:
        """Every setting that changes the raw IQ signal. A background calibration is only
        valid for the exact signature it was recorded with."""
        return (
            self.start_point, self.num_points, self.step_length, self.sweeps_per_frame,
            self.hwaas, self.profile, self.receiver_gain, self.prf, self.enable_tx,
            self.use_loopback_reference, self.phase_enhancement, self.iq_imbalance_compensation,
        )

    @property
    def range_subsweep_index(self) -> int:
        """Index of the wall/range subsweep in result.subframes."""
        return 1 if self.use_loopback_reference else 0

    def raw_distances_m(self, base_step_length_m: float = BASE_STEP_LENGTH_M):
        """Distances WITHOUT the zero-point offset (start_point * base step)."""
        import numpy as np

        points = self.start_point + np.arange(self.num_points) * self.step_length
        return points * base_step_length_m

    def distances_m(self):
        """Raw distances (no offset). Prefer SensorManager.distances_m(), which subtracts
        the calibrated zero-point offset."""
        return self.raw_distances_m()

    def _loopback_subsweep(self):
        from close_range import LOOPBACK_SUBSWEEP

        # Profile 4 is not allowed with 19.5 MHz, so then use Profile 1 at the same PRF.
        if self.prf == "PRF_19_5_MHz":
            profile, prf = 1, self.prf
        else:
            profile, prf = LOOPBACK_SUBSWEEP["profile"], LOOPBACK_SUBSWEEP["prf"]
        return a121.SubsweepConfig(
            start_point=LOOPBACK_SUBSWEEP["start_point"],
            num_points=LOOPBACK_SUBSWEEP["num_points"],
            step_length=LOOPBACK_SUBSWEEP["step_length"],
            profile=a121.Profile(profile),
            hwaas=self.hwaas,
            receiver_gain=LOOPBACK_SUBSWEEP["receiver_gain"],
            enable_tx=True,
            enable_loopback=True,
            phase_enhancement=True,
            iq_imbalance_compensation=True,
            prf=getattr(a121.PRF, prf),
        )

    def _range_subsweep(self, enable_tx=None):
        return a121.SubsweepConfig(
            start_point=self.start_point,
            num_points=self.num_points,
            step_length=self.step_length,
            profile=a121.Profile(self.profile),
            hwaas=self.hwaas,
            receiver_gain=self.receiver_gain,
            enable_tx=self.enable_tx if enable_tx is None else enable_tx,
            enable_loopback=False,
            phase_enhancement=self.phase_enhancement,
            iq_imbalance_compensation=self.iq_imbalance_compensation,
            prf=getattr(a121.PRF, self.prf),
        )

    def to_sensor_config(self):
        """Build a real acconeer.exptool.a121.SensorConfig from these params.

        With use_loopback_reference the frame has two subsweeps:
          subframes[0] = 1-point loopback reference, subframes[1] = the wall range.
        """
        if not ACCONEER_AVAILABLE:
            raise RuntimeError("acconeer.exptool is not installed in this environment")
        if self.step_length not in STEP_LENGTH_OPTIONS:
            raise ValueError(f"Step length must be one of {STEP_LENGTH_OPTIONS}")

        subsweeps = [self._range_subsweep()]
        if self.use_loopback_reference:
            subsweeps.insert(0, self._loopback_subsweep())

        cfg = a121.SensorConfig(
            subsweeps=subsweeps,
            sweeps_per_frame=self.sweeps_per_frame,
            sweep_rate=self.sweep_rate,
            frame_rate=self.frame_rate,
            continuous_sweep_mode=self.continuous_sweep_mode,
            double_buffering=self.double_buffering,
            inter_frame_idle_state=getattr(a121.IdleState, self.inter_frame_idle_state),
            inter_sweep_idle_state=getattr(a121.IdleState, self.inter_sweep_idle_state),
        )
        cfg.validate()
        return cfg


def offset_calibration_config():
    """Acconeer's loopback config for the zero-point offset (get_calibrate_offset_sensor_config)."""
    from close_range import (
        OFFSET_CAL_HWAAS, OFFSET_CAL_NUM_POINTS, OFFSET_CAL_PROFILE,
        OFFSET_CAL_START_POINT, OFFSET_CAL_STEP_LENGTH,
    )

    return a121.SensorConfig(
        start_point=OFFSET_CAL_START_POINT,
        num_points=OFFSET_CAL_NUM_POINTS,
        step_length=OFFSET_CAL_STEP_LENGTH,
        profile=a121.Profile(OFFSET_CAL_PROFILE),
        hwaas=OFFSET_CAL_HWAAS,
        sweeps_per_frame=1,
        enable_loopback=True,
        phase_enhancement=True,
        iq_imbalance_compensation=True,
    )


def split_frame(result_or_frame, params: "SensorParams"):
    """Return (range_subframe, loopback_subframe_or_None) from an a121.Result.

    Shapes: (sweeps, points) and (sweeps, 1).
    """
    import numpy as np

    subframes = getattr(result_or_frame, "subframes", None)
    if subframes is None:
        frame = np.asarray(result_or_frame)
        if params.use_loopback_reference:
            return frame[:, 1:], frame[:, :1]
        return frame, None
    if params.use_loopback_reference and len(subframes) >= 2:
        return np.asarray(subframes[1]), np.asarray(subframes[0])
    return np.asarray(subframes[0]), None


class SensorManager(QObject):
    """
    Single shared instance owning the current SensorParams, the live Client connection,
    and the sensor-level calibrations every tab must share:

      sensor_calibrations  The A121 sensor calibration from the first session. It is passed
                           to every later setup_session(), so Stop/Start or a recording does
                           NOT re-calibrate the sensor and silently move the direct-leakage
                           phase under a stored background (Acconeer does the same).
      zero_offset_m        Zero-point distance offset from the loopback calibration.
    """

    connection_changed = Signal(bool, str)  # (connected, message)
    params_changed = Signal()
    calibration_changed = Signal(str)       # human-readable calibration status

    def __init__(self):
        super().__init__()
        self.params = SensorParams()
        self.client = None
        self._clear_calibration_state()

    def _clear_calibration_state(self):
        # Bumped whenever the sensor calibration is dropped: a background taken under an
        # older epoch no longer matches the sensor's leakage phase and must be retaken.
        self.calibration_epoch = getattr(self, "calibration_epoch", 0) + 1
        self.sensor_calibrations = None
        self.calibration_temperature = None
        self.base_step_length_m = BASE_STEP_LENGTH_M
        self.zero_offset_m = None
        self.loopback_peak_m = None
        self.offset_temperature = None
        # TX-off noise level per sensor-settings signature: {signature: {True: filtered, False: unfiltered}}
        self.noise_levels = {}

    def is_connected(self) -> bool:
        return self.client is not None

    def connect_sensor(self) -> tuple[bool, str]:
        if not ACCONEER_AVAILABLE:
            msg = "acconeer.exptool is not installed in this environment"
            self.connection_changed.emit(False, msg)
            return False, msg
        try:
            self.client = a121.Client.open(ip_address=self.params.ip_address)
            self._clear_calibration_state()
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
        self._clear_calibration_state()
        self.connection_changed.emit(False, "Disconnected")

    def notify_params_changed(self):
        self.params_changed.emit()

    # ------------------------------------------------------------------
    # Sessions and calibrations
    # ------------------------------------------------------------------
    def setup_session(self, sensor_config):
        """client.setup_session() that reuses the cached sensor calibration.

        The first call calibrates the sensor and caches client.calibrations; every later
        call passes them back so the sensor is not re-calibrated between sessions.
        """
        meta = None
        if self.sensor_calibrations:
            try:
                meta = self.client.setup_session(sensor_config, calibrations=self.sensor_calibrations)
            except TypeError:
                # Very old exptool without the calibrations argument: the sensor will be
                # re-calibrated on every session, so a background is only valid in its session.
                meta = None
            except Exception:
                # Calibration rejected (e.g. server restarted): calibrate again below.
                meta = None
        if meta is None:
            had_calibration = self.sensor_calibrations is not None
            meta = self.client.setup_session(sensor_config)
            if had_calibration:
                # The old calibration could not be reused, so the sensor was re-calibrated.
                self.calibration_epoch += 1
            try:
                cals = self.client.calibrations
                if cals:
                    self.sensor_calibrations = dict(cals)
                    cal = next(iter(cals.values()))
                    self.calibration_temperature = getattr(cal, "temperature", None)
            except Exception:
                self.sensor_calibrations = None
        base = getattr(meta, "base_step_length_m", None)
        if base:
            self.base_step_length_m = float(base)
        return meta

    def calibrate_offset(self) -> str:
        """Measure the on-chip loopback pulse and compute the zero-point offset.

        Needs a connected sensor and no running session. Takes well under a second.
        """
        from close_range import distance_offset_m, loopback_peak_location_m

        cfg = offset_calibration_config()
        meta = self.setup_session(cfg)
        self.client.start_session()
        try:
            result = self.client.get_next()
        finally:
            self.client.stop_session()
        base = getattr(meta, "base_step_length_m", None) or self.base_step_length_m
        peak_m = loopback_peak_location_m(
            np.asarray(result.frame), cfg.start_point, cfg.step_length, cfg.profile.value, base
        )
        temperature = getattr(result, "temperature", None)
        self.loopback_peak_m = peak_m
        self.offset_temperature = temperature
        self.zero_offset_m = distance_offset_m(peak_m, self.params.profile, temperature)
        msg = (f"Zero-point offset {self.zero_offset_m * 1000:.1f} mm "
               f"(loopback peak {peak_m * 1000:.1f} mm, Profile {self.params.profile})")
        self.calibration_changed.emit(msg)
        return msg

    def calibrate_noise(self, num_frames: int = 10) -> str:
        """Measure the receiver noise with the transmitter off (same settings otherwise),
        processed exactly like real frames. Needs a connected sensor and no running session."""
        import copy as _copy
        from close_range import noise_level

        p = _copy.deepcopy(self.params)
        p.enable_tx = False
        cfg = p.to_sensor_config()
        self.setup_session(cfg)
        self.client.start_session()
        frames = []
        try:
            for _ in range(max(3, int(num_frames))):
                frames.append(split_frame(self.client.get_next(), p)[0])
        finally:
            self.client.stop_session()
        levels = {
            flt: noise_level(frames, p.profile, p.step_length, p.start_point, flt)
            for flt in (True, False)
        }
        self.noise_levels[self.params.signature()] = levels
        msg = f"Noise level (TX off) {levels[self.params.distance_filter]:.1f}"
        self.calibration_changed.emit(msg)
        return msg

    def current_noise_level(self, params: SensorParams | None = None):
        p = params or self.params
        levels = self.noise_levels.get(p.signature())
        return None if levels is None else levels[bool(p.distance_filter)]

    def ensure_calibrated(self) -> str | None:
        """Offset calibration once per connection (also caches the sensor calibration), and a
        TX-off noise measurement once per sensor-settings combination."""
        msgs = []
        if self.zero_offset_m is None:
            msgs.append(self.calibrate_offset())
        if self.params.signature() not in self.noise_levels and self.params.enable_tx:
            msgs.append(self.calibrate_noise())
        return " · ".join(msgs) if msgs else None

    def reset_calibration(self):
        """Forget the sensor calibration and offset; the next Start re-calibrates both.
        Any stored background becomes invalid (main.py discards it)."""
        self._clear_calibration_state()
        self.calibration_changed.emit("Sensor calibration cleared; it is redone at the next Start.")

    def background_key(self) -> tuple:
        """What a stored background must match: the IQ-relevant settings AND the sensor
        calibration it was taken under."""
        return (self.params.signature(), self.calibration_epoch)

    def total_offset_m(self) -> float:
        """The sensor's own zero-point offset (loopback). Distances are measured from the
        sensor; the physical sensor-to-wall gap is logged as 'Sensor standoff'."""
        return self.zero_offset_m or 0.0

    def distances_m(self, params: SensorParams | None = None):
        """Corrected distances from the sensor: raw (metadata base step) - zero-point offset."""
        p = params or self.params
        return p.raw_distances_m(self.base_step_length_m) - self.total_offset_m()
