"""
main.py

Entry point for the Radar Moisture Tomography app.

The MainWindow owns the one global live-scan acquisition session. The
Amplitude Plot, Live dashboard, and future live graphs can subscribe to that
same feature stream. Data Logging remains a separate recording workflow.
"""

from __future__ import annotations

import sys
import threading
from collections import deque
import time

import copy

import numpy as np
from PySide6.QtCore import QThread, QTimer, Signal, Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from sensor_manager import SensorManager, split_frame
from close_range import leakage_reference, process_frame
from settings_tab import SettingsTab
from record_tab import RecordTab
from health import assess_health
from ui_theme import control_qss
from live_tab import LiveTab, ROLLING_WINDOW_SIZE, PRINT_INTERVAL_MS, extract_features_from_window
from amplitude_plot import AmplitudePlotTab
from diagnostic_tabs import PhaseTab, EnergyTab


class LiveAcquisitionThread(QThread):
    """Global live acquisition thread with phase-stable complex-IQ leakage cancellation.

    Each frame is split into the range subframe (sweeps, points) and, when enabled, the
    1-point loopback reference subframe (sweeps, 1). The background is stored relative to the
    loopback phase and rotated to every frame's loopback phase before subtraction
    (close_range.cancel_leakage, Acconeer's phase-jitter compensation). Sweeps are then
    averaged coherently inside the frame.
    """

    features_ready = Signal(dict, object, object, object, bool, object, object, object, object, float, float, float)
    error = Signal(str)
    calibration_status = Signal(str)
    health_ready = Signal(object)
    background_ready = Signal(object, object)   # complex background IQ, sensor temperature
    profile_ready = Signal(object, object)      # distances, complex corrected profile (window mean)

    def __init__(self, client, distances_m, params):
        super().__init__()
        self.client = client
        self.distances_m = distances_m
        self.params = copy.deepcopy(params)
        self._running = True
        self._state_lock = threading.Lock()
        self._calibration_requested = False
        self._calibration_frames = []
        self._calibration_target_frames = 30
        self._background_iq = None
        self._correction_enabled = False
        self._background_temp = None
        self._last_temperature = None
        self._calibration_temp = None   # temperature of the (reused) sensor calibration
        self.noise_level = None         # TX-off noise level for these settings (features SNR)

    def request_background_calibration(self, frame_count=30):
        """Average complex IQ over multiple frames in a clear, stationary scene."""
        with self._state_lock:
            self._calibration_requested = True
            self._calibration_frames = []
            self._calibration_target_frames = max(5, int(frame_count))
        self.calibration_status.emit(
            f"Calibrating background: collecting {self._calibration_target_frames} frames. Keep the sensor still and point it at the reference scene."
        )

    def set_correction_enabled(self, enabled):
        with self._state_lock:
            self._correction_enabled = bool(enabled)
            has_background = self._background_iq is not None
        if enabled and not has_background:
            self.calibration_status.emit("Correction requested, but no background is calibrated yet. Calibrate first.")
        elif enabled:
            self.calibration_status.emit("Background correction enabled.")
        else:
            self.calibration_status.emit("Background correction disabled; raw signal is used for live features.")

    def run(self):
        raw_amplitude_buffer = deque(maxlen=ROLLING_WINDOW_SIZE)
        corrected_amplitude_buffer = deque(maxlen=ROLLING_WINDOW_SIZE)
        phase_buffer = deque(maxlen=ROLLING_WINDOW_SIZE)
        raw_phase_buffer = deque(maxlen=ROLLING_WINDOW_SIZE)
        corrected_phase_buffer = deque(maxlen=ROLLING_WINDOW_SIZE)
        corrected_complex_buffer = deque(maxlen=ROLLING_WINDOW_SIZE)
        energy_buffer = deque(maxlen=ROLLING_WINDOW_SIZE)
        corrected_energy_buffer = deque(maxlen=ROLLING_WINDOW_SIZE)
        saturated_flags = deque(maxlen=60)
        cal_needed_flags = deque(maxlen=60)
        delayed_flags = deque(maxlen=60)
        session_start_temp = None
        last_emit = 0.0

        try:
            while self._running:
                result = self.client.get_next()
                range_frame, loopback_frame = split_frame(result, self.params)
                # Coherent mean of the sweeps inside one frame (Acconeer: frame.mean(axis=0)).
                single_sweep = np.mean(range_frame, axis=0)

                # Status flags the sensor attaches to every frame.
                temperature = getattr(result, "temperature", None)
                if temperature is not None:
                    self._last_temperature = float(temperature)
                    if session_start_temp is None:
                        session_start_temp = float(temperature)
                saturated_flags.append(bool(getattr(result, "data_saturated", False)))
                cal_needed_flags.append(bool(getattr(result, "calibration_needed", False)))
                delayed_flags.append(bool(getattr(result, "frame_delayed", False)))

                with self._state_lock:
                    if self._calibration_requested:
                        self._calibration_frames.append((
                            np.array(range_frame, dtype=np.complex128, copy=True),
                            None if loopback_frame is None else np.array(loopback_frame, copy=True),
                        ))
                        count = len(self._calibration_frames)
                        target = self._calibration_target_frames
                        if count >= target:
                            ranges = [f[0] for f in self._calibration_frames]
                            loopbacks = [f[1] for f in self._calibration_frames]
                            if all(lb is not None for lb in loopbacks):
                                # Leakage stored relative to the loopback phase.
                                self._background_iq = leakage_reference(ranges, loopbacks)
                            else:
                                self._background_iq = np.mean([r.mean(axis=0) for r in ranges], axis=0)
                            self._background_temp = self._last_temperature
                            self.background_ready.emit(np.array(self._background_iq, copy=True), self._background_temp)
                            self._calibration_frames = []
                            self._calibration_requested = False
                            self.calibration_status.emit(
                                f"Background calibrated from {target} frames. Keep sensor settings and reference setup unchanged."
                            )
                    background_temp = self._background_temp
                    background = None if self._background_iq is None else self._background_iq.copy()
                    correction_enabled = self._correction_enabled

                use_bg = correction_enabled and background is not None and background.shape == single_sweep.shape
                p = self.params
                # Coherent mean -> leakage cancellation -> Acconeer distance filter.
                single_sweep, corrected_iq = process_frame(
                    range_frame, loopback_frame, background if use_bg else None,
                    p.profile, p.step_length, p.start_point, p.distance_filter,
                )
                raw_amplitude = np.abs(single_sweep)
                corrected_amplitude = np.abs(corrected_iq)

                raw_amplitude_buffer.append(raw_amplitude)
                corrected_amplitude_buffer.append(corrected_amplitude)
                energy_buffer.append(np.abs(single_sweep) ** 2)
                # energy of the background-subtracted signal (not raw energy minus background energy)
                corrected_energy_buffer.append(np.abs(corrected_iq) ** 2)
                raw_phase_buffer.append(np.angle(single_sweep))
                corrected_phase_buffer.append(np.angle(corrected_iq))
                corrected_complex_buffer.append(np.asarray(corrected_iq, dtype=complex))
                phase_buffer.append(np.angle(corrected_iq if correction_enabled else single_sweep))

                now = time.time()
                if now - last_emit >= (PRINT_INTERVAL_MS / 1000.0) and len(corrected_amplitude_buffer) >= 3:
                    feature_amplitudes = corrected_amplitude_buffer if correction_enabled else raw_amplitude_buffer
                    background_applied = bool(
                        correction_enabled and background is not None
                        and background.shape == single_sweep.shape
                    )
                    features = extract_features_from_window(
                        np.array(feature_amplitudes),
                        np.array(phase_buffer),
                        self.distances_m,
                        background_corrected=background_applied,
                        noise_level=self.noise_level,
                    )
                    raw_mean = np.mean(np.array(raw_amplitude_buffer), axis=0)
                    corrected_mean = np.mean(np.array(corrected_amplitude_buffer), axis=0)
                    raw_phase_stack = np.array(raw_phase_buffer)
                    corrected_phase_stack = np.array(corrected_phase_buffer)
                    raw_phase_mean = np.angle(np.mean(np.exp(1j * raw_phase_stack), axis=0))
                    corrected_phase_mean = np.angle(np.mean(np.exp(1j * corrected_phase_stack), axis=0))
                    energy_profile = np.mean(np.array(energy_buffer), axis=0)
                    corrected_energy_profile = np.mean(np.array(corrected_energy_buffer), axis=0)
                    # Circular standard deviation at the strongest point in the active amplitude trace.
                    active_amplitudes = np.mean(np.array(feature_amplitudes), axis=0)
                    peak_idx = int(np.argmax(active_amplitudes))
                    selected_phase_stack = np.array(phase_buffer)[:, peak_idx]
                    resultant = abs(np.mean(np.exp(1j * selected_phase_stack)))
                    phase_std = float(np.sqrt(max(0.0, -2.0 * np.log(max(resultant, 1e-12)))))
                    phase_at_peak = float(np.angle(np.mean(np.exp(1j * selected_phase_stack))))
                    peak_distance = float(self.distances_m[peak_idx])
                    self.profile_ready.emit(self.distances_m, np.mean(np.array(corrected_complex_buffer), axis=0))
                    self.features_ready.emit(
                        features, self.distances_m, raw_mean, corrected_mean, correction_enabled,
                        raw_phase_mean, corrected_phase_mean, energy_profile, corrected_energy_profile,
                        phase_at_peak, phase_std, peak_distance
                    )
                    health = assess_health(
                        self._last_temperature,
                        [self._calibration_temp, session_start_temp, background_temp],
                        sum(saturated_flags), sum(cal_needed_flags), sum(delayed_flags),
                    )
                    self.health_ready.emit(health)
                    last_emit = now
        except Exception as exc:
            if self._running:
                self.error.emit(str(exc))

    def stop(self):
        self._running = False


# Stylesheet template. "Regular" is the original desktop look; "compact" is used
# automatically when the window is small (e.g. a 5-inch 800x480 display).
_STYLE_TEMPLATE = """
QWidget {
    background-color: #0d1117;
    color: #c7d0dc;
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: @FONT@px;
}
QMainWindow { background-color: #0d1117; }
QTabWidget::pane { border: none; background-color: #0d1117; }
QTabBar::tab {
    background-color: #12161c;
    color: #8b96a5;
    padding: @TABPAD@;
    border: none;
    font-size: @FONT@px;
    font-weight: 500;
}
QTabBar::tab:selected {
    background-color: #0d1117;
    color: #eef2f7;
    border-bottom: 2px solid #5fb3e8;
}
QTabBar::tab:hover:!selected { color: #c7d0dc; }
QGroupBox {
    background-color: #12161c;
    border: 1px solid #232a35;
    border-radius: 6px;
    margin-top: @GBTOP@px;
    padding-top: @GBPAD@px;
    font-weight: 600;
    color: #8b96a5;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 4px;
    color: #8b96a5;
}
QPushButton {
    background-color: #1c2330;
    border: 1px solid #2d3748;
    border-radius: 4px;
    padding: @BTNPAD@;
    color: #c7d0dc;
}
QPushButton:hover { background-color: #232b3a; border-color: #3d4a5f; }
QPushButton:disabled { color: #4a5568; background-color: #12161c; }
QPushButton#primaryButton {
    background-color: #1a5f8f;
    border: 1px solid #2680bc;
    color: #eef2f7;
    font-weight: 600;
}
QPushButton#primaryButton:hover { background-color: #2172a8; }
QPushButton#globalStartButton {
    background-color: #247a52;
    border: 1px solid #31996a;
    color: #eef2f7;
    font-weight: 700;
    padding: @GBTNPAD@;
}
QPushButton#globalStartButton:hover { background-color: #2d9563; }
QPushButton#globalStopButton {
    background-color: #7d3038;
    border: 1px solid #a7444e;
    color: #eef2f7;
    font-weight: 700;
    padding: @GBTNPAD@;
}
QPushButton#globalStopButton:hover { background-color: #963b45; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #0d1117;
    border: 1px solid #2d3748;
    border-radius: 4px;
    padding: @INPAD@;
    color: #eef2f7;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border-color: #5fb3e8; }
QCheckBox { spacing: 8px; color: #c7d0dc; padding: 3px 0; }
QLabel#summaryLabel { color: #8b96a5; font-family: "Cascadia Code", "Consolas", monospace; }
QLabel#healthLabel { color: #6b7686; }
QLabel#healthLabel[state="ok"] { color: #7fd88f; }
QLabel#healthLabel[state="warn"] { color: #e0a458; }
QLabel#healthLabel[state="bad"] { color: #ff7b72; font-weight: 600; }
QLabel#featureName { color: #6b7686; font-size: @FNAME@px; }
QLabel#featureValue { color: #eef2f7; font-family: "Cascadia Code", "Consolas", monospace; font-size: @FVAL@px; font-weight: 600; }
"""

_STYLE_REGULAR = dict(FONT="13", TABPAD="10px 22px", GBTOP="14", GBPAD="12",
                      BTNPAD="7px 16px", GBTNPAD="8px 18px", INPAD="5px 8px",
                      FNAME="11", FVAL="16")
_STYLE_COMPACT = dict(FONT="11", TABPAD="6px 9px", GBTOP="10", GBPAD="7",
                      BTNPAD="4px 8px", GBTNPAD="5px 10px", INPAD="3px 5px",
                      FNAME="10", FVAL="13")


def build_stylesheet(compact: bool = False) -> str:
    values = _STYLE_COMPACT if compact else _STYLE_REGULAR
    css = _STYLE_TEMPLATE
    for key, val in values.items():
        css = css.replace(f"@{key}@", val)
    # Explicit scroll-bar and spin-box button styling (see ui_theme.py). Needs a running
    # QApplication, so build_stylesheet() must be called after the app object exists.
    return css + control_qss()

# Switch to the compact look below these window sizes (5" display = 800x480).
COMPACT_WIDTH = 900
COMPACT_HEIGHT = 560


def _scrollable(widget: QWidget) -> QScrollArea:
    """Wrap a tab so it scrolls instead of forcing the window larger than the screen."""
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QScrollArea.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    area.setWidget(widget)
    return area


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Radar Moisture Tomography — A121 Console")
        # Size to the screen instead of a fixed 1180x820 so it fits small displays,
        # and keep the minimum tiny so the window stays freely resizable.
        avail = QApplication.primaryScreen().availableGeometry()
        self.resize(min(1180, avail.width()), min(820, avail.height()))
        self.setMinimumSize(320, 240)
        self._compact = None

        self.sensor_manager = SensorManager()
        self.live_thread: LiveAcquisitionThread | None = None

        self._build_ui()
        QShortcut(QKeySequence("F11"), self, activated=self._toggle_fullscreen)

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _apply_density(self):
        """Use the compact stylesheet on small windows, regular on large ones."""
        compact = self.width() < COMPACT_WIDTH or self.height() < COMPACT_HEIGHT
        if compact == self._compact:
            return
        self._compact = compact
        QApplication.instance().setStyleSheet(build_stylesheet(compact))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Deferred so the stylesheet change never happens inside a resize event.
        QTimer.singleShot(0, self._apply_density)

    def _build_ui(self):
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(6, 4, 6, 0)
        root_layout.setSpacing(4)

        header = QHBoxLayout()
        self.global_status = QLabel("Live scan idle")
        self.global_status.setObjectName("summaryLabel")
        # Long status messages are clipped instead of forcing the window wider.
        self.global_status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.global_status.setMinimumWidth(0)
        header.addWidget(self.global_status, 1)
        self.health_label = QLabel("")
        self.health_label.setObjectName("healthLabel")
        self.health_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.health_label.setMinimumWidth(0)
        header.addWidget(self.health_label, 1)

        self.start_measurements_btn = QPushButton("Start Measurement")
        self.start_measurements_btn.setObjectName("globalStartButton")
        self.stop_measurements_btn = QPushButton("Stop")
        self.stop_measurements_btn.setObjectName("globalStopButton")
        self.stop_measurements_btn.setEnabled(False)

        header.addWidget(self.start_measurements_btn)
        header.addWidget(self.stop_measurements_btn)
        self.recalibrate_btn = QPushButton("Recalibrate")
        self.recalibrate_btn.setToolTip(
            "Forget the sensor calibration, zero-point offset and background. Use it when the "
            "header asks for recalibration (temperature drift / calibration_needed)."
        )
        header.addWidget(self.recalibrate_btn)
        root_layout.addLayout(header)

        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.tabBar().setUsesScrollButtons(True)
        tabs.tabBar().setElideMode(Qt.ElideNone)

        self.settings_tab = SettingsTab(self.sensor_manager)
        self.data_logging_tab = RecordTab(self.sensor_manager)
        self.live_tab = LiveTab(self.sensor_manager)
        self.amplitude_plot_tab = AmplitudePlotTab()
        self.phase_tab = PhaseTab()
        self.energy_tab = EnergyTab()
        self.amplitude_plot_tab.calibrate_button.clicked.connect(self.calibrate_background)
        self.amplitude_plot_tab.correction_checkbox.toggled.connect(self.set_background_correction)
        # Background hand-off: the Data Logging tab refuses to record without a valid background.
        self.background_iq = None
        self.background_temp = None
        self.background_signature = None   # sensor settings the background was taken with
        self.session_signature = None      # sensor settings of the running live session
        self.data_logging_tab.set_background_calibration(None)

        # Left-to-right order: Settings, Moisture Level, Amplitude, Phase, Energy,
        # Data Logging. Settings already has its own internal scroll area, so it is
        # added directly; every other tab is wrapped so it scrolls on tiny screens.
        tabs.addTab(self.settings_tab, "Settings")
        tabs.addTab(_scrollable(self.live_tab), "Moisture Level")
        tabs.addTab(_scrollable(self.amplitude_plot_tab), "Amplitude")
        tabs.addTab(_scrollable(self.phase_tab), "Phase")
        tabs.addTab(_scrollable(self.energy_tab), "Energy")
        tabs.addTab(_scrollable(self.data_logging_tab), "Data Logging")

        root_layout.addWidget(tabs, 1)
        self.setCentralWidget(root)

        self.start_measurements_btn.clicked.connect(self.start_live_measurements)
        self.stop_measurements_btn.clicked.connect(self.stop_live_measurements)
        self.recalibrate_btn.clicked.connect(self.recalibrate_sensor)

    def start_live_measurements(self):
        if self.live_thread is not None:
            return

        if not self.sensor_manager.is_connected():
            self.global_status.setText("Not connected — connect the sensor in Settings first.")
            self.live_tab.set_scanning(False, "Not connected — go to Settings and connect first.")
            return

        # A121 only supports one active session on the shared client. Prevent
        # the global live session from being started over an active recording.
        try:
            if self.sensor_manager.client is not None and self.sensor_manager.client.session_is_started:
                self.global_status.setText("Sensor session is already active. Stop the current session first.")
                return

            sensor_config = self.sensor_manager.params.to_sensor_config()
            # Zero-point offset (loopback) once per connection; this also creates the sensor
            # calibration that every later session reuses.
            offset_msg = self.sensor_manager.ensure_calibrated()
            self.sensor_manager.setup_session(sensor_config)
            self.session_signature = self.sensor_manager.background_key()
            stale_background = (
                self.background_iq is not None and self.background_signature != self.session_signature
            )
            if stale_background:
                self.background_iq = None
                self.background_temp = None
                self.background_signature = None
                self.data_logging_tab.set_background_calibration(None)
                self.amplitude_plot_tab.correction_checkbox.blockSignals(True)
                self.amplitude_plot_tab.correction_checkbox.setChecked(False)
                self.amplitude_plot_tab.correction_checkbox.blockSignals(False)
            self.sensor_manager.client.start_session()

            distances_m = self.sensor_manager.distances_m()
            self.live_thread = LiveAcquisitionThread(
                self.sensor_manager.client, distances_m, self.sensor_manager.params
            )
            self.live_thread._calibration_temp = self.sensor_manager.calibration_temperature
            self.live_thread.noise_level = self.sensor_manager.current_noise_level()
            self.live_thread.features_ready.connect(self._on_live_features)
            self.live_thread.profile_ready.connect(self.amplitude_plot_tab.update_complex_profile)
            self.live_thread.calibration_status.connect(self._on_calibration_status)
            self.live_thread.health_ready.connect(self._on_health)
            self.live_thread.background_ready.connect(self._on_background_ready)
            if stale_background:
                self.amplitude_plot_tab.set_calibration_status(
                    "Sensor settings or sensor calibration changed: old background discarded. Calibrate again."
                )
            if self.background_iq is not None:
                self.amplitude_plot_tab.set_calibration_status(
                    "Using the previous background calibration (sensor settings unchanged)."
                )
                # Reuse the last calibration while the sensor settings are unchanged.
                self.live_thread._background_iq = np.array(self.background_iq, copy=True)
                self.live_thread._background_temp = self.background_temp
            self.live_thread._correction_enabled = (
                self.amplitude_plot_tab.correction_checkbox.isChecked() and self.background_iq is not None
            )
            self.live_thread.error.connect(self._on_live_error)
            self.live_thread.finished.connect(self._on_live_thread_finished)
            self.live_thread.start()

            self.start_measurements_btn.setEnabled(False)
            self.stop_measurements_btn.setEnabled(True)
            offset_txt = f" · offset {self.sensor_manager.total_offset_m() * 1000:.1f} mm"
            self.global_status.setText("Live measurements running" + offset_txt)
            if offset_msg:
                self.amplitude_plot_tab.set_calibration_status(offset_msg)
            self.live_tab.set_scanning(True)
        except Exception as exc:
            self.global_status.setText(f"Failed to start: {exc}")
            self.live_tab.set_scanning(False, f"Failed to start: {exc}")
            try:
                if self.sensor_manager.client is not None and self.sensor_manager.client.session_is_started:
                    self.sensor_manager.client.stop_session()
            except Exception:
                pass

    def _discard_background(self):
        self.background_iq = None
        self.background_temp = None
        self.background_signature = None
        self.data_logging_tab.set_background_calibration(None)
        self.amplitude_plot_tab.correction_checkbox.blockSignals(True)
        self.amplitude_plot_tab.correction_checkbox.setChecked(False)
        self.amplitude_plot_tab.correction_checkbox.blockSignals(False)

    def recalibrate_sensor(self):
        """Stop, drop sensor calibration + offset + background. Start redoes the first two;
        the background must be taken again in open air."""
        if self.live_thread is None and self.data_logging_tab.is_recording:
            self.global_status.setText("Stop the Data Logging recording before recalibrating.")
            return
        self.stop_live_measurements()
        self.sensor_manager.reset_calibration()
        self._discard_background()
        msg = ("Calibration cleared. Press Start Measurement (sensor + offset are recalibrated), "
               "then point at open air and Calibrate Background again.")
        self.global_status.setText(msg)
        self.amplitude_plot_tab.set_calibration_status(msg)

    def _on_background_ready(self, background_iq, temperature):
        self.background_iq = np.array(background_iq, dtype=np.complex128, copy=True)
        self.background_temp = temperature
        self.background_signature = self.session_signature
        self.data_logging_tab.set_background_calibration(self.background_iq, self.background_signature)
        # Calibration is global: switch correction on for the plots and the features.
        if not self.amplitude_plot_tab.correction_checkbox.isChecked():
            self.amplitude_plot_tab.correction_checkbox.setChecked(True)

    def _on_calibration_status(self, message):
        self.amplitude_plot_tab.set_calibration_status(message)
        if message.startswith("Background calibrated from") or message.startswith("Correction"):
            self.amplitude_plot_tab.calibrate_button.setEnabled(True)

    def _on_health(self, health):
        self.health_label.setText(health.text)
        self.health_label.setToolTip(health.detail)
        self.health_label.setProperty("state", health.state)
        self.health_label.style().unpolish(self.health_label)
        self.health_label.style().polish(self.health_label)

    def calibrate_background(self):
        """Ask the active acquisition thread to collect a reference background."""
        if self.live_thread is None or not self.live_thread.isRunning():
            self.amplitude_plot_tab.set_calibration_status(
                "Start Measurement before calibrating the background."
            )
            return
        self.amplitude_plot_tab.calibrate_button.setEnabled(False)
        self.live_thread.request_background_calibration(30)

    def set_background_correction(self, enabled):
        """Enable/disable complex-IQ background subtraction in the live thread."""
        if self.live_thread is None or not self.live_thread.isRunning():
            if enabled:
                self.amplitude_plot_tab.set_calibration_status(
                    "Start Measurement and calibrate a background before enabling correction."
                )
                self.amplitude_plot_tab.correction_checkbox.blockSignals(True)
                self.amplitude_plot_tab.correction_checkbox.setChecked(False)
                self.amplitude_plot_tab.correction_checkbox.blockSignals(False)
            return
        self.live_thread.set_correction_enabled(enabled)

    def _on_live_features(self, features, distances_m, raw_amplitude, corrected_amplitude, correction_enabled, raw_phase, corrected_phase, energy_profile, corrected_energy_profile, phase_at_peak, phase_std, peak_distance):
        # Every future live graph should subscribe here rather than the
        # Data Logging tab. This is the single live data fan-out point.
        self.live_tab.update_features(features)
        self.amplitude_plot_tab.update_plot(
            distances_m, raw_amplitude, corrected_amplitude, correction_enabled
        )
        self.phase_tab.update_data(
            distances_m, raw_phase, corrected_phase, phase_at_peak, phase_std,
            peak_distance, correction_enabled, self.sensor_manager.params.use_loopback_reference
        )
        self.energy_tab.update_data(distances_m, energy_profile, corrected_energy_profile, correction_enabled)

    def _on_live_error(self, message):
        self.global_status.setText(f"Live scan error: {message}")
        self.live_tab.set_scanning(False, f"Error: {message}")
        self.stop_live_measurements()

    def _on_live_thread_finished(self):
        # The explicit stop path owns cleanup; this only handles natural exit.
        if self.live_thread is not None and not self.live_thread.isRunning():
            self.start_measurements_btn.setEnabled(True)
            self.stop_measurements_btn.setEnabled(False)

    def stop_live_measurements(self):
        thread = self.live_thread
        self.live_thread = None

        if thread is not None:
            thread.stop()
            thread.wait(1500)
            thread.deleteLater()

        try:
            if self.sensor_manager.client is not None and self.sensor_manager.client.session_is_started:
                self.sensor_manager.client.stop_session()
        except Exception:
            pass

        self.start_measurements_btn.setEnabled(True)
        self.stop_measurements_btn.setEnabled(False)
        self.global_status.setText("Live scan stopped")
        self.health_label.setText("")
        self.live_tab.set_scanning(False)

    def closeEvent(self, event):
        self.stop_live_measurements()
        event.accept()


def main():
    app = QApplication(sys.argv)
    avail = app.primaryScreen().availableGeometry()
    small_screen = avail.width() < COMPACT_WIDTH or avail.height() < COMPACT_HEIGHT
    app.setStyleSheet(build_stylesheet(small_screen))
    window = MainWindow()
    window._compact = small_screen
    if "--fullscreen" in sys.argv:
        window.showFullScreen()
    elif small_screen:
        window.showMaximized()  # fill the 5-inch display
    else:
        window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
