"""
main.py

Entry point for the Radar Moisture Tomography app.

The MainWindow owns the one global live-scan acquisition session. The
Amplitude Plot, Live dashboard, and future live graphs can subscribe to that
same feature stream. Data Logging remains a separate recording workflow.
"""

from __future__ import annotations

import sys
from collections import deque
import time

import numpy as np
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from sensor_manager import SensorManager
from settings_tab import SettingsTab
from record_tab import RecordTab
from live_tab import LiveTab, ROLLING_WINDOW_SIZE, PRINT_INTERVAL_MS, extract_features_from_window
from amplitude_plot import AmplitudePlotTab


class LiveAcquisitionThread(QThread):
    """Global live acquisition thread shared by all live visualizations."""

    features_ready = Signal(dict, object, object)
    error = Signal(str)

    def __init__(self, client, distances_m):
        super().__init__()
        self.client = client
        self.distances_m = distances_m
        self._running = True

    def run(self):
        amplitude_buffer = deque(maxlen=ROLLING_WINDOW_SIZE)
        phase_buffer = deque(maxlen=ROLLING_WINDOW_SIZE)
        last_emit = 0.0

        try:
            while self._running:
                result = self.client.get_next()
                frame = result.frame
                single_sweep = frame[0] if frame.shape[0] == 1 else np.mean(frame, axis=0)

                amplitude_buffer.append(np.abs(single_sweep))
                phase_buffer.append(np.angle(single_sweep))

                now = time.time()
                if now - last_emit >= (PRINT_INTERVAL_MS / 1000.0) and len(amplitude_buffer) >= 3:
                    features = extract_features_from_window(
                        np.array(amplitude_buffer),
                        np.array(phase_buffer),
                        self.distances_m,
                    )
                    self.features_ready.emit(
                        features,
                        self.distances_m,
                        features["mean_amplitude_per_point"],
                    )
                    last_emit = now
        except Exception as exc:
            if self._running:
                self.error.emit(str(exc))

    def stop(self):
        self._running = False


STYLESHEET = """
QWidget {
    background-color: #0d1117;
    color: #c7d0dc;
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 13px;
}
QMainWindow { background-color: #0d1117; }
QTabWidget::pane { border: none; background-color: #0d1117; }
QTabBar::tab {
    background-color: #12161c;
    color: #8b96a5;
    padding: 10px 22px;
    border: none;
    font-size: 13px;
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
    margin-top: 14px;
    padding-top: 12px;
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
    padding: 7px 16px;
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
    padding: 8px 18px;
}
QPushButton#globalStartButton:hover { background-color: #2d9563; }
QPushButton#globalStopButton {
    background-color: #7d3038;
    border: 1px solid #a7444e;
    color: #eef2f7;
    font-weight: 700;
    padding: 8px 18px;
}
QPushButton#globalStopButton:hover { background-color: #963b45; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #0d1117;
    border: 1px solid #2d3748;
    border-radius: 4px;
    padding: 5px 8px;
    color: #eef2f7;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border-color: #5fb3e8; }
QCheckBox { spacing: 8px; color: #c7d0dc; padding: 3px 0; }
QLabel#summaryLabel { color: #8b96a5; font-family: "Cascadia Code", "Consolas", monospace; }
QLabel#featureName { color: #6b7686; font-size: 11px; }
QLabel#featureValue { color: #eef2f7; font-family: "Cascadia Code", "Consolas", monospace; font-size: 16px; font-weight: 600; }
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Radar Moisture Tomography — A121 Console")
        self.resize(1180, 820)

        self.sensor_manager = SensorManager()
        self.live_thread: LiveAcquisitionThread | None = None

        self._build_ui()

    def _build_ui(self):
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(12, 10, 12, 0)
        root_layout.setSpacing(8)

        header = QHBoxLayout()
        header.addStretch(1)

        self.global_status = QLabel("Live scan idle")
        self.global_status.setObjectName("summaryLabel")
        header.addWidget(self.global_status)

        self.start_measurements_btn = QPushButton("Start Measurements")
        self.start_measurements_btn.setObjectName("globalStartButton")
        self.stop_measurements_btn = QPushButton("Stop")
        self.stop_measurements_btn.setObjectName("globalStopButton")
        self.stop_measurements_btn.setEnabled(False)

        header.addWidget(self.start_measurements_btn)
        header.addWidget(self.stop_measurements_btn)
        root_layout.addLayout(header)

        tabs = QTabWidget()
        tabs.setDocumentMode(True)

        self.settings_tab = SettingsTab(self.sensor_manager)
        self.data_logging_tab = RecordTab(self.sensor_manager)
        self.live_tab = LiveTab(self.sensor_manager)
        self.amplitude_plot_tab = AmplitudePlotTab()

        tabs.addTab(self.settings_tab, "Settings")
        tabs.addTab(self.data_logging_tab, "Data Logging")
        tabs.addTab(self.amplitude_plot_tab, "Amplitude Plot")
        tabs.addTab(self.live_tab, "Moisture Level")

        root_layout.addWidget(tabs, 1)
        self.setCentralWidget(root)

        self.start_measurements_btn.clicked.connect(self.start_live_measurements)
        self.stop_measurements_btn.clicked.connect(self.stop_live_measurements)

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
            self.sensor_manager.client.setup_session(sensor_config)
            self.sensor_manager.client.start_session()

            distances_m = self.sensor_manager.params.distances_m()
            self.live_thread = LiveAcquisitionThread(self.sensor_manager.client, distances_m)
            self.live_thread.features_ready.connect(self._on_live_features)
            self.live_thread.error.connect(self._on_live_error)
            self.live_thread.finished.connect(self._on_live_thread_finished)
            self.live_thread.start()

            self.start_measurements_btn.setEnabled(False)
            self.stop_measurements_btn.setEnabled(True)
            self.global_status.setText("Live measurements running")
            self.live_tab.set_scanning(True)
        except Exception as exc:
            self.global_status.setText(f"Failed to start: {exc}")
            self.live_tab.set_scanning(False, f"Failed to start: {exc}")
            try:
                if self.sensor_manager.client is not None and self.sensor_manager.client.session_is_started:
                    self.sensor_manager.client.stop_session()
            except Exception:
                pass

    def _on_live_features(self, features, distances_m, amplitude):
        # Every future live graph should subscribe here rather than the
        # Data Logging tab. This is the single live data fan-out point.
        self.live_tab.update_features(features)
        self.amplitude_plot_tab.update_plot(distances_m, amplitude)

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
        self.live_tab.set_scanning(False)

    def closeEvent(self, event):
        self.stop_live_measurements()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
