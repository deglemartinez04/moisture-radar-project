"""
live_tab.py

Live moisture dashboard. The MainWindow owns the global live acquisition
session and sends extracted features here. This tab no longer starts/stops
the sensor and contains no live graphs; it is the placeholder ML dashboard.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPen
from PySide6.QtWidgets import QGridLayout, QGroupBox, QLabel, QVBoxLayout, QWidget


ROLLING_WINDOW_SIZE = 10
PRINT_INTERVAL_MS = 100


def extract_features_from_window(amplitude_window, phase_window, distances_m):
    """Keep the live feature math identical to the training-feature pipeline."""
    mean_amplitude_per_point = np.mean(amplitude_window, axis=0)
    peak_idx = np.argmax(mean_amplitude_per_point)
    peak_distance_m = distances_m[peak_idx]
    peak_amplitude_mean = mean_amplitude_per_point[peak_idx]
    peak_amplitude_std = np.std(amplitude_window[:, peak_idx])

    peak_phase_values = phase_window[:, peak_idx]
    peak_phase_mean = np.angle(np.mean(np.exp(1j * peak_phase_values)))
    peak_phase_std = np.std(
        np.angle(np.exp(1j * (peak_phase_values - peak_phase_mean)))
    )

    safe_amplitude = np.clip(mean_amplitude_per_point, 1e-6, None)
    log_amplitude = np.log(safe_amplitude)
    slope, _ = np.polyfit(distances_m, log_amplitude, 1)

    total_energy = np.sum(mean_amplitude_per_point)
    noise_floor = np.percentile(mean_amplitude_per_point, 10)
    frame_to_frame_variability = np.mean(np.std(amplitude_window, axis=0))

    return {
        "peak_distance_m": peak_distance_m,
        "peak_amplitude_mean": peak_amplitude_mean,
        "peak_amplitude_std": peak_amplitude_std,
        "peak_phase_mean_rad": peak_phase_mean,
        "peak_phase_std_rad": peak_phase_std,
        "attenuation_slope": slope,
        "total_energy": total_energy,
        "noise_floor": noise_floor,
        "frame_to_frame_variability": frame_to_frame_variability,
        "mean_amplitude_per_point": mean_amplitude_per_point,
        "mean_phase_per_point": np.angle(
            np.mean(np.exp(1j * phase_window), axis=0)
        ),
    }


def run_placeholder_model(features: dict) -> tuple[float, float]:
    """
    Placeholder only. Returns (moisture_percent, confidence_percent).

    Moisture is intentionally derived from a sensor amplitude feature so the
    dashboard visibly changes during testing. It is NOT a real moisture model.
    Confidence is based on signal stability and is also only a visual placeholder.
    """
    amplitude = float(features["peak_amplitude_mean"])
    variability = float(features["frame_to_frame_variability"])

    # Keep the existing amplitude-based placeholder behavior, but present it
    # as a continuous moisture percentage instead of a categorical label.
    moisture_pct = float(np.clip((amplitude / 6000.0) * 100.0, 0.0, 100.0))

    # Lower frame-to-frame variation gives the future model a visually intuitive
    # "more confident" placeholder. Clamp it so the display remains useful.
    relative_variability = variability / max(abs(amplitude), 1.0)
    confidence_pct = float(np.clip(100.0 - relative_variability * 250.0, 0.0, 100.0))

    return moisture_pct, confidence_pct


class CircularPercentage(QWidget):
    """Circular percentage indicator with a colored filled arc."""

    def __init__(self, title: str, neutral_color: str = "#5fb3e8"):
        super().__init__()
        self.title = title
        self.value = 0.0
        self.fill_color = neutral_color
        self.setMinimumSize(230, 230)
        self.setObjectName("circularPercentage")

    def set_value(self, value: float, color: str | None = None):
        self.value = float(np.clip(value, 0.0, 100.0))
        if color is not None:
            self.fill_color = color
        self.update()

    def paintEvent(self, event):
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        side = min(self.width(), self.height())
        rect = self.rect().adjusted(
            (self.width() - side) // 2 + 18,
            (self.height() - side) // 2 + 18,
            -(self.width() - side) // 2 - 18,
            -(self.height() - side) // 2 - 18,
        )

        track_pen = QPen(Qt.GlobalColor.black)
        track_pen.setWidth(18)
        track_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(track_pen)
        painter.drawArc(rect, 90 * 16, -360 * 16)

        fill_pen = QPen(self.fill_color)
        fill_pen.setWidth(18)
        fill_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(fill_pen)
        painter.drawArc(rect, 90 * 16, int(-360 * 16 * self.value / 100.0))

        painter.setPen(Qt.GlobalColor.white)
        value_font = painter.font()
        value_font.setPointSize(27)
        value_font.setBold(True)
        painter.setFont(value_font)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, f"{self.value:.1f}%")

        title_font = painter.font()
        title_font.setPointSize(11)
        title_font.setBold(False)
        painter.setFont(title_font)
        painter.drawText(
            rect.adjusted(0, 48, 0, 0),
            Qt.AlignmentFlag.AlignCenter,
            self.title,
        )


class LiveTab(QWidget):
    """Dashboard fed by the global live scan in MainWindow."""

    def __init__(self, sensor_manager):
        super().__init__()
        self.mgr = sensor_manager
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 20, 24, 20)
        outer.setSpacing(16)

        status_box = QGroupBox("Live Scan Status")
        status_layout = QVBoxLayout(status_box)
        self.status_label = QLabel("Idle — use Start Measurements in the top-right corner.")
        self.status_label.setObjectName("summaryLabel")
        status_layout.addWidget(self.status_label)
        outer.addWidget(status_box)

        results_box = QGroupBox("Live Prediction  — placeholder until the ML model is trained")
        results_layout = QGridLayout(results_box)
        results_layout.setColumnStretch(0, 1)
        results_layout.setColumnStretch(1, 1)

        self.confidence_indicator = CircularPercentage("Confidence", "#5fb3e8")
        self.moisture_indicator = CircularPercentage("Moisture Content", "#7fd88f")
        results_layout.addWidget(self.confidence_indicator, 0, 0, alignment=Qt.AlignmentFlag.AlignCenter)
        results_layout.addWidget(self.moisture_indicator, 0, 1, alignment=Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(results_box, 1)

        features_box = QGroupBox("Extracted Features")
        feat_grid = QGridLayout(features_box)
        self.feature_labels = {}
        feature_names = [
            "peak_distance_m", "peak_amplitude_mean", "peak_amplitude_std",
            "peak_phase_mean_rad", "peak_phase_std_rad", "attenuation_slope",
            "total_energy", "noise_floor", "frame_to_frame_variability",
        ]
        for i, name in enumerate(feature_names):
            row, col = divmod(i, 3)
            title = QLabel(name)
            title.setObjectName("featureName")
            value = QLabel("—")
            value.setObjectName("featureValue")
            cell = QVBoxLayout()
            cell.addWidget(title)
            cell.addWidget(value)
            wrapper = QWidget()
            wrapper.setLayout(cell)
            feat_grid.addWidget(wrapper, row, col)
            self.feature_labels[name] = value
        outer.addWidget(features_box)

    @staticmethod
    def moisture_color(moisture_pct: float) -> str:
        """Map the requested moisture ranges to visual colors."""
        if moisture_pct < 16.0:
            return "#5fb3e8"  # dry / below elevated-moisture range
        if moisture_pct < 20.0:
            return "#7fd88f"  # elevated / green
        if moisture_pct < 30.0:
            return "#e5c85c"  # increasingly wet / yellow
        return "#e56b6f"  # saturation and above / red

    def update_features(self, features: dict):
        """Receive the latest feature set from the global live acquisition."""
        for name, label_widget in self.feature_labels.items():
            val = features.get(name)
            if val is not None:
                label_widget.setText(f"{val:.4f}" if isinstance(val, (float, np.floating)) else str(val))

        moisture_pct, confidence_pct = run_placeholder_model(features)
        self.confidence_indicator.set_value(confidence_pct, "#5fb3e8")
        self.moisture_indicator.set_value(moisture_pct, self.moisture_color(moisture_pct))

    def set_scanning(self, scanning: bool, message: str | None = None):
        if message:
            self.status_label.setText(message)
        else:
            self.status_label.setText("Scanning…" if scanning else "Idle — use Start Measurements in the top-right corner.")

    def reset_results(self):
        self.confidence_indicator.set_value(0.0, "#5fb3e8")
        self.moisture_indicator.set_value(0.0, "#5fb3e8")
        for label in self.feature_labels.values():
            label.setText("—")
