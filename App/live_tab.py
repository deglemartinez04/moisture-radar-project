"""
live_tab.py

Live moisture dashboard. The MainWindow owns the global live acquisition
session and sends extracted features here. This tab no longer starts/stops
the sensor and contains no live graphs; it is the placeholder ML dashboard.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QPainter, QPen
from PySide6.QtWidgets import QGridLayout, QGroupBox, QLabel, QSizePolicy, QVBoxLayout, QWidget

from features import extract_features


ROLLING_WINDOW_SIZE = 10
PRINT_INTERVAL_MS = 100


def extract_features_from_window(amplitude_window, phase_window, distances_m, background_corrected=False,
                                 noise_level=None):
    """Thin wrapper. ALL feature math lives in features.py, which the Data Logging tab also
    uses, so live features and the rows written to data_log.csv are computed identically."""
    return extract_features(
        amplitude_window, phase_window, distances_m, background_corrected=background_corrected,
        noise_level=noise_level,
    )


def run_placeholder_model(features: dict) -> tuple[float, float]:
    """
    Placeholder only. Returns (moisture_percent, confidence_percent).

    Moisture is intentionally derived from a sensor amplitude feature so the
    dashboard visibly changes during testing. It is NOT a real moisture model.
    Confidence is based on signal stability and is also only a visual placeholder.
    """
    amplitude = float(features["peak_amplitude_mean"])
    variability = float(np.sqrt(max(float(features["frame_to_frame_variance"]), 0.0)))

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
        self.setMinimumSize(100, 100)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setObjectName("circularPercentage")

    # Preferred size is the original 230 px; the widget shrinks to fit small displays.
    def sizeHint(self):
        return QSize(200, 200)

    def minimumSizeHint(self):
        return QSize(100, 100)

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
        # Everything scales with the widget so the dial looks identical to the
        # original at 230 px and still fits on a 5-inch display when smaller.
        pen_w = max(7, round(side * 0.078))
        margin = pen_w
        rect = self.rect().adjusted(
            (self.width() - side) // 2 + margin,
            (self.height() - side) // 2 + margin,
            -(self.width() - side) // 2 - margin,
            -(self.height() - side) // 2 - margin,
        )

        track_pen = QPen(Qt.GlobalColor.black)
        track_pen.setWidth(pen_w)
        track_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(track_pen)
        painter.drawArc(rect, 90 * 16, -360 * 16)

        fill_pen = QPen(self.fill_color)
        fill_pen.setWidth(pen_w)
        fill_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(fill_pen)
        painter.drawArc(rect, 90 * 16, int(-360 * 16 * self.value / 100.0))

        painter.setPen(Qt.GlobalColor.white)
        value_font = painter.font()
        value_font.setPointSizeF(max(9.0, side * 0.117))
        value_font.setBold(True)
        painter.setFont(value_font)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, f"{self.value:.1f}%")

        title_font = painter.font()
        title_font.setPointSizeF(max(7.0, side * 0.048))
        title_font.setBold(False)
        painter.setFont(title_font)
        painter.drawText(
            rect.adjusted(0, round(side * 0.21), 0, 0),
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
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(6)

        # Plain label (no group box) saves vertical space on small displays.
        self.status_label = QLabel("Idle — use Start Measurement in the top-right corner.")
        self.status_label.setObjectName("summaryLabel")
        self.status_label.setWordWrap(True)
        outer.addWidget(self.status_label)

        results_box = QGroupBox("Live Prediction  — placeholder until the ML model is trained")
        results_layout = QGridLayout(results_box)
        results_layout.setContentsMargins(6, 6, 6, 6)
        results_layout.setColumnStretch(0, 1)
        results_layout.setColumnStretch(1, 1)

        self.confidence_indicator = CircularPercentage("Confidence", "#5fb3e8")
        self.moisture_indicator = CircularPercentage("Moisture Content", "#7fd88f")
        # Dry / Damp / Saturated label above the two dials (same classes as Data Logging).
        self.class_label = QLabel("—")
        self.class_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        results_layout.addWidget(self.class_label, 0, 0, 1, 2)
        results_layout.addWidget(self.confidence_indicator, 1, 0, alignment=Qt.AlignmentFlag.AlignCenter)
        results_layout.addWidget(self.moisture_indicator, 1, 1, alignment=Qt.AlignmentFlag.AlignCenter)
        self._set_class_label(None)
        outer.addWidget(results_box, 1)

        features_box = QGroupBox("Extracted Features")
        feat_grid = QGridLayout(features_box)
        feat_grid.setContentsMargins(6, 6, 6, 6)
        feat_grid.setSpacing(4)
        self.feature_labels = {}
        # The same primary features the Data Logging tab writes to data_log.csv.
        feature_names = [
            "front_distance_m", "front_amplitude", "peak1_snr_db",
            "rear_energy_ratio_db", "rear_centroid_offset_m", "rear_phase_rad",
            "rear_energy_ratio_db_std", "peak2_distance_m", "peaks_found",
        ]
        for i, name in enumerate(feature_names):
            row, col = divmod(i, 3)
            title = QLabel(name)
            title.setObjectName("featureName")
            value = QLabel("—")
            value.setObjectName("featureValue")
            cell = QVBoxLayout()
            cell.setContentsMargins(0, 0, 0, 0)
            cell.setSpacing(0)
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

    @staticmethod
    def moisture_class(moisture_pct: float) -> str:
        """Same three classes as the Data Logging 'Moisture level' choices."""
        if moisture_pct < 16.0:
            return "Dry"
        if moisture_pct < 30.0:
            return "Damp"
        return "Saturated"

    def _set_class_label(self, moisture_pct):
        if moisture_pct is None:
            self.class_label.setText("—")
            self.class_label.setStyleSheet("color: #6b7686; font-size: 18px; font-weight: 700;")
        else:
            self.class_label.setText(self.moisture_class(moisture_pct))
            self.class_label.setStyleSheet(
                f"color: {self.moisture_color(moisture_pct)}; font-size: 18px; font-weight: 700;"
            )

    def update_features(self, features: dict):
        """Receive the latest feature set from the global live acquisition."""
        for name, label_widget in self.feature_labels.items():
            val = features.get(name)
            if val is None or (isinstance(val, (float, np.floating)) and np.isnan(val)):
                label_widget.setText("—")   # not computable (e.g. fewer than two peaks), never shown as 0
            else:
                label_widget.setText(f"{val:.4g}" if isinstance(val, (float, np.floating)) else str(val))

        moisture_pct, confidence_pct = run_placeholder_model(features)
        self.confidence_indicator.set_value(confidence_pct, "#5fb3e8")
        self.moisture_indicator.set_value(moisture_pct, self.moisture_color(moisture_pct))
        self._set_class_label(moisture_pct)

    def set_scanning(self, scanning: bool, message: str | None = None):
        if message:
            self.status_label.setText(message)
        else:
            self.status_label.setText("Scanning…" if scanning else "Idle — use Start Measurement in the top-right corner.")

    def reset_results(self):
        self.confidence_indicator.set_value(0.0, "#5fb3e8")
        self.moisture_indicator.set_value(0.0, "#5fb3e8")
        self._set_class_label(None)
        for label in self.feature_labels.values():
            label.setText("—")
