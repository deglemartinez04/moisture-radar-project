"""
amplitude_plot.py

Live raw/corrected amplitude comparison and controls for experimental
complex-IQ background subtraction.
"""

from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class AmplitudePlotTab(QWidget):
    def __init__(self):
        super().__init__()
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 20, 24, 20)
        outer.setSpacing(12)

        controls = QHBoxLayout()
        self.calibrate_button = QPushButton("Calibrate Background (30 frames)")
        self.calibrate_button.setObjectName("primaryButton")
        self.correction_checkbox = QCheckBox("Use background correction for live features")
        self.correction_checkbox.setChecked(False)
        controls.addWidget(self.calibrate_button)
        controls.addWidget(self.correction_checkbox)
        controls.addStretch(1)
        outer.addLayout(controls)

        self.status_label = QLabel(
            "Background not calibrated. Start Measurements, then calibrate with the sensor stationary in a clear reference scene."
        )
        self.status_label.setWordWrap(True)
        self.status_label.setObjectName("summaryLabel")
        outer.addWidget(self.status_label)

        plot_box = QGroupBox("Amplitude vs Distance — Raw and Background-Corrected")
        plot_layout = QVBoxLayout(plot_box)

        pg.setConfigOption("background", "#12161c")
        pg.setConfigOption("foreground", "#c7d0dc")

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel("bottom", "Distance", units="m")
        self.plot_widget.setLabel("left", "Amplitude")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.15)
        self.plot_widget.addLegend()
        self.raw_curve = self.plot_widget.plot(
            pen=pg.mkPen(color="#e0a458", width=2), name="Raw amplitude"
        )
        self.corrected_curve = self.plot_widget.plot(
            pen=pg.mkPen(color="#5fb3e8", width=2), name="Corrected amplitude"
        )

        plot_layout.addWidget(self.plot_widget)
        outer.addWidget(plot_box, 1)

    def set_calibration_status(self, message):
        self.status_label.setText(message)

    def update_plot(self, distances_m, raw_amplitude, corrected_amplitude, correction_enabled=False):
        self.raw_curve.setData(distances_m, raw_amplitude)
        self.corrected_curve.setData(distances_m, corrected_amplitude)
        self.corrected_curve.setVisible(bool(correction_enabled))
        self.raw_curve.setVisible(not correction_enabled)
        # During correction, show both curves so its effect remains visible.
        if correction_enabled:
            self.raw_curve.setVisible(True)
            self.corrected_curve.setVisible(True)

    def clear_plot(self):
        self.raw_curve.clear()
        self.corrected_curve.clear()
