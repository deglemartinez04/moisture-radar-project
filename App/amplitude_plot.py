"""
amplitude_plot.py

Live amplitude graph and background-calibration controls.

Two traces: raw amplitude (what the sensor reports) and background-corrected amplitude
(air calibration subtracted: removes leakage and lens echoes).

Optional third trace: |current - reference| of the COMPLEX corrected signal. Record e.g.
drywall alone with "Set reference", then add the stud: the green trace is what the stud
adds. Amplitude alone can hide it (a stud echo out of phase with the drywall echo can even
LOWER the blue curve), but the complex difference shows it. Amplitude is in sensor
units (a.u.): the A121 does not output calibrated volts or dBm.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg

from plot_utils import CompactPlotWidget
from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

NOT_CALIBRATED_TEXT = (
    "Background not calibrated. Press Start Measurement, then calibrate with the sensor "
    "stationary in a clear reference scene."
)


class AmplitudePlotTab(QWidget):
    def __init__(self):
        super().__init__()
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(6)

        controls = QHBoxLayout()
        controls.setSpacing(12)
        self.calibrate_button = QPushButton("Calibrate Background (30 frames)")
        self.calibrate_button.setObjectName("primaryButton")
        self.calibrate_button.setToolTip(
            "Air calibration: point the device at open space with the lens and any standoff attached. "
            "Removes leakage and lens echoes."
        )
        self.correction_checkbox = QCheckBox("Use background correction for live features")
        self.correction_checkbox.setChecked(False)
        controls.addWidget(self.calibrate_button)
        controls.addWidget(self.correction_checkbox)
        self.set_reference_button = QPushButton("Set reference (current scene)")
        self.set_reference_button.setToolTip(
            "Store the current corrected complex profile (e.g. drywall alone). The green trace "
            "then shows |current - reference|: what changed, including phase."
        )
        self.clear_reference_button = QPushButton("Clear reference")
        controls.addWidget(self.set_reference_button)
        controls.addWidget(self.clear_reference_button)
        controls.addStretch(1)
        outer.addLayout(controls)

        self.status_label = QLabel(NOT_CALIBRATED_TEXT)
        self.status_label.setWordWrap(True)
        self.status_label.setObjectName("summaryLabel")
        outer.addWidget(self.status_label)

        plot_box = QGroupBox("Amplitude vs Distance")
        plot_layout = QVBoxLayout(plot_box)
        plot_layout.setContentsMargins(6, 6, 6, 6)

        pg.setConfigOption("background", "#12161c")
        pg.setConfigOption("foreground", "#c7d0dc")

        self.plot_widget = CompactPlotWidget()
        self.plot_widget.setMinimumHeight(120)
        self.plot_widget.setLabel("bottom", "Distance", units="m")
        self.plot_widget.setLabel("left", "Amplitude (a.u.)")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.15)
        self.plot_widget.addLegend()
        self.raw_curve = self.plot_widget.plot(
            pen=pg.mkPen(color="#e0a458", width=2), name="Raw amplitude"
        )
        self.corrected_curve = self.plot_widget.plot(
            pen=pg.mkPen(color="#5fb3e8", width=2), name="Corrected amplitude"
        )

        self.difference_curve = self.plot_widget.plot(
            pen=pg.mkPen(color="#7fd88f", width=2), name="|current − reference|"
        )
        self.difference_curve.setVisible(False)
        self._latest_profile = None
        self._reference = None
        self.set_reference_button.clicked.connect(self._set_reference)
        self.clear_reference_button.clicked.connect(self._clear_reference)

        plot_layout.addWidget(self.plot_widget)
        outer.addWidget(plot_box, 1)

    def set_calibration_status(self, message):
        self.status_label.setText(message)

    def update_plot(self, distances_m, raw_amplitude, corrected_amplitude, correction_enabled=False):
        self.raw_curve.setData(distances_m, raw_amplitude)
        self.corrected_curve.setData(distances_m, corrected_amplitude)
        self.corrected_curve.setVisible(bool(correction_enabled))

    def update_complex_profile(self, distances_m, profile):
        """Latest complex corrected profile (coherent mean over the live window)."""
        self._latest_profile = np.asarray(profile, dtype=complex)
        ref = self._reference
        if ref is not None and ref.shape == self._latest_profile.shape:
            self.difference_curve.setData(distances_m, np.abs(self._latest_profile - ref))
            self.difference_curve.setVisible(True)
        else:
            self.difference_curve.setVisible(False)

    def _set_reference(self):
        if self._latest_profile is None:
            self.set_calibration_status("Start Measurement first; the reference is the current scene.")
            return
        self._reference = self._latest_profile.copy()
        self.set_calibration_status(
            "Reference stored. Change ONE thing (add the stud, wet it, ...) and watch the green trace."
        )

    def _clear_reference(self):
        self._reference = None
        self.difference_curve.setVisible(False)

    def clear_plot(self):
        self.raw_curve.clear()
        self.corrected_curve.clear()
