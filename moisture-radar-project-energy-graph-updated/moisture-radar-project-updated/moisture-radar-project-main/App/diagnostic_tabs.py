"""Live phase and energy diagnostics for the shared A121 acquisition stream."""
from collections import deque
import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QGroupBox


class EnergyTab(QWidget):
    """Live squared-IQ energy profile with explicitly scaled summary metrics."""
    ENERGY_SCALE = 1_000_000.0

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        box = QGroupBox("Radar Energy vs Distance")
        box_layout = QVBoxLayout(box)
        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "Distance", units="mm")
        self.plot.setLabel("left", "Mean signal energy", units="million a.u.")
        self.plot.showGrid(x=True, y=True, alpha=0.22)
        self.plot.setDownsampling(auto=True, mode="peak")
        self.curve = self.plot.plot(
            pen=pg.mkPen("#56c2a5", width=2.2), name="Mean |IQ|² energy"
        )
        self.peak_marker = pg.ScatterPlotItem(
            size=10, pen=pg.mkPen("#f0c674", width=1.5),
            brush=pg.mkBrush("#f0c674"), symbol="o"
        )
        self.plot.addItem(self.peak_marker)
        box_layout.addWidget(self.plot)
        layout.addWidget(box, 1)

        metrics = QGridLayout()
        self.peak_energy_label = self._metric("—")
        self.peak_distance_label = self._metric("—")
        self.integrated_energy_label = self._metric("—")
        for col, (heading, label) in enumerate([
            ("Peak energy", self.peak_energy_label),
            ("Peak distance", self.peak_distance_label),
            ("Integrated energy", self.integrated_energy_label),
        ]):
            metrics.addWidget(QLabel(heading), 0, col)
            metrics.addWidget(label, 1, col)
        layout.addLayout(metrics)
        self.summary = QLabel(
            "Energy is calculated as mean |IQ|² across the rolling frame window. "
            "Values are scaled to millions for readability; arbitrary units are not calibrated moisture measurements."
        )
        self.summary.setObjectName("summaryLabel")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

    @staticmethod
    def _metric(value):
        label = QLabel(value)
        label.setObjectName("featureValue")
        return label

    def update_data(self, distances_m, energy_profile):
        if len(energy_profile) == 0:
            return
        distances_mm = np.asarray(distances_m, dtype=float) * 1000.0
        energy_millions = np.asarray(energy_profile, dtype=float) / self.ENERGY_SCALE
        valid = np.isfinite(distances_mm) & np.isfinite(energy_millions)
        if not np.any(valid):
            return
        distances_mm = distances_mm[valid]
        energy_millions = energy_millions[valid]
        self.curve.setData(distances_mm, energy_millions)
        idx = int(np.argmax(energy_millions))
        self.peak_marker.setData([{"pos": (distances_mm[idx], energy_millions[idx])}])
        self.peak_energy_label.setText(f"{energy_millions[idx]:,.3f} M a.u.")
        self.peak_distance_label.setText(f"{distances_mm[idx]:.1f} mm")
        self.integrated_energy_label.setText(f"{np.sum(energy_millions):,.3f} M a.u.·points")


class PhaseTab(QWidget):
    HISTORY_LENGTH = 240

    def __init__(self):
        super().__init__()
        self.phase_history = deque(maxlen=self.HISTORY_LENGTH)
        self.time_history = deque(maxlen=self.HISTORY_LENGTH)
        self.elapsed_samples = 0
        outer = QVBoxLayout(self)
        plots = QHBoxLayout()
        phase_box = QGroupBox("Phase vs Distance")
        phase_layout = QVBoxLayout(phase_box)
        self.phase_plot = pg.PlotWidget()
        self.phase_plot.setLabel("bottom", "Distance", units="m")
        self.phase_plot.setLabel("left", "Phase", units="rad")
        self.phase_plot.setYRange(-np.pi, np.pi, padding=0.05)
        self.phase_plot.showGrid(x=True, y=True, alpha=0.2)
        self.raw_phase_curve = self.phase_plot.plot(pen=pg.mkPen("#e0a458", width=1.5), name="Raw phase")
        self.corrected_phase_curve = self.phase_plot.plot(pen=pg.mkPen("#5fb3e8", width=2), name="Corrected phase")
        self.phase_plot.addLegend()
        phase_layout.addWidget(self.phase_plot)
        history_box = QGroupBox("Phase Stability Over Time")
        history_layout = QVBoxLayout(history_box)
        self.history_plot = pg.PlotWidget()
        self.history_plot.setLabel("bottom", "Recent sample")
        self.history_plot.setLabel("left", "Phase at peak", units="rad")
        self.history_plot.setYRange(-np.pi, np.pi, padding=0.05)
        self.history_plot.showGrid(x=True, y=True, alpha=0.2)
        self.history_curve = self.history_plot.plot(pen=pg.mkPen("#8bd49c", width=2), name="Phase at peak")
        history_layout.addWidget(self.history_plot)
        plots.addWidget(phase_box, 1)
        plots.addWidget(history_box, 1)
        outer.addLayout(plots, 1)

        indicators = QGroupBox("Live Indicators")
        grid = QGridLayout(indicators)
        self.phase_mean_label = self._value_label()
        self.phase_std_label = self._value_label()
        self.range_label = self._value_label()
        self.correction_label = self._value_label()
        self.loopback_label = self._value_label()
        for col, (name, label) in enumerate([
            ("Peak phase", self.phase_mean_label),
            ("Phase variability", self.phase_std_label),
            ("Peak range", self.range_label),
            ("Background correction", self.correction_label),
            ("Loopback reference", self.loopback_label),
        ]):
            grid.addWidget(QLabel(name), 0, col)
            grid.addWidget(label, 1, col)
        outer.addWidget(indicators)
        note = QLabel("Phase is angular and wraps at ±π. Variability is estimated across the current rolling frame window.")
        note.setWordWrap(True)
        note.setObjectName("summaryLabel")
        outer.addWidget(note)

    @staticmethod
    def _value_label():
        label = QLabel("—")
        label.setObjectName("featureValue")
        return label

    def update_data(self, distances_m, raw_phase, corrected_phase, phase_at_peak, phase_std, peak_distance_m, correction_enabled, loopback_enabled):
        self.raw_phase_curve.setData(distances_m, raw_phase)
        self.corrected_phase_curve.setData(distances_m, corrected_phase)
        self.corrected_phase_curve.setVisible(bool(correction_enabled))
        self.phase_mean_label.setText(f"{phase_at_peak:.3f} rad")
        self.phase_std_label.setText(f"{phase_std:.3f} rad")
        self.range_label.setText(f"{peak_distance_m * 1000:.1f} mm")
        self.correction_label.setText("Enabled" if correction_enabled else "Off")
        self.loopback_label.setText("Configured" if loopback_enabled else "Off")
        self.elapsed_samples += 1
        self.phase_history.append(float(phase_at_peak))
        self.time_history.append(self.elapsed_samples)
        self.history_curve.setData(list(self.time_history), list(self.phase_history))
