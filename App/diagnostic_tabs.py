"""Live phase and energy diagnostics for the shared A121 acquisition stream."""
from collections import deque
import numpy as np
import pyqtgraph as pg
from plot_utils import CompactPlotWidget
from PySide6.QtWidgets import QCheckBox, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QGroupBox


class EnergyTab(QWidget):
    """Live squared-IQ energy profile, raw and background-corrected, with scaled summary metrics."""
    ENERGY_SCALE = 1_000_000.0
    RAW_COLOR = "#e0a458"        # same colours as the Amplitude tab
    CORRECTED_COLOR = "#5fb3e8"

    def __init__(self):
        super().__init__()
        self._last = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)

        options = QHBoxLayout()
        self.log_checkbox = QCheckBox("Log scale (see weak echoes next to strong ones)")
        self.log_checkbox.toggled.connect(self._on_log_toggled)
        options.addWidget(self.log_checkbox)
        options.addStretch(1)
        layout.addLayout(options)

        box = QGroupBox("Radar Energy vs Distance")
        box_layout = QVBoxLayout(box)
        box_layout.setContentsMargins(6, 6, 6, 6)
        self.plot = CompactPlotWidget()
        self.plot.setMinimumHeight(110)
        self.plot.setLabel("bottom", "Distance", units="mm")
        self.plot.setLabel("left", "Mean energy (million a.u.)")
        # Units are already scaled to millions; stop pyqtgraph adding its own SI prefix.
        for axis_name in ("left", "bottom"):
            self.plot.getAxis(axis_name).enableAutoSIPrefix(False)
        self.plot.showGrid(x=True, y=True, alpha=0.22)
        self.plot.setDownsampling(auto=True, mode="peak")
        self.plot.addLegend()
        self.curve = self.plot.plot(pen=pg.mkPen(self.RAW_COLOR, width=2.2), name="Raw |IQ|² energy")
        self.corrected_curve = self.plot.plot(
            pen=pg.mkPen(self.CORRECTED_COLOR, width=2.2), name="Corrected |IQ|² energy"
        )
        self.corrected_curve.setVisible(False)
        self.peak_marker = pg.ScatterPlotItem(
            size=10, pen=pg.mkPen(self.RAW_COLOR, width=1.5), brush=pg.mkBrush(self.RAW_COLOR), symbol="o"
        )
        self.corrected_peak_marker = pg.ScatterPlotItem(
            size=10, pen=pg.mkPen(self.CORRECTED_COLOR, width=1.5), brush=pg.mkBrush(self.CORRECTED_COLOR), symbol="o"
        )
        self.plot.addItem(self.peak_marker)
        self.plot.addItem(self.corrected_peak_marker)
        box_layout.addWidget(self.plot)
        layout.addWidget(box, 1)

        # Metrics: one row for the raw trace, one for the corrected trace.
        metrics = QGridLayout()
        metrics.setHorizontalSpacing(14)
        metrics.setVerticalSpacing(2)
        for col, heading in enumerate(["", "Peak energy", "Peak distance", "Integrated energy"]):
            metrics.addWidget(QLabel(heading), 0, col)
        self._raw_metrics = [self._metric("—") for _ in range(3)]
        self._corrected_metrics = [self._metric("—") for _ in range(3)]
        for row, (name, colour, labels) in enumerate([
            ("Raw", self.RAW_COLOR, self._raw_metrics),
            ("Corrected", self.CORRECTED_COLOR, self._corrected_metrics),
        ], start=1):
            tag = QLabel(name)
            tag.setStyleSheet(f"color: {colour}; font-weight: 600;")
            metrics.addWidget(tag, row, 0)
            for col, label in enumerate(labels, start=1):
                metrics.addWidget(label, row, col)
        layout.addLayout(metrics)

        self.summary = QLabel(
            "Energy = |IQ|² (amplitude squared), averaged over the rolling frame window. Corrected "
            "energy is the energy of the signal AFTER subtracting the background IQ - not raw energy "
            "minus background energy. Values are in millions of arbitrary units."
        )
        self.summary.setObjectName("summaryLabel")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

    @staticmethod
    def _metric(value):
        label = QLabel(value)
        label.setObjectName("featureValue")
        return label

    def _on_log_toggled(self, checked):
        self.plot.setLogMode(x=False, y=bool(checked))
        self.plot.setLabel(
            "left", "Mean energy (million a.u.)" if not checked else "log10 mean energy (M a.u.)"
        )
        if self._last is not None:
            self.update_data(*self._last)

    def _marker_y(self, value):
        """Scatter markers are not log-transformed by pyqtgraph, so do it here."""
        if self.log_checkbox.isChecked():
            return float(np.log10(max(value, 1e-12)))
        return float(value)

    @staticmethod
    def _summarize(distances_mm, energy_millions):
        idx = int(np.argmax(energy_millions))
        return idx, energy_millions[idx], distances_mm[idx], float(np.sum(energy_millions))

    def update_data(self, distances_m, energy_profile, corrected_profile=None, correction_enabled=False):
        if len(energy_profile) == 0:
            return
        self._last = (distances_m, energy_profile, corrected_profile, correction_enabled)
        distances_mm = np.asarray(distances_m, dtype=float) * 1000.0
        raw = np.asarray(energy_profile, dtype=float) / self.ENERGY_SCALE
        valid = np.isfinite(distances_mm) & np.isfinite(raw)
        if not np.any(valid):
            return
        d = distances_mm[valid]
        self.curve.setData(d, raw[valid])
        idx, peak, peak_d, total = self._summarize(d, raw[valid])
        self.peak_marker.setData([{"pos": (d[idx], self._marker_y(raw[valid][idx]))}])
        for label, text in zip(self._raw_metrics, (f"{peak:,.3f} M", f"{peak_d:.1f} mm", f"{total:,.3f} M·pts")):
            label.setText(text)

        show_corrected = bool(correction_enabled) and corrected_profile is not None
        self.corrected_curve.setVisible(show_corrected)
        self.corrected_peak_marker.setVisible(show_corrected)
        if show_corrected:
            corr = np.asarray(corrected_profile, dtype=float) / self.ENERGY_SCALE
            ok = np.isfinite(distances_mm) & np.isfinite(corr)
            if np.any(ok):
                dc = distances_mm[ok]
                self.corrected_curve.setData(dc, corr[ok])
                idx, peak, peak_d, total = self._summarize(dc, corr[ok])
                self.corrected_peak_marker.setData([{"pos": (dc[idx], self._marker_y(corr[ok][idx]))}])
                for label, text in zip(self._corrected_metrics,
                                       (f"{peak:,.3f} M", f"{peak_d:.1f} mm", f"{total:,.3f} M·pts")):
                    label.setText(text)
        else:
            for label in self._corrected_metrics:
                label.setText("— (calibrate background)")


class PhaseTab(QWidget):
    HISTORY_LENGTH = 240

    def __init__(self):
        super().__init__()
        self.phase_history = deque(maxlen=self.HISTORY_LENGTH)
        self.time_history = deque(maxlen=self.HISTORY_LENGTH)
        self.elapsed_samples = 0
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(6)
        plots = QHBoxLayout()
        plots.setSpacing(6)
        phase_box = QGroupBox("Phase vs Distance")
        phase_layout = QVBoxLayout(phase_box)
        phase_layout.setContentsMargins(6, 6, 6, 6)
        self.phase_plot = CompactPlotWidget()
        self.phase_plot.setMinimumHeight(110)
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
        history_layout.setContentsMargins(6, 6, 6, 6)
        self.history_plot = CompactPlotWidget()
        self.history_plot.setMinimumHeight(110)
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
            ("Phase-jitter compensation", self.loopback_label),
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
        self.loopback_label.setText("On (loopback reference)" if loopback_enabled else "Off")
        self.elapsed_samples += 1
        self.phase_history.append(float(phase_at_peak))
        self.time_history.append(self.elapsed_samples)
        self.history_curve.setData(list(self.time_history), list(self.phase_history))
