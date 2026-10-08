"""
amplitude_plot.py

Dedicated amplitude graph for the global live scan. It receives data only
from MainWindow's live acquisition session, never from Data Logging.
"""

from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtWidgets import QGroupBox, QVBoxLayout, QWidget


class AmplitudePlotTab(QWidget):
    def __init__(self):
        super().__init__()
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 20, 24, 20)
        outer.setSpacing(16)

        plot_box = QGroupBox("Amplitude vs Distance — Global Live Scan")
        plot_layout = QVBoxLayout(plot_box)

        pg.setConfigOption("background", "#12161c")
        pg.setConfigOption("foreground", "#c7d0dc")

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel("bottom", "Distance", units="m")
        self.plot_widget.setLabel("left", "Amplitude")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.15)
        self.curve = self.plot_widget.plot(pen=pg.mkPen(color="#5fb3e8", width=2))

        plot_layout.addWidget(self.plot_widget)
        outer.addWidget(plot_box, 1)

    def update_plot(self, distances_m, amplitude):
        self.curve.setData(distances_m, amplitude)

    def clear_plot(self):
        self.curve.clear()
