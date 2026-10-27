"""
plot_utils.py

Small shared helper for the live graphs.

pyqtgraph's PlotWidget reports a large default preferred size (640x480), which
makes a tab taller than a 5-inch display and forces scrollbars even though the
plot could simply shrink. CompactPlotWidget reports a small preferred size so
the plot fills whatever space is available instead.
"""

from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtCore import QSize
from PySide6.QtWidgets import QSizePolicy


class CompactPlotWidget(pg.PlotWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def sizeHint(self):
        return QSize(240, 140)

    def minimumSizeHint(self):
        return QSize(120, 100)
