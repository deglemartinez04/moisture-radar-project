"""
ui_theme.py

Explicit, platform-independent styling for scroll bars and spin boxes.

Why this exists: once a Qt stylesheet styles a widget's border/padding, Qt no longer
draws the native arrow buttons for it. Without explicit rules for the sub-controls
(`::up-button`, `::sub-line`, ...) the up and down buttons end up unsized, stacked on
top of each other or drawn where they cannot be clicked - which showed up as "only the
down button works". Here every button gets an explicit position, size and a visible
icon, so what you see is what you can press. Sizes are touch-friendly.

Icons are small PNGs drawn at start-up into a temp folder (no image files to ship).
"""

from __future__ import annotations

import os
import tempfile

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter, QPolygonF, QPen

ICON_NORMAL = "#c7d0dc"
ICON_DISABLED = "#4a5568"
_ICON_SIZE = 20


def touch_size_px(target_mm: float = 4.0, lo: int = 22, hi: int = 32) -> int:
    """Size of a finger-sized control in pixels, from the real screen DPI (clamped)."""
    override = os.environ.get("RADAR_TOUCH_PX")   # manual override, e.g. RADAR_TOUCH_PX=30
    if override and override.isdigit():
        return int(max(lo, min(hi, int(override))))
    px = lo
    screen = QGuiApplication.primaryScreen()
    if screen is not None:
        phys = screen.physicalSize()
        geo = screen.geometry()
        if phys.width() > 0 and geo.width() > 0:
            px = round(target_mm * geo.width() / phys.width())
    return int(max(lo, min(hi, px)))


def _draw_icon(path: str, kind: str, color: str) -> None:
    img = QImage(_ICON_SIZE, _ICON_SIZE, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    c = QColor(color)
    s = float(_ICON_SIZE)
    if kind in ("up", "down", "left", "right"):
        a, b = 0.28 * s, 0.72 * s          # base span, tip position
        m = 0.5 * s
        pts = {
            "up": [QPointF(a, b), QPointF(s - a, b), QPointF(m, a)],
            "down": [QPointF(a, a), QPointF(s - a, a), QPointF(m, b)],
            "left": [QPointF(b, a), QPointF(b, s - a), QPointF(a, m)],
            "right": [QPointF(a, a), QPointF(a, s - a), QPointF(b, m)],
        }[kind]
        p.setPen(Qt.NoPen)
        p.setBrush(c)
        p.drawPolygon(QPolygonF(pts))
    else:  # "plus" / "minus"
        pen = QPen(c)
        pen.setWidthF(2.4)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        lo, hi = 0.28 * s, 0.72 * s
        p.drawLine(QPointF(lo, 0.5 * s), QPointF(hi, 0.5 * s))
        if kind == "plus":
            p.drawLine(QPointF(0.5 * s, lo), QPointF(0.5 * s, hi))
    p.end()
    img.save(path, "PNG")


def ensure_icons() -> dict:
    """Create the icon PNGs (idempotent) and return {name: url-safe path}."""
    folder = os.path.join(tempfile.gettempdir(), "radar_moisture_icons")
    os.makedirs(folder, exist_ok=True)
    icons = {}
    for kind in ("up", "down", "left", "right", "plus", "minus"):
        for tag, color in (("", ICON_NORMAL), ("_off", ICON_DISABLED)):
            path = os.path.join(folder, f"{kind}{tag}.png")
            _draw_icon(path, kind, color)
            icons[kind + tag] = path.replace("\\", "/")
    return icons


def control_qss(touch_px: int | None = None) -> str:
    """Stylesheet text for QScrollBar, QSpinBox and QDoubleSpinBox."""
    t = touch_px or touch_size_px()
    ic = ensure_icons()
    sb = t                      # scroll bar thickness / arrow button size
    bw = t                      # spin box +/- button width
    return f"""
/* ---- scroll bars: explicit arrow buttons at both ends ---- */
QScrollBar:vertical {{ background: #12161c; width: {sb}px; margin: {sb}px 0 {sb}px 0; border: none; }}
QScrollBar:horizontal {{ background: #12161c; height: {sb}px; margin: 0 {sb}px 0 {sb}px; border: none; }}
QScrollBar::handle:vertical {{ background: #3d4a5f; min-height: {2 * sb}px; border-radius: 4px; margin: 2px 3px; }}
QScrollBar::handle:horizontal {{ background: #3d4a5f; min-width: {2 * sb}px; border-radius: 4px; margin: 3px 2px; }}
QScrollBar::handle:hover {{ background: #5a6a85; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}
QScrollBar::sub-line:vertical {{ subcontrol-origin: margin; subcontrol-position: top; width: {sb}px; height: {sb}px;
    background: #1c2330; border: none; image: url({ic['up']}); }}
QScrollBar::add-line:vertical {{ subcontrol-origin: margin; subcontrol-position: bottom; width: {sb}px; height: {sb}px;
    background: #1c2330; border: none; image: url({ic['down']}); }}
QScrollBar::sub-line:horizontal {{ subcontrol-origin: margin; subcontrol-position: left; width: {sb}px; height: {sb}px;
    background: #1c2330; border: none; image: url({ic['left']}); }}
QScrollBar::add-line:horizontal {{ subcontrol-origin: margin; subcontrol-position: right; width: {sb}px; height: {sb}px;
    background: #1c2330; border: none; image: url({ic['right']}); }}
QScrollBar::sub-line:hover, QScrollBar::add-line:hover {{ background: #232b3a; }}
QScrollBar::sub-line:pressed, QScrollBar::add-line:pressed {{ background: #2d3a52; }}

/* ---- spin boxes: [ - ]  value  [ + ], full-height finger-sized buttons ---- */
QSpinBox, QDoubleSpinBox {{ padding: 0 {bw + 2}px 0 {bw + 2}px; min-height: {t}px; max-height: {t}px;
    qproperty-alignment: AlignCenter; }}
QSpinBox::down-button, QDoubleSpinBox::down-button {{ subcontrol-origin: border; subcontrol-position: left; width: {bw}px; height: {t + 2}px;
    background: #1c2330; border: none; border-right: 1px solid #2d3748;
    border-top-left-radius: 4px; border-bottom-left-radius: 4px; image: url({ic['minus']}); }}
QSpinBox::up-button, QDoubleSpinBox::up-button {{ subcontrol-origin: border; subcontrol-position: right; width: {bw}px; height: {t + 2}px;
    background: #1c2330; border: none; border-left: 1px solid #2d3748;
    border-top-right-radius: 4px; border-bottom-right-radius: 4px; image: url({ic['plus']}); }}
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover,
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {{ background: #232b3a; }}
QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed,
QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed {{ background: #2d3a52; }}
QSpinBox::down-button:disabled, QDoubleSpinBox::down-button:disabled {{ image: url({ic['minus_off']}); background: #12161c; }}
QSpinBox::up-button:disabled, QDoubleSpinBox::up-button:disabled {{ image: url({ic['plus_off']}); background: #12161c; }}
QSpinBox::down-button:off, QDoubleSpinBox::down-button:off {{ image: url({ic['minus_off']}); }}
QSpinBox::up-button:off, QDoubleSpinBox::up-button:off {{ image: url({ic['plus_off']}); }}
"""
