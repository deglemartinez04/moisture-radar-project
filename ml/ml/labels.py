"""Moisture classes and their pin-meter cutoffs -- the one place they are defined.

Agreed by the team (2026-10-13), in % moisture content (MC) from the pin meter:

    Dry         MC <  12
    Damp        12 <= MC <= 20
    Saturated   MC >  20

Change DRY_MAX / DAMP_MAX here and nothing else needs editing.
"""

from __future__ import annotations

DRY_MAX = 12.0     # below this: Dry
DAMP_MAX = 20.0    # up to and including this: Damp; above it: Saturated

LEVELS = ["Dry", "Damp", "Saturated"]   # same names as App/record_tab.py MOISTURE_LEVELS


def level_for_mc(mc_pct: float) -> str:
    """Class for a pin-meter reading in % MC."""
    if mc_pct < DRY_MAX:
        return "Dry"
    if mc_pct <= DAMP_MAX:
        return "Damp"
    return "Saturated"


def describe() -> str:
    return f"Dry < {DRY_MAX:g}%, Damp {DRY_MAX:g}-{DAMP_MAX:g}%, Saturated > {DAMP_MAX:g}%"
