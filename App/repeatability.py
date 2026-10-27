"""
repeatability.py

Answers two questions from data_log.csv (pure Python + numpy, no Qt):

1. Repeatability: when the SAME setup is re-placed and recorded again, how much do the
   features move?  Rows are grouped by (material, moisture level, sample ID, standoff);
   each row is one placement. Reported per feature: mean, spread (std) across placements.

2. Which standoff works best: at each standoff, how well do the classes separate compared
   with that placement-to-placement spread?  For every pair of classes (e.g. Dry vs Damp)
       d' = |mean_a - mean_b| / sqrt((std_a^2 + std_b^2) / 2)
   and the WORST pair is reported (the model is only as good as its hardest pair).
   d' >= 2 is a usable separation, >= 3 is comfortable. Needs >= 2 placements per class.

Record each condition several times, lifting the device / moving the material away and
back between recordings - that is what makes the spread realistic.
"""

from __future__ import annotations

import math
from collections import defaultdict
from itertools import combinations

import numpy as np

KEY_FEATURES = [
    "rear_energy_ratio_db",
    "rear_centroid_offset_m",
    "rear_phase_rad",
    "front_amplitude",
    "front_distance_m",
]
CIRCULAR = {"rear_phase_rad", "phase_delta_rad"}
UNITS = {"front_distance_m": ("mm", 1000.0), "rear_centroid_offset_m": ("mm", 1000.0)}


def _num(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return math.nan
    return v


def _mean_std(values, circular=False):
    v = np.array([x for x in values if not math.isnan(x)], dtype=float)
    if len(v) == 0:
        return math.nan, math.nan, 0
    if circular:
        r = np.mean(np.exp(1j * v))
        std = math.sqrt(max(0.0, -2.0 * math.log(max(abs(r), 1e-12)))) if len(v) > 1 else math.nan
        return float(np.angle(r)), std, len(v)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else math.nan), len(v)


def _diff(a, b, circular=False):
    if circular:
        return abs(math.atan2(math.sin(a - b), math.cos(a - b)))
    return abs(a - b)


def _standoff(row):
    v = _num(row.get("measured_distance_mm"))
    return None if math.isnan(v) else round(v, 1)


def repeatability_table(rows, features=KEY_FEATURES):
    """List of groups: {key, n, stats: {feature: (mean, std)}}."""
    groups = defaultdict(list)
    for r in rows:
        key = (r.get("material", ""), r.get("moisture_level", ""), r.get("sample_id", ""), _standoff(r))
        groups[key].append(r)
    out = []
    for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
        g = groups[key]
        stats = {}
        for f in features:
            m, s, _ = _mean_std([_num(r.get(f)) for r in g], f in CIRCULAR)
            stats[f] = (m, s)
        out.append({"key": key, "n": len(g), "stats": stats})
    return out


def standoff_separation(rows, class_key="moisture_level", features=KEY_FEATURES, within_key="material"):
    """Per (within_key value, standoff): worst-pair d' for each feature."""
    buckets = defaultdict(lambda: defaultdict(list))
    for r in rows:
        buckets[(r.get(within_key, ""), _standoff(r))][r.get(class_key, "")].append(r)
    out = []
    for (within, standoff), classes in sorted(buckets.items(), key=lambda kv: (str(kv[0][0]), kv[0][1] or 0)):
        n_rows = {c: len(v) for c, v in classes.items()}
        scores = {}
        for f in features:
            circ = f in CIRCULAR
            ms = {c: _mean_std([_num(r.get(f)) for r in v], circ) for c, v in classes.items()}
            worst = math.nan
            if len(ms) >= 2:
                vals = []
                for a, b in combinations(sorted(ms), 2):
                    (ma, sa, na), (mb, sb, nb) = ms[a], ms[b]
                    if na < 2 or nb < 2 or math.isnan(sa) or math.isnan(sb):
                        vals = []
                        break
                    pooled = math.sqrt((sa ** 2 + sb ** 2) / 2.0)
                    vals.append(_diff(ma, mb, circ) / max(pooled, 1e-12))
                worst = min(vals) if vals else math.nan
            scores[f] = worst
        out.append({"within": within, "standoff": standoff, "classes": n_rows, "scores": scores})
    return out


def _fmt(f, v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    unit, scale = UNITS.get(f, ("", 1.0))
    return f"{v * scale:.3g}{unit}"


SHORT = {
    "rear_energy_ratio_db": "rear/front dB",
    "rear_centroid_offset_m": "rear centroid",
    "rear_phase_rad": "rear phase",
    "front_amplitude": "front amp",
    "front_distance_m": "front dist",
}


def format_report(rows) -> str:
    if not rows:
        return "No rows in data_log.csv yet."
    lines = ["REPEATABILITY (each row = one placement; spread = std across placements)", ""]
    for g in repeatability_table(rows):
        mat, moist, sample, stand = g["key"]
        lines.append(f"{mat} | {moist} | {sample} | standoff {stand} mm | n={g['n']}")
        parts = []
        for f, (m, s) in g["stats"].items():
            parts.append(f"  {SHORT.get(f, f)}: {_fmt(f, m)} ± {_fmt(f, s)}")
        lines.extend(parts)
    lines += ["", "STANDOFF COMPARISON - worst-pair separation d' between moisture levels",
              "(d' >= 2 usable, >= 3 comfortable; needs >= 2 placements per level)", ""]
    for c in standoff_separation(rows):
        cls = ", ".join(f"{k}:{v}" for k, v in sorted(c["classes"].items()))
        sc = "  ".join(f"{SHORT.get(f, f)} {_fmt('', v)}" for f, v in c["scores"].items())
        lines.append(f"{c['within']} @ {c['standoff']} mm  [{cls}]")
        lines.append(f"  {sc}")
    lines += ["", "STUD DETECTION - separation between materials at each standoff", ""]
    for c in standoff_separation(rows, class_key="material", within_key="moisture_level"):
        cls = ", ".join(f"{k}:{v}" for k, v in sorted(c["classes"].items()))
        sc = "  ".join(f"{SHORT.get(f, f)} {_fmt('', v)}" for f, v in c["scores"].items())
        lines.append(f"{c['within']} @ {c['standoff']} mm  [{cls}]")
        lines.append(f"  {sc}")
    return "\n".join(lines)
