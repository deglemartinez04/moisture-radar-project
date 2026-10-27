#!/usr/bin/env python3
"""
contrast_score.py - score how well a geometry/settings combination separates
wall conditions, from raw Acconeer A121 .h5 recordings (App > Data Logging >
"Save raw radar data to an .h5 file").

WHY: with a lens, a standoff and a wall that changes every test, "the graph looks
calmer" is not a measurement. This script gives one number per comparison:
  * how big the complex (amplitude + phase) difference between two conditions is,
  * relative to the noise, INCLUDING how much the reading changes when the device
    is lifted off and put back (repositioning), which is usually the real limit.
Run it once per geometry (flush / 4 cm / 8 cm / 10 cm standoff ...) or per settings
change (HWAAS, PRF, gain ...) and keep the setup with the best score.

RECORDING PROTOCOL (per geometry, same sensor settings for all files):
  baseline : open air, lens and standoff attached, nothing within ~30 cm  (1 file)
  wall     : drywall only, dry                                            (2 files, re-placed)
  stud     : drywall + dry stud                                           (2 files, re-placed)
  wet      : drywall + wet stud                                           (2 files, re-placed)
  Record >= 100 frames each (10 s at 30 Hz). Re-placing = lift the device away and
  put it back on the standoff; that is what captures repositioning error.

USAGE:
  python contrast_score.py --baseline air.h5 \\
      --test wall=wall_a.h5,wall_b.h5 --test stud=stud_a.h5,stud_b.h5 \\
      --test wet=wet_a.h5,wet_b.h5 [--min-mm 80] [--max-mm 180] \\
      [--skip-frames 5] [--plot out.png] [--csv out.csv]

HOW TO READ THE OUTPUT:
  |delta|   size of the difference between the two conditions (sensor units)
  z         |delta| divided by the rms noise magnitude. Above ~2.4 is a real
            difference at 99.7% confidence; 4+ is comfortable.
  Note "single placement": only one file for a condition, so repositioning error
  is NOT included and the score is optimistic.
  Note "leakage zone": the best bin is closer than 80 mm (about 2 x the 4 cm
  Profile 1 pulse width), where lens/sensor leakage dominates.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from itertools import combinations

import h5py
import numpy as np

BASE_STEP_M = 0.00250227400101721
LEAKAGE_ZONE_MM = 80.0
RAYLEIGH_999 = float(np.sqrt(-2.0 * np.log(0.003)))  # 99.7% quantile of 2-D Gaussian noise magnitude


# ----------------------------------------------------------------------------- loading
@dataclass
class Recording:
    path: str
    z: np.ndarray                 # (frames, points) complex, sweeps already averaged
    distances_mm: np.ndarray
    cfg: dict
    temp_min: float | None
    temp_max: float | None
    n_saturated: int
    n_cal_needed: int
    n_delayed: int


def _as_text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def load_recording(path: str, skip_frames: int = 0) -> Recording:
    with h5py.File(path, "r") as f:
        try:
            entry = f["sessions/session_0/group_0/entry_0"]
            result = entry["result"]
            cfg_json = json.loads(_as_text(f["sessions/session_0/session_config"][()]))
            meta = json.loads(_as_text(entry["metadata"][()]))
        except KeyError as exc:
            sys.exit(f"{path}: not a single-sensor A121 recording written by the app ({exc}).")
        fr = result["frame"][:]
        z = fr["real"].astype(np.float64) + 1j * fr["imag"].astype(np.float64)   # (frames, sweeps, points)
        sensor_cfg = next(iter(cfg_json["groups"][0].values()))
        subs = sensor_cfg["subsweeps"]
        loopback_ref = len(subs) >= 2 and bool(subs[0].get("enable_loopback", False))
        if loopback_ref:
            # App layout: subsweep 0 = 1-point loopback reference, subsweep 1 = range.
            # Reference every sweep to its loopback phase (Acconeer phase-jitter compensation)
            # so recordings from different sessions/placements are phase-comparable.
            n_lb = int(subs[0]["num_points"])
            lb_angle = np.angle(z[:, :, :1])
            z = z[:, :, n_lb:] * np.exp(-1j * lb_angle)
            sub = subs[1]
        else:
            sub = subs[0]
        z = z.mean(axis=1)                       # coherent mean of the sweeps in each frame
        step_m = float(meta.get("base_step_length_m", BASE_STEP_M))
        distances_mm = (sub["start_point"] + np.arange(sub["num_points"]) * sub["step_length"]) * step_m * 1000.0
        cfg = dict(sub)
        cfg["loopback_reference"] = loopback_ref
        cfg["frame_rate"] = sensor_cfg.get("frame_rate")
        cfg["sweeps_per_frame"] = sensor_cfg.get("sweeps_per_frame")
        temp = result["temperature"][:] if "temperature" in result else None
        sat = int(result["data_saturated"][:].sum()) if "data_saturated" in result else 0
        cal = int(result["calibration_needed"][:].sum()) if "calibration_needed" in result else 0
        dly = int(result["frame_delayed"][:].sum()) if "frame_delayed" in result else 0
    if skip_frames:
        z = z[skip_frames:]
    if z.shape[0] < 10:
        sys.exit(f"{path}: only {z.shape[0]} usable frames - record at least 100.")
    return Recording(path, z, distances_mm, cfg,
                     None if temp is None else float(temp.min()), None if temp is None else float(temp.max()),
                     sat, cal, dly)


# ----------------------------------------------------------------------------- statistics
def frame_stats(z: np.ndarray):
    """Complex mean and per-axis noise std at each distance point."""
    mu = z.mean(axis=0)
    sigma = np.sqrt(0.5 * (z.real.var(axis=0, ddof=1) + z.imag.var(axis=0, ddof=1)))
    return mu, sigma


@dataclass
class Condition:
    label: str
    recs: list = field(default_factory=list)

    def compute(self):
        mus, vars_stat = [], []
        for r in self.recs:
            mu, sigma = frame_stats(r.z)
            mus.append(mu)
            vars_stat.append(sigma ** 2 / r.z.shape[0])      # per-axis variance of that recording's mean
        mus = np.stack(mus)
        self.mu = mus.mean(axis=0)
        n_place = len(self.recs)
        stat = np.mean(vars_stat, axis=0)
        if n_place >= 2:
            # Spread between placements already contains the statistical noise AND repositioning error.
            spread = 0.5 * (mus.real.var(axis=0, ddof=1) + mus.imag.var(axis=0, ddof=1))
            per_placement = np.maximum(spread, stat)
        else:
            per_placement = stat
        self.var_mean = per_placement / n_place             # per-axis variance of the condition mean
        self.placements = n_place
        return self


def compare(a: Condition, b: Condition, distances_mm, lo_mm, hi_mm):
    delta = a.mu - b.mu
    var = a.var_mean + b.var_mean
    noise_rms = np.sqrt(2.0 * var)                           # rms magnitude of the 2-D noise
    z = np.abs(delta) / np.maximum(noise_rms, 1e-12)
    level = RAYLEIGH_999 * np.sqrt(var)
    in_window = (distances_mm >= lo_mm) & (distances_mm <= hi_mm)
    sig = (np.abs(delta) > level) & in_window
    zw = np.where(in_window, z, -np.inf)
    best = int(np.argmax(zw))
    notes = []
    if min(a.placements, b.placements) < 2:
        notes.append("single placement: optimistic")
    if distances_mm[best] < LEAKAGE_ZONE_MM:
        notes.append("best bin in leakage zone")
    return dict(delta=delta, z=z, level=level, sig=sig, best=best, in_window=in_window, notes=notes)


# ----------------------------------------------------------------------------- reporting
def jitter_estimate(r: Recording):
    """Phase jitter (rad) at the strongest point: tangential minus radial noise, relative to amplitude."""
    mu = r.z.mean(axis=0)
    k = int(np.argmax(np.abs(mu)))
    u = mu[k] / abs(mu[k])
    d = (r.z[:, k] - mu[k]) * np.conj(u)
    tan, rad = d.imag.std(ddof=1), d.real.std(ddof=1)
    return k, float(np.sqrt(max(tan ** 2 - rad ** 2, 0.0)) / abs(mu[k]))


def settings_key(r: Recording):
    keys = ("start_point", "num_points", "step_length", "profile", "hwaas", "receiver_gain", "prf",
            "phase_enhancement", "iq_imbalance_compensation", "loopback_reference", "frame_rate", "sweeps_per_frame")
    return tuple(r.cfg.get(k) for k in keys), keys


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True, help="open-air recording (lens + standoff attached); FILE[,FILE...] to include repositioning")
    ap.add_argument("--test", action="append", required=True, metavar="NAME=FILE[,FILE...]",
                    help="a condition; list several files (re-placed device) for a realistic score")
    ap.add_argument("--min-mm", type=float, default=0.0, help="ignore points closer than this when scoring")
    ap.add_argument("--max-mm", type=float, default=1e9, help="ignore points farther than this when scoring")
    ap.add_argument("--skip-frames", type=int, default=0, help="drop the first N frames of every file")
    ap.add_argument("--plot", metavar="PNG", help="save a summary figure (needs matplotlib)")
    ap.add_argument("--csv", metavar="CSV", help="save per-distance results")
    args = ap.parse_args()

    base = Condition("baseline", [load_recording(p, args.skip_frames) for p in args.baseline.split(",")]).compute()
    conds = [base]
    for spec in args.test:
        if "=" not in spec:
            ap.error(f"--test needs NAME=FILE[,FILE...], got {spec!r}")
        name, files = spec.split("=", 1)
        conds.append(Condition(name, [load_recording(p, args.skip_frames) for p in files.split(",")]).compute())

    ref = base.recs[0]
    ref_key, key_names = settings_key(ref)
    for c in conds:
        for r in c.recs:
            if not np.allclose(r.distances_mm, ref.distances_mm):
                sys.exit(f"{r.path}: distance axis differs from the baseline. All files must use the same range settings.")
            key, _ = settings_key(r)
            if key != ref_key:
                diffs = [n for n, a, b in zip(key_names, key, ref_key) if a != b]
                print(f"WARNING {r.path}: settings differ from baseline in {diffs}; comparison may be invalid.")

    d = ref.distances_mm
    print("\nRECORDINGS")
    print(f"{'condition':10s} {'files':>5s} {'frames':>6s} {'temp C':>9s} {'sat':>4s} {'cal?':>4s} {'dly':>4s} {'noise/axis':>10s} {'jitter rad@peak':>16s}")
    for c in conds:
        frames = sum(r.z.shape[0] for r in c.recs)
        tmin = min(r.temp_min for r in c.recs if r.temp_min is not None) if any(r.temp_min is not None for r in c.recs) else None
        tmax = max(r.temp_max for r in c.recs if r.temp_max is not None) if any(r.temp_max is not None for r in c.recs) else None
        noise = np.median(np.concatenate([frame_stats(r.z)[1] for r in c.recs]))
        jit = np.mean([jitter_estimate(r)[1] for r in c.recs])
        print(f"{c.label:10s} {len(c.recs):5d} {frames:6d} {('%g-%g' % (tmin, tmax)) if tmin is not None else 'n/a':>9s} "
              f"{sum(r.n_saturated for r in c.recs):4d} {sum(r.n_cal_needed for r in c.recs):4d} "
              f"{sum(r.n_delayed for r in c.recs):4d} {noise:10.1f} {jit:16.3f}")
        if sum(r.n_saturated for r in c.recs) or sum(r.n_cal_needed for r in c.recs):
            print(f"   ^ {c.label}: saturation or calibration_needed flags were set - discard or re-record.")
    print(f"\nrange {d[0]:.0f}-{d[-1]:.0f} mm, {len(d)} points | settings: " +
          ", ".join(f"{n}={v}" for n, v in zip(key_names, ref_key) if n in ("profile", "hwaas", "prf", "receiver_gain", "step_length", "phase_enhancement")))

    pairs = [(c, base) for c in conds[1:]] + [(a, b) for a, b in combinations(conds[1:], 2)]
    results = {}
    print("\nCOMPARISONS  (scored over %.0f-%.0f mm)" % (max(args.min_mm, d[0]), min(args.max_mm, d[-1])))
    print(f"{'comparison':22s} {'best mm':>8s} {'|delta|':>8s} {'z':>6s} {'sig pts':>8s}  notes")
    for a, b in pairs:
        r = compare(a, b, d, args.min_mm, args.max_mm)
        results[(a.label, b.label)] = r
        k = r["best"]
        print(f"{a.label + ' - ' + b.label:22s} {d[k]:8.0f} {abs(r['delta'][k]):8.1f} {r['z'][k]:6.1f} "
              f"{int(r['sig'].sum()):5d}/{int(r['in_window'].sum()):<2d}  {'; '.join(r['notes'])}")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["comparison", "distance_mm", "delta_abs", "delta_phase_rad", "noise_level_99_7", "z", "significant"])
            for (an, bn), r in results.items():
                for i in range(len(d)):
                    w.writerow([f"{an}-{bn}", f"{d[i]:.1f}", f"{abs(r['delta'][i]):.2f}",
                                f"{np.angle(r['delta'][i]):.3f}", f"{r['level'][i]:.2f}", f"{r['z'][i]:.2f}", int(r['sig'][i])])
        print(f"\nper-point results written to {args.csv}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed - skipping --plot (pip install matplotlib)")
        else:
            fig, ax = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
            for c in conds:
                ax[0].plot(d, np.abs(c.mu), label=c.label)
            ax[0].set_ylabel("|mean IQ| (a.u.)"); ax[0].legend(); ax[0].grid(alpha=.3)
            ax[0].set_title("Mean amplitude per condition")
            for c in conds[1:]:
                r = results[(c.label, "baseline")]
                line, = ax[1].plot(d, np.abs(r["delta"]), label=f"{c.label} - baseline")
                ax[1].plot(d, r["level"], "--", color=line.get_color(), alpha=.6)
            ax[1].set_ylabel("|delta| (a.u.)  (dashed = 99.7% noise level)"); ax[1].set_xlabel("Distance (mm)")
            ax[1].legend(); ax[1].grid(alpha=.3)
            ax[1].axvspan(d[0], LEAKAGE_ZONE_MM, color="red", alpha=.07)
            fig.tight_layout(); fig.savefig(args.plot, dpi=130)
            print(f"figure written to {args.plot}")


if __name__ == "__main__":
    main()
