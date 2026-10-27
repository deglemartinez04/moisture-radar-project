"""Write SIMULATED labeled .h5 recordings to rehearse the training workflow.

These files are not measurements. They show how build_dataset.py and
train_model.py behave on a dataset shaped like the one the team will collect;
they say nothing about whether the real sensor can measure moisture.
Material names are prefixed "SIM" so they can't be mixed up with real files.

Physics (simple, plane-wave):
- Scene: wood board (front face = G1, ~0.10 m) in front of a fixed back
  reflector (G2, ~0.23 m), matching the geometry of Wood_High_01.h5.
- Moisture content (MC %) sets the wood's complex permittivity
  eps = eps' - j eps''  (both rise with MC; water is ~80 at these frequencies).
- G1 strength   ~ |Gamma|,  Gamma = (1 - sqrt(eps)) / (1 + sqrt(eps))
- G2 strength   ~ (1 - Gamma^2) * exp(-2 * alpha * thickness),  alpha = k0 * Im(sqrt(eps))
- G2 delay      = thickness * (Re(sqrt(eps)) - 1): G2 appears farther and its phase shifts.
- IQ phase of a reflector at range R is 4*pi*R / lambda (lambda = 5 mm at 60 GHz).
The permittivity-vs-MC coefficients are rough, chosen so a ~25% MC board gives
the G2/G1 ratio seen in Wood_High_01 (~0.63).

Per recording (= one placement of the sensor) it varies: standoff, tilt (gain),
board thickness, local MC around the pin-meter value, and noise. Per frame:
hand jitter in range and amplitude, plus receiver noise.

Writes <out>/SIMWood_<Level>_<n>.h5 and <out>/mc_readings.csv (the simulated
pin-meter value for each file).

Usage:
    python ml/make_synthetic.py                       # 10 recordings per level
    python ml/make_synthetic.py --trials 3            # what 3 per level looks like
    python ml/make_synthetic.py --placement-noise 2   # sloppier placements
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.h5io import write_recording  # noqa: E402
from ml.labels import DAMP_MAX, DRY_MAX  # noqa: E402

START_POINT, NUM_POINTS, STEP_LENGTH = 24, 40, 2   # App project defaults
BASE_STEP_M = 0.00250227400101721
WAVELENGTH_M = 3e8 / 60.5e9
K0 = 2 * np.pi / WAVELENGTH_M

# Pin-meter MC % drawn uniformly across each class's band (team cutoffs, ml/labels.py).
# Boards near a cutoff are included on purpose: they are the hard cases.
LEVEL_MC = {
    "Dry": (6.0, DRY_MAX - 0.1),
    "Damp": (DRY_MAX, DAMP_MAX),
    "Saturated": (DAMP_MAX + 0.1, 32.0),
}

G1_RANGE_M = 0.100        # wood front face
G2_RANGE_M = 0.230        # back reflector, air path
THICKNESS_M = 0.012       # board thickness
PULSE_WIDTH_M = 0.016     # envelope sigma of one reflection (Profile 1, fit to Wood_High_01)
SCALE = 81000.0           # overall amplitude scale (fit to Wood_High_01)
BACK_REFLECTIVITY = 0.62


def permittivity(mc_pct):
    """Very rough wood permittivity at 60 GHz vs moisture content."""
    return complex(1.9 + 0.055 * mc_pct, -(0.04 + 0.0045 * mc_pct))


def scene_profile(distances_m, mc_pct, standoff_m, thickness_m, gain):
    """Complex IQ range profile (one noiseless sweep) for one placement."""
    n = np.sqrt(permittivity(mc_pct))          # complex refractive index, Im < 0 = loss
    gamma = (1 - n) / (1 + n)
    alpha = -K0 * n.imag                        # field attenuation, Np/m
    g1_range = G1_RANGE_M + standoff_m
    g2_range = G2_RANGE_M + standoff_m + thickness_m * (n.real - 1)  # extra delay inside wood
    g1 = SCALE * gain * abs(gamma) * np.exp(1j * np.angle(-gamma))
    g2 = SCALE * gain * BACK_REFLECTIVITY * abs(1 - gamma ** 2) * np.exp(-2 * alpha * thickness_m)

    leakage = 9000 * np.exp(-0.5 * ((distances_m - 0.060) / 0.007) ** 2) * np.exp(0.4j)
    return leakage, (g1_range, g1), (g2_range, g2)


def simulate_recording(rng, mc_pct, n_frames, placement_noise):
    d = (START_POINT + np.arange(NUM_POINTS) * STEP_LENGTH) * BASE_STEP_M
    standoff = rng.normal(0, 0.0015 * placement_noise)
    thickness = THICKNESS_M + rng.normal(0, 0.0004 * placement_noise)
    gain = rng.normal(1.0, 0.05 * placement_noise)                 # tilt / pressure
    local_mc = mc_pct + rng.normal(0, 1.5 * placement_noise)       # pin meter vs radar spot
    leakage, (r1, a1), (r2, a2) = scene_profile(d, local_mc, standoff, thickness, gain)
    noise_sigma = rng.uniform(30, 60)

    frames = np.empty((n_frames, NUM_POINTS), dtype=complex)
    for i in range(n_frames):
        jitter = rng.normal(0, 0.000013)                           # 13 um: Wood_High_01 phase std 0.034 rad
        wobble = 1 + rng.normal(0, 0.004)
        sweep = leakage.copy()
        for r, a in ((r1, a1), (r2, a2)):
            r = r + jitter
            sweep += a * wobble * np.exp(-0.5 * ((d - r) / PULSE_WIDTH_M) ** 2) * np.exp(1j * 2 * K0 * r)
        frames[i] = sweep + noise_sigma * (rng.normal(size=NUM_POINTS) + 1j * rng.normal(size=NUM_POINTS))
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--out-dir", default="ml/synthetic")
    parser.add_argument("--trials", type=int, default=10, help="recordings per moisture level")
    parser.add_argument("--frames", type=int, default=600, help="frames per recording")
    parser.add_argument("--placement-noise", type=float, default=1.0,
                        help="scale of placement-to-placement variation (1 = careful, 2+ = sloppy)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    readings = []
    for level, (lo, hi) in LEVEL_MC.items():
        for trial in range(1, args.trials + 1):
            mc = round(float(rng.uniform(lo, hi)), 1)
            name = f"SIMWood_{level}_{trial}.h5"
            write_recording(out / name, simulate_recording(rng, mc, args.frames, args.placement_noise),
                            START_POINT, NUM_POINTS, STEP_LENGTH, BASE_STEP_M,
                            profile="PROFILE_1", hwaas=32, receiver_gain=16, prf="PRF_15_6_MHz")
            readings.append({"h5_file": name, "moisture_level": level, "pin_meter_mc_pct": mc})
    with open(out / "mc_readings.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["h5_file", "moisture_level", "pin_meter_mc_pct"])
        writer.writeheader()
        writer.writerows(readings)
    print(f"Wrote {len(readings)} SIMULATED recordings + mc_readings.csv to {out}/")


if __name__ == "__main__":
    main()
