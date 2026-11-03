"""Compare the AppV10/V14 logistic regression with this repo's Random Forest.

Both algorithms see the same V14 feature table (28 features) and the same held-out
folds, so the only difference is the algorithm. Tasks mirror V14's train_model.py:
stud, wetness (Dry/Damp/Saturated) and wet-vs-dry.

LR   : standardise -> L2 logistic regression, class_weight=balanced, C picked by an
       inner grouped CV on each training fold (nested, so the test fold never picks C).
RF   : ml/train_model.make_model settings (300 trees, min_samples_leaf=2).
RF-b : the same forest with class_weight=balanced, to separate algorithm from weighting.

Usage:  python ml/compare_models.py --out ml/v14/comparison_report.txt
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent / "v14"
NORM_Q = [f"q{o}_{c}" for o in (10, 15, 20, 25, 30, 35, 40, 45, 50) for c in ("re", "im")]
FEATURES = NORM_Q + [
    "norm_rear_mag_db", "norm_rear_phase_sin", "norm_rear_phase_cos",
    "rear_energy_ratio_db", "rear_centroid_offset_m", "rear_peak_offset_m",
    "rear_phase_sin", "rear_phase_cos", "front_distance_m", "front_amplitude_db",
]
WET = ["Dry", "Damp", "Saturated"]
C_GRID = (0.1, 0.3, 1.0, 3.0, 10.0)
SEEDS = range(5)
DATASETS = [("A (all useful data)", "dataset_A_all_useful.csv", [40.0, 50.0, 60.0], C_GRID),
            ("B (old dry + new wet)", "dataset_B_dry_plus_new.csv", [50.0], tuple(c for c in C_GRID if c <= 1.0))]


def lr(C):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))


def rf(seed, balanced):
    return make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=-1,
        class_weight="balanced" if balanced else None))


def fit_predict(make, Xtr, ytr, Xte):
    if len(np.unique(ytr)) < 2:
        return np.full(len(Xte), ytr[0], dtype=object)
    return make().fit(Xtr, ytr).predict(Xte)


def best_C(Xtr, ytr, gtr, grid):
    n = min(5, len(np.unique(gtr)))
    if n < 2 or len(np.unique(ytr)) < 2:
        return grid[len(grid) // 2]
    scores = []
    for C in grid:
        p = np.empty(len(ytr), dtype=object)
        for a, b in GroupKFold(n).split(Xtr, ytr, gtr):
            p[b] = fit_predict(lambda: lr(C), Xtr[a], ytr[a], Xtr[b])
        scores.append(balanced_accuracy_score(ytr, p))
    return grid[int(np.argmax(scores))]


def group_key(kind, df):
    return df["standoff_mm"] if "standoff" in kind else df["meta_date"]


def splits(kind, X, y, groups, df):
    if kind == "grouped 5-fold":
        return list(GroupKFold(min(5, len(np.unique(groups)))).split(X, y, groups))
    return list(LeaveOneGroupOut().split(X, y, group_key(kind, df).to_numpy()))


def run(df, y, kind, grid):
    X, groups = df[FEATURES].to_numpy(float), df["cv_group"].to_numpy()
    out = {"LR": np.empty(len(y), dtype=object),
           "RF": [np.empty(len(y), dtype=object) for _ in SEEDS],
           "RF-b": [np.empty(len(y), dtype=object) for _ in SEEDS]}
    for tr, te in splits(kind, X, y, groups, df):
        C = best_C(X[tr], y[tr], groups[tr], grid)
        out["LR"][te] = fit_predict(lambda: lr(C), X[tr], y[tr], X[te])
        for i, s in enumerate(SEEDS):
            out["RF"][i][te] = fit_predict(lambda: rf(s, False), X[tr], y[tr], X[te])
            out["RF-b"][i][te] = fit_predict(lambda: rf(s, True), X[tr], y[tr], X[te])
    return out


def sanity_lr(df, y, grid):
    """Their method: pick C on the test folds themselves (optimistic) -- reproduces their report."""
    X, groups = df[FEATURES].to_numpy(float), df["cv_group"].to_numpy()
    best = 0
    for C in grid:
        p = np.empty(len(y), dtype=object)
        for a, b in GroupKFold(min(5, len(np.unique(groups)))).split(X, y, groups):
            p[b] = fit_predict(lambda: lr(C), X[a], y[a], X[b])
        best = max(best, balanced_accuracy_score(y, p))
    return best


def block(task, df, y, labels, grid, lines):
    lines.append(f"\n  --- {task}: {len(y)} rows, {df['cv_group'].nunique()} blocks; "
                 + ", ".join(f"{c}={int((y == c).sum())}" for c in labels))
    lines.append(f"  (check) LR with C chosen on the test folds, as in the V14 report: "
                 f"balanced acc {sanity_lr(df, y, grid):.2f}")
    for kind in ("grouped 5-fold", "leave-one-standoff-out", "leave-one-day-out"):
        if kind != "grouped 5-fold" and group_key(kind, df).nunique() < 2:
            continue
        out = run(df, y, kind, grid)
        lines.append(f"\n  {kind}  (balanced accuracy / accuracy)")
        for name, preds in (("LR (theirs)", [out["LR"]]), ("RF (ours)", out["RF"]), ("RF balanced", out["RF-b"])):
            ba = [balanced_accuracy_score(y, p) for p in preds]
            acc = [np.mean(p == y) for p in preds]
            spread = f"  [5 seeds {min(ba):.2f}-{max(ba):.2f}]" if len(preds) > 1 else ""
            lines.append(f"    {name:<12} {np.mean(ba):.2f} / {np.mean(acc):.2f}{spread}")
        lr_ok, rf_ok = out["LR"] == y, out["RF-b"][0] == y
        b, c = int((lr_ok & ~rf_ok).sum()), int((~lr_ok & rf_ok).sum())
        pv = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
        lines.append(f"    only LR right: {b}, only RF-balanced right: {c}  (McNemar p={pv:.2f})")
        if kind != "grouped 5-fold":
            key = group_key(kind, df).to_numpy()
            for k in sorted(np.unique(key)):
                m = key == k
                lines.append(f"      unseen {k}: LR {np.mean(out['LR'][m] == y[m]):.2f}, "
                             f"RF-b {np.mean(out['RF-b'][0][m] == y[m]):.2f}  (n={m.sum()})")
        elif len(labels) > 2:
            for name, p in (("LR", out["LR"]), ("RF-b", out["RF-b"][0])):
                cm = confusion_matrix(y, p, labels=labels)
                lines.append(f"    {name} confusion (rows=true, cols=pred; {'/'.join(labels)}): {cm.tolist()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(HERE / "comparison_report.txt"))
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    lines = ["LR (V14 AppV10 model) vs RF (this repo) on the V14 feature tables",
             "Same 28 features, same folds. Small data (23-60 wetness rows): treat differences as tentative.",
             "Labels were partly fixed by hand (see App/ml/README_ML.md in the V14 zips)."]
    for title, fname, wet_standoffs, grid in DATASETS:
        d = pd.read_csv(HERE / fname)
        d = d[d["quality_ok"] == 1].reset_index(drop=True)
        lines.append(f"\n=== Dataset {title} ({len(d)} rows pass quality) ===")
        stud = d[d["has_stud"].isin([0, 1])].reset_index(drop=True)
        block("STUD", stud, np.where(stud["has_stud"] == 1, "Stud", "No stud").astype(object),
              ["No stud", "Stud"], grid, lines)
        wet = d[(d["material"] == "Wood + drywall") & d["wetness"].isin(WET)]
        if wet_standoffs:
            wet = wet[wet["standoff_mm"].isin(wet_standoffs)]
        wet = wet.reset_index(drop=True)
        yw = wet["wetness"].to_numpy(object)
        block("WETNESS (Dry/Damp/Saturated)", wet, yw, WET, grid, lines)
        block("WET vs DRY", wet, np.where(yw == "Dry", "Dry", "Wet").astype(object), ["Dry", "Wet"], grid, lines)
    text = "\n".join(lines)
    print(text)
    Path(args.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
