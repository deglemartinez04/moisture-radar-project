"""Train the Dry / Damp / Saturated model on 2026-10-22 (dry) + 2026-10-29 + 2026-10-31 (drying series).

Uses the V14 feature table (ml/v14/dataset_A_all_useful.csv), whose labels for the two drying
sessions were already re-derived from each board's own echo (the raw logs say "Saturated" for all).

Caveat that shapes the design: every wet row was recorded at 50 mm, but the dry rows span
12-60 mm. A model could learn "50 mm = wet". So the default run drops the two standoff-dependent
features (front_distance_m, front_amplitude_db) and keeps only the normalised shape features.
`--all-features` shows what happens with them.

Evaluation: two folds, each holds out one whole wet session (S4 = 10-29 or S3 = 10-31, i.e. one
board) plus half of the dry standoffs. Nothing from a test board or test standoff is in training.
Rows labelled "Surface dry" (surface dried, interior still wet) are never trained on; their
predictions are listed only for information.

Usage:  python ml/train_subset.py [--model rf|lr] [--all-features] [--out DIR]
"""
import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_models import FEATURES, WET, lr, rf  # noqa: E402

HERE = Path(__file__).resolve().parent / "v14"
DAYS = ["2026-10-22", "2026-10-29", "2026-10-31"]
LEVEL = ["front_distance_m", "front_amplitude_db"]
DRY_FOLDS = {"S4": [12.0, 30.0, 50.0], "S3": [20.0, 40.0, 60.0]}   # dry standoffs held out with each wet session


def export(pipe, features, path, n_rows):
    scaler, clf = pipe.named_steps["standardscaler"], pipe.named_steps["logisticregression"]
    model = {"task": "wetness", "type": "standardized_logistic_regression", "features": features,
             "classes": [str(c) for c in clf.classes_], "mean": scaler.mean_.tolist(),
             "scale": scaler.scale_.tolist(), "coef": clf.coef_.tolist(), "intercept": clf.intercept_.tolist(),
             "trained_at": datetime.now().isoformat(timespec="seconds"), "n_rows": n_rows,
             "dataset": "10-22 dry + 10-29 + 10-31 (ml/train_subset.py)"}
    Path(path).write_text(json.dumps(model, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all-features", action="store_true", help="keep front_distance_m / front_amplitude_db")
    ap.add_argument("--C", type=float, default=0.3)
    ap.add_argument("--model", choices=("lr", "rf"), default="rf", help="which model to fit on all rows and save")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.out is None:
        a.out = str(HERE / ("model_rf_1022_1029_1031" if a.model == "rf" else "model_1022_1029_1031"))
    warnings.filterwarnings("ignore")
    feats = FEATURES if a.all_features else [f for f in FEATURES if f not in LEVEL]

    d = pd.read_csv(HERE / "dataset_A_all_useful.csv")
    d = d[(d["quality_ok"] == 1) & d["meta_date"].isin(DAYS) & (d["material"] == "Wood + drywall")].reset_index(drop=True)
    train_pool = d[d["wetness"].isin(WET)].reset_index(drop=True)
    surface_dry = d[d["wetness"] == "Surface dry"].reset_index(drop=True)
    lines = [f"Training subset: {', '.join(DAYS)}, Wood + drywall, {len(feats)} features "
             f"({'with' if a.all_features else 'without'} standoff-dependent features), logistic C={a.C}",
             "Rows: " + ", ".join(f"{c}={int((train_pool.wetness == c).sum())}" for c in WET)
             + f"  (+{len(surface_dry)} 'Surface dry' rows, not trained on)",
             "Wet rows are all at 50 mm; dry rows span 12-60 mm.",
             "Dry blocks by standoff: " + ", ".join(
                 f"{int(s)}mm={n}" for s, n in train_pool[train_pool.wetness == 'Dry'].standoff_mm.value_counts().sort_index().items())]

    # two folds: hold out a whole wet session (one board) + some dry standoffs
    y = train_pool["wetness"].to_numpy(object)
    X = train_pool[feats].to_numpy(float)
    preds = {k: np.empty(len(y), dtype=object) for k in ("LR", "RF")}
    for sess, dry_s in DRY_FOLDS.items():
        te = ((train_pool.wetness != "Dry") & (train_pool.sample_id == sess)) | \
             ((train_pool.wetness == "Dry") & train_pool.standoff_mm.isin(dry_s))
        te = te.to_numpy()
        for name, make in (("LR", lambda: lr(a.C)), ("RF", lambda: rf(0, True))):
            preds[name][te] = make().fit(X[~te], y[~te]).predict(X[te])
        lines.append(f"\nFold {sess}: test = wet session {sess} + dry at {dry_s} mm  "
                     f"(n_test={int(te.sum())}, train: " + ", ".join(
                         f"{c}={int((y[~te] == c).sum())}" for c in WET) + ")")
    lines.append("\nHeld-out results (every row is tested once, on a board and standoff the model never saw)")
    for name in ("LR", "RF"):
        p = preds[name]
        lines.append(f"  {name}: balanced accuracy {balanced_accuracy_score(y, p):.2f}, accuracy {np.mean(p == y):.2f}  "
                     f"(chance 0.33)")
        lines.append(f"    confusion rows=true, cols=pred ({'/'.join(WET)}): {confusion_matrix(y, p, labels=WET).tolist()}")
        for c in WET:
            m = y == c
            lines.append(f"    {c:<9} recall {np.mean(p[m] == c):.2f} (n={m.sum()})")

    final = rf(0, True).fit(X, y) if a.model == "rf" else lr(a.C).fit(X, y)
    if len(surface_dry):
        ps = final.predict(surface_dry[feats].to_numpy(float))
        lines.append(f"\nFinal {a.model.upper()} on the 'Surface dry' rows (surface dry, interior still wet): "
                     + ", ".join(f"{c}={int((ps == c).sum())}" for c in WET))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.model == "rf":
        joblib.dump(final, out / "wetness_model.joblib")
        (out / "model_meta.json").write_text(json.dumps({
            "model": "RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight=balanced, random_state=0)"
                     " after median imputation", "features": feats, "classes": [str(c) for c in final.classes_],
            "n_rows": len(y), "trained_at": datetime.now().isoformat(timespec="seconds"),
            "dataset": "10-22 dry + 10-29 + 10-31 (ml/train_subset.py)"}, indent=1), encoding="utf-8")
        saved = "wetness_model.joblib (needs scikit-learn; App/moisture_model.py cannot load it)"
    else:
        export(final, feats, out / "wetness_model.json", len(y))
        saved = "wetness_model.json"
    lines.append(f"\nSaved {out / saved.split()[0]} -- {a.model.upper()} fit on all {len(y)} rows. Not copied into App/.")
    text = "\n".join(lines)
    print(text)
    (out / "training_report.txt").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
