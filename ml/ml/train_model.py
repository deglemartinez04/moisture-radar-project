"""Train the Random Forest moisture classifier from build_dataset.py output.

Evaluation is grouped by recording: all windows from one .h5 file are either
in the training folds or in the test fold, never both. Windows from the same
recording are near-duplicates, so a random split would report ~100% accuracy
that does not hold on a new wall.

Reports:
- window accuracy (one prediction per 10-frame window, what the live tab shows)
- recording accuracy (majority vote over a recording's windows)
- bootstrap 95% interval, resampling whole recordings
- confusion matrix and feature importances

Then fits on all data and saves the model + metadata the App loads.

Usage:
    python ml/train_model.py ml/dataset.csv
    python ml/train_model.py ml/dataset.csv --out-dir ml/models
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.features import MODEL_FEATURES  # noqa: E402

MIN_RECORDINGS_PER_CLASS = 2
SEED = 0


def make_model():
    return RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )


def check_enough_data(df):
    per_class = df.groupby("moisture_level")["recording_id"].nunique()
    problems = []
    if len(per_class) < 2:
        problems.append(f"need at least 2 moisture classes, have {list(per_class.index)}")
    short = per_class[per_class < MIN_RECORDINGS_PER_CLASS]
    for label, n in short.items():
        problems.append(f"'{label}' has {n} recording(s), need >= {MIN_RECORDINGS_PER_CLASS}")
    if problems:
        print("Recordings per class (windows with a detected wall reflection only):")
        print(per_class.to_string() if len(per_class) else "  (none)")
        sys.exit("Not enough data to train:\n  - " + "\n  - ".join(problems))
    return per_class


def bootstrap_accuracy(df, pred_col, n_boot=1000):
    rng = np.random.default_rng(SEED)
    recs = df["recording_id"].unique()
    by_rec = {r: g for r, g in df.groupby("recording_id")}
    scores = []
    for _ in range(n_boot):
        sample = pd.concat([by_rec[r] for r in rng.choice(recs, size=len(recs), replace=True)])
        scores.append((sample[pred_col] == sample["moisture_level"]).mean())
    return np.percentile(scores, [2.5, 97.5])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", help="CSV from build_dataset.py")
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "models"))
    parser.add_argument("--keep-no-target", action="store_true",
                        help="also train on windows where no wall reflection was found")
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    if not args.keep_no_target:
        dropped = int((df["target_detected"] == 0).sum())
        if dropped:
            print(f"Dropping {dropped} of {len(df)} windows with no wall reflection detected.")
        df = df[df["target_detected"] == 1]

    per_class = check_enough_data(df)
    labels = sorted(per_class.index)
    X, y, groups = df[MODEL_FEATURES].to_numpy(float), df["moisture_level"].to_numpy(), df["recording_id"].to_numpy()

    n_splits = int(min(5, per_class.min()))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    df = df.copy()
    df["pred"] = ""
    for train_idx, test_idx in cv.split(X, y, groups):
        model = make_model().fit(X[train_idx], y[train_idx])
        df.iloc[test_idx, df.columns.get_loc("pred")] = model.predict(X[test_idx])

    window_acc = (df["pred"] == df["moisture_level"]).mean()
    lo, hi = bootstrap_accuracy(df, "pred")
    votes = df.groupby("recording_id").agg(
        truth=("moisture_level", "first"), vote=("pred", lambda p: p.value_counts().idxmax()))
    rec_acc = (votes["truth"] == votes["vote"]).mean()

    print(f"\n{len(df)} windows from {df['recording_id'].nunique()} recordings, "
          f"{n_splits}-fold cross-validation grouped by recording")
    print(f"Window accuracy:    {window_acc:.1%}  (95% bootstrap interval {lo:.1%} - {hi:.1%})")
    print(f"Recording accuracy: {rec_acc:.1%}  ({(votes['truth'] == votes['vote']).sum()}/{len(votes)} majority votes correct)")
    print(f"Chance level:       {1 / len(labels):.1%}")

    cm = confusion_matrix(df["moisture_level"], df["pred"], labels=labels)
    print("\nConfusion matrix (rows = true, columns = predicted, windows):")
    print(pd.DataFrame(cm, index=labels, columns=labels).to_string())
    wrong = votes[votes["truth"] != votes["vote"]]
    if len(wrong):
        print("\nRecordings misclassified by majority vote:")
        print(wrong.to_string())

    final = make_model().fit(X, y)
    importance = pd.Series(final.feature_importances_, index=MODEL_FEATURES).sort_values(ascending=False)
    print("\nFeature importances (final model):")
    print(importance.round(3).to_string())

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, out / "moisture_model.joblib")
    meta_path = Path(args.dataset).with_suffix(".meta.json")
    dataset_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(Path(args.dataset).resolve()),
        "features": MODEL_FEATURES,
        "labels": [str(c) for c in final.classes_],
        "window_frames": dataset_meta.get("window_frames"),
        "sensor_config": dataset_meta.get("sensor_config"),
        "n_windows": int(len(df)),
        "n_recordings": int(df["recording_id"].nunique()),
        "cv_window_accuracy": round(float(window_acc), 4),
        "cv_window_accuracy_95ci": [round(float(lo), 4), round(float(hi), 4)],
        "cv_recording_accuracy": round(float(rec_acc), 4),
    }
    (out / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved {out / 'moisture_model.joblib'} and model_meta.json")


if __name__ == "__main__":
    main()
