"""Test ordinal and two-stage models against the plain logistic regression for Dry/Damp/Saturated.

Models (all take the same features and the same folds; see compare_models.py for the data):
  LR           plain multinomial logistic regression, balanced class weights, C=0.3 (the baseline)
  Ridge-ord    ridge regression on Dry=0 / Damp=1 / Saturated=2, cut at 0.5 and 1.5.
               alpha is picked by an inner grouped CV on each training fold.
  Frank-Hall   ordinal logistic: two binary logistic regressions (>Dry, >Damp), combined into
               P(Dry), P(Damp), P(Saturated)
  Two-stage LR stage 1 wet-vs-dry (logistic), stage 2 Damp-vs-Saturated on wet rows (logistic)
  Two-stage RF stage 1 wet-vs-dry (random forest, the model that was best at wet-vs-dry),
               stage 2 as above

Two evaluations:
  A) dataset A wetness rows (60 rows, 40/50/60 mm, all 28 features): grouped 5-fold,
     leave-one-standoff-out, leave-one-day-out
  B) the 10-22 + 10-29 + 10-31 subset (36 rows, shape-only features): two folds, each holding out one
     whole wet session plus half of the dry standoffs (same as train_subset.py)

Besides balanced accuracy, reports mean error in class steps (Dry<->Saturated counts 2) and the number
of Dry<->Saturated mistakes, since an ordinal model should avoid those.

Usage:  python ml/compare_ordinal.py --out ml/v14/ordinal_report.txt
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_models import FEATURES, WET, group_key, splits  # noqa: E402
from train_subset import DAYS, DRY_FOLDS, LEVEL  # noqa: E402

HERE = Path(__file__).resolve().parent / "v14"
ORD = {c: i for i, c in enumerate(WET)}
C_FIXED = 0.3
ALPHAS = (1.0, 10.0, 100.0, 1000.0)


def prep():
    return [SimpleImputer(strategy="median"), StandardScaler()]


def lr_bin(C=C_FIXED):
    return make_pipeline(*prep(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))


def rf_bin(seed=0):
    return make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(
        n_estimators=200, min_samples_leaf=2, random_state=seed, n_jobs=-1, class_weight="balanced"))


def fit_bin(make, X, yb):
    """Binary classifier, or a constant if the training data has only one class."""
    if len(np.unique(yb)) < 2:
        return ("const", yb[0])
    return ("model", make().fit(X, yb))


def pred_bin(m, X):
    return np.full(len(X), m[1]) if m[0] == "const" else m[1].predict(X)


def proba_bin(m, X):
    if m[0] == "const":
        return np.full(len(X), float(m[1]))
    return m[1].predict_proba(X)[:, list(m[1].classes_).index(1)] if 1 in m[1].classes_ else np.zeros(len(X))


class Baseline:
    def fit(self, X, y, groups):
        self.m = fit_bin(lr_bin, X, y)
        return self

    def predict(self, X):
        return pred_bin(self.m, X)


class RidgeOrd:
    def fit(self, X, y, groups):
        t = np.array([ORD[v] for v in y], float)
        w = compute_sample_weight("balanced", y)
        self.alpha = self._pick(X, t, y, w, groups)
        self.m = make_pipeline(*prep(), Ridge(alpha=self.alpha)).fit(X, t, ridge__sample_weight=w)
        return self

    @staticmethod
    def _pick(X, t, y, w, groups):
        n = min(5, len(np.unique(groups)))
        if n < 2:
            return 100.0
        best = None
        for al in ALPHAS:
            p = np.empty(len(y), dtype=object)
            for a, b in GroupKFold(n).split(X, y, groups):
                m = make_pipeline(*prep(), Ridge(alpha=al)).fit(X[a], t[a], ridge__sample_weight=w[a])
                p[b] = cut(m.predict(X[b]))
            s = balanced_accuracy_score(y, p)
            if best is None or s > best[0]:
                best = (s, al)
        return best[1]

    def predict(self, X):
        return cut(self.m.predict(X))


def cut(v):
    return np.array([WET[0] if x < 0.5 else WET[1] if x < 1.5 else WET[2] for x in v], dtype=object)


class FrankHall:
    def fit(self, X, y, groups):
        t = np.array([ORD[v] for v in y])
        self.gt0 = fit_bin(lr_bin, X, (t >= 1).astype(int))
        self.gt1 = fit_bin(lr_bin, X, (t >= 2).astype(int))
        return self

    def predict(self, X):
        p1, p2 = proba_bin(self.gt0, X), proba_bin(self.gt1, X)
        probs = np.column_stack([1 - p1, np.clip(p1 - p2, 0, None), p2])
        return np.array(WET, dtype=object)[probs.argmax(1)]


class TwoStage:
    def __init__(self, stage1):
        self.stage1 = stage1

    def fit(self, X, y, groups):
        wet = y != "Dry"
        self.s1 = fit_bin(self.stage1, X, wet.astype(int))
        self.s2 = fit_bin(lr_bin, X[wet], (y[wet] == "Saturated").astype(int)) if wet.any() else ("const", 0)
        return self

    def predict(self, X):
        is_wet = pred_bin(self.s1, X) == 1
        sat = pred_bin(self.s2, X) == 1
        return np.where(~is_wet, "Dry", np.where(sat, "Saturated", "Damp")).astype(object)


MODELS = {"LR (baseline)": Baseline, "Ridge-ord": RidgeOrd, "Frank-Hall": FrankHall,
          "Two-stage LR": lambda: TwoStage(lr_bin), "Two-stage RF": lambda: TwoStage(rf_bin)}


def score_lines(y, preds, names):
    out = []
    t = np.array([ORD[v] for v in y])
    for n in names:
        p = np.array([ORD[v] for v in preds[n]])
        out.append(f"    {n:<14} bal.acc {balanced_accuracy_score(y, preds[n]):.2f}  acc {np.mean(preds[n] == y):.2f}  "
                   f"mean step error {np.mean(np.abs(p - t)):.2f}  Dry<->Saturated mistakes {int((np.abs(p - t) == 2).sum())}")
    return out


def vs_baseline(y, preds, names):
    out = []
    base = preds["LR (baseline)"] == y
    for n in names[1:]:
        ok = preds[n] == y
        b, c = int((base & ~ok).sum()), int((~base & ok).sum())
        pv = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
        out.append(f"    {n:<14} better on {c}, worse on {b} rows vs baseline (McNemar p={pv:.2f}; rows are not independent)")
    return out


def cv_run(X, y, groups, folds):
    preds = {n: np.empty(len(y), dtype=object) for n in MODELS}
    for tr, te in folds:
        for n, mk in MODELS.items():
            preds[n][te] = mk().fit(X[tr], y[tr], groups[tr]).predict(X[te])
    return preds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(HERE / "ordinal_report.txt"))
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    names = list(MODELS)
    lines = ["Ordinal and two-stage models vs plain logistic regression (Dry/Damp/Saturated)",
             "Chance balanced accuracy is 0.33. Small data: differences of a few points are noise."]

    d = pd.read_csv(HERE / "dataset_A_all_useful.csv")
    d = d[d["quality_ok"] == 1].reset_index(drop=True)
    wa = d[(d["material"] == "Wood + drywall") & d["wetness"].isin(WET) & d["standoff_mm"].isin([40.0, 50.0, 60.0])].reset_index(drop=True)
    y, X, g = wa["wetness"].to_numpy(object), wa[FEATURES].to_numpy(float), wa["cv_group"].to_numpy()
    lines.append(f"\n=== A) Dataset A wetness: {len(y)} rows ({', '.join(f'{c}={int((y == c).sum())}' for c in WET)}), 28 features ===")
    for kind in ("grouped 5-fold", "leave-one-standoff-out", "leave-one-day-out"):
        preds = cv_run(X, y, g, splits(kind, X, y, g, wa))
        lines += [f"\n  {kind}"] + score_lines(y, preds, names) + vs_baseline(y, preds, names)
        if kind == "leave-one-day-out":
            key = group_key(kind, wa).to_numpy()
            for k in sorted(np.unique(key)):
                m = key == k
                lines.append(f"      unseen {k} (n={m.sum()}): " + ", ".join(
                    f"{n.split(' (')[0]} {np.mean(preds[n][m] == y[m]):.2f}" for n in names))

    s = d[d["meta_date"].isin(DAYS) & (d["material"] == "Wood + drywall")]
    s = s[s["wetness"].isin(WET)].reset_index(drop=True)
    feats = [f for f in FEATURES if f not in LEVEL]
    ys, Xs, gs = s["wetness"].to_numpy(object), s[feats].to_numpy(float), s["cv_group"].to_numpy()
    folds = []
    for sess, dry_s in DRY_FOLDS.items():
        te = (((s.wetness != "Dry") & (s.sample_id == sess)) | ((s.wetness == "Dry") & s.standoff_mm.isin(dry_s))).to_numpy()
        folds.append((np.where(~te)[0], np.where(te)[0]))
    preds = cv_run(Xs, ys, gs, folds)
    lines.append(f"\n=== B) 10-22 + 10-29 + 10-31 subset: {len(ys)} rows ({', '.join(f'{c}={int((ys == c).sum())}' for c in WET)}), "
                 f"26 shape features, held-out board + standoffs ===")
    lines += score_lines(ys, preds, names) + vs_baseline(ys, preds, names)
    lines.append("    Damp recall: " + ", ".join(f"{n.split(' (')[0]} {np.mean(preds[n][ys == 'Damp'] == 'Damp'):.2f}" for n in names)
                 + f"  (n={int((ys == 'Damp').sum())})")

    text = "\n".join(lines)
    print(text)
    Path(a.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
