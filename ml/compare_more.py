"""Try more classifiers on Dry/Damp/Saturated, with the same data, folds and metrics as compare_ordinal.py.

All hyperparameters are fixed (no tuning): with 36-60 rows any tuning would mostly fit noise.
Every model gets median imputation + standardisation first and balanced class weights where the
model supports them.

  LR            baseline logistic regression (C=0.3)
  SVM linear    support vector machine, linear kernel
  SVM RBF       support vector machine, RBF kernel
  LDA           linear discriminant analysis with shrinkage
  PCA+LDA       5 principal components, then LDA
  PLS-DA        partial least squares on one-hot classes (3 components)
  kNN           5 nearest neighbours, distance-weighted
  GP            Gaussian process classifier (RBF kernel)
  Extra-Trees   extremely randomised trees, 300 trees
  Grad-Boost    histogram gradient boosting
  Naive Bayes   Gaussian naive Bayes

Usage:  python ml/compare_more.py --out ml/v14/more_models_report.txt
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_models import FEATURES, WET, group_key, splits  # noqa: E402
from compare_ordinal import score_lines, vs_baseline  # noqa: E402
from train_subset import DAYS, DRY_FOLDS, LEVEL  # noqa: E402

HERE = Path(__file__).resolve().parent / "v14"


def pipe(clf):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), clf)


class Sk:
    def __init__(self, make):
        self.make = make

    def fit(self, X, y, groups):
        if len(np.unique(y)) < 2:
            self.const = y[0]
            return self
        self.const = None
        self.m = self.make().fit(X, y)
        return self

    def predict(self, X):
        return np.full(len(X), self.const, dtype=object) if self.const is not None else self.m.predict(X)


class PlsDa:
    def fit(self, X, y, groups):
        self.classes = np.array(sorted(np.unique(y)), dtype=object)
        if len(self.classes) < 2:
            return self
        Y = (y[:, None] == self.classes[None, :]).astype(float)
        w = 1.0 / Y.sum(0)                       # balance classes by weighting rows
        self.pre = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(X)
        Z = self.pre.transform(X)
        self.m = PLSRegression(n_components=min(3, Z.shape[1], len(y) - 1), scale=False).fit(Z, Y * (Y @ w)[:, None])
        return self

    def predict(self, X):
        if len(self.classes) < 2:
            return np.full(len(X), self.classes[0], dtype=object)
        return self.classes[self.m.predict(self.pre.transform(X)).argmax(1)]


MODELS = {
    "LR (baseline)": lambda: Sk(lambda: pipe(LogisticRegression(C=0.3, max_iter=5000, class_weight="balanced"))),
    "SVM linear": lambda: Sk(lambda: pipe(SVC(kernel="linear", C=0.3, class_weight="balanced"))),
    "SVM RBF": lambda: Sk(lambda: pipe(SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced"))),
    "LDA": lambda: Sk(lambda: pipe(LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))),
    "PCA+LDA": lambda: Sk(lambda: pipe(make_pipeline(PCA(n_components=5), LinearDiscriminantAnalysis()))),
    "PLS-DA": PlsDa,
    "kNN": lambda: Sk(lambda: pipe(KNeighborsClassifier(n_neighbors=5, weights="distance"))),
    "GP": lambda: Sk(lambda: pipe(GaussianProcessClassifier(kernel=1.0 * RBF(5.0), random_state=0))),
    "Extra-Trees": lambda: Sk(lambda: pipe(ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                                                                random_state=0, n_jobs=-1))),
    "Grad-Boost": lambda: Sk(lambda: pipe(HistGradientBoostingClassifier(max_iter=100, min_samples_leaf=3, max_depth=3,
                                                                         class_weight="balanced", random_state=0))),
    "Naive Bayes": lambda: Sk(lambda: pipe(GaussianNB())),
}


def cv_run(X, y, groups, folds):
    preds = {n: np.empty(len(y), dtype=object) for n in MODELS}
    for tr, te in folds:
        for n, mk in MODELS.items():
            try:
                preds[n][te] = mk().fit(X[tr], y[tr], groups[tr]).predict(X[te])
            except Exception as e:                      # a model that cannot fit this fold counts as all-wrong
                preds[n][te] = "ERROR"
                print(f"  ! {n} failed on a fold: {type(e).__name__}: {e}", file=sys.stderr)
    return preds


def ranking(y, preds, tag):
    from sklearn.metrics import balanced_accuracy_score
    rows = sorted(((balanced_accuracy_score(y, p), n) for n, p in preds.items()), reverse=True)
    return f"  {tag}: " + ", ".join(f"{n} {s:.2f}" for s, n in rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(HERE / "more_models_report.txt"))
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    names = list(MODELS)
    lines = ["More classifiers for Dry/Damp/Saturated (same data and folds as compare_ordinal.py)",
             "Chance balanced accuracy is 0.33. Hyperparameters fixed, not tuned. Small data: a few points is noise."]
    summary = []

    d = pd.read_csv(HERE / "dataset_A_all_useful.csv")
    d = d[d["quality_ok"] == 1].reset_index(drop=True)
    wa = d[(d["material"] == "Wood + drywall") & d["wetness"].isin(WET) & d["standoff_mm"].isin([40.0, 50.0, 60.0])].reset_index(drop=True)
    y, X, g = wa["wetness"].to_numpy(object), wa[FEATURES].to_numpy(float), wa["cv_group"].to_numpy()
    lines.append(f"\n=== A) Dataset A wetness: {len(y)} rows, 28 features ===")
    for kind in ("grouped 5-fold", "leave-one-standoff-out", "leave-one-day-out"):
        preds = cv_run(X, y, g, splits(kind, X, y, g, wa))
        lines += [f"\n  {kind}"] + score_lines(y, preds, names) + vs_baseline(y, preds, names)
        summary.append(ranking(y, preds, "A " + kind))

    s = d[d["meta_date"].isin(DAYS) & (d["material"] == "Wood + drywall") & d["wetness"].isin(WET)].reset_index(drop=True)
    feats = [f for f in FEATURES if f not in LEVEL]
    ys, Xs, gs = s["wetness"].to_numpy(object), s[feats].to_numpy(float), s["cv_group"].to_numpy()
    folds = []
    for sess, dry_s in DRY_FOLDS.items():
        te = (((s.wetness != "Dry") & (s.sample_id == sess)) | ((s.wetness == "Dry") & s.standoff_mm.isin(dry_s))).to_numpy()
        folds.append((np.where(~te)[0], np.where(te)[0]))
    preds = cv_run(Xs, ys, gs, folds)
    lines.append(f"\n=== B) 10-22 + 10-29 + 10-31 subset: {len(ys)} rows, 26 shape features, held-out board + standoffs ===")
    lines += score_lines(ys, preds, names) + vs_baseline(ys, preds, names)
    lines.append("    Damp recall: " + ", ".join(f"{n} {np.mean(preds[n][ys == 'Damp'] == 'Damp'):.2f}" for n in names))
    summary.append(ranking(ys, preds, "B subset"))

    lines += ["\n=== Ranking by balanced accuracy per test ==="] + summary
    text = "\n".join(lines)
    print(text)
    Path(a.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
