"""Compare nested cross-validation with the original (non-nested) evaluation.

For each model with results/nested_cv/<model>.json, prints the mean over the
five outer folds of AUC, accuracy, F1, precision, recall and specificity
(threshold 0.5), the bootstrap 95% interval of the mean fold AUC, and the
original mean fold AUC from results/<model>/pred_proba.json on the same folds.

With --ablation, prints the mean fold AUC and its interval for all variables,
the four regional MRI indicators only, and all other variables only
(results/nested_cv/<model>_mri.json and <model>_nonmri.json).

Usage (from the repository root): python scripts/nested_cv_summary.py [--ablation]
"""
import csv
import json
import os
import random
import statistics as st
import sys

from auc_bootstrap import ROOT, auc

MODELS = ["catboost", "xgboost", "lgbm", "rf", "svm", "lr"]


def fold_metrics(y, p):
    pred = [int(v >= 0.5) for v in p]
    tp = sum(a == 1 and b == 1 for a, b in zip(y, pred))
    tn = sum(a == 0 and b == 0 for a, b in zip(y, pred))
    fp = sum(a == 0 and b == 1 for a, b in zip(y, pred))
    fn = sum(a == 1 and b == 0 for a, b in zip(y, pred))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn)
    return {"AUC": auc(y, p), "ACC": (tp + tn) / len(y),
            "F1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "Prec": prec, "Rec": rec, "Spec": tn / (tn + fp)}


def boot_ci(ys, ps, n=2000, seed=0):
    random.seed(seed)
    vals = []
    for _ in range(n):
        a = []
        for y, p in zip(ys, ps):
            idx = [random.randrange(len(y)) for _ in y]
            yy = [y[i] for i in idx]
            if 0 < sum(yy) < len(yy):
                a.append(auc(yy, [p[i] for i in idx]))
        vals.append(st.mean(a))
    vals.sort()
    return vals[int(0.025 * n)], vals[int(0.975 * n) - 1]


def main():
    labels = [[int(r["group"]) for r in csv.DictReader(open(os.path.join(ROOT, "data", "val", f"y_val_fold{i}.csv")))]
              for i in range(1, 6)]
    for m in MODELS:
        path = os.path.join(ROOT, "results", "nested_cv", f"{m}.json")
        if not os.path.exists(path):
            print(f"{m:9s} (no nested results)")
            continue
        folds = json.load(open(path))["folds"]
        ys = [f["y_true"] for f in folds]
        ps = [f["proba"] for f in folds]
        assert ys == labels, "outer folds differ from data/val"
        mets = [fold_metrics(y, p) for y, p in zip(ys, ps)]
        mean = {k: st.mean(x[k] for x in mets) for k in mets[0]}
        lo, hi = boot_ci(ys, ps)
        orig = json.load(open(os.path.join(ROOT, "results", m, "pred_proba.json")))
        orig_auc = st.mean(auc(y, p) for y, p in zip(labels, orig))
        print(f"{m:9s} nested AUC {mean['AUC']:.4f} [{lo:.3f}, {hi:.3f}]  original {orig_auc:.4f}  "
              + "  ".join(f"{k} {v:.4f}" for k, v in mean.items() if k != "AUC"))


def ablation():
    for m in MODELS:
        cells = []
        for suffix, label in (("", "all"), ("_mri", "MRI only"), ("_nonmri", "non-MRI only")):
            path = os.path.join(ROOT, "results", "nested_cv", f"{m}{suffix}.json")
            if not os.path.exists(path):
                cells.append(f"{label} n/a")
                continue
            folds = json.load(open(path))["folds"]
            ys, ps = [f["y_true"] for f in folds], [f["proba"] for f in folds]
            lo, hi = boot_ci(ys, ps)
            cells.append(f"{label} {st.mean(auc(y, p) for y, p in zip(ys, ps)):.4f} [{lo:.3f}, {hi:.3f}]")
        print(f"{m:9s} " + "  ".join(cells))


if __name__ == "__main__":
    ablation() if "--ablation" in sys.argv else main()
