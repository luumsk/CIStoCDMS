"""Bootstrap confidence intervals of the cross-validated AUC.

For each model, reads the held-out probabilities in results/<model>/pred_proba.json
and the labels in data/val/y_val_fold*.csv, and reports the mean AUC over the
five folds (the value in the results table) and a 95% percentile interval:
the predictions of each fold are resampled with replacement and the mean fold
AUC is recomputed for each resample.

Usage (from the repository root): python scripts/auc_bootstrap.py [n_resamples]
"""
import csv
import json
import os
import random
import statistics as st
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
MODELS = ["catboost", "xgboost", "lgbm", "rf", "svm"]


def auc(y, p):
    """Mann-Whitney AUC with ties counted as one half."""
    pos = [b for a, b in zip(y, p) if a == 1]
    neg = [b for a, b in zip(y, p) if a == 0]
    s = sum(1 if a > b else 0.5 if a == b else 0 for a in pos for b in neg)
    return s / (len(pos) * len(neg))


def main(n_resamples=2000):
    labels = [
        [int(r["group"]) for r in csv.DictReader(open(os.path.join(ROOT, "data", "val", f"y_val_fold{i}.csv")))]
        for i in range(1, 6)
    ]
    lo, hi = int(0.025 * n_resamples), int(0.975 * n_resamples) - 1
    random.seed(0)
    for m in MODELS:
        probs = json.load(open(os.path.join(ROOT, "results", m, "pred_proba.json")))
        fold_auc = [auc(y, p) for y, p in zip(labels, probs)]
        boot = []
        for _ in range(n_resamples):
            a = []
            for y, p in zip(labels, probs):
                idx = [random.randrange(len(y)) for _ in y]
                yy = [y[i] for i in idx]
                if 0 < sum(yy) < len(yy):
                    a.append(auc(yy, [p[i] for i in idx]))
            boot.append(st.mean(a))
        boot.sort()
        print(f"{m:9s} mean fold AUC {st.mean(fold_auc):.4f}  95% CI [{boot[lo]:.3f}, {boot[hi]:.3f}]")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2000)
