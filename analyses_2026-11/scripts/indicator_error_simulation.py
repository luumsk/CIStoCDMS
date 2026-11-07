"""How much does CIS->CDMS prediction degrade when the periventricular
indicator comes from a segmentation that misses lesions?

Setting (a deployment scenario, stated as a simulation): the models are trained
exactly as in the dissertation, on the clinically recorded indicators. At test
time, the periventricular indicator of each held-out patient is replaced by an
automated one. A patient whose indicator is 1 is turned to 0 with probability
r (all periventricular lesions missed); optionally a 0 is turned to 1 with
probability q (a false-positive component near the ventricles).

For each of the six models and each outer fold, the configuration selected by
nested CV (results/nested_cv/<model>.json) is refitted on the outer training
set exactly as in scripts/nested_cv.py, and evaluated on the outer validation
fold with the perturbed indicator. AUC = mean over the 5 outer folds, as in the
dissertation table; each setting is repeated with 200 random draws.

Two sets of r values:
  * a sweep r = 0, 0.1, ..., 1.0 (q = 0), and q = 0.1, 0.2 at r = 0;
  * "measured" r: the share of *small* periventricular lesions missed by each
    loss on the Chapter 3 test scans (Table tab:pv_missed, D = 3 mm, raw):
    MSLesSeg DiceCE 0.209, Tversky 0.119, FocalTversky 0.119, CATMIL 0.102;
    3D-MR-MS DiceCE 0.827, Tversky 0.833, FocalTversky 0.833, CATMIL 0.800.
    Applying a per-lesion rate to the per-patient indicator is the worst case
    in which the indicator rests on a single small lesion, as can occur at the
    first clinical episode. It is an upper bound on the damage, not an estimate
    for this cohort, whose images are not available.

The r = 0 row must reproduce the nested-CV AUCs of the dissertation.

Usage: python analyses_2026-11/scripts/indicator_error_simulation.py [n_draws] [model ...]
Outputs: analyses_2026-11/results/indicator_error_*.csv
"""
import json
import os
import sys

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from nested_cv import TARGET, paths, search_space  # noqa: E402

OUT = os.path.join(ROOT, "analyses_2026-11", "results")
MODELS = ["catboost", "xgboost", "lgbm", "rf", "svm", "lr"]
TITLES = dict(zip(MODELS, ["CatBoost", "XGBoost", "LGBM", "RF", "SVM", "LR"]))
PV = "Periventricular_MRI"
MEASURED = {
    ("MSLesSeg", "DiceCE"): 0.209, ("MSLesSeg", "Tversky"): 0.119,
    ("MSLesSeg", "FocalTversky"): 0.119, ("MSLesSeg", "CATMIL"): 0.102,
    ("3D-MR-MS", "DiceCE"): 0.827, ("3D-MR-MS", "Tversky"): 0.833,
    ("3D-MR-MS", "FocalTversky"): 0.833, ("3D-MR-MS", "CATMIL"): 0.800,
}


def settings():
    s = [("sweep", f"r={r:.1f}", r, 0.0) for r in np.round(np.arange(0, 1.01, 0.1), 1)]
    s += [("sweep_fp", f"q={q:.1f}", 0.0, q) for q in (0.1, 0.2)]
    s += [("measured", f"{d}/{loss}", r, 0.0) for (d, loss), r in MEASURED.items()]
    return s


def perturb(x, r, q, rng):
    x = x.copy()
    u = rng.random(len(x))
    x[(x == 1) & (u < r)] = 0
    x[(x == 0) & (u < q)] = 1
    return x


def main(n_draws=200, models=MODELS):
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for m in models:
        folds = json.load(open(os.path.join(ROOT, "results", "nested_cv", f"{m}.json")))["folds"]
        fitted = []
        for k, f in enumerate(folds):
            X_tr = pd.read_csv(paths["X_train_paths"][k])
            y_tr = pd.read_csv(paths["y_train_paths"][k])[TARGET].to_numpy().astype(int)
            X_va = pd.read_csv(paths["X_val_paths"][k])
            y_va = pd.read_csv(paths["y_val_paths"][k])[TARGET].to_numpy().astype(int)
            model = search_space(m, optuna.trial.FixedTrial(f["params"]), X_tr.shape[1])
            model.fit(X_tr, y_tr)
            p0 = model.predict_proba(X_va)[:, 1]
            fitted.append((model, X_va, y_va, roc_auc_score(y_va, p0), f["outer_auc"]))
        repro = [abs(a - b) for _, _, _, a, b in fitted]
        print(f"{m}: refit AUC per fold vs stored, max |diff| = {max(repro):.4f}", flush=True)
        for group, name, r, q in settings():
            rng = np.random.default_rng(42)
            draws = 1 if (r == 0 and q == 0) else n_draws
            aucs, sens, flips, conv_lost = [], [], [], []
            for _ in range(draws):
                fold_auc, fold_sens, n_flip, n_lost, n_all, n_conv = [], [], 0, 0, 0, 0
                for model, X_va, y_va, _, _ in fitted:
                    Xp = X_va.copy()
                    Xp[PV] = perturb(X_va[PV].to_numpy(), r, q, rng)
                    p = model.predict_proba(Xp)[:, 1]
                    p_clean = model.predict_proba(X_va)[:, 1]
                    fold_auc.append(roc_auc_score(y_va, p))
                    fold_sens.append(((p >= 0.5) & (y_va == 1)).sum() / (y_va == 1).sum())
                    n_flip += int(((p >= 0.5) != (p_clean >= 0.5)).sum())
                    # converters correctly flagged with the clinical indicator, missed with the automated one
                    n_lost += int(((p_clean >= 0.5) & (p < 0.5) & (y_va == 1)).sum())
                    n_all += len(y_va)
                    n_conv += int((y_va == 1).sum())
                aucs.append(np.mean(fold_auc))
                sens.append(np.mean(fold_sens))
                flips.append(n_flip / n_all)
                conv_lost.append(n_lost / n_conv)
            aucs, sens, flips, conv_lost = map(np.asarray, (aucs, sens, flips, conv_lost))
            rows.append({"model": TITLES[m], "group": group, "setting": name, "r_miss": r, "q_false_pos": q,
                         "n_draws": draws, "auc_mean": aucs.mean(), "auc_q025": np.quantile(aucs, 0.025),
                         "auc_q975": np.quantile(aucs, 0.975),
                         "sensitivity_mean": sens.mean(),
                         "share_patients_decision_flipped": flips.mean(),
                         "share_converters_newly_missed": conv_lost.mean(),
                         "refit_max_abs_diff_vs_stored": max(repro)})
            print(f"  {name:24s} AUC {aucs.mean():.4f} [{np.quantile(aucs, .025):.4f}, {np.quantile(aucs, .975):.4f}]"
                  f"  sens {sens.mean():.3f}  flipped {flips.mean():.3f}  converters lost {conv_lost.mean():.3f}",
                  flush=True)
        # Reference: the same model family evaluated with the indicator removed entirely
        # is the nested 'nonmri'-style floor only for all four MRI variables; here we give
        # the indicator-at-constant-0 floor, which is r = 1.0 in the sweep.
    df = pd.DataFrame(rows)
    tag = "" if list(models) == MODELS else "_" + "_".join(models)
    df.to_csv(os.path.join(OUT, f"indicator_error_simulation{tag}.csv"), index=False)
    piv = df[df.group != "measured"].pivot_table(index="setting", columns="model", values="auc_mean", sort=False)
    print(piv.round(3).to_string())
    print(df[df.group == "measured"].pivot_table(index="setting", columns="model", values="auc_mean",
                                                 sort=False).round(3).to_string())


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if a else 200, a[1:] or MODELS)
