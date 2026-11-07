"""Can re-tuning the decision threshold recover the converters lost with a missed
periventricular indicator?  And what do false-positive indicators cost?

Same models, refits, and perturbation as indicator_error_simulation.py (no training
beyond the identical refit of the nested-CV configurations).

1. Missed indicators (positive -> 0 with probability r). For each outer fold and
   draw, the threshold is re-chosen ON THE TEST FOLD ITSELF as the largest threshold
   whose sensitivity is at least the clean sensitivity at 0.5 of that fold. This is
   the most favourable re-tuning possible (an oracle); the specificity it leaves is
   therefore an upper bound on what re-tuning can preserve.
   Reported: clean specificity at 0.5, specificity at 0.5 with the automated
   indicator, and specificity at the re-tuned threshold (matched sensitivity).
2. False-positive indicators (negative -> 1 with probability q), threshold 0.5:
   non-converters newly flagged as converters, as a share of all non-converters,
   and specificity.

Usage: python indicator_threshold_check.py [n_draws]
Output: ../results/indicator_threshold_check.csv
"""
import json
import os
import sys

import numpy as np
import optuna
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from nested_cv import TARGET, paths, search_space  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from indicator_error_simulation import MODELS, PV, TITLES, perturb  # noqa: E402

OUT = os.path.join(ROOT, "analyses_2026-11", "results")
SETTINGS = [("miss", "r=0.2", 0.2, 0.0), ("miss", "r=0.5", 0.5, 0.0), ("miss", "r=1.0", 1.0, 0.0),
            ("miss", "MSLesSeg/DiceCE", 0.209, 0.0), ("miss", "MSLesSeg/CATMIL", 0.102, 0.0),
            ("miss", "3D-MR-MS/DiceCE", 0.827, 0.0), ("miss", "3D-MR-MS/CATMIL", 0.800, 0.0),
            ("fp", "q=0.1", 0.0, 0.1), ("fp", "q=0.2", 0.0, 0.2)]


def sens_spec(p, y, t):
    pos = p >= t
    return (pos & (y == 1)).sum() / (y == 1).sum(), (~pos & (y == 0)).sum() / (y == 0).sum()


def matched_threshold(p, y, target_sens):
    """Largest threshold with sensitivity >= target (oracle, on the test fold)."""
    conv = np.sort(p[y == 1])[::-1]
    need = int(np.ceil(target_sens * len(conv) - 1e-9))
    return conv[need - 1] if need > 0 else np.inf


def main(n_draws=200):
    rows = []
    for m in MODELS:
        folds = json.load(open(os.path.join(ROOT, "results", "nested_cv", f"{m}.json")))["folds"]
        fitted = []
        for k, f in enumerate(folds):
            X_tr = pd.read_csv(paths["X_train_paths"][k])
            y_tr = pd.read_csv(paths["y_train_paths"][k])[TARGET].to_numpy().astype(int)
            X_va = pd.read_csv(paths["X_val_paths"][k])
            y_va = pd.read_csv(paths["y_val_paths"][k])[TARGET].to_numpy().astype(int)
            model = search_space(m, optuna.trial.FixedTrial(f["params"]), X_tr.shape[1])
            model.fit(X_tr, y_tr)
            fitted.append((model, X_va, y_va, model.predict_proba(X_va)[:, 1]))
        clean = [sens_spec(p0, y, 0.5) for _, _, y, p0 in fitted]
        clean_sens, clean_spec = np.mean([s for s, _ in clean]), np.mean([s for _, s in clean])
        # fair baseline: the same oracle re-tuning applied to the clean predictions
        clean_spec_rt = np.mean([sens_spec(p0, y, matched_threshold(p0, y, s))[1]
                                 for (_, _, y, p0), (s, _) in zip(fitted, clean)])
        for group, name, r, q in SETTINGS:
            rng = np.random.default_rng(42)
            draws = 1 if r == 1.0 else n_draws
            spec05, spec_rt, sens05, new_fa = [], [], [], []
            for _ in range(draws):
                f_spec05, f_spec_rt, f_sens05, n_fa, n_neg = [], [], [], 0, 0
                for (model, X_va, y_va, p0), (s_clean, _) in zip(fitted, clean):
                    Xp = X_va.copy()
                    Xp[PV] = perturb(X_va[PV].to_numpy(), r, q, rng)
                    p = model.predict_proba(Xp)[:, 1]
                    s, sp = sens_spec(p, y_va, 0.5)
                    f_sens05.append(s)
                    f_spec05.append(sp)
                    f_spec_rt.append(sens_spec(p, y_va, matched_threshold(p, y_va, s_clean))[1])
                    n_fa += int(((p >= 0.5) & (p0 < 0.5) & (y_va == 0)).sum())
                    n_neg += int((y_va == 0).sum())
                spec05.append(np.mean(f_spec05))
                spec_rt.append(np.mean(f_spec_rt))
                sens05.append(np.mean(f_sens05))
                new_fa.append(n_fa / n_neg)
            row = {"model": TITLES[m], "group": group, "setting": name, "r_miss": r, "q_false_pos": q,
                   "n_draws": draws, "clean_sensitivity": clean_sens, "clean_specificity": clean_spec,
                   "clean_specificity_retuned": clean_spec_rt,
                   "sensitivity_at_0.5": np.mean(sens05), "specificity_at_0.5": np.mean(spec05),
                   "specificity_retuned_matched_sens": np.mean(spec_rt),
                   "share_nonconverters_newly_flagged": np.mean(new_fa)}
            rows.append(row)
            print(f"{TITLES[m]:8s} {name:18s} sens {row['sensitivity_at_0.5']:.3f}/{clean_sens:.3f}  "
                  f"spec@0.5 {row['specificity_at_0.5']:.3f}/{clean_spec:.3f}  "
                  f"spec retuned {row['specificity_retuned_matched_sens']:.3f}/{clean_spec_rt:.3f}  "
                  f"new FA {row['share_nonconverters_newly_flagged']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "indicator_threshold_check.csv"), index=False)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
