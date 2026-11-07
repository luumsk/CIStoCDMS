"""Does a model TRAINED on automated (noisy) periventricular indicators recover the
cost of missed indicators?

indicator_error_simulation.py trains on the clinical indicators and perturbs only the
test fold (deployment of an existing model). Here the same perturbation (positive -> 0
with probability r, independently per patient) is applied to BOTH the outer training
fold and the outer test fold, and each model is refitted with the hyperparameters
selected by nested CV (results/nested_cv/<model>.json; no new search). The model can
therefore learn to rely less on the indicator.

Reported per setting (mean over draws; metrics averaged over the 5 outer folds):
  auc, sensitivity and specificity at 0.5,
  converters newly missed = converters flagged by the CLEAN model on clean data but not
  by the noisy-trained model on noisy data, as a share of all converters (same
  definition as the deployment simulation),
  specificity at the oracle threshold restoring the clean sensitivity (as in
  indicator_threshold_check.py).

Usage: python indicator_noisy_training.py [n_draws]
Output: ../results/indicator_noisy_training.csv
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from indicator_error_simulation import MODELS, PV, TITLES, perturb  # noqa: E402
from indicator_threshold_check import matched_threshold, sens_spec  # noqa: E402

OUT = os.path.join(ROOT, "analyses_2026-11", "results")
SETTINGS = [("r=0.2", 0.2), ("r=0.5", 0.5), ("r=0.8", 0.8), ("r=1.0", 1.0),
            ("MSLesSeg/DiceCE", 0.209), ("MSLesSeg/CATMIL", 0.102)]


def main(n_draws=20):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    rows = []
    for m in MODELS:
        folds = json.load(open(os.path.join(ROOT, "results", "nested_cv", f"{m}.json")))["folds"]
        data, clean = [], []
        for k, f in enumerate(folds):
            X_tr = pd.read_csv(paths["X_train_paths"][k])
            y_tr = pd.read_csv(paths["y_train_paths"][k])[TARGET].to_numpy().astype(int)
            X_va = pd.read_csv(paths["X_val_paths"][k])
            y_va = pd.read_csv(paths["y_val_paths"][k])[TARGET].to_numpy().astype(int)
            model = search_space(m, optuna.trial.FixedTrial(f["params"]), X_tr.shape[1])
            model.fit(X_tr, y_tr)
            p0 = model.predict_proba(X_va)[:, 1]
            data.append((f["params"], X_tr, y_tr, X_va, y_va, p0))
            clean.append(sens_spec(p0, y_va, 0.5)[0])
        clean_spec_rt = np.mean([sens_spec(p0, y, matched_threshold(p0, y, s))[1]
                                 for (_, _, _, _, y, p0), s in zip(data, clean)])
        for name, r in SETTINGS:
            rng = np.random.default_rng(42)
            draws = 1 if r == 1.0 else n_draws
            res = {k: [] for k in ("auc", "sens", "spec", "lost", "spec_rt")}
            for _ in range(draws):
                f_auc, f_sens, f_spec, f_rt, n_lost, n_conv = [], [], [], [], 0, 0
                for (params, X_tr, y_tr, X_va, y_va, p0), s_clean in zip(data, clean):
                    Xt, Xv = X_tr.copy(), X_va.copy()
                    Xt[PV] = perturb(X_tr[PV].to_numpy(), r, 0.0, rng)
                    Xv[PV] = perturb(X_va[PV].to_numpy(), r, 0.0, rng)
                    model = search_space(m, optuna.trial.FixedTrial(params), X_tr.shape[1])
                    model.fit(Xt, y_tr)
                    p = model.predict_proba(Xv)[:, 1]
                    f_auc.append(roc_auc_score(y_va, p))
                    s, sp = sens_spec(p, y_va, 0.5)
                    f_sens.append(s)
                    f_spec.append(sp)
                    f_rt.append(sens_spec(p, y_va, matched_threshold(p, y_va, s_clean))[1])
                    n_lost += int(((p0 >= 0.5) & (p < 0.5) & (y_va == 1)).sum())
                    n_conv += int((y_va == 1).sum())
                for key, v in zip(res, (np.mean(f_auc), np.mean(f_sens), np.mean(f_spec),
                                        n_lost / n_conv, np.mean(f_rt))):
                    res[key].append(v)
            row = {"model": TITLES[m], "setting": name, "r_miss": r, "n_draws": draws,
                   "clean_sensitivity": np.mean(clean), "clean_specificity_retuned": clean_spec_rt,
                   "auc_mean": np.mean(res["auc"]), "sensitivity_at_0.5": np.mean(res["sens"]),
                   "specificity_at_0.5": np.mean(res["spec"]),
                   "share_converters_newly_missed": np.mean(res["lost"]),
                   "specificity_retuned_matched_sens": np.mean(res["spec_rt"])}
            rows.append(row)
            print(f"{TITLES[m]:8s} {name:16s} AUC {row['auc_mean']:.3f}  sens {row['sensitivity_at_0.5']:.3f}"
                  f"/{row['clean_sensitivity']:.3f}  lost {row['share_converters_newly_missed']:.3f}  "
                  f"spec retuned {row['specificity_retuned_matched_sens']:.3f}/{clean_spec_rt:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "indicator_noisy_training.csv"), index=False)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
