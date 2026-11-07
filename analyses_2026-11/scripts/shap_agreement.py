"""Patient-level agreement of SHAP attributions across the six nested-CV models.

The dissertation reports that all six models rank the periventricular indicator
first in *global* importance (mean |SHAP|). This script tests the stronger claim
that the models also agree *patient by patient*, i.e. that the attribution of
each individual prediction has the same structure across model families.

Input: results/nested_cv/shap_<model>.json (5 outer folds, held-out patients;
written by scripts/nested_shap.py) and data/val/X_val_fold*.csv for feature
names and values. No model is refitted.

Measures (all on the 273 held-out patients, folds concatenated):
  1. Linear CKA between the patient x feature SHAP matrices of each model pair
     (invariant to the different output scales: log-odds vs probability), with
     a permutation null (patients shuffled in one model, 1000 draws).
  2. Per-patient Spearman correlation of the 20-dimensional SHAP vectors for
     each model pair; median and IQR over patients.
  3. Per-patient top-1 feature: share of patients whose largest |SHAP| is the
     periventricular indicator, for each model, and share of patients for whom
     all six models name the same top-1 feature.
  4. Direction: mean SHAP of the periventricular indicator for patients with
     indicator 1 vs 0, per model (sign agreement).

Usage (from the repository root, or anywhere):
    python analyses_2026-11/scripts/shap_agreement.py
Outputs: analyses_2026-11/results/shap_agreement_*.csv and .json
"""
import itertools
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "analyses_2026-11", "results")
MODELS = ["catboost", "xgboost", "lgbm", "rf", "svm", "lr"]
TITLES = dict(zip(MODELS, ["CatBoost", "XGBoost", "LGBM", "RF", "SVM", "LR"]))
PV = "Periventricular_MRI"
RNG = np.random.default_rng(42)


def load():
    X = pd.concat([pd.read_csv(os.path.join(ROOT, "data", "val", f"X_val_fold{f}.csv")) for f in range(1, 6)],
                  ignore_index=True)
    S = {}
    for m in MODELS:
        folds = json.load(open(os.path.join(ROOT, "results", "nested_cv", f"shap_{m}.json")))
        S[m] = np.vstack([np.asarray(f, dtype=float) for f in folds])
        assert S[m].shape == X.shape, (m, S[m].shape, X.shape)
    return X, S


def cka(a, b):
    a = a - a.mean(0)
    b = b - b.mean(0)
    hsic = np.linalg.norm(a.T @ b, "fro") ** 2
    return hsic / (np.linalg.norm(a.T @ a, "fro") * np.linalg.norm(b.T @ b, "fro"))


def main(n_perm=1000):
    os.makedirs(OUT, exist_ok=True)
    X, S = load()
    feats = list(X.columns)
    n = len(X)
    rows, per_patient = [], {}
    for a, b in itertools.combinations(MODELS, 2):
        obs = cka(S[a], S[b])
        null = np.array([cka(S[a], S[b][RNG.permutation(n)]) for _ in range(n_perm)])
        rho = np.array([spearmanr(S[a][i], S[b][i]).statistic for i in range(n)])
        # Null: same comparison between *different* patients (a derangement), which keeps any
        # agreement that comes only from the shared global importance order.
        perm = RNG.permutation(n)
        while (perm == np.arange(n)).any():
            perm = RNG.permutation(n)
        rho_null = np.array([spearmanr(S[a][i], S[b][perm[i]]).statistic for i in range(n)])
        per_patient[f"{a}-{b}"] = rho
        rows.append({"model_a": TITLES[a], "model_b": TITLES[b], "cka": obs,
                     "cka_null_mean": null.mean(), "cka_null_p99": np.quantile(null, 0.99),
                     "cka_perm_p": (1 + (null >= obs).sum()) / (1 + n_perm),
                     "patient_spearman_median": np.nanmedian(rho),
                     "patient_spearman_q25": np.nanquantile(rho, 0.25),
                     "patient_spearman_q75": np.nanquantile(rho, 0.75),
                     "other_patient_spearman_median": np.nanmedian(rho_null),
                     "share_same_patient_beats_other": float(np.nanmean(rho > rho_null))})
    pairs = pd.DataFrame(rows)
    pairs.to_csv(os.path.join(OUT, "shap_agreement_pairs.csv"), index=False)

    top1 = pd.DataFrame({TITLES[m]: [feats[j] for j in np.abs(S[m]).argmax(1)] for m in MODELS})
    top1.to_csv(os.path.join(OUT, "shap_agreement_top1_per_patient.csv"), index=False)
    all_same = (top1.nunique(axis=1) == 1)
    pv_val = X[PV].to_numpy()
    direction = []
    for m in MODELS:
        j = feats.index(PV)
        direction.append({"model": TITLES[m],
                          "share_top1_periventricular": float((top1[TITLES[m]] == PV).mean()),
                          "mean_shap_pv_if_1": float(S[m][pv_val == 1, j].mean()),
                          "mean_shap_pv_if_0": float(S[m][pv_val == 0, j].mean()),
                          "n_pv1": int((pv_val == 1).sum()), "n_pv0": int((pv_val == 0).sum())})
    direction = pd.DataFrame(direction)
    direction.to_csv(os.path.join(OUT, "shap_agreement_direction.csv"), index=False)

    all_rho = np.concatenate(list(per_patient.values()))
    summary = {
        "n_patients": n, "n_features": len(feats), "n_model_pairs": len(pairs), "n_perm": n_perm,
        "cka_min": float(pairs.cka.min()), "cka_median": float(pairs.cka.median()), "cka_max": float(pairs.cka.max()),
        "cka_null_p99_max": float(pairs.cka_null_p99.max()),
        "cka_all_pairs_p_max": float(pairs.cka_perm_p.max()),
        "patient_spearman_median_all_pairs": float(np.nanmedian(all_rho)),
        "patient_spearman_pairwise_median_range": [float(pairs.patient_spearman_median.min()),
                                                    float(pairs.patient_spearman_median.max())],
        "share_patients_all_six_same_top1": float(all_same.mean()),
        "share_patients_all_six_top1_is_periventricular": float((all_same & (top1.iloc[:, 0] == PV)).mean()),
        "share_top1_periventricular_per_model": dict(zip(direction.model, direction.share_top1_periventricular)),
        "pv_direction_positive_all_models": bool((direction.mean_shap_pv_if_1 > direction.mean_shap_pv_if_0).all()),
    }
    json.dump(summary, open(os.path.join(OUT, "shap_agreement_summary.json"), "w"), indent=2)
    print(pairs.round(3).to_string(index=False))
    print(direction.round(3).to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
