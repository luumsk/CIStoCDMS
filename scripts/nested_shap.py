"""SHAP attribution for the models selected by nested cross-validation.

For each model and outer fold, the configuration selected in
results/nested_cv/<model>.json is refitted on the outer training set exactly as
in nested_cv.py, and SHAP values are computed for the outer validation fold with
the same explainers as notebooks/explain.ipynb: TreeExplainer (raw output,
tree-path-dependent) for the tree ensembles, LinearExplainer (correlation-
dependent, training set as masker) for logistic regression, and KernelExplainer
on the class-1 probability (training set as background) for the SVM.

Writes results/nested_cv/shap_<model>.json and two figures in img/:
nested_shap_global_ranking.jpeg (mean |SHAP| averaged over the six models) and
nested_shap_by_models.jpeg (min-max normalized per model), drawn as in
notebooks/explain.ipynb.

Usage (from the repository root): python scripts/nested_shap.py [model ...]
"""
import json
import os
import sys

import matplotlib
import numpy as np
import optuna
import pandas as pd
import shap

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from nested_cv import ROOT, TARGET, paths, search_space  # noqa: E402

MODELS = ["catboost", "xgboost", "lgbm", "rf", "svm", "lr"]
TITLES = ["CatBoost", "XGBoost", "LGBM", "RF", "SVM", "LR"]
OUT = os.path.join(ROOT, "results", "nested_cv")


def fold_shap(name, params, X_tr, y_tr, X_va):
    model = search_space(name, optuna.trial.FixedTrial(params), X_tr.shape[1])
    model.fit(X_tr, y_tr)
    if name in ("catboost", "xgboost", "lgbm", "rf"):
        sv = shap.TreeExplainer(model, model_output="raw", feature_perturbation="tree_path_dependent").shap_values(X_va)
        sv = np.asarray(sv)
        if sv.ndim == 3:  # RF (and some versions of LightGBM) return one array per class
            sv = sv[..., 1]
        return sv
    if name == "lr":
        return shap.LinearExplainer(model, masker=X_tr, feature_perturbation="correlation_dependent").shap_values(X_va)
    return shap.KernelExplainer(lambda x: model.predict_proba(x)[:, 1], X_tr, nsamples="auto").shap_values(X_va)


def compute(name):
    folds = json.load(open(os.path.join(OUT, f"{name}.json")))["folds"]
    values = []
    for fold, f in enumerate(folds):
        X_tr = pd.read_csv(paths["X_train_paths"][fold])
        y_tr = pd.read_csv(paths["y_train_paths"][fold])[TARGET].to_numpy().astype(int)
        X_va = pd.read_csv(paths["X_val_paths"][fold])
        values.append(np.asarray(fold_shap(name, f["params"], X_tr, y_tr, X_va)).tolist())
        print(f"{name} fold {fold + 1} done", flush=True)
    with open(os.path.join(OUT, f"shap_{name}.json"), "w") as fh:
        json.dump(values, fh)


def figures():
    names = pd.read_csv(paths["X_val_paths"][0]).columns
    df = pd.DataFrame({m: np.abs(np.concatenate([np.asarray(f) for f in json.load(open(os.path.join(OUT, f"shap_{m}.json")))])).mean(0)
                       for m in MODELS}, index=names)
    os.makedirs(os.path.join(ROOT, "img"), exist_ok=True)
    matplotlib.rcParams["font.family"] = ["serif"]

    norm_df = ((df - df.min()) / (df.max() - df.min())).sort_index()
    plt.figure(figsize=(10, 8))
    sns.heatmap(norm_df, annot=True, cmap="Blues", linewidths=.5, cbar=False, fmt=".4f")
    plt.ylabel("")
    plt.xticks(ticks=plt.xticks()[0], labels=TITLES, ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, "img", "nested_shap_by_models.jpeg"), format="jpeg", dpi=600)
    plt.close()

    mean = df.mean(axis=1).sort_values(ascending=True)
    norm = plt.Normalize(mean.min(), mean.max())
    plt.figure(figsize=(10, 6))
    plt.barh(mean.index, mean, color=plt.cm.Blues(norm(mean * 1.75)))
    ax = plt.gca()
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="both", which="both", length=0)
    plt.xticks([])
    for p in ax.patches:
        ax.text(p.get_width() + 0.03, p.get_y() + p.get_height() / 2, f"{p.get_width():.4f}",
                ha="center", va="center", fontsize=8, color="black")
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, "img", "nested_shap_global_ranking.jpeg"), format="jpeg", dpi=600)
    plt.close()

    print(mean.sort_values(ascending=False).round(4).to_string())
    print("rank 1 per model:", {m: norm_df[m].idxmax() for m in MODELS})


if __name__ == "__main__":
    for m in (sys.argv[1:] or MODELS):
        compute(m)
    if all(os.path.exists(os.path.join(OUT, f"shap_{m}.json")) for m in MODELS):
        figures()
