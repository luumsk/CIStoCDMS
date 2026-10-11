"""Nested cross-validation for the six classifiers.

The original notebooks tune hyperparameters with Optuna on the same five folds
that are used to report performance, and CatBoost additionally selects its best
iteration on the evaluation fold (eval_set + use_best_model). Both make the
reported estimates optimistic.

Here the outer loop is the same five folds (data/train, data/val), so results are
directly comparable. For each outer fold, hyperparameters are tuned with Optuna
only on the outer training set, by stratified 5-fold inner cross-validation, with
the same search spaces and the same objective as the notebooks (mean of AUC,
accuracy and F1). The selected model is refit on the whole outer training set,
without any evaluation set, and evaluated once on the outer validation fold.

Usage (from the repository root):
    python scripts/nested_cv.py <model> [n_trials] [n_jobs] [features]
    model:    catboost | xgboost | lgbm | rf | svm | lr
    features: all (default) | mri (the four regional MRI indicators) |
              nonmri (all other variables), for the ablation

Writes results/nested_cv/<model>.json (or <model>_<features>.json for the
ablation) with the held-out probabilities of each outer fold and the selected
hyperparameters.
"""
import json
import os
import sys

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from xgboost import XGBClassifier

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)
from src.config import RANDOM_STATE, TARGET  # noqa: E402
from src.paths import paths  # noqa: E402

INNER_FOLDS = 5
MRI_FEATURES = ["Periventricular_MRI", "Cortical_MRI", "Infratentorial_MRI", "Spinal_Cord_MRI"]


def search_space(name, trial, n_features=20):
    """Search spaces copied from notebooks/<model>.ipynb; one thread per model fit."""
    if name == "catboost":
        return CatBoostClassifier(
            iterations=trial.suggest_int("iterations", 500, 2000),
            depth=trial.suggest_int("depth", 2, 9),
            learning_rate=trial.suggest_float("learning_rate", 0.1, 0.9),
            random_strength=trial.suggest_int("random_strength", 10, 20),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 0.5),
            border_count=trial.suggest_int("border_count", 512, 2048),
            l2_leaf_reg=trial.suggest_int("l2_leaf_reg", 2, 30),
            loss_function="Logloss", eval_metric="AUC", random_seed=RANDOM_STATE,
            verbose=False, thread_count=1,
        )
    if name == "xgboost":
        return XGBClassifier(
            verbosity=0, objective="binary:logistic", eval_metric="logloss",
            booster="gbtree", grow_policy="depthwise",
            reg_lambda=trial.suggest_float("lambda", 1e-8, 1.0),
            subsample=trial.suggest_float("subsample", 0.2, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            max_depth=trial.suggest_int("max_depth", 1, 3),
            reg_alpha=trial.suggest_float("alpha", 1e-8, 0.5),
            learning_rate=trial.suggest_float("eta", 1e-8, 1.0),
            gamma=trial.suggest_float("gamma", 1e-8, 1.0),
            n_jobs=1,
        )
    if name == "lgbm":
        return LGBMClassifier(
            objective="binary", metric="binary_error", boosting_type="gbdt", verbose=-1,
            num_leaves=trial.suggest_int("num_leaves", 2, 256),
            learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
            n_estimators=trial.suggest_int("n_estimators", 10, 1000),
            max_depth=trial.suggest_int("max_depth", 1, 30),
            min_child_weight=trial.suggest_float("min_child_weight", 1e-3, 10.0),
            subsample=trial.suggest_float("subsample", 0.1, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.1, 1.0),
            n_jobs=1,
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 10, 200),
            max_depth=trial.suggest_int("max_depth", 2, 32, log=True),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
            class_weight=trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample"]),
            max_features=trial.suggest_int("max_features", 1, min(5, n_features)),
            criterion="log_loss", oob_score=True, random_state=RANDOM_STATE, n_jobs=1,
        )
    if name == "svm":
        kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])
        extra = {"degree": trial.suggest_int("degree", 2, 5)} if kernel == "poly" else {}
        return SVC(
            C=trial.suggest_float("C", 100, 1000), kernel=kernel,
            gamma=trial.suggest_categorical("gamma", ["scale", "auto"]),
            class_weight=trial.suggest_categorical("class_weight", ["balanced", None]),
            probability=True, random_state=RANDOM_STATE, **extra,
        )
    if name == "lr":
        return LogisticRegression(
            C=trial.suggest_float("C", 1e-3, 100),
            penalty=trial.suggest_categorical("penalty", ["l1", "l2"]),
            max_iter=trial.suggest_int("max_iter", 10, 2000),
            random_state=RANDOM_STATE, solver="liblinear",
        )
    raise ValueError(name)


def score(y, proba):
    """Objective of the notebooks: mean of AUC, accuracy and F1 (threshold 0.5)."""
    pred = (proba >= 0.5).astype(int)
    return (roc_auc_score(y, proba) + accuracy_score(y, pred) + f1_score(y, pred)) / 3


def fit_predict(model, X_train, y_train, X_eval):
    model.fit(X_train, y_train)
    return model.predict_proba(X_eval)[:, 1]


def tune(name, X, y, n_trials, n_jobs):
    n_features = X.shape[1]
    inner = list(StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE).split(X, y))

    def objective(trial):
        vals = []
        for tr, va in inner:
            model = search_space(name, trial, n_features)
            vals.append(score(y[va], fit_predict(model, X.iloc[tr], y[tr], X.iloc[va])))
        return float(np.mean(vals))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    return study.best_trial


def select(X, features):
    if features == "mri":
        return X[MRI_FEATURES]
    if features == "nonmri":
        return X.drop(columns=MRI_FEATURES)
    return X


def main(name, n_trials=100, n_jobs=4, features="all"):
    out = {"model": name, "n_trials": n_trials, "inner_folds": INNER_FOLDS, "features": features, "folds": []}
    for fold in range(5):
        X_tr = select(pd.read_csv(paths["X_train_paths"][fold]), features)
        y_tr = pd.read_csv(paths["y_train_paths"][fold])[TARGET].to_numpy().astype(int)
        X_va = select(pd.read_csv(paths["X_val_paths"][fold]), features)
        y_va = pd.read_csv(paths["y_val_paths"][fold])[TARGET].to_numpy().astype(int)
        best = tune(name, X_tr, y_tr, n_trials, n_jobs)
        model = search_space(name, optuna.trial.FixedTrial(best.params), X_tr.shape[1])
        proba = fit_predict(model, X_tr, y_tr, X_va)
        out["folds"].append({
            "params": best.params, "inner_score": best.value,
            "y_true": y_va.tolist(), "proba": proba.tolist(),
            "outer_auc": roc_auc_score(y_va, proba),
        })
        print(f"{name} fold {fold + 1}: inner {best.value:.4f}, outer AUC {out['folds'][-1]['outer_auc']:.4f}", flush=True)
    os.makedirs(os.path.join(ROOT, "results", "nested_cv"), exist_ok=True)
    suffix = "" if features == "all" else f"_{features}"
    with open(os.path.join(ROOT, "results", "nested_cv", f"{name}{suffix}.json"), "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args[0], *(int(a) for a in args[1:3]), *args[3:4])
