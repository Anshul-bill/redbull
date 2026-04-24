"""Traditional ML benchmark: SVM, RF, GBM, XGBoost, CatBoost, LightGBM.

Runs StratifiedKFold CV with per-fold scaling to avoid leakage into the
cross-validation estimate, reports accuracy / macro-F1 / ROC-AUC (OVR)
as mean +/- std, and writes a leaderboard to outputs/model_comparison.csv.
The Deep Neural Network is intentionally evaluated in dnn_pipeline.py.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from xgboost import XGBClassifier

from config import CV_FOLDS, OUTPUTS_DIR, RANDOM_STATE
from preprocess import load_raw, split_xy
from train_lightgbm import build_lightgbm


def _build_models():
    return {
        "SVM": SVC(probability=True, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "GBM": GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(eval_metric="mlogloss", random_state=RANDOM_STATE),
        "CatBoost": CatBoostClassifier(
            iterations=100, verbose=0, random_state=RANDOM_STATE, allow_writing_files=False
        ),
        "LightGBM": build_lightgbm(),
    }


def _safe_roc_auc(y_true, proba, all_classes) -> float:
    """ROC-AUC OVR that tolerates folds missing some classes."""
    y_bin = label_binarize(y_true, classes=all_classes)
    if y_bin.shape[1] == 1:
        return float("nan")
    present = y_bin.sum(axis=0) > 0
    if present.sum() < 2:
        return float("nan")
    return roc_auc_score(y_bin[:, present], proba[:, present], average="macro")


def _encode_for_xgb(y, xgb_classes):
    """XGBoost needs contiguous 0..k-1 labels; map train labels through the known class list."""
    lookup = {c: i for i, c in enumerate(xgb_classes)}
    return np.array([lookup[v] for v in y])


def run_comparison():
    df = load_raw()
    X, y = split_xy(df)
    X_values = X.values
    y_values = y.values
    all_classes = np.sort(np.unique(y_values))

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    print("\n" + "=" * 78)
    print(f"{'Model':<16} | {'Accuracy':<16} | {'Macro F1':<16} | {'ROC-AUC (OVR)':<16}")
    print("-" * 78)

    for name, _ in _build_models().items():
        fold_acc, fold_f1, fold_auc = [], [], []

        for train_idx, test_idx in skf.split(X_values, y_values):
            X_tr, X_te = X_values[train_idx], X_values[test_idx]
            y_tr, y_te = y_values[train_idx], y_values[test_idx]

            scaler = StandardScaler().fit(X_tr)
            X_tr_s = scaler.transform(X_tr)
            X_te_s = scaler.transform(X_te)

            model = _build_models()[name]

            if name == "XGBoost":
                y_tr_enc = _encode_for_xgb(y_tr, all_classes)
                model.fit(X_tr_s, y_tr_enc)
                proba_model = model.predict_proba(X_te_s)
                proba = np.zeros((len(y_te), len(all_classes)))
                for i, c in enumerate(all_classes):
                    if i < proba_model.shape[1]:
                        proba[:, i] = proba_model[:, i]
                y_pred = all_classes[np.argmax(proba, axis=1)]
            else:
                model.fit(X_tr_s, y_tr)
                y_pred = model.predict(X_te_s)
                proba_model = model.predict_proba(X_te_s)
                proba = np.zeros((len(y_te), len(all_classes)))
                model_classes = list(getattr(model, "classes_", all_classes))
                for i, c in enumerate(model_classes):
                    idx = int(np.where(all_classes == c)[0][0])
                    proba[:, idx] = proba_model[:, i]

            fold_acc.append(accuracy_score(y_te, y_pred))
            fold_f1.append(f1_score(y_te, y_pred, average="macro", zero_division=0))
            fold_auc.append(_safe_roc_auc(y_te, proba, all_classes))

        acc_mean, acc_std = float(np.mean(fold_acc)), float(np.std(fold_acc))
        f1_mean, f1_std = float(np.mean(fold_f1)), float(np.std(fold_f1))
        auc_mean = float(np.nanmean(fold_auc))
        auc_std = float(np.nanstd(fold_auc))

        rows.append(
            {
                "model": name,
                "accuracy_mean": acc_mean,
                "accuracy_std": acc_std,
                "macro_f1_mean": f1_mean,
                "macro_f1_std": f1_std,
                "roc_auc_mean": auc_mean,
                "roc_auc_std": auc_std,
            }
        )
        print(
            f"{name:<16} | {acc_mean:.4f} +/- {acc_std:.3f} | "
            f"{f1_mean:.4f} +/- {f1_std:.3f} | {auc_mean:.4f} +/- {auc_std:.3f}"
        )

    leaderboard = pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False)
    out_path = os.path.join(OUTPUTS_DIR, "model_comparison.csv")
    leaderboard.to_csv(out_path, index=False)
    print(f"\nLeaderboard written to {out_path}")
    print(f"Champion by macro-F1: {leaderboard.iloc[0]['model']}")


if __name__ == "__main__":
    run_comparison()
