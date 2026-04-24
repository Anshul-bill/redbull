"""Deep evaluation of the traditional-track champion selected from the leaderboard.

Reads outputs/model_comparison.csv, picks the row with the highest macro-F1,
refits that model on the held-out train split with scaling, and produces
confusion matrix, per-class ROC curves, and feature-importance plots.
The Deep Neural Network is evaluated separately in dnn_pipeline.py.
"""
from __future__ import annotations

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from config import FEATURES, MODELS_DIR, OUTPUTS_DIR
from preprocess import build_scaler, get_stratified_split, load_raw, split_xy


def _pick_champion():
    leaderboard_path = os.path.join(OUTPUTS_DIR, "model_comparison.csv")
    if not os.path.exists(leaderboard_path):
        raise FileNotFoundError(
            f"{leaderboard_path} not found. Run model_comparison.py first."
        )
    leaderboard = pd.read_csv(leaderboard_path)
    champion_row = leaderboard.sort_values("macro_f1_mean", ascending=False).iloc[0]
    return str(champion_row["model"])


def _instantiate(name: str):
    if name == "SVM":
        from sklearn.svm import SVC
        from config import RANDOM_STATE
        return SVC(probability=True, random_state=RANDOM_STATE)
    if name == "Random Forest":
        from sklearn.ensemble import RandomForestClassifier
        from config import RANDOM_STATE
        return RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    if name == "GBM":
        from sklearn.ensemble import GradientBoostingClassifier
        from config import RANDOM_STATE
        return GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)
    if name == "XGBoost":
        from xgboost import XGBClassifier
        from config import RANDOM_STATE
        return XGBClassifier(eval_metric="mlogloss", random_state=RANDOM_STATE)
    if name == "CatBoost":
        from catboost import CatBoostClassifier
        from config import RANDOM_STATE
        return CatBoostClassifier(
            iterations=200, verbose=0, random_state=RANDOM_STATE, allow_writing_files=False
        )
    if name == "LightGBM":
        from train_lightgbm import build_lightgbm
        return build_lightgbm()
    raise ValueError(f"Unknown champion name: {name}")


def _encode_xgb(y, classes):
    lookup = {c: i for i, c in enumerate(classes)}
    return np.array([lookup[v] for v in y])


def _align_proba(proba_model, model_classes, all_classes, n_rows):
    proba = np.zeros((n_rows, len(all_classes)))
    for i, c in enumerate(model_classes):
        idx = int(np.where(all_classes == c)[0][0])
        proba[:, idx] = proba_model[:, i]
    return proba


def main():
    champion_name = _pick_champion()
    print(f"Champion selected from leaderboard: {champion_name}")

    df = load_raw()
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = get_stratified_split(X, y)
    all_classes = np.sort(np.unique(np.concatenate([y_train.values, y_test.values])))

    scaler = build_scaler(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = _instantiate(champion_name)
    if champion_name == "XGBoost":
        y_train_enc = _encode_xgb(y_train.values, all_classes)
        model.fit(X_train_s, y_train_enc)
        proba_raw = model.predict_proba(X_test_s)
        proba = _align_proba(proba_raw, list(all_classes)[: proba_raw.shape[1]], all_classes, len(y_test))
        y_pred = all_classes[np.argmax(proba, axis=1)]
    else:
        model.fit(X_train_s, y_train.values)
        y_pred = model.predict(X_test_s)
        proba_raw = model.predict_proba(X_test_s)
        proba = _align_proba(proba_raw, list(model.classes_), all_classes, len(y_test))

    print("\n" + "=" * 48)
    print("      CHAMPION MODEL EVALUATION (held-out test)")
    print("=" * 48)
    print(f"Model:         {champion_name}")
    print(f"Accuracy:      {accuracy_score(y_test, y_pred):.4f}")
    print(f"Macro F1:      {f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
    print(f"Weighted F1:   {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
    try:
        y_bin = label_binarize(y_test, classes=all_classes)
        present = y_bin.sum(axis=0) > 0
        auc = roc_auc_score(y_bin[:, present], proba[:, present], average="macro")
        print(f"ROC-AUC (OVR): {auc:.4f}")
    except Exception as exc:
        print(f"ROC-AUC (OVR): n/a ({exc})")
    print(f"Cohen's Kappa: {cohen_kappa_score(y_test, y_pred):.4f}")
    print(f"Log Loss:      {log_loss(y_test, proba, labels=all_classes):.4f}")
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=all_classes)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=all_classes, yticklabels=all_classes)
    plt.title(f"Confusion Matrix — {champion_name}")
    plt.xlabel("Predicted Score")
    plt.ylabel("True Score")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "confusion_matrix.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    y_bin = label_binarize(y_test, classes=all_classes)
    for i, cls in enumerate(all_classes):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
        plt.plot(fpr, tpr, label=f"class {cls}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves (OVR) — {champion_name}")
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "roc_curves.png"), dpi=200)
    plt.close()

    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    else:
        perm = permutation_importance(model, X_test_s, y_test, n_repeats=5, random_state=42, n_jobs=-1)
        importances = pd.Series(perm.importances_mean, index=FEATURES).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    importances.plot(kind="bar", color="teal")
    plt.title(f"Feature Importance — {champion_name}")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "feature_importance.png"), dpi=200)
    plt.close()

    joblib.dump(model, os.path.join(MODELS_DIR, "champion.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    with open(os.path.join(OUTPUTS_DIR, "champion_name.txt"), "w", encoding="utf-8") as f:
        f.write(champion_name + "\n")
    print(f"\nSaved model -> {MODELS_DIR}/champion.joblib, scaler -> {MODELS_DIR}/scaler.joblib")
    print(f"Plots written to {OUTPUTS_DIR}/")


if __name__ == "__main__":
    main()
