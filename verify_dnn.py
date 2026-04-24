"""Three-part verification of the Phase-2 DNN result.

1. Permutation test -- shuffle Total_Label_Score once, refit one ensemble.
   If accuracy stays near chance (1/11 ~= 0.09), the pipeline isn't leaking.
2. Multi-seed split robustness -- rerun the 5-seed ensemble with
   random_state = 7 and 123 and compare test metrics against the
   original (random_state = 42) result.
3. Training curve inspection -- confirmed separately by viewing
   outputs/dnn_training_curves.png.

Uses the best hyperparameters already found by the grid search, so each
verification run is just a 5-seed ensemble fit (~1-3 min per run).
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import train_test_split

from config import FEATURES, TARGET, score_to_bucket
import dnn_pipeline as dp
from preprocess import (
    build_scaler,
    build_sector_scaler,
    load_raw_with_sectors,
    sector_normalize,
)

BEST_HP = {
    "d_token": 64,
    "n_blocks": 2,
    "n_heads": 4,
    "dropout": 0.2,
    "lr": 2e-3,
    "batch_size": 32,
}


def _prepare(seed: int, shuffle_labels: bool = False):
    X_df, y_ser, sectors = load_raw_with_sectors()
    y_arr = y_ser.values.copy()
    if shuffle_labels:
        rng = np.random.default_rng(seed + 9999)
        rng.shuffle(y_arr)

    idx = np.arange(len(X_df))
    idx_train, idx_test = train_test_split(
        idx, test_size=0.2, random_state=seed, stratify=y_arr
    )
    X_all = X_df.values
    s_all = sectors.values

    X_train, X_test = X_all[idx_train], X_all[idx_test]
    y_train, y_test = y_arr[idx_train], y_arr[idx_test]
    sec_train, sec_test = s_all[idx_train], s_all[idx_test]

    global_scaler = build_scaler(X_train)
    sector_scalers = build_sector_scaler(X_train, sec_train)
    Xtr_g = global_scaler.transform(X_train)
    Xte_g = global_scaler.transform(X_test)
    Xtr_s = sector_normalize(X_train, sec_train, sector_scalers, global_scaler)
    Xte_s = sector_normalize(X_test, sec_test, sector_scalers, global_scaler)
    Xtr_full = np.concatenate([Xtr_g, Xtr_s], axis=1).astype(np.float32)
    Xte_full = np.concatenate([Xte_g, Xte_s], axis=1).astype(np.float32)

    classes = np.sort(np.unique(np.concatenate([y_train, y_test])))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)
    y_train_idx = np.array([class_to_idx[v] for v in y_train])
    y_test_idx = np.array([class_to_idx[v] for v in y_test])

    bucket_train = np.array([score_to_bucket(int(v)) for v in y_train])
    from sklearn.utils.class_weight import compute_class_weight
    import torch

    cw = compute_class_weight(
        "balanced", classes=np.arange(dp.N_BUCKETS), y=bucket_train
    )
    bucket_w = torch.tensor(cw, dtype=torch.float32, device=dp.DEVICE)

    return Xtr_full, Xte_full, y_train_idx, y_test_idx, classes, num_classes, bucket_w


def _train_ensemble(hp, Xtr_full, y_train_idx, num_classes, bucket_w, base_seed):
    n = Xtr_full.shape[0]
    val_n = max(int(0.15 * n), 16)
    rng = np.random.default_rng(base_seed)
    perm = rng.permutation(n)
    val_idx, tr_idx = perm[:val_n], perm[val_n:]

    members = []
    for i in range(dp.N_ENSEMBLE_SEEDS):
        m, _ = dp._train_one(
            hp,
            Xtr_full[tr_idx],
            y_train_idx[tr_idx],
            Xtr_full[val_idx],
            y_train_idx[val_idx],
            num_classes,
            bucket_w,
            seed=base_seed + i,
        )
        members.append(m)
    return members


def _evaluate(members, Xte_full, y_test_idx, classes, num_classes):
    probs = dp._ensemble_probs(members, Xte_full, num_classes)
    pred_idx = probs.argmax(axis=1)
    y_pred = classes[pred_idx]
    y_true = classes[y_test_idx]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
    }


def run_one(label: str, seed: int, shuffle_labels: bool = False):
    print(f"\n{'=' * 60}")
    print(f"  {label}  (seed={seed}, shuffle_labels={shuffle_labels})")
    print("=" * 60)
    t0 = time.time()
    Xtr, Xte, y_tr, y_te, classes, K, bw = _prepare(seed, shuffle_labels)
    print(f"  prepared: Xtr={Xtr.shape}  Xte={Xte.shape}  K={K}")
    members = _train_ensemble(BEST_HP, Xtr, y_tr, K, bw, base_seed=seed)
    metrics = _evaluate(members, Xte, y_te, classes, K)
    metrics["elapsed_sec"] = time.time() - t0
    metrics["label"] = label
    metrics["seed"] = seed
    metrics["shuffled"] = shuffle_labels
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:14s}: {v:.4f}")
        else:
            print(f"  {k:14s}: {v}")
    return metrics


def main():
    results = []

    # 1. Permutation test (labels shuffled). Should collapse to ~chance (1/11 = 0.091).
    results.append(run_one("PERMUTATION (shuffled labels)", seed=42, shuffle_labels=True))

    # 2. Split robustness with two different seeds.
    results.append(run_one("SPLIT seed=7", seed=7))
    results.append(run_one("SPLIT seed=123", seed=123))

    # 3. For reference, the original Phase 2 result (already in outputs/dnn_best_hparams.json).
    with open("outputs/dnn_best_hparams.json", "r", encoding="utf-8") as f:
        orig = json.load(f)["test_metrics"]
    orig_row = {"label": "ORIGINAL seed=42", "seed": 42, "shuffled": False, **orig}
    orig_row["elapsed_sec"] = None
    results.append(orig_row)

    df = pd.DataFrame(results)
    df.to_csv("outputs/dnn_verification.csv", index=False)
    print("\n\n==============  VERIFICATION SUMMARY  ==============")
    print(df.to_string(index=False))
    print("\nWritten: outputs/dnn_verification.csv")


if __name__ == "__main__":
    main()
