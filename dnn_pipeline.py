"""Deep Neural Network track (Phase 2) — FT-Transformer + CORN ordinal loss.

Self-contained DNN study kept separate from the traditional model
comparison. Phase 2 swaps the Residual MLP backbone for an FT-Transformer
(Gorishniy et al. 2021) and replaces vanilla multi-class CE with CORN
ordinal loss (Shi, Cao, Raschka 2023), because ``Total_Label_Score`` is
ordinal 0-10 and the previous run's failure mode was middle-class
confusion.

Adds on top of the Phase-1 training recipe:

  - Input is a 40-dim concatenation of (20 globally-scaled features) +
    (20 sector-relative z-scores). Small sectors fall back to the global
    scaler; built by ``preprocess.build_sector_scaler``.
  - SMOTE oversampling on minority classes inside every training fold.
  - Multi-task auxiliary head predicting a 3-class risk bucket
    (High / Medium / Low from ``config.score_to_bucket``) -- acts as a
    regularizer alongside the main CORN head.
  - MC Dropout uncertainty pass: 30 stochastic forward passes per
    ensemble member at inference, aggregated over the 5-seed ensemble.
  - SHAP explainability via ``shap.KernelExplainer`` wrapped around the
    ensemble's CORN softmax. Top-5 / bottom-5 test companies get
    per-feature waterfalls; full per-feature SHAP matrix for every test
    row written to CSV.

Carried over from Phase 1: AdamW + CosineAnnealingWarmRestarts, mixup
(applied on token embeddings now), gradient clipping, early stopping,
5-seed ensemble.
"""
from __future__ import annotations

import itertools
import json
import os
import time
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import torch
import torch.nn as nn
import torch.nn.functional as F
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

from config import FEATURES, MODELS_DIR, OUTPUTS_DIR, RANDOM_STATE, score_to_bucket
from preprocess import (
    build_scaler,
    build_sector_scaler,
    get_stratified_split_with_sectors,
    load_raw_with_sectors,
    sector_normalize,
)

warnings.filterwarnings("ignore", category=UserWarning)

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HPARAM_GRID = {
    "d_token": [32, 64],
    "n_blocks": [2, 3],
    "n_heads": [4, 8],
    "dropout": [0.1, 0.2],
    "lr": [1e-3, 2e-3],
    "batch_size": [32],
}

MAX_EPOCHS = 200
EARLY_STOP_PATIENCE = 20
MIN_LR = 1e-6
INNER_CV_FOLDS = 3
LABEL_SMOOTHING = 0.1
MIXUP_ALPHA = 0.2
MIXUP_PROB = 0.5
GRAD_CLIP = 1.0
N_ENSEMBLE_SEEDS = 5
N_MC_DROPOUT = 30
AUX_LOSS_WEIGHT = 0.3
WEIGHT_DECAY = 1e-4
N_BUCKETS = 3  # High / Medium / Low


# ---------- FT-Transformer ----------
class FeatureTokenizer(nn.Module):
    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.bias, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_features) -> (B, n_features, d_token)
        return x.unsqueeze(-1) * self.weight + self.bias


class TransformerBlock(nn.Module):
    def __init__(self, d_token: int, n_heads: int, dropout: float, ff_factor: int = 2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_token)
        self.attn = nn.MultiheadAttention(
            d_token, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_token)
        self.ff = nn.Sequential(
            nn.Linear(d_token, d_token * ff_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token * ff_factor, d_token),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.dropout(attn_out)
        h = self.norm2(x)
        x = x + self.dropout(self.ff(h))
        return x


class FTTransformer(nn.Module):
    def __init__(self, n_features: int, num_classes: int, n_buckets: int, hp: dict):
        super().__init__()
        d = hp["d_token"]
        self.tokenizer = FeatureTokenizer(n_features, d)
        self.cls_token = nn.Parameter(torch.empty(1, 1, d))
        nn.init.normal_(self.cls_token, std=0.02)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d, hp["n_heads"], hp["dropout"])
                for _ in range(hp["n_blocks"])
            ]
        )
        self.norm = nn.LayerNorm(d)
        self.ordinal_head = nn.Linear(d, num_classes - 1)  # K-1 thresholds
        self.aux_head = nn.Linear(d, n_buckets)

    def forward(self, x: torch.Tensor):
        tok = self.tokenizer(x)  # (B, n_features, d)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        h = torch.cat([cls, tok], dim=1)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h[:, 0])
        return self.ordinal_head(h), self.aux_head(h)


# ---------- CORN ordinal loss ----------
def corn_loss(logits: torch.Tensor, y: torch.Tensor, K: int) -> torch.Tensor:
    """CORN conditional training loss (Shi, Cao, Raschka 2023).

    For each threshold k, only samples with y >= k are informative and
    their target is 1 if y > k else 0.
    """
    loss_total = logits.new_tensor(0.0)
    n_terms = 0
    for k in range(K - 1):
        mask = y >= k
        if mask.sum() == 0:
            continue
        target = (y[mask] > k).float()
        loss_k = F.binary_cross_entropy_with_logits(
            logits[mask, k], target, reduction="mean"
        )
        loss_total = loss_total + loss_k
        n_terms += 1
    return loss_total / max(n_terms, 1)


def corn_probs(logits: torch.Tensor, K: int) -> torch.Tensor:
    """Convert (B, K-1) threshold logits to (B, K) class probabilities.

    Uses cumulative-min to enforce rank consistency at inference, then
    renormalises so the probabilities sum to 1.
    """
    p_gt = torch.sigmoid(logits)  # P(y > k)
    # Rank-consistent: ensure p_gt is non-increasing in k.
    p_gt, _ = torch.cummin(p_gt, dim=1)
    B = p_gt.size(0)
    probs = torch.zeros(B, K, device=p_gt.device, dtype=p_gt.dtype)
    probs[:, 0] = 1 - p_gt[:, 0]
    if K > 2:
        probs[:, 1:-1] = p_gt[:, :-1] - p_gt[:, 1:]
    probs[:, -1] = p_gt[:, -1]
    probs = probs.clamp(min=1e-8)
    probs = probs / probs.sum(dim=1, keepdim=True)
    return probs


# ---------- Training ----------
def _mixup_batch(x: torch.Tensor, y: torch.Tensor, y_aux: torch.Tensor, alpha: float):
    if alpha <= 0.0:
        return x, y, y, y_aux, y_aux, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1.0 - lam) * x[idx]
    return x_mix, y, y[idx], y_aux, y_aux[idx], lam


def _apply_smote(X: np.ndarray, y: np.ndarray, seed: int = RANDOM_STATE) -> tuple:
    """Oversample minority classes. Skip classes with < 6 samples (SMOTE needs k>=1 neighbors)."""
    counts = np.bincount(y)
    if (counts > 0).sum() < 2:
        return X, y
    min_required = 6  # k_neighbors=5 + self
    eligible = [c for c in range(len(counts)) if counts[c] >= min_required]
    if len(eligible) < 2:
        return X, y
    max_count = counts.max()
    target = {c: max_count for c in eligible if counts[c] < max_count}
    if not target:
        return X, y
    try:
        sm = SMOTE(sampling_strategy=target, k_neighbors=5, random_state=seed)
        return sm.fit_resample(X, y)
    except ValueError:
        return X, y


def _train_one(
    hp: dict,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    num_classes: int,
    class_weights: torch.Tensor,
    seed: int = RANDOM_STATE,
    apply_smote: bool = True,
    verbose: bool = False,
):
    torch.manual_seed(seed)

    if apply_smote:
        X_tr, y_tr = _apply_smote(X_tr, y_tr, seed=seed)

    y_tr_bucket = np.array([score_to_bucket(int(v)) for v in y_tr])
    y_va_bucket = np.array([score_to_bucket(int(v)) for v in y_va])

    model = FTTransformer(X_tr.shape[1], num_classes, N_BUCKETS, hp).to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optim, T_0=20, T_mult=2, eta_min=MIN_LR
    )

    train_ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.long),
        torch.tensor(y_tr_bucket, dtype=torch.long),
    )
    train_dl = DataLoader(train_ds, batch_size=hp["batch_size"], shuffle=True, drop_last=True)

    X_va_t = torch.tensor(X_va, dtype=torch.float32, device=DEVICE)
    y_va_t = torch.tensor(y_va, dtype=torch.long, device=DEVICE)
    y_va_b_t = torch.tensor(y_va_bucket, dtype=torch.long, device=DEVICE)

    best_val, best_state, bad_epochs = float("inf"), None, 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(MAX_EPOCHS):
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0
        for xb, yb, yb_buc in train_dl:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            yb_buc = yb_buc.to(DEVICE)
            optim.zero_grad()

            if np.random.rand() < MIXUP_PROB:
                xb_m, y_a, y_b, ya_a, ya_b, lam = _mixup_batch(xb, yb, yb_buc, MIXUP_ALPHA)
                ord_logits, aux_logits = model(xb_m)
                loss_main = lam * corn_loss(ord_logits, y_a, num_classes) + (
                    1 - lam
                ) * corn_loss(ord_logits, y_b, num_classes)
                loss_aux = lam * F.cross_entropy(
                    aux_logits, ya_a, label_smoothing=LABEL_SMOOTHING
                ) + (1 - lam) * F.cross_entropy(
                    aux_logits, ya_b, label_smoothing=LABEL_SMOOTHING
                )
            else:
                ord_logits, aux_logits = model(xb)
                loss_main = corn_loss(ord_logits, yb, num_classes)
                loss_aux = F.cross_entropy(
                    aux_logits,
                    yb_buc,
                    weight=class_weights,
                    label_smoothing=LABEL_SMOOTHING,
                )

            loss = loss_main + AUX_LOSS_WEIGHT * loss_aux
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optim.step()

            with torch.no_grad():
                probs = corn_probs(ord_logits, num_classes)
                pred = probs.argmax(dim=1)
            running_loss += loss.item() * xb.size(0)
            running_correct += (pred == yb).sum().item()
            running_total += xb.size(0)

        scheduler.step()
        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)

        model.eval()
        with torch.no_grad():
            ord_v, aux_v = model(X_va_t)
            val_main = corn_loss(ord_v, y_va_t, num_classes)
            val_aux = F.cross_entropy(aux_v, y_va_b_t)
            val_loss = (val_main + AUX_LOSS_WEIGHT * val_aux).item()
            probs_v = corn_probs(ord_v, num_classes)
            val_acc = (probs_v.argmax(dim=1) == y_va_t).float().mean().item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= EARLY_STOP_PATIENCE:
                break

        if verbose and epoch % 10 == 0:
            print(
                f"    epoch {epoch:03d} train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def _predict_probs(model: nn.Module, X: np.ndarray, K: int) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        ord_logits, _ = model(torch.tensor(X, dtype=torch.float32, device=DEVICE))
        return corn_probs(ord_logits, K).cpu().numpy()


def _ensemble_probs(models, X: np.ndarray, K: int) -> np.ndarray:
    return np.mean([_predict_probs(m, X, K) for m in models], axis=0)


def _mc_dropout_probs(models, X: np.ndarray, K: int, n_samples: int = N_MC_DROPOUT) -> np.ndarray:
    """Run n_samples forward passes with dropout active, batchnorm/layernorm frozen."""
    def _set_dropout_train_only(m):
        for mod in m.modules():
            if isinstance(mod, (nn.Dropout,)):
                mod.train()
            else:
                mod.eval()

    all_samples = []
    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    for model in models:
        _set_dropout_train_only(model)
        with torch.no_grad():
            for _ in range(n_samples):
                ord_logits, _ = model(X_t)
                all_samples.append(corn_probs(ord_logits, K).cpu().numpy())
    return np.stack(all_samples)  # (n_models * n_samples, B, K)


def _grid_combos(grid: dict):
    keys = list(grid.keys())
    for values in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, values))


def run():
    # ---------- Data ----------
    X_df, y_ser, sectors = load_raw_with_sectors()
    X_train, X_test, y_train, y_test, sec_train, sec_test = (
        get_stratified_split_with_sectors(X_df, y_ser, sectors)
    )

    global_scaler = build_scaler(X_train)
    sector_scalers = build_sector_scaler(X_train, sec_train)
    print(
        f"[DNN] Sector scalers fit for: {sorted(sector_scalers.keys())}  "
        f"(others fall back to global)"
    )

    X_train_g = global_scaler.transform(X_train)
    X_test_g = global_scaler.transform(X_test)
    X_train_s = sector_normalize(X_train, sec_train, sector_scalers, global_scaler)
    X_test_s = sector_normalize(X_test, sec_test, sector_scalers, global_scaler)
    X_train_full = np.concatenate([X_train_g, X_train_s], axis=1).astype(np.float32)
    X_test_full = np.concatenate([X_test_g, X_test_s], axis=1).astype(np.float32)

    all_classes = np.sort(np.unique(np.concatenate([y_train, y_test])))
    class_to_idx = {c: i for i, c in enumerate(all_classes)}
    num_classes = len(all_classes)
    y_train_idx = np.array([class_to_idx[v] for v in y_train])
    y_test_idx = np.array([class_to_idx[v] for v in y_test])

    bucket_train = np.array([score_to_bucket(int(v)) for v in y_train])
    cw_bucket = compute_class_weight("balanced", classes=np.arange(N_BUCKETS), y=bucket_train)
    bucket_weights = torch.tensor(cw_bucket, dtype=torch.float32, device=DEVICE)
    print(f"[DNN] Bucket class weights: {np.round(cw_bucket, 2).tolist()}")

    combos = list(_grid_combos(HPARAM_GRID))
    print(
        f"[DNN] FT-Transformer grid: {len(combos)} combos x {INNER_CV_FOLDS} folds on "
        f"device={DEVICE}; input dim={X_train_full.shape[1]} (20 global + 20 sector)"
    )

    skf = StratifiedKFold(n_splits=INNER_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    grid_rows = []
    t0 = time.time()
    for i, hp in enumerate(combos, 1):
        fold_scores = []
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_full, y_train_idx)):
            model, _ = _train_one(
                hp,
                X_train_full[tr_idx],
                y_train_idx[tr_idx],
                X_train_full[va_idx],
                y_train_idx[va_idx],
                num_classes,
                bucket_weights,
                seed=RANDOM_STATE + fold,
            )
            probs = _predict_probs(model, X_train_full[va_idx], num_classes)
            y_pred = probs.argmax(axis=1)
            f1 = f1_score(y_train_idx[va_idx], y_pred, average="macro", zero_division=0)
            fold_scores.append(f1)
            grid_rows.append({**hp, "fold": fold, "macro_f1": f1})
        mean_f1 = float(np.mean(fold_scores))
        std_f1 = float(np.std(fold_scores))
        elapsed = time.time() - t0
        print(
            f"  [{i:3d}/{len(combos)}] macro_f1 = {mean_f1:.4f} +/- {std_f1:.3f}  "
            f"({elapsed:.0f}s elapsed)  hp={hp}"
        )

    grid_df = pd.DataFrame(grid_rows)
    summary = (
        grid_df.groupby(list(HPARAM_GRID.keys()))["macro_f1"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    combined = pd.concat(
        [summary.assign(_kind="summary"), grid_df.assign(_kind="per_fold")],
        ignore_index=True,
    )
    combined.to_csv(os.path.join(OUTPUTS_DIR, "dnn_grid_search.csv"), index=False)
    best_hp_row = summary.iloc[0]
    best_hp = {k: best_hp_row[k] for k in HPARAM_GRID.keys()}
    best_hp = {
        k: (int(v) if k in ("d_token", "n_blocks", "n_heads", "batch_size") else float(v))
        for k, v in best_hp.items()
    }
    print(f"\n[DNN] Best hparams by CV macro-F1: {best_hp}")

    # ---------- 5-seed ensemble ----------
    n = X_train_full.shape[0]
    val_n = max(int(0.15 * n), 16)
    rng = np.random.default_rng(RANDOM_STATE)
    perm = rng.permutation(n)
    val_idx, tr_idx = perm[:val_n], perm[val_n:]

    members, histories = [], []
    print(f"\n[DNN] Training {N_ENSEMBLE_SEEDS}-seed ensemble on best hparams...")
    for i in range(N_ENSEMBLE_SEEDS):
        seed = RANDOM_STATE + i
        m_i, h_i = _train_one(
            best_hp,
            X_train_full[tr_idx],
            y_train_idx[tr_idx],
            X_train_full[val_idx],
            y_train_idx[val_idx],
            num_classes,
            bucket_weights,
            seed=seed,
            verbose=(i == 0),
        )
        members.append(m_i)
        histories.append(h_i)
        print(
            f"  seed {seed}: trained {len(h_i['train_loss'])} epochs "
            f"(best val_loss={min(h_i['val_loss']):.4f})"
        )

    # ---------- Evaluation ----------
    probs = _ensemble_probs(members, X_test_full, num_classes)
    y_pred_idx = probs.argmax(axis=1)
    y_pred = all_classes[y_pred_idx]
    y_test_arr = np.asarray(y_test)

    acc = accuracy_score(y_test_arr, y_pred)
    macro_f1 = f1_score(y_test_arr, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test_arr, y_pred, average="weighted", zero_division=0)
    kappa = cohen_kappa_score(y_test_arr, y_pred)
    try:
        y_bin = label_binarize(y_test_arr, classes=all_classes)
        present = y_bin.sum(axis=0) > 0
        auc = roc_auc_score(y_bin[:, present], probs[:, present], average="macro")
    except Exception:
        auc = float("nan")

    print("\n" + "=" * 56)
    print("                 DNN MODEL REPORT (Phase 2)")
    print("=" * 56)
    print(
        f"Backbone:      FT-Transformer d_token={best_hp['d_token']} "
        f"blocks={best_hp['n_blocks']} heads={best_hp['n_heads']}"
    )
    print(f"Loss:          CORN ordinal + {AUX_LOSS_WEIGHT} * bucket aux CE")
    print(f"Input:         20 global-scaled + 20 sector-scaled = 40 features")
    print(f"Ensemble:      {N_ENSEMBLE_SEEDS} seeds, averaged softmax")
    print(f"Best hparams:  {json.dumps(best_hp)}")
    print(f"Accuracy:      {acc:.4f}")
    print(f"Macro F1:      {macro_f1:.4f}")
    print(f"Weighted F1:   {weighted_f1:.4f}")
    print(f"ROC-AUC (OVR): {auc:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print("\n--- Classification Report ---")
    print(classification_report(y_test_arr, y_pred, zero_division=0))

    # ---------- Plots ----------
    history = histories[0]
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("DNN Loss (seed 0)")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_title("DNN Accuracy (seed 0)")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "dnn_training_curves.png"), dpi=200)
    plt.close()

    cm = confusion_matrix(y_test_arr, y_pred, labels=all_classes)
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="PuBu",
        xticklabels=all_classes, yticklabels=all_classes,
    )
    plt.title(f"DNN Confusion Matrix (FT-Transformer ensemble x{N_ENSEMBLE_SEEDS})")
    plt.xlabel("Predicted Score")
    plt.ylabel("True Score")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "dnn_confusion_matrix.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    y_bin = label_binarize(y_test_arr, classes=all_classes)
    for i, cls in enumerate(all_classes):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        plt.plot(fpr, tpr, label=f"class {cls}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"DNN ROC Curves (OVR, ensemble x{N_ENSEMBLE_SEEDS})")
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "dnn_roc_curves.png"), dpi=200)
    plt.close()

    # ---------- MC Dropout uncertainty ----------
    print(f"\n[DNN] MC Dropout: {N_MC_DROPOUT} passes x {N_ENSEMBLE_SEEDS} models...")
    mc = _mc_dropout_probs(members, X_test_full, num_classes, n_samples=N_MC_DROPOUT)
    mc_mean = mc.mean(axis=0)       # (B, K)
    mc_std = mc.std(axis=0)         # (B, K) — per-class std
    pred_class_std = mc_std[np.arange(len(mc_std)), mc_mean.argmax(axis=1)]
    pred_entropy = -(mc_mean * np.log(mc_mean + 1e-12)).sum(axis=1)

    mc_df = pd.DataFrame({
        "true_score": y_test_arr,
        "pred_score": all_classes[mc_mean.argmax(axis=1)],
        "pred_prob": mc_mean.max(axis=1),
        "pred_std": pred_class_std,
        "entropy": pred_entropy,
    }).sort_values("entropy", ascending=False)
    mc_df.to_csv(os.path.join(OUTPUTS_DIR, "dnn_uncertainty.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(mc_df["pred_prob"], mc_df["entropy"], alpha=0.6, s=25)
    axes[0].set_xlabel("Predicted-class probability (mean over MC samples)")
    axes[0].set_ylabel("Predictive entropy (nats)")
    axes[0].set_title("MC Dropout: confidence vs entropy per test company")
    axes[1].hist(mc_df["entropy"], bins=30, color="slateblue", edgecolor="white")
    axes[1].set_xlabel("Predictive entropy")
    axes[1].set_ylabel("Companies")
    axes[1].set_title("Distribution of DNN uncertainty over the test set")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "dnn_uncertainty.png"), dpi=200)
    plt.close()
    print(
        f"  mean entropy={pred_entropy.mean():.3f}  "
        f"top-10 least-confident written to dnn_uncertainty.csv"
    )

    # ---------- SHAP explainability ----------
    print(f"\n[DNN] SHAP: KernelExplainer over ensemble...")
    feature_names = [f"{f}__global" for f in FEATURES] + [f"{f}__sector" for f in FEATURES]

    def ensemble_fn(x_np: np.ndarray) -> np.ndarray:
        # shap passes numpy; return softmax probs from the ensemble
        return _ensemble_probs(members, x_np.astype(np.float32), num_classes)

    bg_idx = rng.choice(len(X_train_full), size=min(100, len(X_train_full)), replace=False)
    background = X_train_full[bg_idx]
    explainer = shap.KernelExplainer(ensemble_fn, background)

    # Compute SHAP for all test rows at modest nsamples (budget-aware).
    t_shap = time.time()
    shap_values = explainer.shap_values(X_test_full, nsamples=60, silent=True)
    print(f"  SHAP computed in {time.time() - t_shap:.0f}s")

    # shap returns either list of arrays (per class) for multi-output or (B, F, K) array
    if isinstance(shap_values, list):
        shap_arr = np.stack(shap_values, axis=-1)  # (B, F, K)
    else:
        shap_arr = np.asarray(shap_values)
        if shap_arr.ndim == 2:
            shap_arr = shap_arr[..., None]

    # Per-row SHAP for the predicted class
    pred_idx = probs.argmax(axis=1)
    shap_pred = np.stack([shap_arr[i, :, pred_idx[i]] for i in range(len(pred_idx))])
    shap_matrix_df = pd.DataFrame(shap_pred, columns=feature_names)
    shap_matrix_df.insert(0, "pred_score", all_classes[pred_idx])
    shap_matrix_df.insert(0, "true_score", y_test_arr)
    shap_matrix_df.to_csv(os.path.join(OUTPUTS_DIR, "dnn_shap_matrix.csv"), index=False)

    # Top-5 "Low Risk" (predicted high score) and bottom-5 "High Risk" (predicted low score)
    low_risk_prob = probs[:, -N_BUCKETS:].sum(axis=1)  # top classes
    high_risk_prob = probs[:, :N_BUCKETS].sum(axis=1)   # bottom classes
    top5 = np.argsort(-low_risk_prob)[:5]
    bot5 = np.argsort(-high_risk_prob)[:5]

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for ax, i in zip(axes[0], top5):
        vals = shap_pred[i]
        order = np.argsort(-np.abs(vals))[:8]
        ax.barh(
            [feature_names[j] for j in order[::-1]],
            vals[order[::-1]],
            color=["seagreen" if v >= 0 else "indianred" for v in vals[order[::-1]]],
        )
        ax.set_title(
            f"LOW RISK\ntrue={y_test_arr[i]} pred={all_classes[pred_idx[i]]} "
            f"p={probs[i].max():.2f}",
            fontsize=9,
        )
        ax.tick_params(axis="y", labelsize=7)
    for ax, i in zip(axes[1], bot5):
        vals = shap_pred[i]
        order = np.argsort(-np.abs(vals))[:8]
        ax.barh(
            [feature_names[j] for j in order[::-1]],
            vals[order[::-1]],
            color=["seagreen" if v >= 0 else "indianred" for v in vals[order[::-1]]],
        )
        ax.set_title(
            f"HIGH RISK\ntrue={y_test_arr[i]} pred={all_classes[pred_idx[i]]} "
            f"p={probs[i].max():.2f}",
            fontsize=9,
        )
        ax.tick_params(axis="y", labelsize=7)
    plt.suptitle("DNN SHAP attributions — top-5 Low Risk vs bottom-5 High Risk")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "dnn_shap_examples.png"), dpi=200)
    plt.close()

    # Permutation importance via mean-|shap| for the report
    perm_importance = np.abs(shap_pred).mean(axis=0)
    imp_series = pd.Series(perm_importance, index=feature_names).sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    imp_series.head(20).plot(kind="bar", color="slateblue")
    plt.title("DNN feature importance (mean |SHAP|, top 20)")
    plt.ylabel("Mean |SHAP|")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "dnn_permutation_importance.png"), dpi=200)
    plt.close()

    # ---------- Persist ----------
    for i, m in enumerate(members):
        torch.save(
            {
                "state_dict": m.state_dict(),
                "hparams": best_hp,
                "classes": all_classes.tolist(),
                "n_features": X_train_full.shape[1],
                "seed": RANDOM_STATE + i,
            },
            os.path.join(MODELS_DIR, f"dnn_member_{i}.pt"),
        )
    torch.save(
        {
            "state_dict": members[0].state_dict(),
            "hparams": best_hp,
            "classes": all_classes.tolist(),
            "n_features": X_train_full.shape[1],
        },
        os.path.join(MODELS_DIR, "dnn_best.pt"),
    )
    joblib.dump(global_scaler, os.path.join(MODELS_DIR, "dnn_scaler.joblib"))
    joblib.dump(sector_scalers, os.path.join(MODELS_DIR, "sector_scalers.joblib"))

    with open(os.path.join(OUTPUTS_DIR, "dnn_best_hparams.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "backbone": "FT-Transformer",
                "hparams": best_hp,
                "ensemble_size": N_ENSEMBLE_SEEDS,
                "loss": f"CORN + {AUX_LOSS_WEIGHT} * bucket_aux_CE",
                "input": "20 global + 20 sector = 40 features",
                "training": {
                    "label_smoothing": LABEL_SMOOTHING,
                    "mixup_alpha": MIXUP_ALPHA,
                    "mixup_prob": MIXUP_PROB,
                    "grad_clip": GRAD_CLIP,
                    "scheduler": "CosineAnnealingWarmRestarts(T_0=20, T_mult=2)",
                    "optimizer": "AdamW",
                    "smote": "imblearn SMOTE (training folds only)",
                    "aux_head": "3-class risk bucket (High/Medium/Low)",
                },
                "mc_dropout": {
                    "n_samples": N_MC_DROPOUT,
                    "mean_entropy": float(pred_entropy.mean()),
                },
                "test_metrics": {
                    "accuracy": acc,
                    "macro_f1": macro_f1,
                    "weighted_f1": weighted_f1,
                    "roc_auc_ovr_macro": auc,
                    "cohen_kappa": kappa,
                },
            },
            f,
            indent=2,
        )
    print(f"\nArtifacts saved to {OUTPUTS_DIR}/ and {MODELS_DIR}/.")


if __name__ == "__main__":
    run()
