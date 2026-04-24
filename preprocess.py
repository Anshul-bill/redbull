"""Data loading and preprocessing used by every downstream script.

Replaces the empty ``model-preprocess-code.py``. Keeps preprocessing
reproducible in code instead of buried inside the Excel file.

Phase 2 additions:
- ``load_raw_with_sectors`` exposes the Sector column for sector-aware
  normalization used by ``dnn_pipeline.py``.
- ``build_sector_scaler`` / ``sector_normalize`` produce per-sector
  z-scores; sectors below ``MIN_SECTOR_SIZE`` fall back to the global
  scaler to avoid overfitting to very small peer groups.
"""
from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import DATA_PATH, FEATURES, RANDOM_STATE, SECTOR_COL, SHEET, TARGET

MIN_SECTOR_SIZE = 20


def load_raw() -> pd.DataFrame:
    """Read the normalized training sheet and validate its columns."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}. Ensure norm-data.xlsx is inside data/."
        )

    df = pd.read_excel(DATA_PATH, sheet_name=SHEET)
    expected = FEATURES + [TARGET]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"norm-data.xlsx is missing columns: {missing}")

    before = len(df)
    df = df.dropna(subset=expected).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"[preprocess] Dropped {dropped} rows with NaNs in features/target")
    return df


def split_xy(df: pd.DataFrame):
    """Return (X, y) as a DataFrame and Series."""
    return df[FEATURES], df[TARGET].astype(int)


def load_raw_with_sectors() -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (X, y, sectors). Rows missing Sector get bucket 'Unknown'."""
    df = load_raw()
    X, y = split_xy(df)
    if SECTOR_COL in df.columns:
        sectors = df[SECTOR_COL].fillna("Unknown").astype(str)
    else:
        sectors = pd.Series(["Unknown"] * len(df), index=df.index)
    return X, y, sectors


def get_stratified_split(X, y, test_size: float = 0.2):
    """Stratified 80/20 split with the project's fixed seed."""
    return train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )


def get_stratified_split_with_sectors(X, y, sectors, test_size: float = 0.2):
    """Same stratified split, propagating sectors alongside."""
    idx = np.arange(len(X))
    idx_train, idx_test = train_test_split(
        idx, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    X_arr = X.values if hasattr(X, "values") else np.asarray(X)
    y_arr = y.values if hasattr(y, "values") else np.asarray(y)
    s_arr = sectors.values if hasattr(sectors, "values") else np.asarray(sectors)
    return (
        X_arr[idx_train], X_arr[idx_test],
        y_arr[idx_train], y_arr[idx_test],
        s_arr[idx_train], s_arr[idx_test],
    )


def build_scaler(X_train) -> StandardScaler:
    """Fit a global StandardScaler on the training split only."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def build_sector_scaler(X_train, sectors_train) -> dict:
    """Fit per-sector StandardScalers on the training split.

    Sectors with fewer than ``MIN_SECTOR_SIZE`` training rows are omitted
    from the dict; callers must fall back to the global scaler for them.
    """
    X_arr = np.asarray(X_train, dtype=float)
    s_arr = np.asarray(sectors_train)
    scalers: dict = {}
    for sector in np.unique(s_arr):
        mask = s_arr == sector
        if mask.sum() < MIN_SECTOR_SIZE:
            continue
        s = StandardScaler().fit(X_arr[mask])
        scalers[str(sector)] = s
    return scalers


def sector_normalize(X, sectors, sector_scalers: dict, global_scaler: StandardScaler) -> np.ndarray:
    """Return sector-relative z-scores. Rows in small/missing sectors fall back
    to the global scaler so every row still gets a (n_features,) vector."""
    X_arr = np.asarray(X, dtype=float)
    s_arr = np.asarray(sectors)
    out = np.empty_like(X_arr, dtype=float)
    for i, sector in enumerate(s_arr):
        scaler = sector_scalers.get(str(sector), global_scaler)
        out[i] = scaler.transform(X_arr[i : i + 1])[0]
    return out
