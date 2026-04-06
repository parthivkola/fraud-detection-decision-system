from __future__ import annotations

import json
import os
import random
from typing import Any, Tuple

import joblib
import numpy as np
import pandas as pd


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str) -> None:
    """Create a directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def save_artifact(obj: Any, path: str) -> None:
    """Save a Python object with joblib."""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        ensure_dir(parent_dir)
    joblib.dump(obj, path)


def load_artifact(path: str) -> Any:
    """Load a joblib artifact."""
    return joblib.load(path)


def save_json(data: dict, path: str) -> None:
    """Save a dict as JSON."""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        ensure_dir(parent_dir)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_json(path: str) -> dict:
    """Load a JSON file into a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_features_target(
    df: pd.DataFrame,
    target_col: str = "Class",
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a dataframe into (X, y). Raises ValueError if target_col is missing."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()
    return X, y