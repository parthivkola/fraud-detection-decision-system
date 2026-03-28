"""
Utility helpers shared across the ML pipeline.

This module contains reusable functions for:
- creating directories
- loading/saving artifacts
- splitting features and target
- setting random seeds
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Tuple

import joblib
import numpy as np
import pandas as pd


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str) -> None:
    """
    Create a directory if it does not already exist.

    Args:
        path: Directory path.
    """
    os.makedirs(path, exist_ok=True)


def save_artifact(obj: Any, path: str) -> None:
    """
    Save a Python object using joblib.

    Args:
        obj: Object to save.
        path: Destination file path.
    """
    parent_dir = os.path.dirname(path)
    if parent_dir:
        ensure_dir(parent_dir)
    joblib.dump(obj, path)


def load_artifact(path: str) -> Any:
    """
    Load a saved joblib artifact.

    Args:
        path: Path to artifact.

    Returns:
        Loaded Python object.
    """
    return joblib.load(path)


def save_json(data: dict, path: str) -> None:
    """
    Save dictionary data as JSON.

    Args:
        data: Dictionary to save.
        path: Destination JSON file path.
    """
    parent_dir = os.path.dirname(path)
    if parent_dir:
        ensure_dir(parent_dir)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_json(path: str) -> dict:
    """
    Load a JSON file.

    Args:
        path: JSON file path.

    Returns:
        Parsed dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_features_target(
    df: pd.DataFrame,
    target_col: str = "Class",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split a dataframe into features and target.

    Args:
        df: Input dataframe.
        target_col: Name of target column.

    Returns:
        Tuple of (X, y).

    Raises:
        ValueError: If target column is missing.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()
    return X, y