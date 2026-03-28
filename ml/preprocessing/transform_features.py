"""
Reusable feature transformation logic.

This module handles:
- log-transforming Amount
- scaling Amount using StandardScaler
- reusing the same fitted scaler during inference

Important:
The same transform logic used during training must be used in the API.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


AMOUNT_COL = "Amount"


def log_transform_amount(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log1p transform to the Amount column.

    log1p(x) = log(1 + x), which is useful for skewed positive values.

    Args:
        df: Input dataframe.

    Returns:
        Dataframe with transformed Amount column.

    Raises:
        ValueError: If Amount column is missing.
    """
    if AMOUNT_COL not in df.columns:
        raise ValueError(f"'{AMOUNT_COL}' column is required for transformation.")

    df = df.copy()
    df[AMOUNT_COL] = np.log1p(df[AMOUNT_COL])
    return df


def fit_amount_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """
    Fit a StandardScaler on the Amount column of training data only.

    Args:
        X_train: Training features dataframe.

    Returns:
        Fitted StandardScaler.
    """
    if AMOUNT_COL not in X_train.columns:
        raise ValueError(f"'{AMOUNT_COL}' column not found in training data.")

    scaler = StandardScaler()
    scaler.fit(X_train[[AMOUNT_COL]])
    return scaler


def apply_amount_scaler(
    df: pd.DataFrame,
    scaler: StandardScaler,
) -> pd.DataFrame:
    """
    Apply a fitted scaler to the Amount column.

    Args:
        df: Input dataframe.
        scaler: Fitted StandardScaler.

    Returns:
        Transformed dataframe.
    """
    if AMOUNT_COL not in df.columns:
        raise ValueError(f"'{AMOUNT_COL}' column not found in dataframe.")

    df = df.copy()
    df[[AMOUNT_COL]] = scaler.transform(df[[AMOUNT_COL]])
    return df


def fit_transform_train(
    X_train: pd.DataFrame,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Fit transformations on training data and return transformed data + scaler.

    Steps:
    - log-transform Amount
    - fit scaler on transformed Amount
    - scale transformed Amount

    Args:
        X_train: Training features dataframe.

    Returns:
        Tuple of (transformed training dataframe, fitted scaler)
    """
    X_train = log_transform_amount(X_train)
    scaler = fit_amount_scaler(X_train)
    X_train = apply_amount_scaler(X_train, scaler)
    return X_train, scaler


def transform_new_data(
    df: pd.DataFrame,
    scaler: StandardScaler,
) -> pd.DataFrame:
    """
    Transform validation/test/inference data using an already-fitted scaler.

    Steps:
    - log-transform Amount
    - apply saved scaler

    Args:
        df: Dataframe to transform.
        scaler: Previously fitted scaler.

    Returns:
        Transformed dataframe.
    """
    df = log_transform_amount(df)
    df = apply_amount_scaler(df, scaler)
    return df