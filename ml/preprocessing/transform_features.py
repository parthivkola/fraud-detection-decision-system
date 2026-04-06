from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


AMOUNT_COL = "Amount"


def log_transform_amount(df: pd.DataFrame) -> pd.DataFrame:
    """Apply log1p to the Amount column."""
    if AMOUNT_COL not in df.columns:
        raise ValueError(f"'{AMOUNT_COL}' column is required for transformation.")

    df = df.copy()
    df[AMOUNT_COL] = np.log1p(df[AMOUNT_COL])
    return df


def fit_amount_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Fit a StandardScaler on the Amount column."""
    if AMOUNT_COL not in X_train.columns:
        raise ValueError(f"'{AMOUNT_COL}' column not found in training data.")

    scaler = StandardScaler()
    scaler.fit(X_train[[AMOUNT_COL]])
    return scaler


def apply_amount_scaler(
    df: pd.DataFrame,
    scaler: StandardScaler,
) -> pd.DataFrame:
    """Scale the Amount column using a fitted scaler."""
    if AMOUNT_COL not in df.columns:
        raise ValueError(f"'{AMOUNT_COL}' column not found in dataframe.")

    df = df.copy()
    df[[AMOUNT_COL]] = scaler.transform(df[[AMOUNT_COL]])
    return df


def fit_transform_train(
    X_train: pd.DataFrame,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """Log-transform and scale Amount on training data. Returns (transformed_df, scaler)."""
    X_train = log_transform_amount(X_train)
    scaler = fit_amount_scaler(X_train)
    X_train = apply_amount_scaler(X_train, scaler)
    return X_train, scaler


def transform_new_data(
    df: pd.DataFrame,
    scaler: StandardScaler,
) -> pd.DataFrame:
    """Log-transform and scale Amount using a previously fitted scaler."""
    df = log_transform_amount(df)
    df = apply_amount_scaler(df, scaler)
    return df