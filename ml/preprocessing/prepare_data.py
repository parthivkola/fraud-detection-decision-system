"""
Basic dataframe preparation logic.

This module is responsible for:
- loading raw data
- validating required columns
- dropping columns not used for modeling
- performing lightweight dataframe checks

Keep this separate from feature transformations so the same logic
can be reused by both training and the API.
"""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd


FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COL = "Class"
DROP_COLUMNS = ["Time"]


def load_data(path: str) -> pd.DataFrame:
    """
    Load CSV data into a pandas DataFrame.

    Args:
        path: Path to CSV file.

    Returns:
        Loaded dataframe.
    """
    return pd.read_csv(path)


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: Optional[Iterable[str]] = None,
) -> None:
    """
    Validate that all required columns are present in the dataframe.

    Args:
        df: Input dataframe.
        required_columns: Columns expected to exist.

    Raises:
        ValueError: If any required column is missing.
    """
    if required_columns is None:
        raise ValueError("required_columns must be provided for validation.")

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")


def drop_unused_columns(
    df: pd.DataFrame,
    columns_to_drop: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Drop columns that should not be used by the model.

    Args:
        df: Input dataframe.
        columns_to_drop: Columns to remove.

    Returns:
        Updated dataframe.
    """
    if columns_to_drop is None:
        columns_to_drop = DROP_COLUMNS

    existing_cols = [col for col in columns_to_drop if col in df.columns]
    return df.drop(columns=existing_cols).copy()


def basic_checks(df: pd.DataFrame) -> None:
    """
    Run simple dataframe checks and print useful info.

    Args:
        df: Input dataframe.
    """
    print("Shape:", df.shape)
    print("Duplicate rows:", df.duplicated().sum())
    print("Missing values:\n", df.isna().sum())


def prepare_training_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare training dataframe before feature transformation.

    Steps:
    - validate required training columns
    - remove duplicate rows
    - drop unused columns like 'Time'

    Args:
        df: Raw dataframe.

    Returns:
        Prepared dataframe.
    """
    df = df.copy()
    validate_required_columns(df, required_columns=FEATURE_COLUMNS + [TARGET_COL])
    df = df.drop_duplicates().copy()
    df = drop_unused_columns(df)
    return df


def prepare_inference_dataframe(
    df: pd.DataFrame,
    expected_feature_columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Prepare inference dataframe before feature transformation.

    This function ensures that incoming data matches the exact schema
    used during training.

    Steps:
    - drop unused columns (e.g., 'Time')
    - validate required feature columns exist
    - remove unexpected extra columns
    - reorder columns to match training schema

    Args:
        df: Incoming inference dataframe.
        expected_feature_columns: Exact feature list used during training.

    Returns:
        Prepared dataframe ready for feature transformation.

    Raises:
        ValueError: If expected feature columns are not provided
            or if required columns are missing.
    """
    if expected_feature_columns is None:
        raise ValueError("expected_feature_columns must be provided for inference.")

    expected_feature_columns = list(expected_feature_columns)
    df = df.copy()

    # Drop unused columns like 'Time'
    df = drop_unused_columns(df)

    # Validate required feature columns
    validate_required_columns(df, required_columns=expected_feature_columns)

    # Remove extra columns not used by the model
    extra_cols = [col for col in df.columns if col not in expected_feature_columns]
    if extra_cols:
        df = df.drop(columns=extra_cols)

    # Reorder columns to match training schema exactly
    df = df[expected_feature_columns].copy()

    # Final sanity check
    if list(df.columns) != expected_feature_columns:
        raise ValueError("Column ordering mismatch after processing.")

    return df