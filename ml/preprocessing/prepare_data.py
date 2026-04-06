from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd


FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COL = "Class"
DROP_COLUMNS = ["Time"]


def load_data(path: str) -> pd.DataFrame:
    """Load CSV data into a DataFrame."""
    return pd.read_csv(path)


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: Optional[Iterable[str]] = None,
) -> None:
    """Raise ValueError if any required columns are missing."""
    if required_columns is None:
        raise ValueError("required_columns must be provided for validation.")

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")


def drop_unused_columns(
    df: pd.DataFrame,
    columns_to_drop: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Drop columns not used by the model."""
    if columns_to_drop is None:
        columns_to_drop = DROP_COLUMNS

    existing_cols = [col for col in columns_to_drop if col in df.columns]
    return df.drop(columns=existing_cols).copy()


def basic_checks(df: pd.DataFrame) -> None:
    """Print shape, duplicate count, and missing value info."""
    print("Shape:", df.shape)
    print("Duplicate rows:", df.duplicated().sum())
    print("Missing values:\n", df.isna().sum())


def prepare_training_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate columns, deduplicate, and drop unused columns."""
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
    Prepare incoming data for inference: drop unused columns, validate schema,
    remove extras, and reorder to match training column order.
    """
    if expected_feature_columns is None:
        raise ValueError("expected_feature_columns must be provided for inference.")

    expected_feature_columns = list(expected_feature_columns)
    df = df.copy()

    df = drop_unused_columns(df)
    validate_required_columns(df, required_columns=expected_feature_columns)

    extra_cols = [col for col in df.columns if col not in expected_feature_columns]
    if extra_cols:
        df = df.drop(columns=extra_cols)

    df = df[expected_feature_columns].copy()

    if list(df.columns) != expected_feature_columns:
        raise ValueError("Column ordering mismatch after processing.")

    return df