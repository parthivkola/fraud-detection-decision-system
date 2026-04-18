"""Offline model evaluation — threshold tuning, confusion matrix, classification report."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def find_best_threshold(
    y_true: pd.Series,
    y_probs: np.ndarray,
    min_precision: float = 0.80,
    recall_tolerance: float = 0.02,
) -> Tuple[float, Dict[str, float]]:
    """
    Find the best threshold balancing precision and recall.

    Keeps only thresholds with precision >= min_precision, picks
    the one with highest recall (within tolerance), then breaks ties
    by precision and F1. Falls back to best F1 overall if nothing
    meets the precision floor.
    """
    thresholds = np.arange(0.05, 0.96, 0.05)
    candidates = []
    all_rows = []

    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        row = {
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

        all_rows.append(row)

        if precision >= min_precision:
            candidates.append(row)

    if not candidates:
        best = max(all_rows, key=lambda x: (x["f1"], x["recall"], x["precision"]))
        return best["threshold"], {
            "precision": best["precision"],
            "recall": best["recall"],
            "f1": best["f1"],
        }

    max_recall = max(row["recall"] for row in candidates)

    near_best_recall = [
        row for row in candidates
        if row["recall"] >= max_recall - recall_tolerance
    ]

    best = max(
        near_best_recall,
        key=lambda x: (x["precision"], x["f1"], x["threshold"]),
    )

    return best["threshold"], {
        "precision": best["precision"],
        "recall": best["recall"],
        "f1": best["f1"],
    }


def evaluate_at_threshold(
    y_true: pd.Series,
    y_probs: np.ndarray,
    threshold: float,
) -> Dict[str, object]:
    """Evaluate predictions at a given threshold. Returns a dict of metrics."""
    y_pred = (y_probs >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "roc_auc": roc_auc_score(y_true, y_probs),
        "pr_auc": average_precision_score(y_true, y_probs),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def compare_thresholds(
    y_true: pd.Series,
    y_probs: np.ndarray,
    thresholds: list[float],
) -> pd.DataFrame:
    """Compare precision/recall/F1 across multiple thresholds."""
    rows = []

    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
            }
        )

    return pd.DataFrame(rows).sort_values(by="threshold").reset_index(drop=True)