"""
Risk level logic: maps a fraud probability to a risk level and decision.

Boundaries are designed for a model trained with high scale_pos_weight (~877)
which pushes probabilities to extremes. The threshold (default 0.9) is loaded
from model_metadata.json at startup.

Risk bands:
    LOW      (prob < 0.3)              → approve
    MEDIUM   (0.3 ≤ prob < 0.6)        → review
    HIGH     (0.6 ≤ prob < threshold)  → review
    CRITICAL (prob ≥ threshold)        → block
"""

from __future__ import annotations

from typing import Tuple


def assess_risk(probability: float, threshold: float) -> Tuple[str, str]:
    """
    Map a fraud probability to (risk_level, decision).

    Args:
        probability: Model's predicted fraud probability (0–1).
        threshold: Decision threshold from model_metadata.json.

    Returns:
        Tuple of (risk_level, decision).
    """
    if probability >= threshold:
        return "CRITICAL", "block"
    elif probability >= 0.6:
        return "HIGH", "review"
    elif probability >= 0.3:
        return "MEDIUM", "review"
    else:
        return "LOW", "approve"
