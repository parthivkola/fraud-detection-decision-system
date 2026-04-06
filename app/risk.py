"""
Risk bands:
    LOW      (prob < 0.3)              → approve
    MEDIUM   (0.3 ≤ prob < 0.6)        → review
    HIGH     (0.6 ≤ prob < threshold)  → review
    CRITICAL (prob ≥ threshold)        → block
"""

from __future__ import annotations

from typing import Tuple


def assess_risk(probability: float, threshold: float) -> Tuple[str, str]:
    """Return (risk_level, decision) for a given fraud probability."""
    if probability >= threshold:
        return "CRITICAL", "block"
    elif probability >= 0.6:
        return "HIGH", "review"
    elif probability >= 0.3:
        return "MEDIUM", "review"
    else:
        return "LOW", "approve"
