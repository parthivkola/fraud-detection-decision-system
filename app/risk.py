from __future__ import annotations
from typing import Tuple


def risk_level(prob: float, threshold: float) -> str:
    if prob < threshold * 0.3:
        return "LOW"
    if prob < threshold:
        return "MEDIUM"
    return "CRITICAL"


def decision(prob: float, threshold: float) -> str:
    if prob < threshold * 0.3:
        return "approve"
    if prob < threshold:
        return "review"
    return "block"


def assess_risk(prob: float, threshold: float) -> Tuple[str, str]:
    """Return (risk_level, decision) tuple for a given probability and threshold."""
    return risk_level(prob, threshold), decision(prob, threshold)
