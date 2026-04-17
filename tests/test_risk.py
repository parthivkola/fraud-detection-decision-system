"""Tests for the risk assessment engine."""

from app.risk import assess_risk


class TestRiskAssessment:
    def test_risk_low(self):
        level, decision = assess_risk(0.1, threshold=0.9)
        assert level == "LOW"
        assert decision == "approve"

    def test_risk_medium(self):
        level, decision = assess_risk(0.4, threshold=0.9)
        assert level == "MEDIUM"
        assert decision == "review"

    def test_risk_high(self):
        level, decision = assess_risk(0.7, threshold=0.9)
        assert level == "HIGH"
        assert decision == "review"

    def test_risk_critical(self):
        level, decision = assess_risk(0.95, threshold=0.9)
        assert level == "CRITICAL"
        assert decision == "block"

    def test_risk_at_boundary(self):
        level, decision = assess_risk(0.3, threshold=0.9)
        assert level == "MEDIUM"
        assert decision == "review"

    def test_risk_at_threshold(self):
        level, decision = assess_risk(0.9, threshold=0.9)
        assert level == "CRITICAL"
        assert decision == "block"
