"""
Escalation Engine.
Combines multiple risk signals into a calibrated auto-handle vs. escalate decision with stated reasons.
"""

from typing import Dict, Any, List
from src.escalation.signals import SignalExtractor
from src.config import ESCALATION_THRESHOLD


class EscalationEngine:
    def __init__(self, threshold: float = ESCALATION_THRESHOLD):
        # Optimized for recall: Missing an urgent/angry escalation is 5x worse than a false alarm
        self.threshold = threshold
        self.extractor = SignalExtractor()

    def evaluate(
        self,
        customer_message: str,
        intent: str,
        intent_confidence: float,
        thread_turn_count: int = 1,
    ) -> Dict[str, Any]:
        """
        Evaluate whether the ticket can be automated or requires a human support agent.
        Returns:
            {
                "decision": "escalate" | "auto_handle",
                "escalation_score": float,
                "threshold": float,
                "reasons": List[str],
                "signal_breakdown": Dict[str, float]
            }
        """
        conf_sig = self.extractor.extract_confidence_signal(intent_confidence)
        sent_sig = self.extractor.extract_sentiment_signal(customer_message)
        sens_sig = self.extractor.extract_sensitivity_signal(customer_message)
        prior_sig = self.extractor.extract_intent_prior_signal(intent)

        # Thread length penalty (if it's turn 4+ and still not resolved, escalate)
        turn_risk = min(1.0, max(0.0, (thread_turn_count - 1) * 0.25))
        turn_reason = f"Prolonged unresolved dialogue ({thread_turn_count} turns)" if turn_risk >= 0.5 else None

        # Weighted combination
        weights = {
            "sensitivity": 0.35,
            "sentiment": 0.25,
            "confidence": 0.20,
            "intent_prior": 0.10,
            "turn_penalty": 0.10,
        }

        # Any single catastrophic flag triggers immediate hard escalation
        reasons: List[str] = []
        is_hard_override = False

        if sens_sig["risk_score"] >= 0.8:
            is_hard_override = True
            reasons.append(sens_sig["reason"])
        if sent_sig["risk_score"] >= 0.85:
            is_hard_override = True
            reasons.append(sent_sig["reason"])

        composite_score = (
            weights["sensitivity"] * sens_sig["risk_score"] +
            weights["sentiment"] * sent_sig["risk_score"] +
            weights["confidence"] * conf_sig["risk_score"] +
            weights["intent_prior"] * prior_sig["risk_score"] +
            weights["turn_penalty"] * turn_risk
        )

        # Collect soft reasons
        for sig, r in [
            (conf_sig, conf_sig["reason"]),
            (sent_sig, sent_sig["reason"]),
            (prior_sig, prior_sig["reason"]),
            ({"risk_score": turn_risk}, turn_reason),
        ]:
            if r and r not in reasons:
                reasons.append(r)

        should_escalate = is_hard_override or (composite_score >= self.threshold)

        if not should_escalate and not reasons:
            reasons.append("Standard automated self-service candidate; low safety/frustration risk")

        return {
            "decision": "escalate" if should_escalate else "auto_handle",
            "escalation_score": round(float(composite_score), 3),
            "threshold": self.threshold,
            "reasons": reasons,
            "signal_breakdown": {
                "sensitivity": round(sens_sig["risk_score"], 2),
                "sentiment": round(sent_sig["risk_score"], 2),
                "confidence_risk": round(conf_sig["risk_score"], 2),
                "intent_prior": round(prior_sig["risk_score"], 2),
                "turn_penalty": round(turn_risk, 2),
            },
        }
