"""
Multi-signal extraction for the escalation engine.
Computes 4 distinct orthogonal risk vectors:
1. Model Confidence Signal
2. Customer Sentiment & Frustration Signal
3. Sensitivity & Legal / PII Signal
4. Complexity & Repetition Signal
"""

import re
from typing import Dict, Any, List
from src.intent.taxonomy import INTENT_METADATA


class SignalExtractor:
    @staticmethod
    def extract_confidence_signal(classifier_confidence: float) -> Dict[str, Any]:
        """Lower classifier confidence indicates higher escalation risk."""
        risk = max(0.0, 1.0 - classifier_confidence)
        reason = f"Low intent classification certainty ({classifier_confidence:.2f})" if risk > 0.4 else None
        return {"risk_score": risk, "reason": reason}

    @staticmethod
    def extract_sentiment_signal(text: str) -> Dict[str, Any]:
        """Detect customer anger, hostility, or severe frustration."""
        t_lower = text.lower()
        anger_cues = [
            ("sue", 0.9), ("lawyer", 0.9), ("legal", 0.85), ("court", 0.85),
            ("unacceptable", 0.7), ("furious", 0.75), ("horrible", 0.6), ("scam", 0.8),
            ("robbery", 0.8), ("fraud", 0.85), ("garbage", 0.65), ("worst", 0.6),
            ("never buy again", 0.75), ("threat", 0.85)
        ]
        
        detected_cues = []
        max_score = 0.0
        for cue, weight in anger_cues:
            if re.search(r'\b' + re.escape(cue) + r'\b', t_lower):
                detected_cues.append(cue)
                max_score = max(max_score, weight)

        # Check excessive punctuation or uppercase shouts
        words = [w for w in text.split() if len(w) > 2 and w.isalpha()]
        shout_ratio = (sum(1 for w in words if w.isupper()) / len(words)) if words else 0.0
        if shout_ratio > 0.4:
            max_score = max(max_score, 0.65)
            detected_cues.append("SHOUTING")

        reason = f"Hostile sentiment detected: [{', '.join(detected_cues)}]" if detected_cues else None
        return {"risk_score": max_score, "reason": reason}

    @staticmethod
    def extract_sensitivity_signal(text: str) -> Dict[str, Any]:
        """Detect sensitive topics: security compromise, stolen devices, medical/child safety."""
        t_lower = text.lower()
        triggers = [
            ("hacked", 0.9), ("stolen", 0.8), ("identity theft", 0.95),
            ("medical", 0.85), ("hospital", 0.85), ("emergency", 0.9),
            ("child", 0.7), ("fire", 0.9), ("smoke", 0.9), ("exploded", 0.95),
            ("burn", 0.85)
        ]

        detected = []
        max_score = 0.0
        for cue, weight in triggers:
            if re.search(r'\b' + re.escape(cue) + r'\b', t_lower):
                detected.append(cue)
                max_score = max(max_score, weight)

        reason = f"Critical safety or security risk: [{', '.join(detected)}]" if detected else None
        return {"risk_score": max_score, "reason": reason}

    @staticmethod
    def extract_intent_prior_signal(intent: str) -> Dict[str, Any]:
        """Inherent domain complexity based on intent taxonomy metadata."""
        meta = INTENT_METADATA.get(intent, {})
        base_prob = meta.get("typical_escalation_prob", 0.25)
        reason = f"Intent '{intent}' carries elevated baseline escalation requirements" if base_prob >= 0.50 else None
        return {"risk_score": base_prob, "reason": reason}
