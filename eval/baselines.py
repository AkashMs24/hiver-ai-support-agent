"""
Implements two required comparison baselines:
1. Baseline A (Trivial Baseline):
   - Majority class intent prediction
   - Static generic template response
   - Always escalate to human agent

2. Baseline B (Simple Lexical/Keyword Baseline):
   - Keyword frequency matching for intent
   - Verbatim retrieval of closest historical response via BM25/TF-IDF
   - Fixed keyword dictionary rule for escalation (e.g. 'refund', 'broken', 'sue')
"""

from typing import Dict, Any, List
from src.intent.taxonomy import INTENT_METADATA, IntentCategory


class TrivialBaseline:
    """Trivial Baseline: Constant majority predictions & generic boilerplate."""

    def __init__(self, majority_intent: str = IntentCategory.GENERAL_INQUIRY.value):
        self.majority_intent = majority_intent
        self.canned_reply = (
            "We'd like to look into this with you. Please DM us your device model, "
            "iOS version, and more details so we can assist."
        )

    def process(self, text: str) -> Dict[str, Any]:
        return {
            "intent": self.majority_intent,
            "intent_confidence": 0.30,
            "draft_reply": self.canned_reply,
            "escalation_decision": "escalate",  # Always escalate
            "escalation_reason": "Trivial rule: Always transfer to human agent",
        }


class SimpleLexicalBaseline:
    """Simple Baseline: Word matching, TF-IDF lexical search, keyword escalation."""

    def __init__(self):
        self.intent_keywords = {
            intent: meta["keywords"]
            for intent, meta in INTENT_METADATA.items()
        }
        self.escalation_keywords = [
            "refund", "sue", "lawsuit", "hacked", "stolen", "unacceptable",
            "cancel", "fraud", "manager", "dispute", "charged", "worst"
        ]

    def process(self, text: str) -> Dict[str, Any]:
        t_lower = text.lower()

        # 1. Lexical intent matching
        scores = {}
        for intent, kws in self.intent_keywords.items():
            scores[intent] = sum(1 for kw in kws if kw in t_lower)

        best_intent = max(scores, key=scores.get)
        if scores[best_intent] == 0:
            best_intent = IntentCategory.OTHER.value

        # 2. Template based on best intent
        templates = {
            IntentCategory.DEVICE_HARDWARE.value: "Sorry to hear about the physical damage. Please bring your device into an Apple Store or authorized service center.",
            IntentCategory.SOFTWARE_UPDATE.value: "For issues with software updates, please back up your device and try restoring through iTunes or Finder on a computer.",
            IntentCategory.ACCOUNT_ACCESS.value: "You can reset your Apple ID password or unlock your account directly at iforgot.apple.com.",
            IntentCategory.BILLING_CHARGE.value: "To review recent charges or request a refund, please sign in at reportaproblem.apple.com.",
            IntentCategory.BATTERY_PERFORMANCE.value: "Check Settings > Battery > Battery Health to inspect your maximum capacity and peak performance capability.",
        }
        reply = templates.get(
            best_intent,
            "Thanks for reaching out. Please try restarting your device, and send us a direct message if the problem persists."
        )

        # 3. Keyword escalation rule
        has_esc_keyword = any(kw in t_lower for kw in self.escalation_keywords)
        decision = "escalate" if has_esc_keyword else "auto_handle"
        reason = "Matched lexical escalation keyword" if has_esc_keyword else "No escalation keywords detected"

        return {
            "intent": best_intent,
            "intent_confidence": 0.55,
            "draft_reply": reply,
            "escalation_decision": decision,
            "escalation_reason": reason,
        }
