"""
End-to-end Pipeline combining:
1. Intent Classifier
2. Historical RAG Retriever & Grounded Generator
3. Multi-signal Escalation Engine
"""

from typing import Dict, Any, Optional
from src.intent.classifier import IntentClassifier
from src.reply.index import ReplyIndex
from src.reply.retriever import ReplyRetriever
from src.reply.generator import ReplyGenerator
from src.escalation.engine import EscalationEngine
from src.data.preprocessor import clean_tweet_text


class SupportAgentPipeline:
    def __init__(self):
        self.classifier = IntentClassifier()
        self.reply_index = ReplyIndex()
        self.retriever = ReplyRetriever(self.reply_index)
        self.generator = ReplyGenerator()
        self.escalation = EscalationEngine()

    def process_message(
        self,
        customer_message: str,
        thread_turn_count: int = 1,
        conversation_history: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Full inference run on incoming customer query:
        1. Clean & normalize
        2. Classify intent
        3. Retrieve grounded support examples & generate draft
        4. Decide escalation status + reasons
        """
        cleaned_text = clean_tweet_text(customer_message)

        # 1. Intent Classification
        intent, intent_conf, method = self.classifier.classify(cleaned_text)

        # 2. Historical Retrieval & Grounded Reply
        retrieved = self.retriever.retrieve(cleaned_text, top_k=3)
        draft_reply, sources, gen_conf = self.generator.generate_reply(
            customer_message=cleaned_text,
            intent=intent,
            retrieved_examples=retrieved,
        )

        # 3. Escalation Decision
        esc_result = self.escalation.evaluate(
            customer_message=cleaned_text,
            intent=intent,
            intent_confidence=intent_conf,
            thread_turn_count=thread_turn_count,
        )

        return {
            "input_text": customer_message,
            "cleaned_text": cleaned_text,
            "intent": {
                "category": intent,
                "confidence": round(intent_conf, 3),
                "classification_method": method,
            },
            "reply": {
                "draft_reply": draft_reply,
                "grounded_sources": sources,
                "char_length": len(draft_reply),
                "confidence": gen_conf,
            },
            "escalation": {
                "decision": esc_result["decision"],
                "score": esc_result["escalation_score"],
                "reasons": esc_result["reasons"],
                "signal_breakdown": esc_result["signal_breakdown"],
            },
        }
