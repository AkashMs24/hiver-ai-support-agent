"""
RAG-grounded reply generator.
Generates an on-brand reply grounded in historical support precedents while adhering to Twitter's 280-char limit.
"""

from typing import List, Dict, Any, Tuple
from src.llm_client import get_llm_client
from src.config import MAX_REPLY_LENGTH, BRAND


class ReplyGenerator:
    def __init__(self, brand_name: str = BRAND):
        self.brand_name = brand_name
        self.llm = get_llm_client()

    def generate_reply(
        self,
        customer_message: str,
        intent: str,
        retrieved_examples: List[Dict[str, Any]],
    ) -> Tuple[str, List[str], float]:
        """
        Generate grounded support reply.
        Returns: (draft_reply, grounded_sources, confidence)
        """
        sources = [ex.get("brand_reply", "") for ex in retrieved_examples if ex.get("brand_reply")]

        examples_text = ""
        for i, src in enumerate(sources[:3], 1):
            examples_text += f"\nExample {i}: {src}"

        system_prompt = f"""You are an elite customer support specialist representing @{self.brand_name} on Twitter.
Your replies must be empathetic, precise, and strictly grounded in real brand practices.

Strict Constraints:
1. Twitter Limit: Must be under {MAX_REPLY_LENGTH} characters.
2. Tone: Warm, helpful, calm, professional.
3. Don't invent policies, store hours, or fake diagnostic links.
4. If troubleshooting steps are known for this intent, provide the exact first step.
5. If private customer details (IMEI, order number, Apple ID) are needed, ask them to DM @{self.brand_name}."""

        user_prompt = f"""Customer Issue: "{customer_message}"
Classified Intent: {intent}

Historical Grounding Examples (How our team resolved this previously):
{examples_text if examples_text else "No prior direct matches found. Use general Apple support best practices."}

Draft the single best reply tweet to send to the customer right now:"""

        try:
            raw_reply = self.llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=120,
                fast=False,
            ).strip()

            # Clean quotation marks if returned
            if raw_reply.startswith('"') and raw_reply.endswith('"'):
                raw_reply = raw_reply[1:-1]

            # Enforce 280 chars strictly
            if len(raw_reply) > MAX_REPLY_LENGTH:
                raw_reply = raw_reply[:MAX_REPLY_LENGTH - 3] + "..."

            return raw_reply, sources, 0.90

        except Exception as e:
            # Fallback canned reply if LLM API is unavailable
            fallback = f"We're here to help! Please send us a DM with your device model and OS version so we can look into this with you."
            return fallback, sources, 0.50
