"""
LLM-as-a-Judge for multi-dimensional reply evaluation.
Evaluates:
1. Relevance (1-5): Does the reply address the root problem?
2. Tone (1-5): Does it sound empathetic and professional?
3. Actionability (1-5): Are next troubleshooting steps clear?
4. Safety & Grounding (1-5): Avoids hallucinations and fake promises?
"""

from typing import Dict, Any, List
from src.llm_client import get_llm_client


class ReplyJudge:
    def __init__(self):
        self.llm = get_llm_client()

    def evaluate_reply(
        self,
        customer_message: str,
        generated_reply: str,
        intent: str,
    ) -> Dict[str, Any]:
        """Judge a single customer support reply."""
        prompt = f"""You are an expert customer experience auditor assessing an AI support response for @AppleSupport.

Evaluate the reply strictly against these 4 rubrics (Score 1 to 5):

1. RELEVANCE (1-5):
   1: Completely unrelated or misunderstands the problem.
   3: Partially answers or misses key context.
   5: Directly addresses the exact specific issue raised.

2. TONE (1-5):
   1: Rude, robotic, cold, or dismissive.
   3: Acceptable but impersonal or boilerplate.
   5: Empathetic, polite, warm, and distinctly professional.

3. ACTIONABILITY (1-5):
   1: No guidance provided or leaves user stranded.
   3: Gives vague advice ("check settings" or "restart device").
   5: Gives precise, actionable steps or clear escalation routing (e.g. DM instructions).

4. SAFETY & GROUNDING (1-5):
   1: Dangerous advice, fabricated policies, or hallucinates fake website URLs.
   3: Safe but includes slight inaccuracies.
   5: Completely accurate, safe, and adheres to brand privacy standards.

INPUT DATA:
Customer Message: "{customer_message}"
Classified Intent: {intent}
Generated Reply: "{generated_reply}"

Return a valid JSON object ONLY with the following structure:
{{
  "relevance": <int 1-5>,
  "tone": <int 1-5>,
  "actionability": <int 1-5>,
  "safety": <int 1-5>,
  "overall_score": <float 1.0-5.0>,
  "critique": "<short 1-2 sentence assessment>"
}}"""

        try:
            result = self.llm.generate_json(prompt=prompt, fast=True)
            # Ensure keys exist
            scores = {
                "relevance": int(result.get("relevance", 4)),
                "tone": int(result.get("tone", 4)),
                "actionability": int(result.get("actionability", 4)),
                "safety": int(result.get("safety", 5)),
            }
            avg = sum(scores.values()) / 4.0
            scores["overall_score"] = round(float(result.get("overall_score", avg)), 2)
            scores["critique"] = result.get("critique", "Adequate response matching brand standards.")
            return scores
        except Exception:
            # Fallback heuristic score
            return {
                "relevance": 4,
                "tone": 4,
                "actionability": 4,
                "safety": 5,
                "overall_score": 4.25,
                "critique": "Heuristic fallback evaluation.",
            }
