"""
Unified LLM client that abstracts over Groq and Gemini APIs.
Provides a single interface for all LLM calls in the pipeline.
"""

import json
import time
import logging
from typing import Optional

from src.config import (
    GROQ_API_KEY, GEMINI_API_KEY,
    GROQ_MODEL, GROQ_FAST_MODEL, GEMINI_MODEL,
    get_llm_provider,
)

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified LLM client supporting Groq and Gemini backends."""

    def __init__(self, provider: Optional[str] = None):
        self.provider = provider or get_llm_provider()
        self._client = None
        self._init_client()

    def _init_client(self):
        if self.provider == "groq":
            from groq import Groq
            self._client = Groq(api_key=GROQ_API_KEY)
        elif self.provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            self._client = genai

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.3,
        max_tokens: int = 1024,
        fast: bool = False,
        json_mode: bool = False,
    ) -> str:
        """
        Generate text from the LLM.

        Args:
            prompt: User prompt
            system_prompt: System/instruction prompt
            temperature: Sampling temperature (lower = more deterministic)
            max_tokens: Maximum tokens in response
            fast: Use faster/cheaper model variant
            json_mode: Request JSON output format

        Returns:
            Generated text string
        """
        for attempt in range(3):
            try:
                if self.provider == "groq":
                    return self._groq_generate(
                        prompt, system_prompt, temperature, max_tokens, fast, json_mode
                    )
                elif self.provider == "gemini":
                    return self._gemini_generate(
                        prompt, system_prompt, temperature, max_tokens, json_mode
                    )
            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt + 1}/3): {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s
                else:
                    raise

    def _groq_generate(
        self, prompt, system_prompt, temperature, max_tokens, fast, json_mode
    ) -> str:
        model = GROQ_FAST_MODEL if fast else GROQ_MODEL
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    def _gemini_generate(
        self, prompt, system_prompt, temperature, max_tokens, json_mode
    ) -> str:
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if json_mode:
            generation_config["response_mime_type"] = "application/json"

        model = self._client.GenerativeModel(
            GEMINI_MODEL,
            generation_config=generation_config,
        )
        response = model.generate_content(full_prompt)
        return response.text

    def generate_json(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.1,
        fast: bool = False,
    ) -> dict:
        """Generate and parse JSON output from the LLM."""
        raw = self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            json_mode=True,
            fast=fast,
        )
        # Clean up response — sometimes models add markdown fences
        raw = raw.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        return json.loads(raw.strip())


# Singleton instance — import and use directly
_default_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get or create the default LLM client singleton."""
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client
