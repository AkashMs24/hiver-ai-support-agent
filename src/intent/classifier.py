"""
Hybrid Intent Classifier:
1. Fast Primary: Logistic Regression / Embedding matching for low latency & low compute
2. Confidence Gate: If prediction confidence < threshold, triggers LLM few-shot verification
"""

import os
import joblib
from typing import Tuple, Dict, Any, Optional
import numpy as np
from rich.console import Console

from src.config import CLASSIFIER_PATH, INTENT_CONFIDENCE_THRESHOLD
from src.intent.taxonomy import INTENT_METADATA, IntentCategory, get_all_intents
from src.llm_client import get_llm_client

console = Console()


class IntentClassifier:
    def __init__(self, confidence_threshold: float = INTENT_CONFIDENCE_THRESHOLD):
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.vectorizer = None
        self.embedder = None
        self.is_trained = False
        self._load_or_bootstrap()

    def _load_or_bootstrap(self):
        """Load trained model from disk if available, else initialize prototype weights."""
        if os.path.exists(CLASSIFIER_PATH):
            try:
                saved = joblib.load(CLASSIFIER_PATH)
                self.model = saved["model"]
                self.vectorizer = saved["vectorizer"]
                self.is_trained = True
                return
            except Exception:
                pass

        # Bootstrap with keyword / prototype vectors
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        # Build initial training set using taxonomy metadata & few-shots
        texts, labels = [], []
        for intent, meta in INTENT_METADATA.items():
            for example in meta["few_shot_examples"]:
                texts.append(example)
                labels.append(intent)
            # Add synthetic keyword sequences
            texts.append(" ".join(meta["keywords"]))
            labels.append(intent)

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        X = self.vectorizer.fit_transform(texts)
        self.model = LogisticRegression(max_iter=500, C=1.0)
        self.model.fit(X, labels)
        self.is_trained = True

    def train_on_golden_or_silver(self, texts: list, labels: list):
        """Train or fine-tune the classifier on curated dataset."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=5000)
        X = self.vectorizer.fit_transform(texts)
        self.model = LogisticRegression(max_iter=1000, C=1.5)
        self.model.fit(X, labels)
        self.is_trained = True

        joblib.dump({"model": self.model, "vectorizer": self.vectorizer}, CLASSIFIER_PATH)
        console.print(f"[green]✓ Saved trained intent classifier to {CLASSIFIER_PATH}[/green]")

    def classify(self, text: str) -> Tuple[str, float, str]:
        """
        Classify customer query intent.
        Returns: (intent_name, confidence_score, method_used)
        """
        if not text or not text.strip():
            return IntentCategory.OTHER.value, 1.0, "rule_empty"

        # Step 1: Fast statistical inference
        X_test = self.vectorizer.transform([text])
        probs = self.model.predict_proba(X_test)[0]
        max_idx = np.argmax(probs)
        top_intent = self.model.classes_[max_idx]
        confidence = float(probs[max_idx])

        # Step 2: Gating check — if confident, return immediately
        if confidence >= self.confidence_threshold:
            return top_intent, confidence, "statistical_fast"

        # Step 3: Ambiguous or low confidence -> fallback to LLM zero/few-shot
        try:
            llm = get_llm_client()
            prompt = f"""You are a customer support intent classifier for @AppleSupport.
Classify the customer message into EXACTLY one of the following canonical intents:
{get_all_intents()}

Customer message: "{text}"

Statistical model suggested "{top_intent}" with low confidence ({confidence:.2f}).
Respond in strict JSON with keys:
"intent": (one of canonical strings above),
"confidence": (float between 0.0 and 1.0),
"reasoning": (short one sentence explanation)"""

            result = llm.generate_json(prompt=prompt, fast=True)
            candidate = result.get("intent", top_intent)
            if candidate in get_all_intents():
                llm_conf = float(result.get("confidence", 0.85))
                return candidate, llm_conf, "llm_gated_fallback"
        except Exception:
            pass

        return top_intent, confidence, "statistical_fallback"
