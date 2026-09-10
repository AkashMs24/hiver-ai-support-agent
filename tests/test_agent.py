import pytest
from src.data.preprocessor import clean_tweet_text, extract_metadata_features
from src.intent.taxonomy import IntentCategory, INTENT_METADATA
from src.intent.classifier import IntentClassifier
from src.escalation.engine import EscalationEngine
from src.escalation.signals import SignalExtractor


def test_clean_tweet_text():
    raw = "Hey @123456 my phone broke!! check __email__ and &amp; help"
    cleaned = clean_tweet_text(raw)
    assert "@user" in cleaned
    assert "[EMAIL]" in cleaned
    assert "&" in cleaned
    assert "@123456" not in cleaned


def test_extract_metadata_features():
    shout_text = "MY PHONE IS TOTALLY BROKEN AND UNACCEPTABLE"
    feats = extract_metadata_features(shout_text)
    assert feats["has_anger_keywords"] is True
    assert feats["shouting_ratio"] > 0.5


def test_intent_classifier_bootstrap():
    classifier = IntentClassifier()
    assert classifier.is_trained is True
    intent, conf, method = classifier.classify("My iPhone screen is cracked and flickering")
    assert intent in [c.value for c in IntentCategory]
    assert 0.0 <= conf <= 1.0


def test_escalation_engine_critical_override():
    engine = EscalationEngine()
    # Stolen device should trigger immediate escalation override
    res = engine.evaluate(
        customer_message="Someone broke into my car and stole my iPhone and laptop!!",
        intent="account_access",
        intent_confidence=0.9,
    )
    assert res["decision"] == "escalate"
    assert len(res["reasons"]) > 0


def test_escalation_engine_auto_handle():
    engine = EscalationEngine()
    # Routine benign inquiry
    res = engine.evaluate(
        customer_message="How do I change my home screen wallpaper?",
        intent="general_inquiry",
        intent_confidence=0.95,
    )
    assert res["decision"] == "auto_handle"
