"""
Preprocessor for customer support tweets.
Cleans noise, handles twitter specific artifacts, normalizes masks.
"""

import re
import html


def clean_tweet_text(text: str) -> str:
    """
    Clean and normalize customer or brand tweet text:
    - Decode HTML entities (&amp; -> &)
    - Normalize whitespace
    - Anonymize user handles (@12345 -> @user)
    - Preserve masked tokens (__email__, __phone__) but normalize to readable brackets
    - Preserve URLs but standardize formatting
    """
    if not text or not isinstance(text, str):
        return ""

    # Unescape HTML
    text = html.unescape(text)

    # Replace Kaggle masks with standardized tags
    text = re.sub(r'__email__', '[EMAIL]', text, flags=re.IGNORECASE)
    text = re.sub(r'__phone__', '[PHONE]', text, flags=re.IGNORECASE)

    # Anonymize user IDs (@\d+ -> @user) while keeping brand mentions if named
    text = re.sub(r'@\d+', '@user', text)

    # Normalize excessive newlines and whitespace
    text = re.sub(r'\r\n|\r|\n', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def extract_metadata_features(text: str) -> dict:
    """
    Extract linguistic and sensitivity features useful for intent and escalation.
    """
    cleaned = clean_tweet_text(text)
    
    # Check for urgent/frustrated keywords
    anger_keywords = ["terrible", "worst", "broken", "fraud", "scam", "sue", "lawsuit", "unacceptable", "furious", "stolen", "useless", "ridiculous"]
    has_anger = any(w in cleaned.lower() for w in anger_keywords)

    # Check for PII presence
    has_pii = "[EMAIL]" in cleaned or "[PHONE]" in cleaned or bool(re.search(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', cleaned))

    # Questions vs statements
    is_question = "?" in cleaned

    # All-caps detection (shouting / escalation trigger)
    words = [w for w in cleaned.split() if len(w) > 2 and w.isalpha()]
    shouting_ratio = (sum(1 for w in words if w.isupper()) / len(words)) if words else 0.0

    return {
        "char_length": len(cleaned),
        "word_count": len(cleaned.split()),
        "has_anger_keywords": has_anger,
        "has_pii": has_pii,
        "is_question": is_question,
        "shouting_ratio": round(shouting_ratio, 2),
    }
