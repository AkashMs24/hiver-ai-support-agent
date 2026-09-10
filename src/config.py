"""
Central configuration for the Hiver Support Agent.
Loads from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# === Paths ===
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
GOLDEN_DATA_DIR = DATA_DIR / "golden"
EVAL_RESULTS_DIR = ROOT_DIR / "eval" / "results"

# Ensure directories exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, GOLDEN_DATA_DIR, EVAL_RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# === LLM Configuration ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
GROQ_FAST_MODEL = os.getenv("GROQ_FAST_MODEL", "llama-3.1-8b-instant")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# === Pipeline Configuration ===
BRAND = os.getenv("BRAND", "AppleSupport")
SUBSAMPLE_SIZE = int(os.getenv("SUBSAMPLE_SIZE", "5000"))
INTENT_CONFIDENCE_THRESHOLD = float(os.getenv("INTENT_CONFIDENCE_THRESHOLD", "0.6"))
ESCALATION_THRESHOLD = float(os.getenv("ESCALATION_THRESHOLD", "0.55"))

# === Embedding Configuration ===
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # Dimension for all-MiniLM-L6-v2

# === Dataset ===
RAW_CSV_FILENAME = "twcs.csv"  # Twitter Customer Support dataset filename
RAW_CSV_PATH = RAW_DATA_DIR / RAW_CSV_FILENAME

# === Processed Data Paths ===
CONVERSATIONS_PATH = PROCESSED_DATA_DIR / "conversations.jsonl"
BRAND_STATS_PATH = PROCESSED_DATA_DIR / "brand_stats.json"
INTENT_CLUSTERS_PATH = PROCESSED_DATA_DIR / "intent_clusters.json"
FAISS_INDEX_PATH = PROCESSED_DATA_DIR / "reply_index.faiss"
FAISS_METADATA_PATH = PROCESSED_DATA_DIR / "reply_metadata.jsonl"
CLASSIFIER_PATH = PROCESSED_DATA_DIR / "intent_classifier.joblib"
EMBEDDINGS_CACHE_PATH = PROCESSED_DATA_DIR / "embeddings_cache.npy"

# === Evaluation ===
GOLDEN_SET_PATH = GOLDEN_DATA_DIR / "golden_set.jsonl"
JUDGE_CALIBRATION_PATH = GOLDEN_DATA_DIR / "judge_calibration.jsonl"

# === Twitter Constraints ===
MAX_REPLY_LENGTH = 280


def get_llm_provider() -> str:
    """Determine which LLM provider to use based on available keys."""
    if GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here":
        return "groq"
    if GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here":
        return "gemini"
    raise ValueError(
        "No LLM API key found. Set GROQ_API_KEY or GEMINI_API_KEY in .env file."
    )


def validate_config():
    """Validate that required configuration is present."""
    errors = []
    if not RAW_CSV_PATH.exists():
        errors.append(
            f"Dataset not found at {RAW_CSV_PATH}. "
            f"Download from https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter "
            f"and place twcs.csv in {RAW_DATA_DIR}/"
        )
    try:
        get_llm_provider()
    except ValueError as e:
        errors.append(str(e))
    return errors
