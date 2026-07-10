"""Central configuration.

Everything tunable lives here so the pipeline is transparent and reproducible.
Values can be overridden with environment variables (loaded from a local .env by
`src.llm.load_dotenv`) without touching code.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
CACHE_DIR = ROOT / ".cache"

SEEDS_PATH = DATA_DIR / "seeds.jsonl"
EMAILS_PATH = DATA_DIR / "emails.jsonl"
HUMAN_RATINGS_PATH = DATA_DIR / "human_ratings.jsonl"
SPLITS_PATH = DATA_DIR / "splits.json"

for _d in (REPORTS_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _env(name: str, default: str) -> str:
    val = os.environ.get(name)
    return val if val not in (None, "") else default


# --------------------------------------------------------------------------- #
# Models (Groq).  Two DIFFERENT models on purpose:
#   - the generator writes replies
#   - the judge scores them  -> reduces self-preference / self-scoring bias.
# IDs are verified against GET /openai/v1/models at build time; override via env.
# --------------------------------------------------------------------------- #
GROQ_BASE_URL = _env("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GEN_MODEL = _env("GEN_MODEL", "llama-3.3-70b-versatile")
JUDGE_MODEL = _env("JUDGE_MODEL", "openai/gpt-oss-120b")
DATASET_MODEL = _env("DATASET_MODEL", "llama-3.3-70b-versatile")

# Sampling
GEN_TEMPERATURE = float(_env("GEN_TEMPERATURE", "0.4"))
JUDGE_TEMPERATURE = float(_env("JUDGE_TEMPERATURE", "0.0"))
DATASET_TEMPERATURE = float(_env("DATASET_TEMPERATURE", "0.8"))

# --------------------------------------------------------------------------- #
# Dataset / splits
# --------------------------------------------------------------------------- #
CATEGORIES = [
    "customer_support",
    "scheduling",
    "sales_inquiry",
    "internal_ops",
    "billing",
]
TARGET_DATASET_SIZE = int(_env("TARGET_DATASET_SIZE", "120"))
TEST_SIZE = int(_env("TEST_SIZE", "25"))
SPLIT_SEED = int(_env("SPLIT_SEED", "13"))

# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
RETRIEVE_K = int(_env("RETRIEVE_K", "3"))

# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
# Composite weights.  Deliberately weighted toward REFERENCE-FREE quality (the
# LLM rubric judge + key-point coverage) because the single gold reply is only
# one of many valid replies, so lexical overlap with it is a weak signal.
# Reference metrics act as a cheap, deterministic sanity floor.  Rationale and
# the validation of these weights live in the README + meta_eval.
COMPOSITE_WEIGHTS = {
    "llm_judge": 0.50,
    "key_point_coverage": 0.25,
    "tfidf_cosine": 0.15,
    "rouge_l_f1": 0.10,
}
# A reply at/above this composite is counted as "good enough to suggest".
PASS_THRESHOLD = float(_env("PASS_THRESHOLD", "0.65"))

# The five rubric criteria the judge scores (1-5 each).
JUDGE_CRITERIA = [
    "relevance",        # does it respond to what THIS email asked?
    "completeness",     # are all asks / questions handled?
    "correctness",      # grounded; no invented facts / commitments
    "tone",             # professional, appropriately warm/firm
    "actionability",    # clear next steps; easy for the recipient to act on
]

# --------------------------------------------------------------------------- #
# Meta-evaluation
# --------------------------------------------------------------------------- #
META_EVAL_SUBSET = int(_env("META_EVAL_SUBSET", "10"))   # emails used for discrimination
JUDGE_RELIABILITY_SAMPLES = int(_env("JUDGE_RELIABILITY_SAMPLES", "3"))
JUDGE_RELIABILITY_ITEMS = int(_env("JUDGE_RELIABILITY_ITEMS", "5"))

# --------------------------------------------------------------------------- #
# LLM client behaviour
# --------------------------------------------------------------------------- #
LLM_MAX_RETRIES = int(_env("LLM_MAX_RETRIES", "5"))
LLM_TIMEOUT = float(_env("LLM_TIMEOUT", "90"))
LLM_USE_CACHE = _env("LLM_USE_CACHE", "1") == "1"
