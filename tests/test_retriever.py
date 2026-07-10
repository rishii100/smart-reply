"""Unit tests for the TF-IDF retriever.  NO API key needed.

Tests verify:
  - Retrieval returns results
  - More relevant queries rank higher
  - exclude_ids works
  - k parameter is respected
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retriever import TFIDFRetriever, tokenize


# Sample emails for testing.
_EMAILS = [
    {"id": "t1", "category": "billing", "subject": "Refund request",
     "incoming": "I was charged twice for my subscription. Please refund the duplicate.",
     "reply": "We've refunded the duplicate charge."},
    {"id": "t2", "category": "scheduling", "subject": "Meeting tomorrow",
     "incoming": "Can we move our 3pm meeting to 4pm tomorrow?",
     "reply": "Sure, I've updated the calendar invite."},
    {"id": "t3", "category": "customer_support", "subject": "App crash on login",
     "incoming": "The app crashes every time I try to log in on Android.",
     "reply": "We're aware of the bug and a fix is coming."},
    {"id": "t4", "category": "billing", "subject": "Invoice correction",
     "incoming": "Our invoice is missing the VAT number. Can you reissue?",
     "reply": "Reissued with VAT number."},
    {"id": "t5", "category": "sales_inquiry", "subject": "Pricing for 50 users",
     "incoming": "What's the pricing for a team of 50? Do you offer volume discounts?",
     "reply": "For 50 seats, here's our pricing breakdown."},
]


def test_basic_retrieval():
    """Retriever returns results for a relevant query."""
    ret = TFIDFRetriever(_EMAILS, k=3)
    results = ret.retrieve("I need a refund for a duplicate charge")
    assert len(results) > 0, "No results returned"
    assert len(results) <= 3, f"Too many results: {len(results)}"
    # Top result should be billing-related.
    top_id = results[0][0]["id"]
    assert top_id in ("t1", "t4"), f"Expected billing email, got {top_id}"


def test_relevance_ranking():
    """More relevant queries should rank higher."""
    ret = TFIDFRetriever(_EMAILS, k=5)
    results = ret.retrieve("duplicate charge refund billing")
    ids = [r[0]["id"] for r in results]
    # t1 (about duplicate charge refund) should be ranked first.
    assert ids[0] == "t1", f"Expected t1 first, got {ids}"


def test_exclude_ids():
    """exclude_ids should prevent specific emails from being returned."""
    ret = TFIDFRetriever(_EMAILS, k=3)
    results = ret.retrieve("refund duplicate charge", exclude_ids={"t1"})
    ids = [r[0]["id"] for r in results]
    assert "t1" not in ids, f"t1 should be excluded but got {ids}"


def test_k_limit():
    """Results should be limited to k."""
    ret = TFIDFRetriever(_EMAILS, k=2)
    results = ret.retrieve("email about a meeting or billing")
    assert len(results) <= 2


def test_empty_query():
    """Empty query should return empty results."""
    ret = TFIDFRetriever(_EMAILS, k=3)
    results = ret.retrieve("")
    assert len(results) == 0


def test_tokenizer():
    """Tokenizer should lowercase, split, and remove stop words."""
    tokens = tokenize("Hello, this is A Test! Email about REFUNDS.")
    assert "hello" in tokens
    assert "test" in tokens
    assert "refunds" in tokens
    assert "this" not in tokens  # stop word
    assert "is" not in tokens    # stop word


def _run_all():
    tests = [
        test_basic_retrieval, test_relevance_ranking,
        test_exclude_ids, test_k_limit, test_empty_query, test_tokenizer,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  ✅ {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {t.__name__}: {e}")
        except Exception as e:
            print(f"  ❌ {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
    return passed == len(tests)


if __name__ == "__main__":
    print("Running retriever tests…")
    success = _run_all()
    sys.exit(0 if success else 1)
