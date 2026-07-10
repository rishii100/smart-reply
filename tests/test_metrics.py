"""Unit tests for pure-python text metrics.  NO API key needed.

Tests verify that each metric:
  - Returns 1.0 for identical texts
  - Returns 0.0 for empty/completely disjoint texts
  - Returns a value in [0, 1]
  - Increases as overlap increases
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics import rouge_l_f1, token_f1, jaccard, tfidf_cosine, all_reference_metrics


def test_identical():
    """Identical texts should score 1.0."""
    text = "Hello, how are you doing today? I hope everything is going well."
    assert rouge_l_f1(text, text) > 0.99, f"rouge_l_f1 identical: {rouge_l_f1(text, text)}"
    assert token_f1(text, text) > 0.99, f"token_f1 identical: {token_f1(text, text)}"
    assert jaccard(text, text) > 0.99, f"jaccard identical: {jaccard(text, text)}"
    assert tfidf_cosine(text, text) > 0.99, f"tfidf_cosine identical: {tfidf_cosine(text, text)}"


def test_empty():
    """Empty text against non-empty should score 0.0."""
    text = "This is a test email about project deadlines."
    assert rouge_l_f1("", text) == 0.0
    assert rouge_l_f1(text, "") == 0.0
    assert token_f1("", text) == 0.0
    assert jaccard("", text) == 0.0
    assert tfidf_cosine("", text) == 0.0


def test_disjoint():
    """Completely disjoint texts should score near 0.0."""
    a = "quantum physics nuclear electron particle accelerator"
    b = "gardening roses tulips fertiliser watering sunlight"
    assert rouge_l_f1(a, b) < 0.01
    assert token_f1(a, b) < 0.01
    assert jaccard(a, b) < 0.01
    assert tfidf_cosine(a, b) < 0.01


def test_partial_overlap():
    """Partial overlap should give intermediate scores."""
    a = "Thank you for your email. We will process your refund within 5 days."
    b = "Thank you for your email. The refund has been processed and should appear soon."
    for metric in [rouge_l_f1, token_f1, jaccard, tfidf_cosine]:
        score = metric(a, b)
        assert 0.1 < score < 0.99, f"{metric.__name__}: {score}"


def test_range():
    """All metrics should return values in [0, 1]."""
    pairs = [
        ("short", "a much longer text about many different topics and subjects"),
        ("the quick brown fox jumps", "the lazy dog sleeps quietly"),
        ("meeting at 3pm tomorrow", "let's schedule a meeting for 3pm tomorrow"),
    ]
    for a, b in pairs:
        metrics = all_reference_metrics(a, b)
        for name, val in metrics.items():
            assert 0.0 <= val <= 1.0, f"{name}({a!r}, {b!r}) = {val}"


def test_ordering():
    """More similar text should score higher."""
    ref = "Please refund my payment of $49. The invoice number is INV-1234."
    good = "We've refunded your $49 payment for invoice INV-1234."
    bad = "The weather today is sunny with clear skies."

    for metric in [rouge_l_f1, token_f1, jaccard, tfidf_cosine]:
        good_score = metric(good, ref)
        bad_score = metric(bad, ref)
        assert good_score > bad_score, (
            f"{metric.__name__}: good={good_score:.4f} <= bad={bad_score:.4f}"
        )


def _run_all():
    tests = [
        test_identical, test_empty, test_disjoint,
        test_partial_overlap, test_range, test_ordering,
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
    print("Running metrics tests…")
    success = _run_all()
    sys.exit(0 if success else 1)
