"""Pure-Python reference-based text metrics.

All metrics are implemented from scratch against the standard library.  This
avoids torch/scikit-learn wheels that may not build on Python 3.14 and keeps the
eval harness reproducible with zero installs.

Exported functions all return floats in [0, 1].
"""
from __future__ import annotations

import math
import re
from collections import Counter


# --------------------------------------------------------------------------- #
# Shared tokenizer
# --------------------------------------------------------------------------- #
def _tokens(text: str) -> list[str]:
    """Lowercase word tokens (alpha-numeric)."""
    return re.findall(r"[a-z0-9]+", text.lower())


# --------------------------------------------------------------------------- #
# ROUGE-L (F1)
# --------------------------------------------------------------------------- #
def _lcs_length(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence (DP, O(n*m))."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    # Space-optimised to two rows.
    prev = [0] * (m + 1)
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (m + 1)
    return prev[m]


def rouge_l_f1(candidate: str, reference: str) -> float:
    """Compute ROUGE-L F1 between candidate and reference texts."""
    c_tok = _tokens(candidate)
    r_tok = _tokens(reference)
    if not c_tok or not r_tok:
        return 0.0
    lcs = _lcs_length(c_tok, r_tok)
    prec = lcs / len(c_tok)
    rec = lcs / len(r_tok)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


# --------------------------------------------------------------------------- #
# Token-level F1 (unigram overlap)
# --------------------------------------------------------------------------- #
def token_f1(candidate: str, reference: str) -> float:
    """Unigram precision/recall F1 between two texts."""
    c_tok = Counter(_tokens(candidate))
    r_tok = Counter(_tokens(reference))
    if not c_tok or not r_tok:
        return 0.0
    common = sum((c_tok & r_tok).values())
    prec = common / sum(c_tok.values())
    rec = common / sum(r_tok.values())
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


# --------------------------------------------------------------------------- #
# Jaccard similarity (set-level)
# --------------------------------------------------------------------------- #
def jaccard(candidate: str, reference: str) -> float:
    c_set = set(_tokens(candidate))
    r_set = set(_tokens(reference))
    if not c_set and not r_set:
        return 1.0
    if not c_set or not r_set:
        return 0.0
    return len(c_set & r_set) / len(c_set | r_set)


# --------------------------------------------------------------------------- #
# TF-IDF cosine similarity (self-contained, not using retriever.py)
# --------------------------------------------------------------------------- #
def tfidf_cosine(candidate: str, reference: str) -> float:
    """TF-IDF cosine between exactly two documents.

    Computes IDF from the two-document "corpus" — not the same as corpus-level
    IDF, but gives a principled token weighting that down-weights function words.
    """
    c_tok = _tokens(candidate)
    r_tok = _tokens(reference)
    if not c_tok or not r_tok:
        return 0.0

    # Build two-document IDF.
    docs = [set(c_tok), set(r_tok)]
    all_terms = docs[0] | docs[1]
    N = 2
    idf = {}
    for t in all_terms:
        df = sum(1 for d in docs if t in d)
        idf[t] = math.log((N + 1) / (df + 1)) + 1

    def _vec(tokens: list[str]) -> dict[str, float]:
        tf: dict[str, float] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        return {t: (1 + math.log(c)) * idf.get(t, 0) for t, c in tf.items()}

    va, vb = _vec(c_tok), _vec(r_tok)
    dot = sum(va[k] * vb[k] for k in va if k in vb)
    na = math.sqrt(sum(v * v for v in va.values())) or 1e-9
    nb = math.sqrt(sum(v * v for v in vb.values())) or 1e-9
    return dot / (na * nb)


# --------------------------------------------------------------------------- #
# Aggregate helper
# --------------------------------------------------------------------------- #
def all_reference_metrics(candidate: str, reference: str) -> dict[str, float]:
    """Compute all reference-based metrics and return as a dict."""
    return {
        "rouge_l_f1": rouge_l_f1(candidate, reference),
        "token_f1": token_f1(candidate, reference),
        "jaccard": jaccard(candidate, reference),
        "tfidf_cosine": tfidf_cosine(candidate, reference),
    }
