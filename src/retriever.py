"""Pure-Python TF-IDF + cosine-similarity retriever.

Zero external dependencies.  Given a corpus of past (incoming, reply) pairs,
finds the k most similar incoming emails to a new query via TF-IDF vectors and
cosine similarity.  Good enough for a corpus of ~100 emails and avoids the
reproducibility risk of heavy ML wheels on Python 3.14.

Trade-off vs. embedding models:
  * TF-IDF is purely lexical — it misses semantic similarity between different
    phrasings of the same intent.  A sentence-transformer embedding would fix
    that, but adds a ~500 MB dependency.  The README documents this as an
    optional upgrade path.
  * For the typical professional-email domain (shared vocabulary, similar
    structure), TF-IDF works surprisingly well.
"""
from __future__ import annotations

import math
import re
from typing import Any

from . import config


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #
_STOP = frozenset(
    "a an the is are was were be been being have has had do does did will would "
    "shall should may might can could of in to for on with at by from as into "
    "through during before after above below between under again further then "
    "once here there when where why how all each every both few more most other "
    "some such no nor not only own same so than too very and but if or because "
    "until while about up out off over this that these those it its".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alpha, drop stop words and short tokens."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOP and len(t) > 1]


# --------------------------------------------------------------------------- #
# TF-IDF
# --------------------------------------------------------------------------- #
class TFIDFRetriever:
    """Build an in-memory TF-IDF index over email texts and retrieve by cosine."""

    def __init__(self, emails: list[dict[str, Any]], k: int | None = None):
        self.k = k or config.RETRIEVE_K
        self.emails = list(emails)
        self._vocab: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._vectors: list[dict[str, float]] = []
        self._build()

    # ----- index ----- #
    def _text(self, email: dict[str, Any]) -> str:
        """Concatenate subject + incoming for richer matching."""
        parts = [email.get("subject", ""), email.get("incoming", "")]
        return " ".join(parts)

    def _build(self) -> None:
        N = len(self.emails)
        if N == 0:
            return

        # Compute document frequencies.
        doc_tokens: list[list[str]] = []
        df: dict[str, int] = {}
        for e in self.emails:
            toks = tokenize(self._text(e))
            doc_tokens.append(toks)
            for t in set(toks):
                df[t] = df.get(t, 0) + 1

        # Build vocab and IDF.
        for i, term in enumerate(sorted(df)):
            self._vocab[term] = i
            self._idf[term] = math.log((N + 1) / (df[term] + 1)) + 1  # smoothed

        # Build TF-IDF vectors (sparse dicts).
        for toks in doc_tokens:
            tf: dict[str, float] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            vec = {t: (1 + math.log(c)) * self._idf.get(t, 0) for t, c in tf.items()}
            self._vectors.append(vec)

    # ----- query ----- #
    def _query_vec(self, text: str) -> dict[str, float]:
        toks = tokenize(text)
        tf: dict[str, float] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        return {t: (1 + math.log(c)) * self._idf.get(t, 0) for t, c in tf.items() if t in self._idf}

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        dot = sum(a[k] * b[k] for k in a if k in b)
        norm_a = math.sqrt(sum(v * v for v in a.values())) or 1e-9
        norm_b = math.sqrt(sum(v * v for v in b.values())) or 1e-9
        return dot / (norm_a * norm_b)

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        exclude_ids: set[str] | None = None,
    ) -> list[tuple[dict[str, Any], float]]:
        """Return the k most similar (email, score) pairs to the query text.

        `exclude_ids` prevents retrieval of specific email IDs (e.g. the gold
        email itself during evaluation).
        """
        k = k or self.k
        exclude_ids = exclude_ids or set()
        qvec = self._query_vec(query)
        if not qvec:
            return []

        scored = []
        for i, vec in enumerate(self._vectors):
            email = self.emails[i]
            if email["id"] in exclude_ids:
                continue
            scored.append((email, self._cosine(qvec, vec)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
