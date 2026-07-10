"""LLM-based rubric judge and key-point coverage checker.

Two complementary reference-free signals:

1. **Rubric judge** — a *different* Groq model from the generator scores the
   reply on 5 criteria (1-5 each) with written rationale.  Using a different
   model reduces self-preference bias.

2. **Key-point coverage** — the LLM extracts the must-address points from the
   incoming email, then a checker counts how many the reply covers.  This gives
   a transparent, auditable completeness signal.

Both return normalised [0, 1] scores.
"""
from __future__ import annotations

import json
from typing import Any

from . import config
from .llm import chat, chat_json


# --------------------------------------------------------------------------- #
# 1. Rubric judge
# --------------------------------------------------------------------------- #
_JUDGE_SYSTEM = """\
You are an expert email quality evaluator. You will be given an incoming email
and a candidate reply. Score the reply on EACH of the following criteria using
a 1–5 scale, and provide a one-sentence rationale for each score.

Criteria:
  relevance      — Does the reply respond to what THIS incoming email actually asked?
  completeness   — Are ALL asks, questions, and requests handled?
  correctness    — Is the reply grounded in fact? Does it avoid invented details?
  tone           — Is the tone appropriately professional, warm, or firm?
  actionability  — Does the reply give clear next steps the recipient can act on?

Scoring guide:
  5 = Excellent — fully satisfies the criterion
  4 = Good — minor gap
  3 = Acceptable — noticeable gap but still usable
  2 = Poor — significant issue
  1 = Bad — fails the criterion

Return ONLY a JSON object (no markdown fences) with this exact structure:
{
  "relevance":     {"score": <int 1-5>, "rationale": "<string>"},
  "completeness":  {"score": <int 1-5>, "rationale": "<string>"},
  "correctness":   {"score": <int 1-5>, "rationale": "<string>"},
  "tone":          {"score": <int 1-5>, "rationale": "<string>"},
  "actionability": {"score": <int 1-5>, "rationale": "<string>"}
}
"""


def judge_reply(
    incoming: str,
    reply: str,
    *,
    subject: str = "",
    cache_bust: str | None = None,
) -> dict[str, Any]:
    """Score a reply with the rubric judge.

    Returns the raw JSON dict from the LLM (scores + rationales) plus a
    normalised ``overall`` field in [0, 1].
    """
    user = (
        f"Incoming email subject: {subject}\n\n"
        f"Incoming email:\n{incoming}\n\n"
        f"Candidate reply:\n{reply}"
    )
    result = chat_json(
        [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
        model=config.JUDGE_MODEL,
        temperature=config.JUDGE_TEMPERATURE,
        max_tokens=512,
        cache_bust=cache_bust,
    )

    # Normalise: average of criterion scores, mapped from [1,5] → [0,1].
    scores = []
    for crit in config.JUDGE_CRITERIA:
        entry = result.get(crit, {})
        s = entry.get("score", 3) if isinstance(entry, dict) else 3
        s = max(1, min(5, int(s)))
        if isinstance(entry, dict):
            entry["score"] = s
        scores.append(s)

    result["overall"] = round((sum(scores) / len(scores) - 1) / 4, 4)  # [0,1]
    return result


# --------------------------------------------------------------------------- #
# 2. Key-point extraction + coverage
# --------------------------------------------------------------------------- #
_EXTRACT_SYSTEM = """\
You are an analyst. Given an incoming email, extract a JSON list of the key
points, questions, or requests that a reply MUST address to be complete.
Each item should be a short phrase (max 15 words).

Return ONLY a JSON object: {"key_points": ["point 1", "point 2", ...]}
"""

_COVER_SYSTEM = """\
You are a coverage checker.  Given an incoming email's key points and a
candidate reply, determine which key points the reply addresses.

Return ONLY a JSON object:
{"coverage": [{"point": "<key point>", "covered": true/false, "evidence": "<brief quote or explanation>"}]}
"""


def key_point_coverage(
    incoming: str,
    reply: str,
    *,
    subject: str = "",
) -> dict[str, Any]:
    """Extract key points from the incoming email and check reply coverage.

    Returns::

        {
            "key_points": [...],
            "coverage": [{"point": ..., "covered": bool, "evidence": ...}, ...],
            "score": <float 0-1>,  # fraction of key points covered
        }
    """
    # Step 1: extract key points.
    extract_msg = (
        f"Subject: {subject}\n\nIncoming email:\n{incoming}"
    )
    extracted = chat_json(
        [
            {"role": "system", "content": _EXTRACT_SYSTEM},
            {"role": "user", "content": extract_msg},
        ],
        model=config.JUDGE_MODEL,
        temperature=0.0,
        max_tokens=512,
    )
    key_points = extracted.get("key_points", [])
    if not key_points:
        return {"key_points": [], "coverage": [], "score": 1.0}

    # Step 2: check coverage.
    cover_msg = (
        f"Key points:\n{json.dumps(key_points)}\n\n"
        f"Candidate reply:\n{reply}"
    )
    covered = chat_json(
        [
            {"role": "system", "content": _COVER_SYSTEM},
            {"role": "user", "content": cover_msg},
        ],
        model=config.JUDGE_MODEL,
        temperature=0.0,
        max_tokens=512,
    )
    cov_list = covered.get("coverage", [])
    n_covered = sum(1 for c in cov_list if c.get("covered", False))
    score = n_covered / len(key_points) if key_points else 1.0

    return {
        "key_points": key_points,
        "coverage": cov_list,
        "score": round(score, 4),
    }
