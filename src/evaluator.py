"""Evaluator: combines all signals into per-response breakdowns + composite score.

The composite score blends reference-free (LLM judge, key-point coverage) with
reference-based (TF-IDF cosine, ROUGE-L) metrics.  Weights are deliberately
skewed toward reference-free quality because the gold reply is only ONE valid
reply — lexical overlap with it is a necessary but weak signal.

Weights (from config.py):
  llm_judge:            0.50   ← primary quality signal
  key_point_coverage:   0.25   ← transparent completeness check
  tfidf_cosine:         0.15   ← cheap lexical sanity floor
  rouge_l_f1:           0.10   ← classic reference metric, weakest alone
"""
from __future__ import annotations

from typing import Any

from . import config
from .metrics import all_reference_metrics
from .judge import judge_reply, key_point_coverage


def evaluate_one(
    generation: dict[str, Any],
    *,
    cache_bust: str | None = None,
) -> dict[str, Any]:
    """Evaluate a single generated reply and return full breakdown.

    `generation` should have keys: email_id, incoming, gold_reply,
    generated_reply, subject (optional), category (optional).
    """
    incoming = generation["incoming"]
    gold = generation.get("gold_reply", "")
    candidate = generation["generated_reply"]
    subject = generation.get("subject", "")

    # 1. Reference-based metrics (vs gold).
    ref_metrics = all_reference_metrics(candidate, gold) if gold else {
        "rouge_l_f1": 0.0, "token_f1": 0.0, "jaccard": 0.0, "tfidf_cosine": 0.0,
    }

    # 2. LLM rubric judge (reference-free).
    judge_result = judge_reply(
        incoming, candidate, subject=subject, cache_bust=cache_bust,
    )

    # 3. Key-point coverage (reference-free).
    coverage_result = key_point_coverage(incoming, candidate, subject=subject)

    # 4. Composite score.
    signals = {
        "llm_judge": judge_result["overall"],
        "key_point_coverage": coverage_result["score"],
        "tfidf_cosine": ref_metrics["tfidf_cosine"],
        "rouge_l_f1": ref_metrics["rouge_l_f1"],
    }

    composite = sum(
        config.COMPOSITE_WEIGHTS.get(k, 0) * v for k, v in signals.items()
    )
    composite = round(composite, 4)

    return {
        "email_id": generation["email_id"],
        "category": generation.get("category", "unknown"),
        "subject": subject,
        "incoming": incoming,
        "gold_reply": gold,
        "generated_reply": candidate,
        "retrieved_ids": generation.get("retrieved_ids", []),
        "reference_metrics": ref_metrics,
        "judge": judge_result,
        "key_point_coverage": coverage_result,
        "composite_signals": signals,
        "composite_score": composite,
        "pass": composite >= config.PASS_THRESHOLD,
    }


def evaluate_all(
    generations: list[dict[str, Any]],
    *,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Evaluate a list of generations with progress reporting."""
    results = []
    for i, gen in enumerate(generations, 1):
        if verbose:
            print(f"  [{i}/{len(generations)}] Evaluating {gen['email_id']}…")
        result = evaluate_one(gen)
        if verbose:
            print(f"    composite={result['composite_score']:.3f}  "
                  f"judge={result['judge']['overall']:.3f}  "
                  f"coverage={result['key_point_coverage']['score']:.3f}")
        results.append(result)
    return results


def summarise(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute overall and per-category summary statistics."""
    if not results:
        return {}

    def _mean(vals: list[float]) -> float:
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    composites = [r["composite_score"] for r in results]
    pass_count = sum(1 for r in results if r["pass"])

    # Per-criterion means.
    criterion_means = {}
    for crit in config.JUDGE_CRITERIA:
        vals = []
        for r in results:
            entry = r["judge"].get(crit, {})
            if isinstance(entry, dict):
                vals.append(entry.get("score", 3))
        criterion_means[crit] = _mean(vals)

    # Per-category breakdown.
    by_cat: dict[str, list[float]] = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r["composite_score"])
    category_scores = {cat: _mean(vals) for cat, vals in sorted(by_cat.items())}

    # Signal means.
    signal_means = {}
    for key in config.COMPOSITE_WEIGHTS:
        vals = [r["composite_signals"][key] for r in results]
        signal_means[key] = _mean(vals)

    return {
        "total_evaluated": len(results),
        "overall_composite": _mean(composites),
        "composite_std": round(
            (sum((c - _mean(composites)) ** 2 for c in composites) / len(composites)) ** 0.5, 4
        ),
        "pass_rate": round(pass_count / len(results), 4),
        "pass_threshold": config.PASS_THRESHOLD,
        "signal_means": signal_means,
        "criterion_means": criterion_means,
        "category_scores": category_scores,
        "score_distribution": {
            "min": round(min(composites), 4),
            "max": round(max(composites), 4),
            "median": round(sorted(composites)[len(composites) // 2], 4),
        },
        "composite_weights": dict(config.COMPOSITE_WEIGHTS),
    }
