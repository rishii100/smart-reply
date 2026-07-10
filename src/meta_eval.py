"""Meta-evaluation: validates that the accuracy metric reflects REAL quality.

Three independent checks prove the metric isn't just producing a number:

1. **Discrimination / adversarial** — for a subset of test emails, create
   controlled quality variants (gold, off-topic, empty, wrong-facts, rude-tone,
   truncated) and score them.  The metric should rank gold highest and
   corruptions lowest.  Reports per-condition means and pairwise accuracy (the
   fraction of (gold, corruption) pairs ranked correctly).

2. **Human-anchor correlation** — uses hand-rated samples in
   `human_ratings.jsonl`.  Computes Spearman and Pearson correlation between
   each automatic metric and the human score — showing that the composite/judge
   correlate strongly while raw ROUGE is weak.

3. **Judge reliability** — runs the judge k times on the same items and reports
   score standard deviation and inter-metric agreement.
"""
from __future__ import annotations

import math
import random
from typing import Any

from . import config
from .evaluator import evaluate_one
from .dataset import load_human_ratings
from .judge import judge_reply
from .metrics import all_reference_metrics


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _std(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


def _pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 3:
        return 0.0
    mx, my = _mean(x), _mean(y)
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    sx = math.sqrt(sum((xi - mx) ** 2 for xi in x)) or 1e-9
    sy = math.sqrt(sum((yi - my) ** 2 for yi in y)) or 1e-9
    return cov / (sx * sy)


def _spearman(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation."""
    def _rank(vals: list[float]) -> list[float]:
        indexed = sorted(enumerate(vals), key=lambda t: t[1])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(indexed):
            j = i
            while j < len(indexed) and indexed[j][1] == indexed[i][1]:
                j += 1
            avg_rank = (i + j - 1) / 2 + 1
            for k in range(i, j):
                ranks[indexed[k][0]] = avg_rank
            i = j
        return ranks

    rx, ry = _rank(x), _rank(y)
    return _pearson(rx, ry)


# --------------------------------------------------------------------------- #
# 1. Discrimination test
# --------------------------------------------------------------------------- #
def _make_corruptions(
    email: dict[str, Any],
    all_emails: list[dict[str, Any]],
    rng: random.Random,
) -> dict[str, str]:
    """Create controlled quality variants of the gold reply."""
    gold = email["reply"]
    # Off-topic: pick a random OTHER email's reply.
    others = [e for e in all_emails if e["id"] != email["id"]]
    off_topic = rng.choice(others)["reply"] if others else "No reply."

    # Truncated: first 20% of words.
    words = gold.split()
    trunc = " ".join(words[: max(3, len(words) // 5)]) + "…"

    # Rude tone.
    rude = (
        f"Look, I don't have time for this. Here's the short version:\n"
        f"{' '.join(words[:len(words)//2])}\n"
        f"Deal with it yourself."
    )

    # Wrong facts: swap numbers and proper nouns.
    wrong = gold.replace("Monday", "Saturday").replace("Tuesday", "Sunday")
    wrong = wrong.replace("$49", "$4900").replace("$18", "$180")
    wrong = wrong.replace("3 users", "300 users").replace("48 hours", "48 weeks")
    if wrong == gold:
        wrong = "Our policy states the opposite of what you described. " + gold[:100]

    return {
        "gold": gold,
        "off_topic": off_topic,
        "empty": "",
        "truncated": trunc,
        "rude_tone": rude,
        "wrong_facts": wrong,
    }


def run_discrimination(
    test_emails: list[dict[str, Any]],
    all_emails: list[dict[str, Any]],
    *,
    n: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Evaluate controlled corruptions and measure pairwise accuracy."""
    n = n or config.META_EVAL_SUBSET
    rng = random.Random(config.SPLIT_SEED)
    subset = rng.sample(test_emails, min(n, len(test_emails)))

    condition_scores: dict[str, list[float]] = {
        k: [] for k in ["gold", "off_topic", "empty", "truncated", "rude_tone", "wrong_facts"]
    }
    pairwise_correct = 0
    pairwise_total = 0
    details: list[dict[str, Any]] = []

    for i, email in enumerate(subset, 1):
        if verbose:
            print(f"  [discrimination {i}/{len(subset)}] {email['id']}")
        corruptions = _make_corruptions(email, all_emails, rng)
        email_detail: dict[str, Any] = {"email_id": email["id"], "scores": {}}

        for cond_name, reply_text in corruptions.items():
            gen = {
                "email_id": email["id"],
                "category": email.get("category", "unknown"),
                "subject": email.get("subject", ""),
                "incoming": email["incoming"],
                "gold_reply": email["reply"],
                "generated_reply": reply_text,
            }
            result = evaluate_one(gen)
            score = result["composite_score"]
            condition_scores[cond_name].append(score)
            email_detail["scores"][cond_name] = score

            # Pairwise: gold should score higher than each corruption.
            if cond_name != "gold":
                pairwise_total += 1
                gold_score = condition_scores["gold"][-1] if cond_name != "gold" else score
                if cond_name == list(corruptions.keys())[-1]:
                    pass  # gold already added
                # Compare gold vs this corruption for this email
        
        # Now compute pairwise for this email
        gold_s = email_detail["scores"]["gold"]
        for cond_name, cond_s in email_detail["scores"].items():
            if cond_name != "gold":
                pairwise_total_temp = 1
                if gold_s > cond_s:
                    pairwise_correct += 1

        details.append(email_detail)

    condition_means = {k: round(_mean(v), 4) for k, v in condition_scores.items()}
    # Recompute pairwise properly
    pairwise_total = 0
    pairwise_correct = 0
    for d in details:
        gold_s = d["scores"].get("gold", 0)
        for cond, s in d["scores"].items():
            if cond != "gold":
                pairwise_total += 1
                if gold_s > s:
                    pairwise_correct += 1

    pairwise_acc = round(pairwise_correct / pairwise_total, 4) if pairwise_total else 0.0

    return {
        "n_emails": len(subset),
        "condition_means": condition_means,
        "pairwise_accuracy": pairwise_acc,
        "pairwise_correct": pairwise_correct,
        "pairwise_total": pairwise_total,
        "details": details,
    }


# --------------------------------------------------------------------------- #
# 2. Human-anchor correlation
# --------------------------------------------------------------------------- #
def run_human_correlation(verbose: bool = True) -> dict[str, Any]:
    """Correlate automatic metrics with human-rated samples."""
    ratings = load_human_ratings()
    if not ratings:
        if verbose:
            print("  [human-correlation] No human_ratings.jsonl found — skipping.")
        return {"n_samples": 0, "note": "No human ratings available."}

    human_scores: list[float] = []
    auto_scores: dict[str, list[float]] = {
        "composite": [], "llm_judge": [], "key_point_coverage": [],
        "rouge_l_f1": [], "tfidf_cosine": [], "token_f1": [], "jaccard": [],
    }

    for i, item in enumerate(ratings, 1):
        if verbose:
            print(f"  [human-correlation {i}/{len(ratings)}] rating={item.get('human_score')}")
        human_scores.append(float(item["human_score"]))

        gen = {
            "email_id": item.get("id", f"hr-{i}"),
            "category": item.get("category", "unknown"),
            "subject": item.get("subject", ""),
            "incoming": item["incoming"],
            "gold_reply": item.get("gold_reply", ""),
            "generated_reply": item["candidate_reply"],
        }
        result = evaluate_one(gen)
        auto_scores["composite"].append(result["composite_score"])
        auto_scores["llm_judge"].append(result["judge"]["overall"])
        auto_scores["key_point_coverage"].append(result["key_point_coverage"]["score"])
        ref = result["reference_metrics"]
        for k in ["rouge_l_f1", "tfidf_cosine", "token_f1", "jaccard"]:
            auto_scores[k].append(ref.get(k, 0.0))

    correlations: dict[str, dict[str, float]] = {}
    for metric_name, vals in auto_scores.items():
        correlations[metric_name] = {
            "pearson": round(_pearson(human_scores, vals), 4),
            "spearman": round(_spearman(human_scores, vals), 4),
        }

    return {
        "n_samples": len(ratings),
        "correlations": correlations,
    }


# --------------------------------------------------------------------------- #
# 3. Judge reliability
# --------------------------------------------------------------------------- #
def run_judge_reliability(
    test_emails: list[dict[str, Any]],
    *,
    k: int | None = None,
    n_items: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the judge k times on n items and report consistency."""
    k = k or config.JUDGE_RELIABILITY_SAMPLES
    n_items = n_items or config.JUDGE_RELIABILITY_ITEMS
    rng = random.Random(config.SPLIT_SEED + 1)
    subset = rng.sample(test_emails, min(n_items, len(test_emails)))

    item_results: list[dict[str, Any]] = []
    all_stds: list[float] = []

    for i, email in enumerate(subset, 1):
        if verbose:
            print(f"  [reliability {i}/{len(subset)}] {email['id']} × {k} runs")
        scores: list[float] = []
        for run in range(k):
            result = judge_reply(
                email["incoming"],
                email["reply"],
                subject=email.get("subject", ""),
                cache_bust=f"reliability-{run}",
            )
            scores.append(result["overall"])

        std = _std(scores)
        all_stds.append(std)
        item_results.append({
            "email_id": email["id"],
            "scores": [round(s, 4) for s in scores],
            "mean": round(_mean(scores), 4),
            "std": round(std, 4),
        })

    return {
        "k_runs": k,
        "n_items": len(subset),
        "items": item_results,
        "mean_std": round(_mean(all_stds), 4),
        "interpretation": (
            "Low mean_std (< 0.05) indicates the judge is consistent across runs. "
            "High std would suggest noisy/unreliable scoring."
        ),
    }


# --------------------------------------------------------------------------- #
# Full meta-eval
# --------------------------------------------------------------------------- #
def run_meta_eval(
    test_emails: list[dict[str, Any]],
    all_emails: list[dict[str, Any]],
    *,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run all three meta-evaluation checks."""
    if verbose:
        print("\n=== Meta-Evaluation: Validating the Metric ===\n")

    if verbose:
        print("--- 1. Discrimination (adversarial corruption test) ---")
    disc = run_discrimination(test_emails, all_emails, verbose=verbose)
    if verbose:
        print(f"  Pairwise accuracy: {disc['pairwise_accuracy']:.1%}")
        print(f"  Condition means: {disc['condition_means']}\n")

    if verbose:
        print("--- 2. Human-anchor correlation ---")
    human = run_human_correlation(verbose=verbose)
    if verbose and human.get("correlations"):
        for m, c in human["correlations"].items():
            print(f"    {m}: pearson={c['pearson']:.3f}  spearman={c['spearman']:.3f}")
        print()

    if verbose:
        print("--- 3. Judge reliability ---")
    rel = run_judge_reliability(test_emails, verbose=verbose)
    if verbose:
        print(f"  Mean score std across {rel['k_runs']} runs: {rel['mean_std']:.4f}\n")

    return {
        "discrimination": disc,
        "human_correlation": human,
        "judge_reliability": rel,
    }
