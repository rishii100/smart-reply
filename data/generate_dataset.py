"""Generate the full email dataset from hand-authored seeds.

For each of the 5 categories, this script few-shots the seed examples and asks
the LLM to synthesise additional realistic (incoming, reply) pairs.  The output
is written to `data/emails.jsonl`.

The script is re-runnable and deterministic (fixed prompts + on-disk cache).
The committed `emails.jsonl` means graders can run eval without regenerating.

Usage:
    python -m data.generate_dataset          # generate ~120 emails
    python -m data.generate_dataset --count 50  # smaller run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Adjust path so we can import src.*
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.llm import chat_json, load_dotenv
from src.dataset import load_seeds, save_jsonl


_SYSTEM = """\
You are a professional email dataset generator. You produce realistic email
(incoming, reply) pairs for a given category. The emails should feel like they
come from a real professional workplace — varied senders, specific details,
natural language, occasional typos. Replies should be helpful, specific, and
address every point in the incoming email.

IMPORTANT RULES:
1. Each pair must be UNIQUE — different scenario, different names, different details.
2. Vary the tone: some casual, some formal, some terse, some long.
3. Vary difficulty: some simple one-question emails, some complex multi-ask.
4. Use realistic names, company names, product names, dollar amounts, dates.
5. Do NOT reuse scenarios from the examples.
6. Each reply should be substantive — not just "OK" or "Thanks".
"""


def _generate_batch(
    category: str,
    seeds: list[dict],
    count: int,
    batch_num: int,
) -> list[dict]:
    """Generate a batch of emails for one category."""
    # Build few-shot examples from seeds.
    examples = []
    for s in seeds:
        examples.append(
            f'Subject: {s.get("subject", "")}\n'
            f'Incoming: {s["incoming"]}\n'
            f'Reply: {s["reply"]}\n'
            f'Tone: {s.get("tone", "professional")}\n'
            f'Difficulty: {s.get("difficulty", "medium")}'
        )

    prompt = (
        f"Generate {count} unique email (incoming, reply) pairs for the "
        f"category: {category}.\n\n"
        f"Here are examples of the style and quality expected:\n\n"
        + "\n---\n".join(examples)
        + f"\n\nGenerate {count} NEW and UNIQUE pairs. Return a JSON object:\n"
        '{"emails": [{"subject": "...", "incoming": "...", "reply": "...", '
        '"tone": "...", "difficulty": "easy|medium|hard"}, ...]}'
    )

    result = chat_json(
        [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ],
        model=config.DATASET_MODEL,
        temperature=config.DATASET_TEMPERATURE,
        max_tokens=4096,
    )

    emails = result.get("emails", [])
    out = []
    for i, e in enumerate(emails):
        out.append({
            "id": f"gen-{category[:3]}-{batch_num:02d}-{i+1:03d}",
            "category": category,
            "tone": e.get("tone", "professional"),
            "difficulty": e.get("difficulty", "medium"),
            "subject": e.get("subject", ""),
            "incoming": e.get("incoming", ""),
            "reply": e.get("reply", ""),
        })
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate email dataset from seeds")
    parser.add_argument("--count", type=int, default=config.TARGET_DATASET_SIZE,
                        help="Target total dataset size")
    args = parser.parse_args()

    load_dotenv()
    seeds = load_seeds()
    if not seeds:
        print("ERROR: No seeds found at", config.SEEDS_PATH)
        sys.exit(1)

    print(f"Loaded {len(seeds)} seeds")
    print(f"Target dataset size: {args.count}")

    # Group seeds by category.
    seeds_by_cat: dict[str, list[dict]] = {}
    for s in seeds:
        seeds_by_cat.setdefault(s["category"], []).append(s)

    per_cat = max(4, args.count // len(config.CATEGORIES))
    all_emails = list(seeds)  # Start with seeds.

    for cat in config.CATEGORIES:
        cat_seeds = seeds_by_cat.get(cat, seeds[:3])
        generated_count = 0
        batch = 0

        while generated_count < per_cat:
            batch += 1
            need = min(per_cat - generated_count, 8)  # batches of ≤8
            print(f"  [{cat}] Generating batch {batch} ({need} emails)…")
            try:
                batch_emails = _generate_batch(cat, cat_seeds, need, batch)
                all_emails.extend(batch_emails)
                generated_count += len(batch_emails)
                print(f"  [{cat}] Got {len(batch_emails)} emails (total {generated_count})")
            except Exception as e:
                print(f"  [{cat}] Batch {batch} failed: {e}")
                break

    # Deduplicate by ID (seeds + generated).
    seen = set()
    unique = []
    for e in all_emails:
        if e["id"] not in seen:
            seen.add(e["id"])
            unique.append(e)

    save_jsonl(unique, config.EMAILS_PATH)
    print(f"\nSaved {len(unique)} emails to {config.EMAILS_PATH}")

    # Also create splits.
    from src.dataset import make_splits, save_splits
    splits = make_splits(unique)
    save_splits(splits)
    print(f"Created splits: {len(splits['retrieval_pool'])} pool, {len(splits['test'])} test")


if __name__ == "__main__":
    main()
