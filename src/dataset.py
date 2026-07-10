"""Dataset loading, splitting, and persistence.

Handles JSONL I/O and deterministic train/test splitting so the retrieval pool
never leaks gold answers for the held-out test set.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from . import config


# --------------------------------------------------------------------------- #
# JSONL helpers
# --------------------------------------------------------------------------- #
def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def save_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """Write a list of dicts as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_seeds() -> list[dict[str, Any]]:
    return load_jsonl(config.SEEDS_PATH)


def load_emails() -> list[dict[str, Any]]:
    return load_jsonl(config.EMAILS_PATH)


def load_human_ratings() -> list[dict[str, Any]]:
    return load_jsonl(config.HUMAN_RATINGS_PATH)


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def make_splits(
    emails: list[dict[str, Any]],
    test_size: int | None = None,
    seed: int | None = None,
) -> dict[str, list[str]]:
    """Split email IDs into retrieval_pool and test sets.

    The split is stratified by category (as far as possible) and deterministic
    given the seed.
    """
    test_size = test_size or config.TEST_SIZE
    seed = seed if seed is not None else config.SPLIT_SEED
    rng = random.Random(seed)

    by_cat: dict[str, list[str]] = {}
    for e in emails:
        by_cat.setdefault(e["category"], []).append(e["id"])

    test_ids: list[str] = []
    pool_ids: list[str] = []

    # Proportional allocation per category.
    for cat in sorted(by_cat):
        ids = by_cat[cat]
        rng.shuffle(ids)
        n = max(1, round(test_size * len(ids) / len(emails)))
        test_ids.extend(ids[:n])
        pool_ids.extend(ids[n:])

    return {"retrieval_pool": sorted(pool_ids), "test": sorted(test_ids)}


def save_splits(splits: dict[str, list[str]]) -> None:
    config.SPLITS_PATH.write_text(
        json.dumps(splits, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_splits() -> dict[str, list[str]]:
    if not config.SPLITS_PATH.exists():
        return {}
    return json.loads(config.SPLITS_PATH.read_text(encoding="utf-8"))


def get_pool_and_test(
    emails: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (pool_emails, test_emails).  Creates splits if missing."""
    splits = load_splits()
    if not splits:
        splits = make_splits(emails)
        save_splits(splits)

    id_to_email = {e["id"]: e for e in emails}
    pool = [id_to_email[i] for i in splits["retrieval_pool"] if i in id_to_email]
    test = [id_to_email[i] for i in splits["test"] if i in id_to_email]
    return pool, test
