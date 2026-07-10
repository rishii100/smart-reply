"""Generate replies for the test set (standalone script).

Usage:
    python -m scripts.run_generate
    python -m scripts.run_generate --limit 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.llm import load_dotenv
from src.dataset import load_emails, get_pool_and_test
from src.retriever import TFIDFRetriever
from src.generator import generate_reply
from src.report import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate replies for test set")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str,
                        default=str(config.REPORTS_DIR / "generations.json"))
    args = parser.parse_args()

    load_dotenv()
    emails = load_emails()
    pool, test = get_pool_and_test(emails)

    if args.limit:
        test = test[: args.limit]

    print(f"Generating replies for {len(test)} test emails…")
    retriever = TFIDFRetriever(pool)
    generations = []
    for i, email in enumerate(test, 1):
        print(f"  [{i}/{len(test)}] {email['id']}")
        gen = generate_reply(email, retriever)
        generations.append(gen)

    save_json(generations, Path(args.output))
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
