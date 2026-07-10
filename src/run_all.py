"""End-to-end CLI orchestrator.

Runs the full pipeline: generate replies for the test set, evaluate them, run
meta-evaluation, and write all reports.

Usage:
    python -m src.run_all                    # full run
    python -m src.run_all --limit 10         # quick test
    python -m src.run_all --skip-meta        # skip meta-eval (faster)
    python -m src.run_all --skip-generation  # re-evaluate existing generations
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import config
from .llm import load_dotenv
from .dataset import load_emails, get_pool_and_test, save_jsonl, load_jsonl
from .retriever import TFIDFRetriever
from .generator import generate_reply
from .evaluator import evaluate_all, summarise
from .meta_eval import run_meta_eval
from .report import write_reports, save_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Email Reply: end-to-end generate → evaluate → report"
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Max test emails to process (default: all)")
    parser.add_argument("--skip-meta", action="store_true",
                        help="Skip meta-evaluation (faster)")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip generation, re-evaluate existing generations")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for reports")
    args = parser.parse_args()

    load_dotenv()
    output_dir = Path(args.output_dir) if args.output_dir else config.REPORTS_DIR

    # ------------------------------------------------------------------ #
    # 1. Load data
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("📧 AI Email Reply — End-to-End Pipeline")
    print("=" * 60)

    emails = load_emails()
    if not emails:
        print("\n❌ No emails.jsonl found. Run dataset generation first:")
        print("   python -m data.generate_dataset")
        sys.exit(1)

    pool, test = get_pool_and_test(emails)
    print(f"\n📦 Dataset: {len(emails)} emails total")
    print(f"   Retrieval pool: {len(pool)}")
    print(f"   Test set:       {len(test)}")

    if args.limit:
        test = test[: args.limit]
        print(f"   (limited to {len(test)} for this run)")

    # ------------------------------------------------------------------ #
    # 2. Generate replies
    # ------------------------------------------------------------------ #
    gen_path = output_dir / "generations.json"

    if args.skip_generation and gen_path.exists():
        print(f"\n⏩ Skipping generation — loading from {gen_path}")
        generations = json.loads(gen_path.read_text(encoding="utf-8"))
    else:
        print("\n🤖 Generating replies…")
        retriever = TFIDFRetriever(pool)
        t0 = time.time()
        generations = []
        for i, email in enumerate(test, 1):
            print(f"  [{i}/{len(test)}] {email['id']}…")
            gen = generate_reply(email, retriever)
            generations.append(gen)
            print(f"    ✓ {len(gen['generated_reply'])} chars, "
                  f"retrieved: {gen['retrieved_ids']}")
        elapsed = time.time() - t0
        print(f"\n✅ Generated {len(generations)} replies in {elapsed:.1f}s")
        save_json(generations, gen_path)

    # ------------------------------------------------------------------ #
    # 3. Evaluate
    # ------------------------------------------------------------------ #
    print("\n📊 Evaluating replies…")
    t0 = time.time()
    results = evaluate_all(generations)
    elapsed = time.time() - t0
    print(f"\n✅ Evaluated {len(results)} replies in {elapsed:.1f}s")

    summary = summarise(results)
    print(f"\n{'=' * 60}")
    print(f"   Overall composite: {summary['overall_composite']:.4f}")
    print(f"   Pass rate:         {summary['pass_rate']:.0%}")
    print(f"   Score range:       [{summary['score_distribution']['min']:.4f}, "
          f"{summary['score_distribution']['max']:.4f}]")
    print(f"{'=' * 60}")

    # ------------------------------------------------------------------ #
    # 4. Meta-evaluation
    # ------------------------------------------------------------------ #
    meta = None
    if not args.skip_meta:
        print("\n🔬 Running meta-evaluation…")
        t0 = time.time()
        meta = run_meta_eval(test, emails)
        elapsed = time.time() - t0
        print(f"\n✅ Meta-evaluation complete in {elapsed:.1f}s")
    else:
        print("\n⏩ Skipping meta-evaluation (--skip-meta)")

    # ------------------------------------------------------------------ #
    # 5. Write reports
    # ------------------------------------------------------------------ #
    print("\n📝 Writing reports…")
    paths = write_reports(results, summary, meta, output_dir=output_dir)
    print(f"   per_response.json: {paths['per_response']}")
    print(f"   summary.json:      {paths['summary']}")
    if "meta_eval" in paths:
        print(f"   meta_eval.json:    {paths['meta_eval']}")
    print(f"   report.html:       {paths['html']}")

    # Also save a sample_generations for offline eval
    sample_path = output_dir / "sample_generations.json"
    save_json(generations, sample_path)
    print(f"   sample_generations.json: {sample_path}")

    print(f"\n🎉 Done! Open the report:")
    print(f"   open {paths['html']}")


if __name__ == "__main__":
    main()
