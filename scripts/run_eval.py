"""Evaluate an existing generations file (works offline on committed sample).

Usage:
    python -m scripts.run_eval
    python -m scripts.run_eval --generations reports/sample_generations.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.llm import load_dotenv
from src.dataset import load_emails
from src.evaluator import evaluate_all, summarise
from src.report import write_reports, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate existing generations")
    parser.add_argument("--generations", type=str,
                        default=str(config.REPORTS_DIR / "sample_generations.json"))
    parser.add_argument("--output-dir", type=str, default=str(config.REPORTS_DIR))
    args = parser.parse_args()

    load_dotenv()
    gen_path = Path(args.generations)
    if not gen_path.exists():
        print(f"❌ Not found: {gen_path}")
        print("Run generation first: python -m scripts.run_generate")
        sys.exit(1)

    generations = json.loads(gen_path.read_text(encoding="utf-8"))
    print(f"Evaluating {len(generations)} generations from {gen_path}…")

    results = evaluate_all(generations)
    summary = summarise(results)

    print(f"\nOverall composite: {summary['overall_composite']:.4f}")
    print(f"Pass rate: {summary['pass_rate']:.0%}")

    output_dir = Path(args.output_dir)
    paths = write_reports(results, summary, output_dir=output_dir)
    print(f"\nReport: {paths['html']}")


if __name__ == "__main__":
    main()
