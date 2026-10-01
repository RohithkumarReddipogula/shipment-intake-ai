"""Run the extraction eval against Gemini and save the results.

    python eval/run_eval.py

Writes eval/results/<model>_<prompt_version>.json (per-email detail) and .md (tables
for the README). Re-running with the same model and prompt version overwrites them.

Results are not saved if any email failed to extract (API errors, quota), since those
would be scored as wrong fields. Pass --allow-errors to save anyway.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from app.config import get_settings
from app.evaluation import EvalAborted, evaluate, load_dataset, render_markdown, render_text
from app.extraction.gemini import GeminiExtractor

HERE = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=HERE / "dataset.jsonl")
    parser.add_argument("--out-dir", type=Path, default=HERE / "results")
    parser.add_argument("--delay", type=float, default=2.0, help="seconds between emails (rate limits)")
    parser.add_argument("--allow-errors", action="store_true", help="save results even if some emails failed")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    settings = get_settings()
    extractor = GeminiExtractor.from_settings(settings)
    cases = load_dataset(args.dataset)
    print(f"Evaluating {len(cases)} emails with {extractor.model} (prompt {extractor.prompt_version})\n")

    def progress(r: dict) -> None:
        mark = "ok " if r["all_fields_correct"] and r["decision_ok"] else "ERR"
        print(f"  [{mark}] {r['id']:<28} {r['decision']:<13} {', '.join(r['reasons'])}")

    try:
        report = evaluate(cases, extractor, settings.confidence_threshold,
                          on_case=progress, delay=args.delay)
    except EvalAborted as e:
        sys.exit(f"\nAborted: {e}\nNothing saved.")
    print("\n" + render_text(report))

    if report["errors"] and not args.allow_errors:
        sys.exit(f"\n{len(report['errors'])} emails failed to extract; their fields would count as "
                 "wrong. Nothing saved (use --allow-errors to save anyway).")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Not Path.with_suffix: model names like "gemini-3.8-flash" contain dots.
    name = f"{extractor.model}_{extractor.prompt_version}"
    json_path, md_path = args.out_dir / f"{name}.json", args.out_dir / f"{name}.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"\nSaved {json_path} and {md_path}")


if __name__ == "__main__":
    main()
