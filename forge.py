#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from forgeai.factory import ForgeFactory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build, evaluate, and improve a generated AI agent."
    )
    parser.add_argument("--brief", required=True, help="Description of the desired agent")
    parser.add_argument("--docs", required=True, help="Folder containing source documents")
    parser.add_argument("--run-id", help="Artifact directory name (default: generated)")
    parser.add_argument("--output-root", default="runs", help="Root for run artifacts")
    args = parser.parse_args(argv)

    run_id = args.run_id or "generated"
    try:
        result = ForgeFactory().run(
            args.brief,
            Path(args.docs).resolve(),
            run_id=run_id,
            output_root=Path(args.output_root),
            answer_provider=lambda question: input(f"{question}\n> "),
        )
    except Exception as exc:
        parser.exit(2, f"ForgeAI failed: {type(exc).__name__}: {exc}\n")

    print(f"Run artifacts: {result.run_dir}")
    print(
        "Final evaluation score: "
        f"{result.final_report.final_score:.1%} "
        f"({result.final_report.passed_cases}/{result.final_report.total_cases} passed)"
    )
    print(f"Final agent: {result.run_dir / 'agent' / 'final_agent.json'}")
    print(f"Report: {result.run_dir / 'final_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
