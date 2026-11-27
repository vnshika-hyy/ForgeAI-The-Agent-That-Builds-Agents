#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from forgeai.ingestion.loader import load_folder
from forgeai.llm.client import LLMClient
from forgeai.runtime.agent_runner import AgentResponse, AgentRunner
from forgeai.schemas.agent_definition import AgentDefinition
from forgeai.tools.registry import ToolRegistry


def _print_response(response: AgentResponse) -> None:
    print(response.answer)
    if response.source_chunk_ids:
        print("Sources: " + ", ".join(response.source_chunk_ids))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a generated ForgeAI agent against its supplied documents."
    )
    parser.add_argument("--agent", required=True, help="Path to final_agent.json")
    parser.add_argument("--docs", required=True, help="Folder containing agent documents")
    parser.add_argument("--question", help="Ask one question and exit")
    args = parser.parse_args(argv)

    agent_path = Path(args.agent).resolve()
    documents_path = Path(args.docs).resolve()
    if not agent_path.is_file():
        parser.error(f"Agent definition not found: {agent_path}")
    if not documents_path.is_dir():
        parser.error(f"Documents folder not found: {documents_path}")

    definition = AgentDefinition.model_validate_json(
        agent_path.read_text(encoding="utf-8")
    )
    documents = load_folder(documents_path)
    runner = AgentRunner(
        agent_definition=definition,
        documents=documents,
        llm_client=LLMClient(),
        tool_registry=ToolRegistry(),
    )

    if args.question is not None:
        _print_response(runner.answer(args.question))
        return 0

    print(f"{definition.agent_name} is ready. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            question = input("You: ").strip()
        except EOFError:
            break
        if question.casefold() in {"exit", "quit"}:
            break
        if not question:
            continue
        _print_response(runner.answer(question))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
