import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .evaluation.eval_generator import EvalGenerator
from .evaluation.executor import EvalExecutor
from .evaluation.failure_classifier import FailureClassifier
from .evaluation.judge import Judge
from .generation.agent_generator import AgentGenerator
from .ingestion.chunker import chunk_documents
from .ingestion.loader import Document, load_folder
from .ingestion.vector_store import LocalTextStore
from .intake.groundability import GroundabilityGuard
from .intake.interviewer import IntakeInterviewer
from .llm.client import LLMClient
from .optimization.critic import Critic
from .optimization.optimizer import OptimizationResult, Optimizer
from .reporting.reporter import FinalReport, Reporter
from .runtime.agent_runner import AgentRunner
from .schemas.agent_definition import AgentDefinition
from .schemas.evaluation import EvaluationSet
from .schemas.reports import GroundabilityReport
from .schemas.requirements import Requirements
from .tools.registry import ToolRegistry


@dataclass
class FactoryRunResult:
    run_dir: Path
    requirements: Requirements
    groundability: GroundabilityReport
    evaluation_set: EvaluationSet
    optimization: OptimizationResult
    final_report: FinalReport


class ForgeFactory:
    def __init__(
        self,
        llm_client: LLMClient | None = None,
        tool_registry: ToolRegistry | None = None,
        reporter: Reporter | None = None,
    ) -> None:
        self.llm_client = llm_client or LLMClient()
        self.tool_registry = tool_registry or ToolRegistry()
        self.reporter = reporter or Reporter()

    @staticmethod
    def _safe_run_id(run_id: str) -> str:
        normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", run_id.strip())
        if not normalized or normalized in {".", ".."}:
            raise ValueError("run_id must contain a safe filename character.")
        return normalized

    @staticmethod
    def _write_json(path: Path, value: object) -> None:
        def serialize(item: object) -> object:
            if hasattr(item, "model_dump"):
                return item.model_dump(mode="json")
            if isinstance(item, dict):
                return {key: serialize(val) for key, val in item.items()}
            if isinstance(item, (list, tuple)):
                return [serialize(element) for element in item]
            if isinstance(item, Path):
                return str(item)
            return item

        path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(value, "model_dump_json"):
            serialized = value.model_dump_json(indent=2)
        else:
            serialized = json.dumps(serialize(value), ensure_ascii=False, indent=2)
        path.write_text(serialized, encoding="utf-8")

    def run(
        self,
        brief: str,
        documents: Path | list[Document],
        *,
        run_id: str,
        output_root: Path = Path("runs"),
        answers: dict[str, str] | None = None,
        answer_provider: Callable[[str], str] | None = None,
    ) -> FactoryRunResult:
        if not brief.strip():
            raise ValueError("The user brief cannot be empty.")
        document_list = (
            load_folder(documents)
            if isinstance(documents, Path)
            else documents
        )
        run_dir = output_root / self._safe_run_id(run_id)
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"Run directory already contains artifacts: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "brief.txt").write_text(brief.strip() + "\n", encoding="utf-8")

        interviewer = IntakeInterviewer(self.llm_client)
        questions = interviewer.generate_questions(brief, document_list)
        clarification_answers = dict(answers or {})
        for question in questions:
            if question not in clarification_answers and answer_provider is not None:
                clarification_answers[question] = answer_provider(question)

        requirements = interviewer.build_requirements(
            brief,
            document_list,
            clarification_answers,
        )
        self._write_json(run_dir / "requirements.json", requirements)

        grounding_chunks = chunk_documents(document_list)
        groundability = GroundabilityGuard(
            self.llm_client,
            LocalTextStore(grounding_chunks),
        ).assess(requirements, document_list)
        self._write_json(run_dir / "groundability.json", groundability)

        initial_agent = AgentGenerator(
            self.llm_client,
            self.tool_registry,
        ).generate(requirements, groundability, document_list)
        self._write_json(run_dir / "agent" / "initial_agent.json", initial_agent)

        evaluation_set = EvalGenerator(self.llm_client).generate(
            requirements,
            groundability,
            initial_agent,
            document_list,
        )
        self._write_json(run_dir / "evals" / "generated_evals.json", evaluation_set)

        def evaluate(
            agent_definition: AgentDefinition,
            cases: EvaluationSet,
        ):
            runner = AgentRunner(
                agent_definition,
                document_list,
                self.llm_client,
                self.tool_registry,
            )
            executor = EvalExecutor(runner, Judge(self.llm_client))
            return executor.run(cases)

        optimizer = Optimizer(FailureClassifier(self.llm_client), Critic())
        optimization = optimizer.optimize(
            initial_agent,
            evaluation_set,
            evaluate,
            groundability,
        )

        agent_dir = run_dir / "agent"
        for index, version in enumerate(optimization.agent_versions):
            name = "iteration_0_agent.json" if index == 0 else f"iteration_{index}_agent.json"
            self._write_json(agent_dir / name, version)
        self._write_json(agent_dir / "final_agent.json", optimization.agent_definition)

        eval_dir = run_dir / "evals"
        for index, results in enumerate(optimization.results_history):
            self._write_json(eval_dir / f"iteration_{index}_results.json", results)

        optimization_dir = run_dir / "optimization"
        classifications_payload = [
            [item.model_dump(mode="json") for item in group]
            for group in optimization.classification_history
        ]
        self._write_json(
            optimization_dir / "failure_classifications.json",
            classifications_payload,
        )
        self._write_json(
            optimization_dir / "iteration_log.json",
            {
                "stop_reason": optimization.stop_reason,
                "iterations": [
                    item.model_dump(mode="json")
                    for item in optimization.scorecards
                ],
            },
        )
        self._write_json(
            optimization_dir / "scorecards.json",
            [item.model_dump(mode="json") for item in optimization.scorecards],
        )

        report = self.reporter.build(optimization, groundability)
        self.reporter.write(report, run_dir)
        return FactoryRunResult(
            run_dir=run_dir,
            requirements=requirements,
            groundability=groundability,
            evaluation_set=evaluation_set,
            optimization=optimization,
            final_report=report,
        )
