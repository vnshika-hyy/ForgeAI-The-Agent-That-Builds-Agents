from forgeai.factory import ForgeFactory


class FakeLLMClient:
    def generate_json(self, prompt: str, schema: type):
        name = schema.__name__
        if name == "ClarificationQuestions":
            data = {"questions": []}
        elif name == "Requirements":
            data = {
                "agent_name": "Factory Test Agent",
                "purpose": "Answer based on supplied materials.",
                "scope": "Supplied materials",
                "target_audience": "Test users",
                "desired_tone": "Clear and concise",
                "must_answer_topics": [],
                "must_refuse_topics": [],
                "required_capabilities": [],
                "document_sources": [],
                "tool_requirements": [],
                "assumptions": [],
                "identified_grounding_gaps": [],
            }
        elif name == "AgentGenerationResponse":
            data = {
                "system_prompt": "Use only supplied evidence.",
                "retrieval": {
                    "chunk_size": 900,
                    "chunk_overlap": 150,
                    "top_k": 5,
                    "similarity_threshold": 0.0,
                },
                "tools": [],
                "refusal_policy": "Refuse unsupported requests.",
                "citation_policy": "Cite source evidence.",
                "output_contract": "Respond clearly.",
            }
        elif name == "GeneratedEvalBatch":
            if "Generation category: out_of_scope" in prompt:
                data = {
                    "cases": [
                        {
                            "question": "Can you provide information not in the supplied materials?",
                            "case_type": "out_of_scope",
                            "expected_behavior": "refuse",
                            "expected_key_facts": [],
                            "supporting_chunk_ids": [],
                            "notes": "The requested information is not sourced.",
                        }
                    ]
                }
            elif "Generation category: ambiguous" in prompt:
                data = {
                    "cases": [
                        {
                            "question": "Which one do you mean?",
                            "case_type": "ambiguous",
                            "expected_behavior": "refuse",
                            "expected_key_facts": [],
                            "supporting_chunk_ids": [],
                            "notes": "The request does not identify a target.",
                        }
                    ]
                }
            else:
                data = {"cases": []}
        elif name == "AgentTurn":
            data = {
                "action": "answer",
                "answer": "I cannot support this request from the provided materials.",
                "source_chunk_ids": [],
                "refused": True,
            }
        else:
            raise AssertionError(f"Unexpected LLM schema request: {name}")
        return schema.model_validate(data)


def test_mocked_end_to_end_factory_persists_artifacts(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "empty_context.md").write_text(
        "No substantive policy claims are present.", encoding="utf-8"
    )
    output_root = tmp_path / "runs"

    run = ForgeFactory(llm_client=FakeLLMClient()).run(
        "Build a generic assistant.",
        docs,
        run_id="mocked",
        output_root=output_root,
    )

    assert run.final_report.total_cases >= 1
    assert run.final_report.final_score == 1.0
    assert (run.run_dir / "brief.txt").is_file()
    assert (run.run_dir / "requirements.json").is_file()
    assert (run.run_dir / "groundability.json").is_file()
    assert (run.run_dir / "agent" / "initial_agent.json").is_file()
    assert (run.run_dir / "agent" / "final_agent.json").is_file()
    assert (run.run_dir / "evals" / "generated_evals.json").is_file()
    assert (run.run_dir / "evals" / "iteration_0_results.json").is_file()
    assert (run.run_dir / "optimization" / "iteration_log.json").is_file()
    assert (run.run_dir / "optimization" / "scorecards.json").is_file()
    assert (run.run_dir / "final_report.json").is_file()
    assert (run.run_dir / "final_report.md").is_file()
