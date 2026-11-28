"""Tests for Debiased Judge (bias defenses + 4 failure types) and Critic-Optimizer."""

from __future__ import annotations

import json
from pathlib import Path

from forgeai.factory.judge import _pairwise_debiased_compare, evaluate_single_case
from forgeai.factory.optimizer import run_optimization_loop
from forgeai.ingestion import load_corpus
from forgeai.ingestion.chunker import DocumentChunk
from forgeai.llm.base import BudgetTracker, LLMProvider, LLMResponse
from forgeai.retrieval.hybrid import RetrievedChunk
from forgeai.runtime.executor import AgentRuntime, ExecutionTrace
from forgeai.schemas.agent import AgentDefinition, AgentResponse, RetrievalConfig
from forgeai.schemas.evals import EvalCase


class _AlwaysFirstBiasJudge(LLMProvider):
    """Simulates a judge suffering 100% position bias (always picks FIRST)."""

    provider_name = "biased_judge"

    def complete(self, messages, *, temperature=0.1, max_tokens=1200, json_mode=False, json_schema=None):
        return LLMResponse(
            text=json.dumps({"winner": "FIRST", "reason": "always first"}),
            model=self.model_name,
            provider="biased_judge",
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
        )


class _SimpleJudge(LLMProvider):
    provider_name = "judge"

    def complete(self, messages, *, temperature=0.1, max_tokens=1200, json_mode=False, json_schema=None):
        return LLMResponse(
            text=json.dumps({"fact_checks": [], "unsupported_claims": [], "summary": "ok"}),
            model=self.model_name,
            provider="judge",
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
        )


def test_pairwise_position_bias_is_neutralized_to_tie() -> None:
    case = EvalCase(
        case_id="c1",
        category="answerable",
        question="How many leave days?",
        expected_behavior="answer",
        key_facts=["24 annual leave days"],
    )
    chunk = DocumentChunk("id1", "hr.md", "hr.md", "md", "Sec 1", "Leave", "24 annual leave days")
    hit = RetrievedChunk(chunk=chunk, score=0.8, bm25_score=0.7, dense_score=0.9)
    t_a = ExecutionTrace(response=AgentResponse(answer="24 days", citations=[{"source": "hr.md", "excerpt": "24"}]), retrieved_chunks=[hit])
    t_b = ExecutionTrace(response=AgentResponse(answer="24 leave days", citations=[{"source": "hr.md", "excerpt": "24"}]), retrieved_chunks=[hit])

    result = _pairwise_debiased_compare(case, t_a, t_b, _AlwaysFirstBiasJudge("bias"))
    assert result == "TIE"


def test_judge_classifies_all_four_failure_categories() -> None:
    judge = _SimpleJudge("j")
    chunk = DocumentChunk("id1", "hr.md", "hr.md", "md", "Sec 1", "Leave", "Annual leave is 24 days.")
    hit = RetrievedChunk(chunk=chunk, score=0.85, bm25_score=0.8, dense_score=0.9)

    ans_case = EvalCase(
        case_id="a1",
        category="answerable",
        question="How many annual leave days?",
        expected_behavior="answer",
        source_documents=["hr.md"],
        key_facts=["Annual leave is 24 days"],
    )
    oos_case = EvalCase(
        case_id="o1",
        category="out_of_scope",
        question="What is the CEO salary?",
        expected_behavior="refuse",
    )

    # 1. wrong_refusal (failed to refuse OOS)
    v_wr = evaluate_single_case(
        oos_case,
        ExecutionTrace(response=AgentResponse(answer="The CEO makes 500k.", refused=False)),
        judge,
        {"hr.md"},
        use_llm_judge=False,
    )
    assert v_wr.failure_category == "wrong_refusal"

    # 2. retrieval_miss (empty retrieval, no fabricated numbers)
    v_rm = evaluate_single_case(
        ans_case,
        ExecutionTrace(response=AgentResponse(answer="Leave entitlement details were not found in the retrieved sections.", citations=[]), retrieved_chunks=[]),
        judge,
        {"hr.md"},
        use_llm_judge=False,
    )
    assert v_rm.failure_category == "retrieval_miss"

    # 3. hallucination (unverified number when context has 24)
    v_hal = evaluate_single_case(
        ans_case,
        ExecutionTrace(
            response=AgentResponse(answer="Leave is 24 days plus 99 bonus yacht days.", citations=[{"source": "hr.md", "excerpt": "24"}]),
            retrieved_chunks=[hit],
        ),
        judge,
        {"hr.md"},
        use_llm_judge=False,
    )
    assert v_hal.failure_category == "hallucination"

    # 4. prompt_defect (facts retrieved, no hallucination, but invalid JSON contract / missed facts)
    v_pd = evaluate_single_case(
        ans_case,
        ExecutionTrace(
            response=AgentResponse(answer="See the policy document.", citations=[]),
            retrieved_chunks=[hit],
            raw_contract_valid=False,
        ),
        judge,
        {"hr.md"},
        use_llm_judge=False,
    )
    assert v_pd.failure_category == "prompt_defect"


def test_optimizer_rolls_back_when_patch_worsens_score(tmp_path: Path) -> None:
    (tmp_path / "hr.md").write_text("# Leave\nAnnual leave is 24 days.\n", encoding="utf-8")
    corpus = load_corpus(tmp_path)

    class _DegradingOnV2LLM(LLMProvider):
        provider_name = "test"
        def complete(self, messages, *, temperature=0.1, max_tokens=1200, json_mode=False, json_schema=None):
            sys_txt = messages[0]["content"]
            user_txt = messages[-1]["content"]
            if "DOMINANT FAILURE CATEGORY" in user_txt:
                return LLMResponse(
                    text=json.dumps({
                        "diagnosis_summary": "test patch",
                        "target_case_ids": ["c1"],
                        "prompt_addition": "PATCH_MARKER",
                        "retrieval_updates": {"top_k": 5, "hybrid_alpha": 0.5, "min_score_threshold": 0.08, "chunk_size": 600, "chunk_overlap": 100, "use_query_expansion": False},
                        "temperature": 0.05,
                    }),
                    model="t", provider="t", prompt_tokens=10, completion_tokens=10, total_tokens=20
                )
            # Fail c1 deliberately when PATCH_MARKER is present without fixing c2!
            if "CEO" in user_txt:
                ans = {"action": "final", "answer": "500k", "citations": [], "refused": False, "confidence": "low", "tools_used": []}
            elif "PATCH_MARKER" in sys_txt:
                ans = {"action": "final", "answer": "I cannot answer this based on the provided documents.", "citations": [], "refused": True, "confidence": "low", "tools_used": []}
            else:
                ans = {"action": "final", "answer": "Annual leave is 24 days.", "citations": [{"source": "hr.md", "excerpt": "24 days"}], "refused": False, "confidence": "high", "tools_used": []}
            return LLMResponse(text=json.dumps(ans), model="t", provider="t", prompt_tokens=10, completion_tokens=10, total_tokens=20)

    agent = AgentDefinition(
        agent_id="a1", name="Test", brief="HR", provider="test", model_name="t", judge_model_name="t",
        system_prompt="Base prompt", retrieval=RetrievalConfig(min_score_threshold=0.0), docs_dir=str(tmp_path), created_at="now"
    )
    llm = _DegradingOnV2LLM("t")
    budget = BudgetTracker()
    rt = AgentRuntime(agent, llm, corpus=corpus)
    cases = [
        EvalCase(case_id="c1", category="answerable", question="Annual leave days?", expected_behavior="answer", source_documents=["hr.md"], key_facts=["24 days"]),
        EvalCase(case_id="c2", category="out_of_scope", question="CEO salary?", expected_behavior="refuse"),
    ]
    final_agent, cards, iters = run_optimization_loop(agent, cases, rt, llm, llm, budget, max_iterations=1, use_llm_judge=False)
    assert iters[0].accepted is False
    assert "PATCH_MARKER" not in final_agent.system_prompt


def test_judge_clamps_fact_recall_when_small_judge_splits_facts() -> None:
    class _SplittingJudge(LLMProvider):
        provider_name = "split"
        def complete(self, messages, *, temperature=0.1, max_tokens=1200, json_mode=False, json_schema=None):
            return LLMResponse(
                text=json.dumps({
                    "fact_checks": [
                        {"fact": "24", "status": "supported", "evidence": "24"},
                        {"fact": "days", "status": "supported", "evidence": "days"},
                        {"fact": "annual", "status": "supported", "evidence": "annual"},
                    ],
                    "unsupported_claims": [],
                    "summary": "split 1 fact into 3",
                }),
                model="s", provider="split", prompt_tokens=5, completion_tokens=5, total_tokens=10,
            )

    chunk = DocumentChunk("id1", "hr.md", "hr.md", "md", "Sec 1", "Leave", "Annual leave is 24 days.")
    hit = RetrievedChunk(chunk=chunk, score=0.9, bm25_score=0.9, dense_score=0.9)
    case = EvalCase(
        case_id="s1",
        category="answerable",
        question="How many annual leave days?",
        expected_behavior="answer",
        source_documents=["hr.md"],
        key_facts=["Annual leave is 24 days."],
    )
    verdict = evaluate_single_case(
        case,
        ExecutionTrace(
            response=AgentResponse(answer="Annual leave is 24 days.", citations=[{"source": "hr.md", "excerpt": "24 days"}]),
            retrieved_chunks=[hit],
        ),
        _SplittingJudge("s"),
        {"hr.md"},
        use_llm_judge=True,
    )
    assert 0.0 <= verdict.metrics.fact_recall <= 1.0
    assert verdict.passed is True


def test_judge_recognizes_natural_refusal_and_filters_schema_examples() -> None:
    from forgeai.factory.eval_gen import _fact_grounded_in_corpus

    assert _fact_grounded_in_corpus("lead_time_days: examples=['10', '25']", "lead_time_days 10 25") is False

    judge = _SimpleJudge("j")
    oos = EvalCase(
        case_id="o_nat",
        category="out_of_scope",
        question="Compare competitor pricing?",
        expected_behavior="refuse",
    )
    verdict = evaluate_single_case(
        oos,
        ExecutionTrace(
            response=AgentResponse(
                answer="The provided documents do not contain information on competitors, so a comparison cannot be made.",
                refused=False,
            )
        ),
        judge,
        {"a.pdf"},
        use_llm_judge=False,
    )
    assert verdict.actual_refused is True
    assert verdict.passed is True


def test_judge_rejects_fabricated_citation_excerpt() -> None:
    judge = _SimpleJudge("j")
    chunk = DocumentChunk("id1", "hr.md", "hr.md", "md", "Sec 1", "Leave", "Annual leave is 24 days.")
    hit = RetrievedChunk(chunk=chunk, score=0.9, bm25_score=0.9, dense_score=0.9)
    case = EvalCase(
        case_id="c_cit",
        category="answerable",
        question="Annual leave?",
        expected_behavior="answer",
        source_documents=["hr.md"],
        key_facts=["Annual leave is 24 days."],
    )
    bad_cit = evaluate_single_case(
        case,
        ExecutionTrace(
            response=AgentResponse(
                answer="Annual leave is 24 days.",
                citations=[{"source": "hr.md", "excerpt": "Employees receive 99 yacht days annually"}],
            ),
            retrieved_chunks=[hit],
        ),
        judge,
        {"hr.md"},
        use_llm_judge=False,
    )
    assert bad_cit.metrics.citation_valid is False
    assert bad_cit.metrics.groundedness < 1.0
