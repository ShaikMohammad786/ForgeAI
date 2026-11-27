"""Schemas for auto-generated evaluations, hybrid debiased judging, and scorecards."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


EvalCategory = Literal["answerable", "out_of_scope", "edge_case"]
ExpectedBehavior = Literal["answer", "refuse"]
FailureCategory = Literal[
    "none",
    "retrieval_miss",
    "prompt_defect",
    "hallucination",
    "wrong_refusal",
]


class EvalCase(BaseModel):
    """Single auto-generated test case tied to document evidence or out-of-scope rules."""

    case_id: str
    category: EvalCategory
    sub_type: str = "single_doc"
    question: str
    expected_behavior: ExpectedBehavior
    source_documents: list[str] = Field(default_factory=list)
    key_facts: list[str] = Field(default_factory=list)
    expected_tools: list[str] = Field(default_factory=list)
    rationale: str = ""
    verified_in_corpus: bool = True


class EvalSet(BaseModel):
    """Complete evaluation suite generated from a document corpus."""

    generated_at: str
    generator_model: str
    total_cases: int
    cases: list[EvalCase]


class HybridJudgeMetrics(BaseModel):
    """Granular deterministic + atomic NLI metrics defending against LLM judge biases."""

    contract_valid: bool
    refusal_correct: bool
    citation_valid: bool
    numeric_accuracy: float = Field(ge=0.0, le=1.0)
    fact_recall: float = Field(ge=0.0, le=1.0)
    groundedness: float = Field(ge=0.0, le=1.0)
    verbosity_penalty: float = Field(default=0.0, ge=0.0, le=0.4)
    tool_appropriateness: float = Field(default=1.0, ge=0.0, le=1.0)
    composite_score: float = Field(ge=0.0, le=1.0)


class CaseVerdict(BaseModel):
    """Full audit record of evaluating one test case."""

    case_id: str
    category: EvalCategory
    sub_type: str
    question: str
    expected_behavior: ExpectedBehavior
    actual_answer: str
    actual_refused: bool
    actual_citations: list[dict[str, str]] = Field(default_factory=list)
    actual_tools_used: list[str] = Field(default_factory=list)
    retrieved_sources: list[str] = Field(default_factory=list)
    retrieved_top_score: float = 0.0
    passed: bool
    failure_category: FailureCategory = "none"
    failure_reason: str = ""
    metrics: HybridJudgeMetrics
    fact_checks: list[dict[str, Any]] = Field(default_factory=list)
    latency_ms: float = 0.0


class IterationScorecard(BaseModel):
    """Aggregate scorecard for a single baseline or optimization iteration."""

    iteration: int
    label: str
    total_cases: int
    passed_cases: int
    pass_rate: float
    mean_composite_score: float
    mean_groundedness: float
    mean_fact_recall: float
    refusal_accuracy: float
    contract_adherence: float
    by_category: dict[str, dict[str, float]] = Field(default_factory=dict)
    failure_counts: dict[str, int] = Field(default_factory=dict)
    verdicts: list[CaseVerdict] = Field(default_factory=list)
