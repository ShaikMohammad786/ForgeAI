"""Schemas for the Critic-Optimizer loop, budget tracker, regression diffs, and honest report."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PatchChange(BaseModel):
    """Single targeted modification applied to an AgentDefinition."""

    field_path: str
    old_value: Any
    new_value: Any
    rationale: str


class IterationRecord(BaseModel):
    """Audit log entry for one optimization iteration."""

    iteration: int
    dominant_failure_category: str
    failure_ids_addressed: list[str] = Field(default_factory=list)
    diagnosis_summary: str
    changes: list[PatchChange] = Field(default_factory=list)
    score_before: float
    score_after: float
    pass_rate_before: float
    pass_rate_after: float
    accepted: bool
    decision_reason: str
    tokens_used: int = 0
    duration_seconds: float = 0.0


class BudgetSummary(BaseModel):
    """Cumulative token & latency tracking across the factory pipeline."""

    max_tokens: int
    tokens_used: int
    prompt_tokens: int
    completion_tokens: int
    max_seconds: float
    elapsed_seconds: float
    llm_calls: int
    stopped_early_reason: str | None = None


class RegressionDiff(BaseModel):
    """Comparison produced by Regression Mode after document updates."""

    old_docs_count: int
    new_docs_count: int
    added_files: list[str] = Field(default_factory=list)
    modified_files: list[str] = Field(default_factory=list)
    removed_files: list[str] = Field(default_factory=list)
    old_pass_rate: float
    new_pass_rate: float
    newly_broken: list[dict[str, Any]] = Field(default_factory=list)
    newly_fixed: list[dict[str, Any]] = Field(default_factory=list)
    still_failing: list[dict[str, Any]] = Field(default_factory=list)
    now_ungroundable: list[dict[str, Any]] = Field(default_factory=list)


class FinalReport(BaseModel):
    """Honest quality report emitted at completion of a ForgeAI build."""

    agent_name: str
    brief: str
    provider: str
    generator_model: str
    judge_model: str
    final_pass_rate: float
    final_composite_score: float
    scorecards: list[dict[str, Any]] = Field(default_factory=list)
    iterations: list[IterationRecord] = Field(default_factory=list)
    tournament_candidates: list[dict[str, Any]] = Field(default_factory=list)
    remaining_failures: list[dict[str, Any]] = Field(default_factory=list)
    groundability_warnings: list[str] = Field(default_factory=list)
    budget: BudgetSummary
    honest_assessment: str
