"""Schemas for declarative Agent Definitions, Output Contract, and Tournament candidates."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RetrievalConfig(BaseModel):
    """Chunking and hybrid retrieval settings."""

    chunk_size: int = Field(default=600, ge=200, le=2000)
    chunk_overlap: int = Field(default=100, ge=0, le=500)
    top_k: int = Field(default=5, ge=1, le=15)
    hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    min_score_threshold: float = Field(default=0.08, ge=0.0, le=1.0)
    chunking_strategy: Literal["heading_recursive", "paragraph_recursive"] = (
        "heading_recursive"
    )
    use_query_expansion: bool = False


class ToolBinding(BaseModel):
    """Authorized tool attached to the generated agent."""

    name: str
    description: str
    justification: str
    required_capabilities: list[str] = Field(default_factory=list)


class OutputContract(BaseModel):
    """Strict output contract enforced by the generic runtime."""

    format: Literal["structured_json"] = "structured_json"
    required_fields: list[str] = Field(
        default_factory=lambda: [
            "answer",
            "citations",
            "refused",
            "refusal_reason",
            "confidence",
            "tools_used",
        ]
    )
    require_citations_when_answered: bool = True
    max_answer_words: int = 220
    refusal_phrase_prefix: str = (
        "I cannot answer this based on the provided documents"
    )


class AgentResponse(BaseModel):
    """Validated runtime output from a generated ForgeAI agent."""

    answer: str
    citations: list[dict[str, str]] = Field(default_factory=list)
    refused: bool = False
    refusal_reason: str | None = None
    confidence: Literal["high", "medium", "low"] = "medium"
    tools_used: list[str] = Field(default_factory=list)


class PromptCandidate(BaseModel):
    """Candidate system prompt evaluated in the Prompt Tournament."""

    candidate_id: str
    strategy_name: str
    system_prompt: str
    tournament_score: float = 0.0
    pairwise_wins: int = 0
    pairwise_losses: int = 0
    pairwise_ties: int = 0
    deterministic_score: float = 0.0
    notes: str = ""


class AgentDefinition(BaseModel):
    """Complete declarative agent blueprint shipped by ForgeAI."""

    agent_id: str
    name: str
    version: int = 1
    brief: str
    provider: str
    model_name: str
    judge_model_name: str
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    max_tool_steps: int = Field(default=3, ge=1, le=6)
    system_prompt: str
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    tools: list[ToolBinding] = Field(default_factory=list)
    output_contract: OutputContract = Field(default_factory=OutputContract)
    must_answer_topics: list[str] = Field(default_factory=list)
    must_refuse_topics: list[str] = Field(default_factory=list)
    groundability_caveats: list[str] = Field(default_factory=list)
    docs_dir: str
    created_at: str
