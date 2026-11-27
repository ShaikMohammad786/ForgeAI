"""Schemas for Stage 1 (Intake Interviewer) & Stage 2 (Groundability Guard)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ClarifyingQA(BaseModel):
    """Single clarifying question & answer gathered during intake."""

    question: str
    answer: str
    answered_by: Literal["user", "auto_inferred"] = "auto_inferred"
    rationale: str = ""


class GroundabilityCheckItem(BaseModel):
    """Verification of one requested topic/capability against the document corpus."""

    requested_topic: str
    status: Literal["supported", "partial", "unsupported"]
    evidence_score: float = Field(ge=0.0, le=1.0)
    matching_sources: list[str] = Field(default_factory=list)
    matching_snippets: list[str] = Field(default_factory=list)
    rationale: str


class GroundabilityReport(BaseModel):
    """Aggregate groundability result preventing agents from hallucinating corpus gaps."""

    overall_status: Literal["grounded", "partially_grounded", "ungroundable"]
    groundability_ratio: float = Field(ge=0.0, le=1.0)
    items: list[GroundabilityCheckItem] = Field(default_factory=list)
    unsupported_requests: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class DocumentSummary(BaseModel):
    """Lightweight inventory record for an ingested corpus file."""

    filename: str
    relative_path: str
    doc_type: str
    sha256: str
    num_pages_or_rows: int
    num_chunks: int
    columns: list[str] = Field(default_factory=list)
    key_headings: list[str] = Field(default_factory=list)
    preview: str = ""


class RequirementsDoc(BaseModel):
    """Structured requirements output produced by intake + groundability guard."""

    brief: str
    domain_name: str
    scope: str
    audience: str
    tone: str
    must_answer_topics: list[str] = Field(default_factory=list)
    must_refuse_topics: list[str] = Field(default_factory=list)
    recommended_tools: list[str] = Field(default_factory=list)
    rejected_tools: dict[str, str] = Field(default_factory=dict)
    clarifying_qa: list[ClarifyingQA] = Field(default_factory=list)
    groundability: GroundabilityReport
    human_checkpoint_status: Literal[
        "approved", "edited", "auto_approved", "aborted"
    ] = "auto_approved"
    human_notes: str = ""
    corpus_summary: list[DocumentSummary] = Field(default_factory=list)
