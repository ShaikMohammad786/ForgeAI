"""Stage 1: Intake Interviewer + Bonus 3 Human Checkpoint."""

from __future__ import annotations

from typing import Callable

from pydantic import BaseModel, Field

from forgeai.factory.groundability import (
    audit_groundability,
    split_brief_capabilities_and_refusals,
)
from forgeai.ingestion.loader import LoadedCorpus
from forgeai.llm.base import LLMProvider
from forgeai.prompts import render_prompt
from forgeai.schemas.requirements import ClarifyingQA, GroundabilityReport, RequirementsDoc
from forgeai.tools.registry import TOOL_CATALOG, evaluate_tool_eligibility


class _QuestionItem(BaseModel):
    question: str
    default_answer: str
    rationale: str = ""


class _IntakeQuestionsOutput(BaseModel):
    questions: list[_QuestionItem] = Field(default_factory=list)


class _RequirementsSynthesisOutput(BaseModel):
    domain_name: str
    scope: str
    audience: str
    tone: str
    must_answer_topics: list[str] = Field(default_factory=list)
    must_refuse_topics: list[str] = Field(default_factory=list)
    recommended_tools: list[str] = Field(default_factory=list)
    tool_reasons: dict[str, str] = Field(default_factory=dict)


def generate_clarifying_questions(
    brief: str,
    corpus: LoadedCorpus,
    llm: LLMProvider,
    *,
    max_questions: int = 5,
) -> list[_QuestionItem]:
    """Generate <=5 clarifying questions grounded in the brief and corpus."""
    max_q = max(1, min(5, max_questions))
    prompt = render_prompt(
        "intake_questions.j2",
        brief=brief,
        corpus_inventory=corpus.format_inventory(),
        max_questions=max_q,
    )
    try:
        parsed, _ = llm.complete_structured(
            [
                {"role": "system", "content": "You output strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            _IntakeQuestionsOutput,
            temperature=0.1,
            max_tokens=900,
        )
        questions = [q for q in parsed.questions if q.question.strip()][:max_q]
        if questions:
            return questions
    except Exception:
        pass

    # Deterministic fallback questions if small model fails JSON
    fallback = [
        _QuestionItem(
            question="Who is the primary end-user audience and what tone should responses use?",
            default_answer="Target the audience specified in the brief with a clear, concise, policy-accurate tone.",
            rationale="Clarifies audience framing and response style.",
        ),
        _QuestionItem(
            question="When a question asks for details missing from the provided documents, should the assistant strictly refuse?",
            default_answer="Yes, strictly refuse and explain what information is absent rather than guessing.",
            rationale="Defines refusal strictness.",
        ),
        _QuestionItem(
            question="Should every factual answer cite the exact source filename or section?",
            default_answer="Yes, include exact source document citations for every factual claim.",
            rationale="Ensures auditability.",
        ),
    ]
    return fallback[:max_q]


def run_intake_interview(
    brief: str,
    corpus: LoadedCorpus,
    llm: LLMProvider,
    *,
    embed_model_name: str = "BAAI/bge-small-en-v1.5",
    max_questions: int = 5,
    answer_callback: Callable[[list[_QuestionItem]], list[str]] | None = None,
    pre_supplied_answers: list[str] | None = None,
    groundability_report: GroundabilityReport | None = None,
) -> RequirementsDoc:
    """Execute Stage 1 (Intake) + Stage 2 (Groundability Guard) -> RequirementsDoc."""
    questions = generate_clarifying_questions(
        brief, corpus, llm, max_questions=max_questions
    )

    qa_list: list[ClarifyingQA] = []
    if pre_supplied_answers is not None:
        for idx, q in enumerate(questions):
            ans = (
                pre_supplied_answers[idx].strip()
                if idx < len(pre_supplied_answers) and pre_supplied_answers[idx].strip()
                else q.default_answer
            )
            by = "user" if idx < len(pre_supplied_answers) and pre_supplied_answers[idx].strip() else "auto_inferred"
            qa_list.append(
                ClarifyingQA(
                    question=q.question,
                    answer=ans,
                    answered_by=by,  # type: ignore[arg-type]
                    rationale=q.rationale,
                )
            )
    elif answer_callback is not None:
        user_answers = answer_callback(questions)
        for idx, q in enumerate(questions):
            raw_ans = user_answers[idx].strip() if idx < len(user_answers) else ""
            ans = raw_ans if raw_ans else q.default_answer
            by = "user" if raw_ans else "auto_inferred"
            qa_list.append(
                ClarifyingQA(
                    question=q.question,
                    answer=ans,
                    answered_by=by,  # type: ignore[arg-type]
                    rationale=q.rationale,
                )
            )
    else:
        for q in questions:
            qa_list.append(
                ClarifyingQA(
                    question=q.question,
                    answer=q.default_answer,
                    answered_by="auto_inferred",
                    rationale=q.rationale,
                )
            )

    groundability = groundability_report or audit_groundability(
        brief, corpus, llm, embed_model_name=embed_model_name
    )

    eligible_tools, deterministic_rejections = evaluate_tool_eligibility(brief, corpus)
    eligible_names = {t.name for t in eligible_tools}
    eligible_text = (
        "\n".join(f"- {t.name}: {t.description}" for t in eligible_tools)
        if eligible_tools
        else "NONE"
    )

    qa_text = "\n".join(f"Q: {x.question}\nA ({x.answered_by}): {x.answer}" for x in qa_list)
    g_summary = (
        f"Overall: {groundability.overall_status} (ratio={groundability.groundability_ratio})\n"
        + "\n".join(
            f"- {it.requested_topic}: {it.status} ({it.rationale})"
            for it in groundability.items
        )
    )

    synth_prompt = render_prompt(
        "requirements_synthesis.j2",
        brief=brief,
        qa_text=qa_text,
        corpus_inventory=corpus.format_inventory(),
        groundability_summary=g_summary,
        eligible_tools_text=eligible_text,
    )

    try:
        synth, _ = llm.complete_structured(
            [
                {"role": "system", "content": "You output strict JSON only."},
                {"role": "user", "content": synth_prompt},
            ],
            _RequirementsSynthesisOutput,
            temperature=0.05,
            max_tokens=1100,
        )
    except Exception:
        synth = _RequirementsSynthesisOutput(
            domain_name="Document-Grounded Assistant",
            scope=f"Answer grounded user questions strictly from the provided documents for: {brief[:180]}",
            audience="Domain users referenced in the brief",
            tone="concise, accurate, professional",
            must_answer_topics=[
                it.requested_topic
                for it in groundability.items
                if it.status in {"supported", "partial"}
            ]
            or [f"Topics documented in {', '.join(corpus.filenames[:3])}"],
            must_refuse_topics=[
                "Topics not covered in the provided documents",
                "Legal, medical, or speculative advice beyond written rules",
            ],
            recommended_tools=[t.name for t in eligible_tools],
            tool_reasons={t.name: "Matched corpus/brief domain capability" for t in eligible_tools},
        )

    # Hard Groundability Enforcement: move all unsupported items out of must_answer -> must_refuse
    unsupported_lower = {u.lower() for u in groundability.unsupported_requests}
    clean_must_answer = [
        t
        for t in synth.must_answer_topics
        if t.strip() and t.strip().lower() not in unsupported_lower
    ]
    if not clean_must_answer:
        clean_must_answer = [
            it.requested_topic
            for it in groundability.items
            if it.status in {"supported", "partial"}
        ]

    clean_must_refuse = list(synth.must_refuse_topics)
    _, brief_refusal_clauses = split_brief_capabilities_and_refusals(brief)
    for clause in brief_refusal_clauses:
        if clause not in clean_must_refuse:
            clean_must_refuse.append(clause)
    for unsup in groundability.unsupported_requests:
        label = f"[Unsupported in docs] {unsup}"
        if label not in clean_must_refuse and unsup not in clean_must_refuse:
            clean_must_refuse.append(label)
    if "Questions outside the provided document corpus" not in clean_must_refuse:
        clean_must_refuse.append("Questions outside the provided document corpus")

    # Strict Tool Filtering: only keep tools in eligible_names
    final_tools: list[str] = []
    rejected_tools = dict(deterministic_rejections)
    for t_name in synth.recommended_tools:
        if t_name in eligible_names and t_name in TOOL_CATALOG:
            if t_name not in final_tools:
                final_tools.append(t_name)
        elif t_name in TOOL_CATALOG and t_name not in rejected_tools:
            rejected_tools[t_name] = "Rejected: incompatible with brief or corpus capabilities."

    # Record LLM-excluded eligible tools cleanly too
    for spec in eligible_tools:
        if spec.name not in final_tools and spec.name not in rejected_tools:
            rejected_tools[spec.name] = synth.tool_reasons.get(
                spec.name,
                "Excluded by requirements architect as unnecessary for the brief.",
            )

    return RequirementsDoc(
        brief=brief,
        domain_name=synth.domain_name.strip() or "ForgeAI Assistant",
        scope=synth.scope.strip(),
        audience=synth.audience.strip(),
        tone=synth.tone.strip(),
        must_answer_topics=list(dict.fromkeys(clean_must_answer)),
        must_refuse_topics=list(dict.fromkeys(clean_must_refuse)),
        recommended_tools=final_tools,
        rejected_tools=rejected_tools,
        clarifying_qa=qa_list,
        groundability=groundability,
        human_checkpoint_status="auto_approved",
        corpus_summary=[d.summary for d in corpus.documents],
    )
