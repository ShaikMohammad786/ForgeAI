"""Stage 3: Declarative Agent Generator & Bonus 1 Prompt Tournament."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from forgeai.ingestion.loader import LoadedCorpus
from forgeai.llm.base import LLMProvider
from forgeai.prompts import render_prompt
from forgeai.schemas.agent import (
    AgentDefinition,
    CustomToolSpec,
    OutputContract,
    PromptCandidate,
    RetrievalConfig,
    ToolBinding,
)
from forgeai.schemas.requirements import RequirementsDoc
from forgeai.tools.custom_macro import validate_custom_tool_spec
from forgeai.tools.registry import TOOL_CATALOG, evaluate_tool_eligibility, format_tools_for_prompt


class _GeneratedAgentPayload(BaseModel):
    agent_name: str
    system_prompt: str
    chunk_size: int = Field(default=600, ge=250, le=1600)
    chunk_overlap: int = Field(default=100, ge=0, le=400)
    top_k: int = Field(default=5, ge=2, le=12)
    hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    min_score_threshold: float = Field(default=0.10, ge=0.0, le=0.7)


STRATEGY_DIRECTIVES: list[tuple[str, str]] = [
    (
        "strict_grounded_citation",
        "Prioritize literal factual fidelity: quote exact numbers, SKUs, and clauses from retrieved chunks; refuse immediately if evidence is absent.",
    ),
    (
        "structured_policy_analyst",
        "Use structured internal verification: check scope, inspect authorized tools first for data/math, cross-reference multiple retrieved chunks, then produce a crisp cited answer.",
    ),
    (
        "concise_tool_specialist",
        "Optimize for small-model precision: short direct sentences, strict JSON discipline, explicit distinction between retrieved facts and out-of-scope refusals.",
    ),
]


def _build_tool_bindings(
    requirements: RequirementsDoc,
    corpus: LoadedCorpus,
) -> list[ToolBinding]:
    """Attach ONLY tools that pass deterministic capability checks and brief intent."""
    eligible_specs, _ = evaluate_tool_eligibility(requirements.brief, corpus)
    eligible_map = {s.name: s for s in eligible_specs}

    bindings: list[ToolBinding] = []
    for name in requirements.recommended_tools:
        spec = eligible_map.get(name) or TOOL_CATALOG.get(name)
        if spec is None or name not in eligible_map:
            continue
        bindings.append(
            ToolBinding(
                name=spec.name,
                description=spec.description,
                justification=f"Selected during intake for {requirements.domain_name}",
                required_capabilities=[spec.name],
            )
        )
    return bindings


class _CustomMacrosPayload(BaseModel):
    macros: list[CustomToolSpec] = Field(default_factory=list)


def _synthesize_custom_tools(
    requirements: RequirementsDoc,
    corpus: LoadedCorpus,
    llm: LLMProvider,
) -> list[CustomToolSpec]:
    """Synthesize 0-2 AST-validated declarative domain macros when calculation/tabular rules warrant."""
    eligible_specs, _ = evaluate_tool_eligibility(requirements.brief, corpus)
    eligible_names = {s.name for s in eligible_specs}
    if not (corpus.has_csv or "calculator" in eligible_names or "statistics_calculator" in eligible_names):
        return []

    previews = [
        {
            "filename": d.summary.filename,
            "doc_type": d.summary.doc_type,
            "preview": d.summary.preview[:420],
        }
        for d in corpus.documents[:8]
    ]
    prompt = render_prompt(
        "custom_tool_synthesis.j2",
        requirements=requirements,
        corpus_previews=previews,
        has_tabular=corpus.has_csv,
    )
    try:
        payload, _ = llm.complete_structured(
            [
                {"role": "system", "content": "You output strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            _CustomMacrosPayload,
            temperature=0.1,
            max_tokens=900,
        )
    except Exception:
        return []

    reserved = set(TOOL_CATALOG.keys())
    valid: list[CustomToolSpec] = []
    seen_names: set[str] = set()
    for item in payload.macros:
        norm_name = item.name.strip().lower()
        if norm_name in seen_names:
            continue
        candidate = item.model_copy(update={"name": norm_name})
        ok, _ = validate_custom_tool_spec(
            candidate,
            docs_dir=corpus.docs_dir,
            reserved_names=reserved,
            has_tabular=corpus.has_csv,
        )
        if ok:
            valid.append(candidate)
            seen_names.add(norm_name)
        if len(valid) >= 2:
            break
    return valid


def _fallback_system_prompt(
    requirements: RequirementsDoc,
    tools: list[ToolBinding],
    refusal_prefix: str,
    strategy_name: str,
    custom_tools: list[CustomToolSpec] | None = None,
) -> str:
    all_names = [t.name for t in tools] + [ct.name for ct in (custom_tools or [])]
    tool_names = ", ".join(all_names) if all_names else "none"
    must_ans = "; ".join(requirements.must_answer_topics[:6])
    must_ref = "; ".join(requirements.must_refuse_topics[:6])
    return (
        f"You are {requirements.domain_name} ({strategy_name}).\n"
        f"Audience: {requirements.audience}. Tone: {requirements.tone}.\n"
        f"Scope: {requirements.scope}\n\n"
        f"AUTHORIZED TOOLS: {tool_names}.\n"
        f"MUST-ANSWER TOPICS: {must_ans}.\n"
        f"MUST-REFUSE TOPICS: {must_ref}.\n\n"
        "STRICT RULES:\n"
        "1. Answer ONLY using facts explicitly present in the RETRIEVED CONTEXT or TOOL OUTPUTS.\n"
        "2. Never fabricate policy clauses, numbers, prices, SKUs, or dates.\n"
        f"3. If the question asks about any MUST-REFUSE topic or lacks supporting evidence in the context, set refused=true and begin answer with: \"{refusal_prefix}\".\n"
        "4. Include accurate source citations with exact filenames from the retrieved chunks.\n"
        "5. Output ONLY valid JSON with keys: answer, citations, refused, refusal_reason, confidence, tools_used."
    )


def generate_agent_candidates(
    requirements: RequirementsDoc,
    corpus: LoadedCorpus,
    llm: LLMProvider,
    *,
    judge_model_name: str,
    num_candidates: int = 3,
) -> tuple[AgentDefinition, list[PromptCandidate]]:
    """Generate baseline AgentDefinition plus 2-3 candidate prompts for the Prompt Tournament."""
    output_contract = OutputContract()
    tools = _build_tool_bindings(requirements, corpus)
    custom_tools = _synthesize_custom_tools(requirements, corpus, llm)
    tools_desc = format_tools_for_prompt(tools, custom_tools=custom_tools)

    candidates: list[PromptCandidate] = []
    chosen_retrieval: RetrievalConfig | None = None
    chosen_name = requirements.domain_name

    for idx, (strat_name, directive) in enumerate(
        STRATEGY_DIRECTIVES[: max(1, min(3, num_candidates))], start=1
    ):
        prompt = render_prompt(
            "agent_generator.j2",
            requirements=requirements,
            tools_description=tools_desc,
            strategy_name=strat_name,
            strategy_directive=directive,
            refusal_prefix=output_contract.refusal_phrase_prefix,
        )
        try:
            payload, _ = llm.complete_structured(
                [
                    {"role": "system", "content": "You output strict JSON only."},
                    {"role": "user", "content": prompt},
                ],
                _GeneratedAgentPayload,
                temperature=0.15 if idx > 1 else 0.05,
                max_tokens=1200,
            )
            sys_prompt = payload.system_prompt.strip()
            if len(sys_prompt) < 80:
                sys_prompt = _fallback_system_prompt(
                    requirements, tools, output_contract.refusal_phrase_prefix, strat_name, custom_tools
                )
            if chosen_retrieval is None:
                chosen_name = payload.agent_name.strip() or requirements.domain_name
                chosen_retrieval = RetrievalConfig(
                    chunk_size=max(350, min(1000, payload.chunk_size)),
                    chunk_overlap=max(50, min(200, payload.chunk_overlap)),
                    top_k=max(4, min(8, payload.top_k)),
                    hybrid_alpha=round(max(0.25, min(0.75, payload.hybrid_alpha)), 2),
                    min_score_threshold=round(max(0.06, min(0.25, payload.min_score_threshold)), 3),
                    chunking_strategy="heading_recursive",
                )
        except Exception:
            sys_prompt = _fallback_system_prompt(
                requirements, tools, output_contract.refusal_phrase_prefix, strat_name, custom_tools
            )

        candidates.append(
            PromptCandidate(
                candidate_id=f"cand_{idx}_{strat_name}",
                strategy_name=strat_name,
                system_prompt=sys_prompt,
            )
        )

    if chosen_retrieval is None:
        chosen_retrieval = RetrievalConfig(
            chunk_size=550,
            chunk_overlap=100,
            top_k=5,
            hybrid_alpha=0.45 if corpus.has_csv else 0.55,
            min_score_threshold=0.10,
            chunking_strategy="heading_recursive",
        )

    agent = AgentDefinition(
        agent_id=f"agent-{uuid.uuid4().hex[:8]}",
        name=chosen_name,
        version=1,
        brief=requirements.brief,
        provider=llm.provider_name,
        model_name=llm.model_name,
        judge_model_name=judge_model_name,
        temperature=0.08,
        max_tool_steps=3 if (tools or custom_tools) else 1,
        system_prompt=candidates[0].system_prompt,
        retrieval=chosen_retrieval,
        tools=tools,
        custom_tools=custom_tools,
        output_contract=output_contract,
        must_answer_topics=requirements.must_answer_topics,
        must_refuse_topics=requirements.must_refuse_topics,
        groundability_caveats=requirements.groundability.warnings,
        docs_dir=str(corpus.docs_dir),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    return agent, candidates
