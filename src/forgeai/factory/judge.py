"""Stage 5: Hybrid Debiased Judge & Prompt Tournament Comparator."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from forgeai.llm.base import LLMProvider
from forgeai.prompts import render_prompt
from forgeai.runtime.executor import AgentRuntime, ExecutionTrace
from forgeai.schemas.agent import AgentDefinition, PromptCandidate
from forgeai.schemas.evals import (
    CaseVerdict,
    EvalCase,
    FailureCategory,
    HybridJudgeMetrics,
    IterationScorecard,
)


class _AtomicFactVerdict(BaseModel):
    fact: str
    status: str = "missing"
    evidence: str = ""


class _LLMAtomicNLIOutput(BaseModel):
    fact_checks: list[_AtomicFactVerdict] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    summary: str = ""


class _LLMPairwiseOutput(BaseModel):
    winner: str = "TIE"
    reason: str = ""


_REFUSAL_RE = re.compile(
    r"\b(cannot answer|unable to answer|can't answer|cannot provide|cannot be (?:made|provided|determined|answered)|cannot approve|not covered|not mentioned|no information|no details|do(?:es)? not contain|not available in|outside (?:the |my )?scope|do not have information|cannot verify|not specified in the provided|strictly prohibited)\b",
    re.IGNORECASE,
)
_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
    "have", "has", "will", "must", "can", "per", "not", "all", "any", "into",
}


def _extract_numbers(text: str) -> list[str]:
    return re.findall(r"\b\d+(?:\.\d+)?%?\b", text.replace(",", ""))


def _det_fact_match(fact: str, answer: str) -> bool:
    ans_lower = answer.lower()
    ans_nums = set(_extract_numbers(answer))
    fact_nums = _extract_numbers(fact)
    for n in fact_nums:
        if n not in ans_nums and n.rstrip("%") not in ans_nums:
            return False

    words = [
        w.lower()
        for w in re.findall(r"[a-zA-Z0-9_-]{3,}", fact)
        if w.lower() not in _STOPWORDS
    ]
    if not words:
        return bool(fact_nums)
    hits = sum(1 for w in words if w in ans_lower)
    return (hits / len(words)) >= 0.45


def _compute_numeric_accuracy(key_facts: list[str], answer: str) -> float:
    expected_nums: list[str] = []
    for f in key_facts:
        expected_nums.extend(_extract_numbers(f))
    if not expected_nums:
        return 1.0
    ans_nums = set(_extract_numbers(answer))
    ans_raw = answer.replace(",", "")
    matched = sum(
        1 for n in expected_nums if n in ans_nums or n.rstrip("%") in ans_raw
    )
    return round(matched / len(expected_nums), 3)


def _check_citations(
    citations: list[dict[str, str]],
    valid_sources: set[str],
    retrieved_sources: list[str],
) -> bool:
    if not citations:
        return False
    valid_lower = {s.lower() for s in valid_sources}
    ret_lower = {s.lower() for s in retrieved_sources}
    for c in citations:
        src = str(c.get("source") or "").strip().lower()
        if not src:
            continue
        if src in valid_lower or any(src in v or v in src for v in valid_lower):
            if not ret_lower or src in ret_lower or any(src in r or r in src for r in ret_lower):
                return True
    return False


def _evidence_contains_claim(claim: str, evidence_text: str) -> bool:
    ev_lower = evidence_text.lower()
    claim_nums = _extract_numbers(claim)
    if claim_nums and any(n not in ev_lower for n in claim_nums):
        return False
    words = [
        w.lower()
        for w in re.findall(r"[a-zA-Z0-9_-]{4,}", claim)
        if w.lower() not in _STOPWORDS
    ]
    if not words:
        return True
    hits = sum(1 for w in words if w in ev_lower)
    return (hits / len(words)) >= 0.45


def evaluate_single_case(
    case: EvalCase,
    trace: ExecutionTrace,
    judge_llm: LLMProvider,
    valid_sources: set[str],
    *,
    max_answer_words: int = 220,
    min_score_threshold: float = 0.08,
    use_llm_judge: bool = True,
) -> CaseVerdict:
    """Score one agent execution using deterministic checks + debiased atomic NLI."""
    resp = trace.response
    actual_refused = bool(resp.refused or _REFUSAL_RE.search(resp.answer or ""))
    expected_refuse = case.expected_behavior == "refuse"
    refusal_correct = actual_refused == expected_refuse

    retrieved_sources = list(dict.fromkeys(h.chunk.source for h in trace.retrieved_chunks))
    top_score = trace.retrieved_chunks[0].score if trace.retrieved_chunks else 0.0

    evidence_parts = [h.chunk.format_for_prompt() for h in trace.retrieved_chunks]
    for t_out in trace.tool_outputs:
        evidence_parts.append(f"[Tool: {t_out['tool']}] {t_out['output']}")
    evidence_text = "\n\n".join(evidence_parts) or "NO EVIDENCE RETRIEVED."

    contract_valid = bool(trace.raw_contract_valid and (resp.answer or "").strip())
    cit_valid = _check_citations(resp.citations, valid_sources, retrieved_sources)
    if expected_refuse and actual_refused:
        cit_valid = True  # Refusals do not require document citations

    # Tool appropriateness
    tool_app = 1.0
    if case.expected_tools and not expected_refuse:
        if any(t in resp.tools_used for t in case.expected_tools):
            tool_app = 1.0
        else:
            tool_app = 0.5

    # Verbosity bias defense
    words_count = len((resp.answer or "").split())
    verbosity_penalty = 0.0
    if words_count > max_answer_words:
        verbosity_penalty = min(0.25, ((words_count - max_answer_words) / max_answer_words) * 0.25)

    fact_checks: list[dict[str, Any]] = []
    unsupported_claims: list[str] = []

    if expected_refuse:
        fact_recall = 1.0 if actual_refused else 0.0
        numeric_acc = 1.0 if actual_refused else 0.0
        groundedness = 1.0 if actual_refused else 0.0
    elif actual_refused:
        # False refusal on an answerable case: never reward groundedness for dodging
        fact_recall = 0.0
        numeric_acc = 0.0
        groundedness = 0.0
    else:
        numeric_acc = _compute_numeric_accuracy(case.key_facts, resp.answer)
        det_hits = [
            1.0 if _det_fact_match(f, resp.answer) else 0.0 for f in case.key_facts
        ]
        det_recall = sum(det_hits) / max(1, len(det_hits))

        llm_recall = det_recall
        if use_llm_judge and case.key_facts:
            key_facts_text = "\n".join(f"- {f}" for f in case.key_facts)
            prompt = render_prompt(
                "judge_atomic_nli.j2",
                question=case.question,
                expected_behavior=case.expected_behavior,
                key_facts_text=key_facts_text,
                evidence_text=evidence_text[:3200],
                actual_answer=resp.answer,
                actual_refused=actual_refused,
            )
            try:
                parsed, _ = judge_llm.complete_structured(
                    [
                        {"role": "system", "content": "You output strict JSON only."},
                        {"role": "user", "content": prompt},
                    ],
                    _LLMAtomicNLIOutput,
                    temperature=0.0,
                    max_tokens=900,
                )
                if parsed.fact_checks:
                    sup_count = sum(
                        1
                        for fc in parsed.fact_checks
                        if fc.status.lower().strip() == "supported"
                    )
                    # Small judges occasionally split 1 key_fact into multiple checks; bound to [0, 1].
                    denom = max(1, len(case.key_facts), len(parsed.fact_checks))
                    llm_recall = min(1.0, max(0.0, sup_count / denom))
                    fact_checks = [fc.model_dump() for fc in parsed.fact_checks]
                # Filter judge's unsupported claims against actual evidence text so paraphrases aren't penalized
                unsupported_claims = [
                    cl
                    for cl in parsed.unsupported_claims
                    if cl.strip() and not _evidence_contains_claim(cl, evidence_text)
                ]
            except Exception:
                pass

        if not fact_checks:
            fact_checks = [
                {
                    "fact": f,
                    "status": "supported" if _det_fact_match(f, resp.answer) else "missing",
                    "evidence": "deterministic_check",
                }
                for f in case.key_facts
            ]

        fact_recall = round(min(1.0, max(0.0, 0.5 * det_recall + 0.5 * llm_recall)), 3)
        numeric_acc = min(1.0, max(0.0, numeric_acc))

        # Groundedness check: unverified numbers/SKUs are high-severity hallucination signals
        ans_nums = set(_extract_numbers(resp.answer))
        ev_nums = set(_extract_numbers(evidence_text))
        hallucinated_nums = [n for n in ans_nums if n not in ev_nums]
        penalty = 0.25 * len(unsupported_claims) + 0.45 * len(hallucinated_nums)
        groundedness = round(max(0.0, 1.0 - penalty), 3)

    if expected_refuse:
        composite = round(
            max(0.0, (0.75 if refusal_correct else 0.0) + (0.25 if contract_valid else 0.0) - verbosity_penalty),
            4,
        )
        passed = refusal_correct and contract_valid
    else:
        composite = round(
            max(
                0.0,
                min(
                    1.0,
                    0.35 * fact_recall
                    + 0.25 * groundedness
                    + 0.15 * numeric_acc
                    + 0.15 * (1.0 if cit_valid else 0.0)
                    + 0.10 * (1.0 if contract_valid else 0.0)
                    - verbosity_penalty,
                ),
            ),
            4,
        )
        num_gate = 0.30 if fact_recall >= 0.65 else 0.50
        passed = (
            refusal_correct
            and contract_valid
            and fact_recall >= 0.50
            and groundedness >= 0.60
            and numeric_acc >= num_gate
        )

    # Failure taxonomy classification (exact 4 categories!)
    failure_category: FailureCategory = "none"
    failure_reason = ""
    if not passed:
        if not refusal_correct:
            failure_category = "wrong_refusal"
            failure_reason = (
                "False refusal on answerable question."
                if actual_refused
                else "Failed to refuse out-of-scope question."
            )
        elif groundedness < 0.60:
            failure_category = "hallucination"
            failure_reason = (
                f"Ungrounded claims/numbers detected: {unsupported_claims[:2] or 'unverified numeric claim'}"
            )
        else:
            # Distinguish retrieval_miss vs prompt_defect
            expected_docs = set(case.source_documents)
            corpus_hit_expected = bool(expected_docs & set(retrieved_sources)) if expected_docs else bool(retrieved_sources)
            facts_in_retrieved = any(
                _det_fact_match(f, evidence_text) for f in case.key_facts
            ) if case.key_facts else True

            if (not retrieved_sources) or (top_score < min_score_threshold) or (not corpus_hit_expected and not facts_in_retrieved) or (not facts_in_retrieved):
                failure_category = "retrieval_miss"
                failure_reason = (
                    f"Required facts/sources ({case.source_documents}) missing from top-{len(trace.retrieved_chunks)} retrieved chunks (top_score={top_score:.2f})."
                )
            else:
                failure_category = "prompt_defect"
                reasons = []
                if not contract_valid:
                    reasons.append("invalid JSON output contract")
                if not cit_valid:
                    reasons.append("missing/invalid citation")
                if fact_recall < 0.50:
                    reasons.append(f"missed expected facts despite retrieval (recall={fact_recall:.2f})")
                if numeric_acc < 0.50:
                    reasons.append(f"numeric inaccuracy ({numeric_acc:.2f})")
                failure_reason = "; ".join(reasons) or "Prompt failed to utilize retrieved evidence."

    metrics = HybridJudgeMetrics(
        contract_valid=contract_valid,
        refusal_correct=refusal_correct,
        citation_valid=cit_valid,
        numeric_accuracy=numeric_acc,
        fact_recall=fact_recall,
        groundedness=groundedness,
        verbosity_penalty=round(verbosity_penalty, 3),
        tool_appropriateness=tool_app,
        composite_score=composite,
    )

    return CaseVerdict(
        case_id=case.case_id,
        category=case.category,
        sub_type=case.sub_type,
        question=case.question,
        expected_behavior=case.expected_behavior,
        actual_answer=resp.answer,
        actual_refused=actual_refused,
        actual_citations=resp.citations,
        actual_tools_used=resp.tools_used,
        retrieved_sources=retrieved_sources,
        retrieved_top_score=top_score,
        passed=passed,
        failure_category=failure_category,
        failure_reason=failure_reason,
        metrics=metrics,
        fact_checks=fact_checks,
        latency_ms=trace.latency_ms,
    )


def evaluate_agent_on_cases(
    runtime: AgentRuntime,
    cases: list[EvalCase],
    judge_llm: LLMProvider,
    *,
    iteration: int,
    label: str,
    use_llm_judge: bool = True,
) -> IterationScorecard:
    """Evaluate AgentRuntime over a list of EvalCases in two model-batched passes (prevents VRAM thrashing)."""
    valid_sources = set(runtime.corpus.filenames)
    # Phase A: run all agent generations together while generator model stays resident in VRAM
    traces: list[ExecutionTrace] = [runtime.ask(case.question) for case in cases]
    # Phase B: judge all collected traces together while judge model stays resident in VRAM
    verdicts: list[CaseVerdict] = [
        evaluate_single_case(
            case,
            trace,
            judge_llm,
            valid_sources,
            max_answer_words=runtime.agent.output_contract.max_answer_words,
            min_score_threshold=runtime.agent.retrieval.min_score_threshold,
            use_llm_judge=use_llm_judge,
        )
        for case, trace in zip(cases, traces)
    ]

    n = max(1, len(verdicts))
    passed_count = sum(1 for v in verdicts if v.passed)
    by_cat: dict[str, dict[str, float]] = {}
    for cat in ("answerable", "out_of_scope", "edge_case"):
        sub = [v for v in verdicts if v.category == cat]
        if sub:
            by_cat[cat] = {
                "total": float(len(sub)),
                "passed": float(sum(1 for v in sub if v.passed)),
                "pass_rate": round(sum(1 for v in sub if v.passed) / len(sub), 4),
                "mean_score": round(sum(v.metrics.composite_score for v in sub) / len(sub), 4),
            }

    failure_counts = {
        "retrieval_miss": sum(1 for v in verdicts if v.failure_category == "retrieval_miss"),
        "prompt_defect": sum(1 for v in verdicts if v.failure_category == "prompt_defect"),
        "hallucination": sum(1 for v in verdicts if v.failure_category == "hallucination"),
        "wrong_refusal": sum(1 for v in verdicts if v.failure_category == "wrong_refusal"),
    }

    return IterationScorecard(
        iteration=iteration,
        label=label,
        total_cases=len(verdicts),
        passed_cases=passed_count,
        pass_rate=round(passed_count / n, 4),
        mean_composite_score=round(sum(v.metrics.composite_score for v in verdicts) / n, 4),
        mean_groundedness=round(sum(v.metrics.groundedness for v in verdicts) / n, 4),
        mean_fact_recall=round(sum(v.metrics.fact_recall for v in verdicts) / n, 4),
        refusal_accuracy=round(sum(1 for v in verdicts if v.metrics.refusal_correct) / n, 4),
        contract_adherence=round(sum(1 for v in verdicts if v.metrics.contract_valid) / n, 4),
        by_category=by_cat,
        failure_counts=failure_counts,
        verdicts=verdicts,
    )


def _pairwise_debiased_compare(
    case: EvalCase,
    trace_a: ExecutionTrace,
    trace_b: ExecutionTrace,
    judge_llm: LLMProvider,
) -> str:
    """Run position-swapped pairwise comparison (A,B) & (B,A) returning 'A', 'B', or 'TIE'."""
    ev_text = "\n".join(h.chunk.format_for_prompt()[:400] for h in trace_a.retrieved_chunks[:3])
    kf_text = "\n".join(f"- {k}" for k in case.key_facts) or "Refuse if out of scope."

    def _call_order(first_ans: str, second_ans: str) -> str:
        prompt = render_prompt(
            "judge_pairwise.j2",
            question=case.question,
            expected_behavior=case.expected_behavior,
            key_facts_text=kf_text,
            evidence_text=ev_text[:1600],
            response_first=first_ans[:700],
            response_second=second_ans[:700],
        )
        try:
            res, _ = judge_llm.complete_structured(
                [
                    {"role": "system", "content": "You output strict JSON only."},
                    {"role": "user", "content": prompt},
                ],
                _LLMPairwiseOutput,
                temperature=0.0,
                max_tokens=300,
            )
            w = res.winner.upper().strip()
            return w if w in {"FIRST", "SECOND", "TIE"} else "TIE"
        except Exception:
            return "TIE"

    # Forward (A=FIRST, B=SECOND) and Swapped (B=FIRST, A=SECOND)
    fwd = _call_order(trace_a.response.answer, trace_b.response.answer)
    rev = _call_order(trace_b.response.answer, trace_a.response.answer)

    a_fwd = 1 if fwd == "FIRST" else (-1 if fwd == "SECOND" else 0)
    a_rev = 1 if rev == "SECOND" else (-1 if rev == "FIRST" else 0)
    net = a_fwd + a_rev
    if net > 0:
        return "A"
    if net < 0:
        return "B"
    # Position bias detected when both forward & reverse prefer FIRST or both prefer SECOND -> TIE
    return "TIE"


def run_prompt_tournament(
    base_agent: AgentDefinition,
    candidates: list[PromptCandidate],
    eval_cases: list[EvalCase],
    runtime: AgentRuntime,
    judge_llm: LLMProvider,
) -> tuple[AgentDefinition, list[PromptCandidate]]:
    """Bonus 1: Compete 2-3 prompt candidates with position-debiased scoring."""
    if len(candidates) <= 1 or not eval_cases:
        return base_agent, candidates

    # Pick stratified 4-case subset (1 answerable, 1 out_of_scope, 2 edge_case)
    # Preserves full category coverage + swapped (A,B)/(B,A) bias defense with 33% fewer calls.
    subset: list[EvalCase] = []
    subset.extend([c for c in eval_cases if c.category == "answerable"][:1])
    subset.extend([c for c in eval_cases if c.category == "out_of_scope"][:1])
    subset.extend([c for c in eval_cases if c.category == "edge_case"][:2])
    if len(subset) < min(4, len(eval_cases)):
        seen_ids = {c.case_id for c in subset}
        for c in eval_cases:
            if c.case_id not in seen_ids:
                subset.append(c)
            if len(subset) >= 4:
                break

    traces_by_cand: dict[str, list[ExecutionTrace]] = {}
    valid_sources = set(runtime.corpus.filenames)

    for cand in candidates:
        trial_agent = base_agent.model_copy(update={"system_prompt": cand.system_prompt})
        runtime.update_agent(trial_agent)
        cand_traces: list[ExecutionTrace] = []
        scores: list[float] = []
        for c in subset:
            tr = runtime.ask(c.question)
            cand_traces.append(tr)
            ver = evaluate_single_case(
                c,
                tr,
                judge_llm,
                valid_sources,
                max_answer_words=trial_agent.output_contract.max_answer_words,
                min_score_threshold=trial_agent.retrieval.min_score_threshold,
                use_llm_judge=False,  # Fast deterministic scoring; pairwise debiased judge follows
            )
            scores.append(ver.metrics.composite_score)
        traces_by_cand[cand.candidate_id] = cand_traces
        cand.deterministic_score = round(sum(scores) / max(1, len(scores)), 4)

    # Round-robin position-swapped pairwise matches across subset
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            ca, cb = candidates[i], candidates[j]
            for k, case in enumerate(subset):
                winner = _pairwise_debiased_compare(
                    case,
                    traces_by_cand[ca.candidate_id][k],
                    traces_by_cand[cb.candidate_id][k],
                    judge_llm,
                )
                if winner == "A":
                    ca.pairwise_wins += 1
                    cb.pairwise_losses += 1
                elif winner == "B":
                    cb.pairwise_wins += 1
                    ca.pairwise_losses += 1
                else:
                    ca.pairwise_ties += 1
                    cb.pairwise_ties += 1

    for cand in candidates:
        total_matches = max(1, cand.pairwise_wins + cand.pairwise_losses + cand.pairwise_ties)
        win_rate = (cand.pairwise_wins + 0.5 * cand.pairwise_ties) / total_matches
        cand.tournament_score = round(0.65 * cand.deterministic_score + 0.35 * win_rate, 4)

    ranked = sorted(candidates, key=lambda c: c.tournament_score, reverse=True)
    winner_cand = ranked[0]
    winning_agent = base_agent.model_copy(update={"system_prompt": winner_cand.system_prompt})
    runtime.update_agent(winning_agent)
    return winning_agent, ranked
