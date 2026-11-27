"""Stage 6: Targeted Critic-Optimizer Loop (<=3 iterations, rollback guard, budget early stop)."""

from __future__ import annotations

import time
from collections import Counter

from pydantic import BaseModel, Field

from forgeai.factory.judge import evaluate_agent_on_cases
from forgeai.llm.base import BudgetExceededError, BudgetTracker, LLMProvider
from forgeai.prompts import render_prompt
from forgeai.runtime.executor import AgentRuntime
from forgeai.schemas.agent import AgentDefinition, RetrievalConfig
from forgeai.schemas.evals import EvalCase, IterationScorecard
from forgeai.schemas.optimizer import IterationRecord, PatchChange


class _RetrievalUpdates(BaseModel):
    top_k: int = 5
    hybrid_alpha: float = 0.5
    min_score_threshold: float = 0.08
    chunk_size: int = 600
    chunk_overlap: int = 100
    use_query_expansion: bool = False


class _OptimizerPatchPayload(BaseModel):
    diagnosis_summary: str = ""
    target_case_ids: list[str] = Field(default_factory=list)
    prompt_addition: str = ""
    retrieval_updates: _RetrievalUpdates = Field(default_factory=_RetrievalUpdates)
    temperature: float = 0.08


_PRIORITY_ORDER = ["hallucination", "wrong_refusal", "retrieval_miss", "prompt_defect"]


def _choose_dominant_failure(scorecard: IterationScorecard) -> tuple[str, list[str]]:
    failed = [v for v in scorecard.verdicts if not v.passed and v.failure_category != "none"]
    if not failed:
        return "none", []
    counts = Counter(v.failure_category for v in failed)
    best_cat = max(
        counts.keys(),
        key=lambda c: (counts[c], -_PRIORITY_ORDER.index(c) if c in _PRIORITY_ORDER else -9),
    )
    ids = [v.case_id for v in failed if v.failure_category == best_cat]
    return best_cat, ids


def _apply_targeted_patch(
    agent: AgentDefinition,
    dominant: str,
    scorecard: IterationScorecard,
    llm: LLMProvider,
) -> tuple[AgentDefinition, list[PatchChange], str, list[str]]:
    """Generate and apply a minimal targeted patch (never a full regeneration)."""
    failed_cases = [v for v in scorecard.verdicts if v.failure_category == dominant][:5]
    failed_text = "\n".join(
        f"- [{v.case_id}] ({v.category}/{v.sub_type}) Q: {v.question}\n"
        f"  Reason: {v.failure_reason} | TopScore={v.retrieved_top_score:.2f} | Sources={v.retrieved_sources}"
        for v in failed_cases
    )

    prompt = render_prompt(
        "optimizer_patch.j2",
        current_prompt=agent.system_prompt,
        retrieval=agent.retrieval,
        dominant_category=dominant,
        failure_counts=scorecard.failure_counts,
        failed_cases_text=failed_text,
        temperature=agent.temperature,
    )

    patch_payload: _OptimizerPatchPayload | None = None
    try:
        patch_payload, _ = llm.complete_structured(
            [
                {"role": "system", "content": "You output strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            _OptimizerPatchPayload,
            temperature=0.05,
            max_tokens=900,
        )
    except Exception:
        patch_payload = None

    changes: list[PatchChange] = []
    new_ret = agent.retrieval.model_copy()
    new_prompt = agent.system_prompt
    new_temp = agent.temperature

    # Deterministic domain-safe bounds paired with LLM diagnosis
    if dominant == "retrieval_miss":
        desired_k = patch_payload.retrieval_updates.top_k if patch_payload else new_ret.top_k + 2
        new_k = min(10, max(new_ret.top_k + 1, desired_k))
        if new_k != new_ret.top_k:
            changes.append(PatchChange(field_path="retrieval.top_k", old_value=new_ret.top_k, new_value=new_k, rationale="Increase candidate context breadth for missed chunks"))
            new_ret.top_k = new_k

        new_thresh = round(max(0.04, new_ret.min_score_threshold - 0.03), 3)
        if new_thresh != new_ret.min_score_threshold:
            changes.append(PatchChange(field_path="retrieval.min_score_threshold", old_value=new_ret.min_score_threshold, new_value=new_thresh, rationale="Relax threshold slightly to avoid dropping borderline relevant chunks"))
            new_ret.min_score_threshold = new_thresh

        if not new_ret.use_query_expansion:
            changes.append(PatchChange(field_path="retrieval.use_query_expansion", old_value=False, new_value=True, rationale="Enable query normalization for better lexical recall"))
            new_ret.use_query_expansion = True

        # Balance BM25 vs dense depending on numeric/SKU failures
        target_alpha = round(max(0.30, min(0.70, patch_payload.retrieval_updates.hybrid_alpha if patch_payload else 0.45)), 2)
        if abs(target_alpha - new_ret.hybrid_alpha) >= 0.05:
            changes.append(PatchChange(field_path="retrieval.hybrid_alpha", old_value=new_ret.hybrid_alpha, new_value=target_alpha, rationale="Rebalance lexical BM25 vs dense retrieval weighting"))
            new_ret.hybrid_alpha = target_alpha

    elif dominant == "hallucination":
        new_temp = 0.0
        if new_temp != agent.temperature:
            changes.append(PatchChange(field_path="temperature", old_value=agent.temperature, new_value=new_temp, rationale="Set temperature=0.0 to suppress speculative generation"))
        new_thresh = round(min(0.22, new_ret.min_score_threshold + 0.03), 3)
        if new_thresh != new_ret.min_score_threshold:
            changes.append(PatchChange(field_path="retrieval.min_score_threshold", old_value=new_ret.min_score_threshold, new_value=new_thresh, rationale="Raise retrieval score gate to block weak noisy snippets"))
            new_ret.min_score_threshold = new_thresh

        addition = (
            patch_payload.prompt_addition.strip()
            if (patch_payload and patch_payload.prompt_addition.strip())
            else "GROUNDING GUARD: State ONLY facts and exact numbers explicitly verified in RETRIEVED CONTEXT or TOOL OBSERVATIONS. Omit any detail not directly observed."
        )
        if addition not in new_prompt:
            changes.append(PatchChange(field_path="system_prompt", old_value=f"(len={len(new_prompt)})", new_value=f"+ {addition[:120]}", rationale="Add surgical anti-hallucination constraint"))
            new_prompt = f"{new_prompt.rstrip()}\n\n{addition}"

    elif dominant == "wrong_refusal":
        false_refusals = sum(1 for v in failed_cases if v.actual_refused)
        missed_refusals = len(failed_cases) - false_refusals
        if false_refusals >= missed_refusals:
            new_thresh = round(max(0.04, new_ret.min_score_threshold - 0.03), 3)
            if new_thresh != new_ret.min_score_threshold:
                changes.append(PatchChange(field_path="retrieval.min_score_threshold", old_value=new_ret.min_score_threshold, new_value=new_thresh, rationale="Lower score gate to reduce false refusals on valid queries"))
                new_ret.min_score_threshold = new_thresh
            addition = (
                patch_payload.prompt_addition.strip()
                if (patch_payload and patch_payload.prompt_addition.strip())
                else "ANSWERABILITY RULE: If RETRIEVED CONTEXT or TOOL OBSERVATIONS contain relevant facts answering the core question, answer concisely with citations instead of refusing."
            )
        else:
            new_thresh = round(min(0.24, new_ret.min_score_threshold + 0.04), 3)
            if new_thresh != new_ret.min_score_threshold:
                changes.append(PatchChange(field_path="retrieval.min_score_threshold", old_value=new_ret.min_score_threshold, new_value=new_thresh, rationale="Raise score gate to enforce refusal on out-of-scope queries"))
                new_ret.min_score_threshold = new_thresh
            addition = (
                patch_payload.prompt_addition.strip()
                if (patch_payload and patch_payload.prompt_addition.strip())
                else f"STRICT REFUSAL RULE: When asked about unsupported or missing policies, ALWAYS set refused=true and start answer with \"{agent.output_contract.refusal_phrase_prefix}\"."
            )
        if addition not in new_prompt:
            changes.append(PatchChange(field_path="system_prompt", old_value=f"(len={len(new_prompt)})", new_value=f"+ {addition[:120]}", rationale="Clarify answer-vs-refuse boundary"))
            new_prompt = f"{new_prompt.rstrip()}\n\n{addition}"

    else:  # prompt_defect
        addition = (
            patch_payload.prompt_addition.strip()
            if (patch_payload and patch_payload.prompt_addition.strip())
            else "FORMAT & COMPLETENESS RULE: Include all key facts, exact numbers, and at least one valid filename in citations; return strict JSON matching the output contract."
        )
        if addition not in new_prompt:
            changes.append(PatchChange(field_path="system_prompt", old_value=f"(len={len(new_prompt)})", new_value=f"+ {addition[:120]}", rationale="Repair prompt compliance and fact/citation completeness"))
            new_prompt = f"{new_prompt.rstrip()}\n\n{addition}"
        if new_ret.top_k < 6:
            changes.append(PatchChange(field_path="retrieval.top_k", old_value=new_ret.top_k, new_value=new_ret.top_k + 1, rationale="Provide +1 chunk for multi-document synthesis"))
            new_ret.top_k += 1

    diagnosis = (
        patch_payload.diagnosis_summary.strip()
        if (patch_payload and patch_payload.diagnosis_summary.strip())
        else f"Targeted fix for dominant failure cluster '{dominant}' ({len(failed_cases)} sample cases)."
    )
    target_ids = [v.case_id for v in failed_cases]

    patched_agent = agent.model_copy(
        update={
            "version": agent.version + 1,
            "system_prompt": new_prompt,
            "retrieval": new_ret,
            "temperature": new_temp,
        }
    )
    return patched_agent, changes, diagnosis, target_ids


def run_optimization_loop(
    initial_agent: AgentDefinition,
    eval_cases: list[EvalCase],
    runtime: AgentRuntime,
    factory_llm: LLMProvider,
    judge_llm: LLMProvider,
    budget: BudgetTracker,
    *,
    max_iterations: int = 3,
    min_gain_threshold: float = 0.02,
    use_llm_judge: bool = True,
) -> tuple[AgentDefinition, list[IterationScorecard], list[IterationRecord]]:
    """Run Baseline + up to 3 targeted Critic-Optimizer iterations with rollback & budget guard."""
    max_iterations = max(0, min(3, max_iterations))
    runtime.update_agent(initial_agent)

    baseline_card = evaluate_agent_on_cases(
        runtime,
        eval_cases,
        judge_llm,
        iteration=0,
        label="Baseline (v1)",
        use_llm_judge=use_llm_judge,
    )
    scorecards: list[IterationScorecard] = [baseline_card]
    records: list[IterationRecord] = []

    best_agent = initial_agent
    best_card = baseline_card

    for it in range(1, max_iterations + 1):
        dominant, all_cat_ids = _choose_dominant_failure(best_card)
        if dominant == "none" or best_card.pass_rate >= 1.0:
            break

        try:
            budget.check_before_call()
        except BudgetExceededError:
            break

        toks_before = budget.total_tokens
        t0 = time.monotonic()

        patched_agent, changes, diagnosis, target_ids = _apply_targeted_patch(
            best_agent, dominant, best_card, factory_llm
        )
        if not changes:
            break

        try:
            runtime.update_agent(patched_agent)
            new_card = evaluate_agent_on_cases(
                runtime,
                eval_cases,
                judge_llm,
                iteration=it,
                label=f"Iteration {it} ({dominant})",
                use_llm_judge=use_llm_judge,
            )
        except BudgetExceededError as exc:
            runtime.update_agent(best_agent)
            budget.stopped_early_reason = str(exc)
            break

        toks_delta = budget.total_tokens - toks_before
        dur = round(time.monotonic() - t0, 2)

        score_delta = new_card.mean_composite_score - best_card.mean_composite_score
        pass_delta = new_card.pass_rate - best_card.pass_rate
        improved = (pass_delta > 0) or (pass_delta == 0 and score_delta >= 0.005)

        if improved:
            accepted = True
            decision = (
                f"Accepted patch (+{pass_delta*100:.1f}pp pass rate, {score_delta:+.3f} composite score)."
            )
            best_agent = patched_agent
            best_card = new_card
            scorecards.append(new_card)
        else:
            accepted = False
            decision = (
                f"Rolled back patch (pass_delta={pass_delta:+.3f}, score_delta={score_delta:+.3f} did not improve over best)."
            )
            runtime.update_agent(best_agent)
            # Still record the attempted scorecard transparently!
            new_card.label = f"Iteration {it} ({dominant}) [ROLLED BACK]"
            scorecards.append(new_card)

        records.append(
            IterationRecord(
                iteration=it,
                dominant_failure_category=dominant,
                failure_ids_addressed=target_ids or all_cat_ids[:5],
                diagnosis_summary=diagnosis,
                changes=changes,
                score_before=scorecards[0].mean_composite_score if it == 1 else best_card.mean_composite_score if not accepted else round(best_card.mean_composite_score - score_delta, 4),
                score_after=new_card.mean_composite_score,
                pass_rate_before=round(best_card.pass_rate - (pass_delta if accepted else 0.0), 4),
                pass_rate_after=new_card.pass_rate,
                accepted=accepted,
                decision_reason=decision,
                tokens_used=toks_delta,
                duration_seconds=dur,
            )
        )

        # Bonus 2: marginal-gain early stop if accepted improvement stopped paying for itself
        if accepted and pass_delta == 0 and score_delta < min_gain_threshold:
            budget.stopped_early_reason = (
                f"Marginal gain early stop after iteration {it}: delta={score_delta:.4f} < {min_gain_threshold}"
            )
            break

    runtime.update_agent(best_agent)
    return best_agent, scorecards, records
