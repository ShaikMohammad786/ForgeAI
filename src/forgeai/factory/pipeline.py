"""End-to-end ForgeAI Factory pipeline & artifact bundle persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from forgeai.config import ForgeSettings, get_settings
from forgeai.factory.eval_gen import generate_eval_set
from forgeai.factory.generator import generate_agent_candidates
from forgeai.factory.intake import run_intake_interview
from forgeai.factory.judge import run_prompt_tournament
from forgeai.factory.optimizer import run_optimization_loop
from forgeai.factory.reporter import build_final_report, render_report_markdown
from forgeai.ingestion.loader import LoadedCorpus, load_corpus
from forgeai.llm.base import BudgetExceededError, BudgetTracker
from forgeai.llm.router import build_factory_providers
from forgeai.runtime.executor import AgentRuntime
from forgeai.schemas.agent import AgentDefinition, PromptCandidate
from forgeai.schemas.evals import EvalSet, IterationScorecard
from forgeai.schemas.optimizer import FinalReport, IterationRecord
from forgeai.schemas.requirements import RequirementsDoc


@dataclass
class PipelineResult:
    """All outputs produced by a ForgeAI factory run."""

    out_dir: Path
    requirements: RequirementsDoc
    agent: AgentDefinition | None
    eval_set: EvalSet | None
    scorecards: list[IterationScorecard]
    iterations: list[IterationRecord]
    candidates: list[PromptCandidate]
    report: FinalReport | None
    halted_ungroundable: bool = False


def _render_requirements_md(req: RequirementsDoc) -> str:
    lines = [
        f"# Requirements Specification — {req.domain_name}",
        "",
        f"**Brief:** {req.brief}",
        f"**Audience:** {req.audience}",
        f"**Tone:** {req.tone}",
        f"**Human Checkpoint:** `{req.human_checkpoint_status}`",
        "",
        "## Scope",
        req.scope,
        "",
        "## Must-Answer Topics",
        *[f"- {t}" for t in req.must_answer_topics],
        "",
        "## Must-Refuse Topics",
        *[f"- {t}" for t in req.must_refuse_topics],
        "",
        "## Tool Selection",
        f"- **Recommended:** `{req.recommended_tools}`",
        *[f"- **Rejected `{k}`:** {v}" for k, v in req.rejected_tools.items()],
        "",
        "## Clarifying Q&A",
    ]
    for i, qa in enumerate(req.clarifying_qa, 1):
        lines.append(f"{i}. **Q:** {qa.question}")
        lines.append(f"   **A ({qa.answered_by}):** {qa.answer}")

    lines.extend(
        [
            "",
            "## Groundability Audit",
            f"- **Status:** `{req.groundability.overall_status}` (ratio={req.groundability.groundability_ratio})",
        ]
    )
    for item in req.groundability.items:
        lines.append(
            f"- **{item.requested_topic}** -> `{item.status}` (score={item.evidence_score:.2f}, sources={item.matching_sources}): {item.rationale}"
        )
    if req.groundability.warnings:
        lines.append("")
        lines.append("### Warnings")
        for w in req.groundability.warnings:
            lines.append(f"- ⚠️ {w}")
    return "\n".join(lines) + "\n"


def _render_iteration_log_md(iterations: list[IterationRecord]) -> str:
    if not iterations:
        return "# Optimization Iteration Log\n\nNo optimization patches were needed or executed.\n"
    lines = ["# Optimization Iteration Log", ""]
    for it in iterations:
        status = "ACCEPTED" if it.accepted else "ROLLED BACK"
        lines.extend(
            [
                f"## Iteration {it.iteration} ({it.dominant_failure_category}) — {status}",
                f"- **Diagnosis:** {it.diagnosis_summary}",
                f"- **Failure IDs Targeted:** `{it.failure_ids_addressed}`",
                f"- **Score:** `{it.score_before:.3f}` -> `{it.score_after:.3f}`",
                f"- **Pass Rate:** `{round(it.pass_rate_before*100,1)}%` -> `{round(it.pass_rate_after*100,1)}%`",
                f"- **Decision:** {it.decision_reason}",
                "- **Targeted Diff:**",
            ]
        )
        for ch in it.changes:
            lines.append(f"  - `{ch.field_path}`: `{ch.old_value}` -> `{ch.new_value}` ({ch.rationale})")
        lines.append("")
    return "\n".join(lines)


def _render_scorecard_md(scorecards: list[IterationScorecard]) -> str:
    lines = [
        "# Evaluation Scorecards",
        "",
        "| Iter | Stage | Passed | Pass Rate | Composite | Groundedness | Fact Recall | Refusal Acc | Contract |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for sc in scorecards:
        lines.append(
            f"| {sc.iteration} | {sc.label} | {sc.passed_cases}/{sc.total_cases} | {round(sc.pass_rate*100,1)}% | {sc.mean_composite_score:.3f} | {sc.mean_groundedness:.3f} | {sc.mean_fact_recall:.3f} | {sc.refusal_accuracy:.3f} | {sc.contract_adherence:.3f} |"
        )
    return "\n".join(lines) + "\n"


def save_bundle_artifacts(
    out_dir: Path,
    *,
    corpus: LoadedCorpus,
    requirements: RequirementsDoc,
    agent: AgentDefinition | None,
    eval_set: EvalSet | None,
    scorecards: list[IterationScorecard],
    iterations: list[IterationRecord],
    candidates: list[PromptCandidate],
    report: FinalReport | None,
) -> None:
    """Persist complete, auditable factory artifacts to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "corpus_manifest.json").write_text(
        json.dumps(corpus.manifest, indent=2), encoding="utf-8"
    )
    (out_dir / "requirements.yaml").write_text(
        yaml.safe_dump(requirements.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    (out_dir / "requirements.md").write_text(
        _render_requirements_md(requirements), encoding="utf-8"
    )

    if agent is not None:
        (out_dir / "agent_definition.yaml").write_text(
            yaml.safe_dump(agent.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )
        (out_dir / "agent_definition.json").write_text(
            json.dumps(agent.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )
    if eval_set is not None:
        (out_dir / "eval_set.json").write_text(
            json.dumps(eval_set.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )
    if candidates:
        (out_dir / "tournament_results.json").write_text(
            json.dumps([c.model_dump(mode="json") for c in candidates], indent=2),
            encoding="utf-8",
        )
    if scorecards:
        (out_dir / "scorecard.json").write_text(
            json.dumps([sc.model_dump(mode="json") for sc in scorecards], indent=2),
            encoding="utf-8",
        )
        (out_dir / "scorecard.md").write_text(
            _render_scorecard_md(scorecards), encoding="utf-8"
        )
    (out_dir / "iteration_log.json").write_text(
        json.dumps([it.model_dump(mode="json") for it in iterations], indent=2),
        encoding="utf-8",
    )
    (out_dir / "iteration_log.md").write_text(
        _render_iteration_log_md(iterations), encoding="utf-8"
    )
    if report is not None and agent is not None:
        (out_dir / "report.md").write_text(
            render_report_markdown(report, agent, requirements), encoding="utf-8"
        )


def run_forge_pipeline(
    *,
    brief: str,
    docs_dir: str | Path,
    out_dir: str | Path,
    provider: str | None = None,
    model_name: str | None = None,
    judge_provider: str | None = None,
    judge_model_name: str | None = None,
    gemini_api_key: str | None = None,
    unattended: bool = True,
    max_questions: int = 5,
    answer_callback: Callable[[list[Any]], list[str]] | None = None,
    pre_supplied_answers: list[str] | None = None,
    checkpoint_callback: Callable[[RequirementsDoc], RequirementsDoc | None] | None = None,
    run_tournament: bool = True,
    max_iterations: int | None = None,
    max_tokens_budget: int | None = None,
    max_seconds_budget: float | None = None,
    min_gain_threshold: float | None = None,
    eval_target_cases: int | None = None,
    use_llm_judge: bool = True,
    progress_callback: Callable[[str, float], None] | None = None,
    settings: ForgeSettings | None = None,
) -> PipelineResult:
    """Execute the full ForgeAI Agent Factory pipeline end-to-end."""
    cfg = settings or get_settings()
    out_path = Path(out_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    def _notify(msg: str, pct: float) -> None:
        if progress_callback:
            progress_callback(msg, pct)

    budget = BudgetTracker(
        max_tokens=max_tokens_budget or cfg.max_tokens_budget,
        max_seconds=max_seconds_budget or cfg.max_seconds_budget,
    )
    factory_llm, judge_llm = build_factory_providers(
        provider=provider,
        model_name=model_name,
        judge_provider=judge_provider,
        judge_model_name=judge_model_name,
        gemini_api_key=gemini_api_key,
        budget=budget,
        settings=cfg,
    )

    _notify("Loading and indexing document corpus...", 0.08)
    corpus = load_corpus(docs_dir)

    _notify("Stage 1 & 2: Running Intake Interviewer & Groundability Guard...", 0.20)
    requirements = run_intake_interview(
        brief,
        corpus,
        factory_llm,
        embed_model_name=cfg.embed_model,
        max_questions=max_questions,
        answer_callback=None if unattended else answer_callback,
        pre_supplied_answers=pre_supplied_answers,
    )

    # Bonus 3: Human Checkpoint
    if not unattended and checkpoint_callback is not None:
        reviewed = checkpoint_callback(requirements)
        if reviewed is None or reviewed.human_checkpoint_status == "aborted":
            requirements.human_checkpoint_status = "aborted"
            save_bundle_artifacts(
                out_path,
                corpus=corpus,
                requirements=requirements,
                agent=None,
                eval_set=None,
                scorecards=[],
                iterations=[],
                candidates=[],
                report=None,
            )
            return PipelineResult(
                out_dir=out_path,
                requirements=requirements,
                agent=None,
                eval_set=None,
                scorecards=[],
                iterations=[],
                candidates=[],
                report=None,
                halted_ungroundable=False,
            )
        requirements = reviewed

    # Requirement 2 hard guard: refuse to synthesize an agent when corpus cannot support the brief at all
    if requirements.groundability.overall_status == "ungroundable":
        save_bundle_artifacts(
            out_path,
            corpus=corpus,
            requirements=requirements,
            agent=None,
            eval_set=None,
            scorecards=[],
            iterations=[],
            candidates=[],
            report=None,
        )
        block_md = (
            f"# ForgeAI Halted — Ungroundable Brief\n\n"
            f"**Brief:** {brief}\n\n"
            "ForgeAI refused to generate an agent because the document corpus lacks sufficient evidence "
            "to support the requested capabilities without hallucinating.\n\n"
            "## Warnings\n"
            + "\n".join(f"- {w}" for w in requirements.groundability.warnings)
            + "\n"
        )
        (out_path / "report.md").write_text(block_md, encoding="utf-8")
        return PipelineResult(
            out_dir=out_path,
            requirements=requirements,
            agent=None,
            eval_set=None,
            scorecards=[],
            iterations=[],
            candidates=[],
            report=None,
            halted_ungroundable=True,
        )

    _notify("Stage 3: Generating Agent Definition & Candidate Prompts...", 0.36)
    base_agent, candidates = generate_agent_candidates(
        requirements,
        corpus,
        factory_llm,
        judge_model_name=judge_llm.model_name,
        num_candidates=3 if run_tournament else 1,
    )

    _notify("Stage 4: Generating Grounded Evaluation Suite (>=20 cases)...", 0.50)
    eval_set = generate_eval_set(
        requirements,
        base_agent,
        corpus,
        factory_llm,
        target_cases=eval_target_cases or cfg.eval_case_target,
    )

    runtime = AgentRuntime(
        base_agent,
        factory_llm,
        corpus=corpus,
        embed_model_name=cfg.embed_model,
    )

    if run_tournament and len(candidates) > 1:
        _notify("Bonus 1: Running Position-Debiased Prompt Tournament...", 0.62)
        try:
            base_agent, candidates = run_prompt_tournament(
                base_agent, candidates, eval_set.cases, runtime, judge_llm
            )
        except BudgetExceededError as exc:
            budget.stopped_early_reason = str(exc)

    _notify("Stage 5 & 6: Running Evaluation & Critic-Optimizer Loop...", 0.78)
    final_agent, scorecards, iterations = run_optimization_loop(
        base_agent,
        eval_set.cases,
        runtime,
        factory_llm,
        judge_llm,
        budget,
        max_iterations=cfg.max_iterations if max_iterations is None else max_iterations,
        min_gain_threshold=cfg.min_gain_threshold if min_gain_threshold is None else min_gain_threshold,
        use_llm_judge=use_llm_judge,
    )

    _notify("Stage 7: Compiling Honest Report & Saving Bundle...", 0.94)
    report = build_final_report(
        final_agent, requirements, scorecards, iterations, candidates, budget
    )
    save_bundle_artifacts(
        out_path,
        corpus=corpus,
        requirements=requirements,
        agent=final_agent,
        eval_set=eval_set,
        scorecards=scorecards,
        iterations=iterations,
        candidates=candidates,
        report=report,
    )
    _notify("Completed ForgeAI build.", 1.0)

    return PipelineResult(
        out_dir=out_path,
        requirements=requirements,
        agent=final_agent,
        eval_set=eval_set,
        scorecards=scorecards,
        iterations=iterations,
        candidates=candidates,
        report=report,
        halted_ungroundable=False,
    )
