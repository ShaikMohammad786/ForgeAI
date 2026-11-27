"""Stage 7: Honest Reporting — scorecards, iteration diff logs, and failure diagnostics."""

from __future__ import annotations

from forgeai.llm.base import BudgetTracker
from forgeai.schemas.agent import AgentDefinition, PromptCandidate
from forgeai.schemas.evals import IterationScorecard
from forgeai.schemas.optimizer import BudgetSummary, FinalReport, IterationRecord
from forgeai.schemas.requirements import RequirementsDoc


def build_final_report(
    agent: AgentDefinition,
    requirements: RequirementsDoc,
    scorecards: list[IterationScorecard],
    iterations: list[IterationRecord],
    candidates: list[PromptCandidate],
    budget: BudgetTracker,
) -> FinalReport:
    """Assemble honest FinalReport strictly from measured evaluation verdicts."""
    accepted_cards = [sc for sc in scorecards if "[ROLLED BACK]" not in sc.label]
    final_card = accepted_cards[-1] if accepted_cards else scorecards[-1]

    remaining_failures: list[dict] = []
    for v in final_card.verdicts:
        if not v.passed:
            remaining_failures.append(
                {
                    "case_id": v.case_id,
                    "category": v.category,
                    "sub_type": v.sub_type,
                    "failure_category": v.failure_category,
                    "question": v.question,
                    "expected_behavior": v.expected_behavior,
                    "actual_refused": v.actual_refused,
                    "actual_answer_excerpt": v.actual_answer[:220],
                    "failure_reason": v.failure_reason,
                    "composite_score": v.metrics.composite_score,
                }
            )

    pct = round(final_card.pass_rate * 100.0, 1)
    if not remaining_failures:
        honest_text = (
            f"Agent '{agent.name}' passed {final_card.passed_cases}/{final_card.total_cases} ({pct}%) "
            "evaluation cases on the generated test suite. Note that passing auto-generated checks does not "
            "guarantee zero unseen edge-case errors under small-model inference; verify groundability caveats before production."
        )
    else:
        counts_str = ", ".join(f"{k}={v}" for k, v in final_card.failure_counts.items() if v > 0)
        honest_text = (
            f"Agent '{agent.name}' achieved {final_card.passed_cases}/{final_card.total_cases} ({pct}%) pass rate "
            f"(composite={final_card.mean_composite_score:.3f}). "
            f"{len(remaining_failures)} cases still fail ({counts_str}). "
            "Each remaining failure is documented transparently below with root cause and evidence."
        )

    compact_cards = [
        {
            "iteration": sc.iteration,
            "label": sc.label,
            "passed_cases": sc.passed_cases,
            "total_cases": sc.total_cases,
            "pass_rate": sc.pass_rate,
            "mean_composite_score": sc.mean_composite_score,
            "mean_groundedness": sc.mean_groundedness,
            "mean_fact_recall": sc.mean_fact_recall,
            "refusal_accuracy": sc.refusal_accuracy,
            "contract_adherence": sc.contract_adherence,
            "by_category": sc.by_category,
            "failure_counts": sc.failure_counts,
        }
        for sc in scorecards
    ]

    budget_sum = BudgetSummary(
        max_tokens=budget.max_tokens,
        tokens_used=budget.total_tokens,
        prompt_tokens=budget.prompt_tokens,
        completion_tokens=budget.completion_tokens,
        max_seconds=budget.max_seconds,
        elapsed_seconds=budget.elapsed_seconds,
        llm_calls=budget.llm_calls,
        stopped_early_reason=budget.stopped_early_reason,
    )

    return FinalReport(
        agent_name=agent.name,
        brief=requirements.brief,
        provider=agent.provider,
        generator_model=agent.model_name,
        judge_model=agent.judge_model_name,
        final_pass_rate=final_card.pass_rate,
        final_composite_score=final_card.mean_composite_score,
        scorecards=compact_cards,
        iterations=iterations,
        tournament_candidates=[
            {
                "candidate_id": c.candidate_id,
                "strategy_name": c.strategy_name,
                "tournament_score": c.tournament_score,
                "deterministic_score": c.deterministic_score,
                "pairwise_wins": c.pairwise_wins,
                "pairwise_losses": c.pairwise_losses,
                "pairwise_ties": c.pairwise_ties,
            }
            for c in candidates
        ],
        remaining_failures=remaining_failures,
        groundability_warnings=requirements.groundability.warnings,
        budget=budget_sum,
        honest_assessment=honest_text,
    )


def render_report_markdown(
    report: FinalReport,
    agent: AgentDefinition,
    requirements: RequirementsDoc,
) -> str:
    """Render clean, candid report.md for audit review."""
    lines: list[str] = [
        f"# ForgeAI Quality Report — {report.agent_name}",
        "",
        f"**Brief:** {report.brief}",
        f"**Provider / Generator Model:** `{report.provider}` / `{report.generator_model}`",
        f"**Judge Model:** `{report.judge_model}`",
        f"**Final Pass Rate:** **{round(report.final_pass_rate * 100, 1)}%** (Composite Score: **{report.final_composite_score:.3f}**)",
        "",
        "## Honest Assessment",
        report.honest_assessment,
        "",
        "## Attached vs Rejected Tools",
        f"- **Attached Tools:** `{[t.name for t in agent.tools] or ['none']}`",
    ]
    for tool_name, reason in requirements.rejected_tools.items():
        lines.append(f"- **Rejected `{tool_name}`:** {reason}")

    lines.extend(["", "## Groundability Audit", f"- **Status:** `{requirements.groundability.overall_status}` (ratio={requirements.groundability.groundability_ratio})"])
    if report.groundability_warnings:
        for w in report.groundability_warnings:
            lines.append(f"- ⚠️ {w}")
    else:
        lines.append("- All requested brief topics verified against corpus documents.")

    if report.tournament_candidates:
        lines.extend(
            [
                "",
                "## Prompt Tournament Results (Position-Debiased)",
                "| Candidate | Strategy | Tournament Score | Det Score | W / L / T |",
                "| :--- | :--- | :---: | :---: | :---: |",
            ]
        )
        for c in report.tournament_candidates:
            lines.append(
                f"| `{c['candidate_id']}` | {c['strategy_name']} | {c['tournament_score']:.3f} | {c['deterministic_score']:.3f} | {c['pairwise_wins']} / {c['pairwise_losses']} / {c['pairwise_ties']} |"
            )

    lines.extend(
        [
            "",
            "## Per-Iteration Scorecard",
            "| Iter | Stage | Pass Rate | Composite | Groundedness | Fact Recall | Refusal Acc | Contract | Failures (R/P/H/W) |",
            "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for sc in report.scorecards:
        fc = sc["failure_counts"]
        f_str = f"{fc.get('retrieval_miss',0)}/{fc.get('prompt_defect',0)}/{fc.get('hallucination',0)}/{fc.get('wrong_refusal',0)}"
        lines.append(
            f"| {sc['iteration']} | {sc['label']} | {sc['passed_cases']}/{sc['total_cases']} ({round(sc['pass_rate']*100,1)}%) | {sc['mean_composite_score']:.3f} | {sc['mean_groundedness']:.3f} | {sc['mean_fact_recall']:.3f} | {sc['refusal_accuracy']:.3f} | {sc['contract_adherence']:.3f} | {f_str} |"
        )

    lines.extend(["", "## Targeted Optimization Log"])
    if not report.iterations:
        lines.append("- No optimization patches applied (baseline already met criteria or budget stopped early).")
    for it in report.iterations:
        status = "ACCEPTED" if it.accepted else "ROLLED BACK"
        lines.append(
            f"### Iteration {it.iteration} — Target: `{it.dominant_failure_category}` [{status}]"
        )
        lines.append(f"- **Diagnosis:** {it.diagnosis_summary}")
        lines.append(f"- **Failure IDs Addressed:** `{it.failure_ids_addressed}`")
        lines.append(
            f"- **Score Before -> After:** `{it.score_before:.3f}` -> `{it.score_after:.3f}` "
            f"(Pass rate: `{round(it.pass_rate_before*100,1)}%` -> `{round(it.pass_rate_after*100,1)}%`)"
        )
        for ch in it.changes:
            lines.append(f"  - Change `{ch.field_path}`: `{ch.old_value}` -> `{ch.new_value}` ({ch.rationale})")
        lines.append(f"- **Decision:** {it.decision_reason}")

    lines.extend(["", "## What Still Fails & Why"])
    if not report.remaining_failures:
        lines.append("All evaluation cases passed on the final accepted configuration.")
    else:
        for rf in report.remaining_failures:
            lines.extend(
                [
                    f"### `{rf['case_id']}` ({rf['category']} / {rf['sub_type']}) — `{rf['failure_category']}`",
                    f"- **Question:** {rf['question']}",
                    f"- **Expected:** `{rf['expected_behavior']}` | **Actual Refused:** `{rf['actual_refused']}` | **Score:** `{rf['composite_score']:.3f}`",
                    f"- **Why It Failed:** {rf['failure_reason']}",
                    f"- **Actual Excerpt:** {rf['actual_answer_excerpt']}",
                ]
            )

    lines.extend(
        [
            "",
            "## Budget & Compute Telemetry",
            f"- **LLM Calls:** {report.budget.llm_calls}",
            f"- **Tokens Used:** {report.budget.tokens_used:,} / {report.budget.max_tokens:,} (prompt={report.budget.prompt_tokens:,}, completion={report.budget.completion_tokens:,})",
            f"- **Elapsed Time:** {report.budget.elapsed_seconds:.1f}s / {report.budget.max_seconds:.1f}s",
            f"- **Early Stop Reason:** {report.budget.stopped_early_reason or 'None (completed cleanly)'}",
        ]
    )
    return "\n".join(lines) + "\n"
