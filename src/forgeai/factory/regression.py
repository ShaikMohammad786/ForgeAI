"""Bonus 5: Regression Mode — re-evaluates prior eval set when documents change."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from forgeai.config import ForgeSettings, get_settings
from forgeai.factory.eval_gen import _fact_grounded_in_corpus
from forgeai.factory.groundability import audit_groundability
from forgeai.factory.judge import evaluate_agent_on_cases
from forgeai.ingestion.loader import diff_manifests, load_corpus
from forgeai.llm.base import BudgetTracker
from forgeai.llm.router import build_factory_providers
from forgeai.runtime.executor import AgentRuntime
from forgeai.schemas.agent import AgentDefinition
from forgeai.schemas.evals import EvalSet, IterationScorecard
from forgeai.schemas.optimizer import RegressionDiff


def run_regression_check(
    *,
    agent_dir: str | Path,
    new_docs_dir: str | Path,
    out_dir: str | Path,
    provider: str | None = None,
    model_name: str | None = None,
    judge_provider: str | None = None,
    judge_model_name: str | None = None,
    gemini_api_key: str | None = None,
    use_llm_judge: bool = True,
    settings: ForgeSettings | None = None,
) -> tuple[RegressionDiff, IterationScorecard]:
    """Detect corpus changes, rebuild index/agent on new docs, re-run old eval set, and report diffs."""
    cfg = settings or get_settings()
    src_dir = Path(agent_dir).resolve()
    dst_dir = Path(out_dir).resolve()
    dst_dir.mkdir(parents=True, exist_ok=True)

    agent_yaml = src_dir / "agent_definition.yaml"
    eval_path = src_dir / "eval_set.json"
    scorecard_path = src_dir / "scorecard.json"
    manifest_path = src_dir / "corpus_manifest.json"

    old_agent = AgentDefinition.model_validate(
        yaml.safe_load(agent_yaml.read_text(encoding="utf-8"))
    )
    old_evals = EvalSet.model_validate_json(eval_path.read_text(encoding="utf-8"))
    old_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    old_cards_raw = (
        json.loads(scorecard_path.read_text(encoding="utf-8"))
        if scorecard_path.exists()
        else []
    )
    old_card = IterationScorecard.model_validate(old_cards_raw[-1]) if old_cards_raw else None

    new_corpus = load_corpus(new_docs_dir)
    added, modified, removed = diff_manifests(old_manifest, new_corpus.manifest)

    budget = BudgetTracker(max_tokens=cfg.max_tokens_budget, max_seconds=cfg.max_seconds_budget)
    chosen_provider = provider or (
        "gemini" if old_agent.provider.startswith("gemini") else old_agent.provider
    )
    factory_llm, judge_llm = build_factory_providers(
        provider=chosen_provider,
        model_name=model_name or old_agent.model_name,
        judge_provider=judge_provider,
        judge_model_name=judge_model_name or old_agent.judge_model_name,
        gemini_api_key=gemini_api_key,
        budget=budget,
        settings=cfg,
    )

    # Re-audit groundability against updated docs & update agent metadata cleanly
    new_groundability = audit_groundability(
        old_agent.brief, new_corpus, factory_llm, embed_model_name=cfg.embed_model
    )
    updated_refuse = list(old_agent.must_refuse_topics)
    for unsup in new_groundability.unsupported_requests:
        tag = f"[Unsupported after doc update] {unsup}"
        if tag not in updated_refuse:
            updated_refuse.append(tag)

    regenerated_agent = old_agent.model_copy(
        update={
            "version": old_agent.version + 1,
            "docs_dir": str(new_corpus.docs_dir),
            "must_refuse_topics": updated_refuse,
            "groundability_caveats": new_groundability.warnings,
        }
    )

    runtime = AgentRuntime(
        regenerated_agent,
        factory_llm,
        corpus=new_corpus,
        docs_dir_override=new_corpus.docs_dir,
        embed_model_name=cfg.embed_model,
    )
    new_card = evaluate_agent_on_cases(
        runtime,
        old_evals.cases,
        judge_llm,
        iteration=regenerated_agent.version,
        label="Regression Check (New Docs)",
        use_llm_judge=use_llm_judge,
    )

    old_by_id = {v.case_id: v for v in old_card.verdicts} if old_card else {}
    new_corpus_lower = "\n".join(d.full_text.lower() for d in new_corpus.documents)

    newly_broken: list[dict] = []
    newly_fixed: list[dict] = []
    still_failing: list[dict] = []
    now_ungroundable: list[dict] = []

    case_lookup = {c.case_id: c for c in old_evals.cases}
    for nv in new_card.verdicts:
        case = case_lookup.get(nv.case_id)
        if case and case.expected_behavior == "answer" and case.key_facts:
            still_in_docs = any(
                _fact_grounded_in_corpus(f, new_corpus_lower) for f in case.key_facts
            )
            if not still_in_docs:
                now_ungroundable.append(
                    {
                        "case_id": nv.case_id,
                        "question": nv.question,
                        "old_sources": case.source_documents,
                        "reason": "Key facts no longer exist in updated corpus.",
                    }
                )

        ov = old_by_id.get(nv.case_id)
        if ov is None:
            continue
        if ov.passed and not nv.passed:
            newly_broken.append(
                {
                    "case_id": nv.case_id,
                    "question": nv.question,
                    "failure_category": nv.failure_category,
                    "failure_reason": nv.failure_reason,
                }
            )
        elif (not ov.passed) and nv.passed:
            newly_fixed.append({"case_id": nv.case_id, "question": nv.question})
        elif (not ov.passed) and (not nv.passed):
            still_failing.append(
                {
                    "case_id": nv.case_id,
                    "question": nv.question,
                    "failure_category": nv.failure_category,
                    "failure_reason": nv.failure_reason,
                }
            )

    diff = RegressionDiff(
        old_docs_count=len(old_manifest),
        new_docs_count=len(new_corpus.manifest),
        added_files=added,
        modified_files=modified,
        removed_files=removed,
        old_pass_rate=old_card.pass_rate if old_card else 0.0,
        new_pass_rate=new_card.pass_rate,
        newly_broken=newly_broken,
        newly_fixed=newly_fixed,
        still_failing=still_failing,
        now_ungroundable=now_ungroundable,
    )

    # Save regression artifacts cleanly
    (dst_dir / "agent_definition.yaml").write_text(
        yaml.safe_dump(regenerated_agent.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    (dst_dir / "corpus_manifest.json").write_text(
        json.dumps(new_corpus.manifest, indent=2), encoding="utf-8"
    )
    (dst_dir / "regression_report.json").write_text(
        json.dumps(diff.model_dump(mode="json"), indent=2), encoding="utf-8"
    )
    (dst_dir / "regression_scorecard.json").write_text(
        json.dumps(new_card.model_dump(mode="json"), indent=2), encoding="utf-8"
    )

    md_lines = [
        f"# ForgeAI Regression Report — {regenerated_agent.name}",
        "",
        f"- **Added Files:** `{added}`",
        f"- **Modified Files:** `{modified}`",
        f"- **Removed Files:** `{removed}`",
        f"- **Pass Rate Change:** `{round(diff.old_pass_rate*100,1)}%` -> `{round(diff.new_pass_rate*100,1)}%`",
        "",
        f"## Newly Broken ({len(newly_broken)})",
        *[f"- `{b['case_id']}` ({b['failure_category']}): {b['question']} — {b['failure_reason']}" for b in newly_broken],
        "",
        f"## Now Ungroundable After Document Changes ({len(now_ungroundable)})",
        *[f"- `{u['case_id']}`: {u['question']} (old docs: {u['old_sources']})" for u in now_ungroundable],
        "",
        f"## Newly Fixed ({len(newly_fixed)})",
        *[f"- `{fx['case_id']}`: {fx['question']}" for fx in newly_fixed],
        "",
        f"## Still Failing ({len(still_failing)})",
        *[f"- `{sf['case_id']}` ({sf['failure_category']}): {sf['question']}" for sf in still_failing],
    ]
    (dst_dir / "regression_report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return diff, new_card
