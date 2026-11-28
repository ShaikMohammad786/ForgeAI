"""Bonus 4: Factory vs Human Baseline comparative benchmark script."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from forgeai.config import get_settings
from forgeai.factory.judge import evaluate_agent_on_cases
from forgeai.llm.router import build_provider
from forgeai.runtime.executor import AgentRuntime
from forgeai.schemas.agent import AgentDefinition
from forgeai.schemas.evals import EvalSet


def compare_factory_vs_human(
    *,
    human_yaml: Path,
    forge_dir: Path,
    out_md: Path,
    provider: str | None = None,
    model_name: str | None = None,
    judge_provider: str | None = None,
    judge_model_name: str | None = None,
) -> dict:
    cfg = get_settings()
    eval_set = EvalSet.model_validate_json((forge_dir / "eval_set.json").read_text(encoding="utf-8"))
    human_agent = AgentDefinition.model_validate(yaml.safe_load(human_yaml.read_text(encoding="utf-8")))
    forge_agent = AgentDefinition.model_validate(
        yaml.safe_load((forge_dir / "agent_definition.yaml").read_text(encoding="utf-8"))
    )

    chosen_prov = provider or ("gemini" if forge_agent.provider.startswith("gemini") else forge_agent.provider)
    chosen_model = model_name or forge_agent.model_name
    chosen_judge = judge_model_name or forge_agent.judge_model_name

    # Run both under the EXACT SAME model + docs + judge for a fair apples-to-apples test!
    human_agent = human_agent.model_copy(
        update={
            "provider": chosen_prov,
            "model_name": chosen_model,
            "judge_model_name": chosen_judge,
            "docs_dir": forge_agent.docs_dir,
        }
    )

    llm = build_provider(provider=chosen_prov, model_name=chosen_model, settings=cfg)
    judge_llm = build_provider(
        provider=judge_provider or chosen_prov,
        model_name=chosen_judge,
        settings=cfg,
    )

    human_rt = AgentRuntime(human_agent, llm, embed_model_name=cfg.embed_model)
    forge_rt = AgentRuntime(forge_agent, llm, embed_model_name=cfg.embed_model)

    human_card = evaluate_agent_on_cases(
        human_rt, eval_set.cases, judge_llm, iteration=0, label="Human Hand-Written Baseline"
    )
    forge_card = evaluate_agent_on_cases(
        forge_rt, eval_set.cases, judge_llm, iteration=forge_agent.version, label="ForgeAI Optimized Agent"
    )

    h_map = {v.case_id: v for v in human_card.verdicts}
    f_map = {v.case_id: v for v in forge_card.verdicts}

    forge_wins = []
    human_wins = []
    for cid, fv in f_map.items():
        hv = h_map.get(cid)
        if not hv:
            continue
        if fv.passed and not hv.passed:
            forge_wins.append((cid, fv.question, hv.failure_category, hv.failure_reason))
        elif hv.passed and not fv.passed:
            human_wins.append((cid, fv.question, fv.failure_category, fv.failure_reason))

    lines = [
        "# Factory vs Human Baseline Comparison (Bonus 4)",
        "",
        f"- **Evaluation Suite:** `{eval_set.total_cases}` auto-generated cases (`{forge_dir / 'eval_set.json'}`)",
        f"- **Model Under Test:** `{chosen_prov}` / `{chosen_model}` | **Judge:** `{chosen_judge}`",
        "",
        "## Head-to-Head Scorecard",
        "| Metric | Human Baseline | ForgeAI Optimized | Delta |",
        "| :--- | :---: | :---: | :---: |",
        f"| Pass Rate | {human_card.passed_cases}/{human_card.total_cases} ({round(human_card.pass_rate*100,1)}%) | {forge_card.passed_cases}/{forge_card.total_cases} ({round(forge_card.pass_rate*100,1)}%) | {round((forge_card.pass_rate - human_card.pass_rate)*100,1):+.1f}pp |",
        f"| Composite Score | {human_card.mean_composite_score:.3f} | {forge_card.mean_composite_score:.3f} | {forge_card.mean_composite_score - human_card.mean_composite_score:+.3f} |",
        f"| Groundedness | {human_card.mean_groundedness:.3f} | {forge_card.mean_groundedness:.3f} | {forge_card.mean_groundedness - human_card.mean_groundedness:+.3f} |",
        f"| Fact Recall | {human_card.mean_fact_recall:.3f} | {forge_card.mean_fact_recall:.3f} | {forge_card.mean_fact_recall - human_card.mean_fact_recall:+.3f} |",
        f"| Refusal Accuracy | {human_card.refusal_accuracy:.3f} | {forge_card.refusal_accuracy:.3f} | {forge_card.refusal_accuracy - human_card.refusal_accuracy:+.3f} |",
        f"| Contract Adherence | {human_card.contract_adherence:.3f} | {forge_card.contract_adherence:.3f} | {forge_card.contract_adherence - human_card.contract_adherence:+.3f} |",
        "",
        f"## Where ForgeAI Beats the Human Baseline ({len(forge_wins)} cases)",
        *(
            [f"- `{cid}`: {q} *(Human failed via `{fc}`: {fr})*" for cid, q, fc, fr in forge_wins]
            if forge_wins
            else ["- None in this run."]
        ),
        "",
        f"## Where the Human Baseline Beats ForgeAI ({len(human_wins)} cases)",
        *(
            [f"- `{cid}`: {q} *(ForgeAI failed via `{fc}`: {fr})*" for cid, q, fc, fr in human_wins]
            if human_wins
            else ["- None in this run."]
        ),
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "human": human_card.model_dump(mode="json", exclude={"verdicts"}),
        "forgeai": forge_card.model_dump(mode="json", exclude={"verdicts"}),
        "forge_wins_count": len(forge_wins),
        "human_wins_count": len(human_wins),
    }
    (out_md.with_suffix(".json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
