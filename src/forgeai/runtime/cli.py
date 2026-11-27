"""Typer CLI for ForgeAI (build, chat, eval, regress, models, ui)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from forgeai.config import get_settings
from forgeai.factory.judge import evaluate_agent_on_cases
from forgeai.factory.pipeline import run_forge_pipeline
from forgeai.factory.regression import run_regression_check
from forgeai.llm.router import build_provider, discover_backends
from forgeai.runtime.executor import AgentRuntime
from forgeai.schemas.evals import EvalSet
from forgeai.schemas.requirements import RequirementsDoc


app = typer.Typer(
    name="forgeai",
    help="ForgeAI: Autonomous Agent Factory that builds, evaluates, optimizes, and runs grounded AI agents.",
    add_completion=False,
)
console = Console()


def _cli_answer_callback(questions: list) -> list[str]:
    console.print("\n[bold cyan]Intake Interview (Press Enter to accept default answer):[/bold cyan]")
    answers: list[str] = []
    for idx, q in enumerate(questions, start=1):
        console.print(f"\n[bold]{idx}. {q.question}[/bold]")
        console.print(f"[dim]Rationale: {q.rationale}[/dim]")
        ans = Prompt.ask("Your answer", default=q.default_answer)
        answers.append(ans)
    return answers


def _cli_checkpoint_callback(req: RequirementsDoc) -> RequirementsDoc | None:
    console.print("\n")
    console.print(
        Panel.fit(
            f"[bold]Domain:[/bold] {req.domain_name}\n"
            f"[bold]Scope:[/bold] {req.scope}\n"
            f"[bold]Audience:[/bold] {req.audience}\n"
            f"[bold]Tone:[/bold] {req.tone}\n"
            f"[bold]Groundability:[/bold] {req.groundability.overall_status} (ratio={req.groundability.groundability_ratio})\n"
            f"[bold]Must Answer:[/bold] {req.must_answer_topics}\n"
            f"[bold]Must Refuse:[/bold] {req.must_refuse_topics}\n"
            f"[bold]Recommended Tools:[/bold] {req.recommended_tools}",
            title="Human Checkpoint — Review Requirements",
        )
    )
    if req.groundability.warnings:
        for w in req.groundability.warnings:
            console.print(f"[yellow]WARNING:[/yellow] {w}")

    choice = Prompt.ask(
        "Proceed with generation? [approve / edit / abort]",
        choices=["approve", "edit", "abort"],
        default="approve",
    )
    if choice == "abort":
        return None
    if choice == "edit":
        new_scope = Prompt.ask("Scope", default=req.scope)
        new_tone = Prompt.ask("Tone", default=req.tone)
        extra_refuse = Prompt.ask(
            "Add comma-separated must-refuse topics (or leave blank)", default=""
        )
        req.scope = new_scope
        req.tone = new_tone
        if extra_refuse.strip():
            req.must_refuse_topics.extend(
                [x.strip() for x in extra_refuse.split(",") if x.strip()]
            )
        req.human_checkpoint_status = "edited"
        return req

    req.human_checkpoint_status = "approved"
    return req


@app.command()
def models() -> None:
    """Inspect local Ollama models and Gemini API configuration."""
    cfg = get_settings()
    status = discover_backends(cfg)
    table = Table(title="ForgeAI Model Backends")
    table.add_column("Provider")
    table.add_column("Status")
    table.add_column("Models")
    table.add_row(
        "Ollama (Local)",
        "[green]Available[/green]" if status.ollama_available else "[red]Offline[/red]",
        ", ".join(status.ollama_models) if status.ollama_models else "(none)",
    )
    table.add_row(
        "Gemini API (.env)",
        "[green]Key Configured[/green]" if status.gemini_available else "[yellow]GEMINI_API_KEY unset[/yellow]",
        ", ".join(status.gemini_models),
    )
    console.print(table)


@app.command()
def build(
    brief: Optional[str] = typer.Option(None, "--brief", "-b", help="One-paragraph brief describing the desired agent."),
    brief_file: Optional[Path] = typer.Option(None, "--brief-file", help="Path to text file containing the brief."),
    docs: Path = typer.Option(..., "--docs", "-d", exists=True, file_okay=False, help="Path to corpus document folder."),
    out: Path = typer.Option(..., "--out", "-o", help="Output directory for generated agent and audit bundle."),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="LLM provider: ollama | gemini | auto"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Generator & agent model name."),
    judge_model: Optional[str] = typer.Option(None, "--judge-model", help="Separate judge model name."),
    unattended: bool = typer.Option(True, "--unattended/--interactive", help="Run unattended or prompt interactively with <=5 questions + checkpoint."),
    max_iterations: int = typer.Option(3, "--max-iterations", min=0, max=3, help="Max Critic-Optimizer iterations (0-3)."),
    max_tokens: int = typer.Option(250000, "--max-tokens", help="Token budget ceiling."),
    max_seconds: float = typer.Option(900.0, "--max-seconds", help="Wall-clock latency budget (seconds)."),
    eval_target: int = typer.Option(24, "--eval-target", min=20, max=50, help="Target number of auto-generated eval cases (>=20)."),
    tournament: bool = typer.Option(True, "--tournament/--no-tournament", help="Run 3-candidate Prompt Tournament."),
    llm_judge: bool = typer.Option(True, "--llm-judge/--det-judge", help="Use hybrid atomic NLI LLM judge + deterministic gates."),
) -> None:
    """Run ForgeAI end-to-end to build, test, optimize, and ship a runnable agent."""
    brief_text = (brief or "").strip()
    if brief_file is not None:
        brief_text = brief_file.read_text(encoding="utf-8").strip()
    if not brief_text:
        raise typer.BadParameter("Provide either --brief or --brief-file.")

    def _prog(msg: str, pct: float) -> None:
        console.print(f"[cyan][{int(pct*100):3d}%][/cyan] {msg}")

    result = run_forge_pipeline(
        brief=brief_text,
        docs_dir=docs,
        out_dir=out,
        provider=provider,
        model_name=model,
        judge_model_name=judge_model,
        unattended=unattended,
        answer_callback=_cli_answer_callback,
        checkpoint_callback=_cli_checkpoint_callback,
        run_tournament=tournament,
        max_iterations=max_iterations,
        max_tokens_budget=max_tokens,
        max_seconds_budget=max_seconds,
        eval_target_cases=eval_target,
        use_llm_judge=llm_judge,
        progress_callback=_prog,
    )

    if result.halted_ungroundable:
        console.print(f"[bold red]Halted:[/bold red] Brief was ungroundable against {docs}. See {out / 'report.md'}")
        raise typer.Exit(code=2)

    if result.report:
        console.print(
            Panel.fit(
                f"[bold green]Build Complete[/bold green]\n"
                f"Agent: {result.agent.name if result.agent else 'N/A'}\n"
                f"Pass Rate: {round(result.report.final_pass_rate * 100, 1)}% "
                f"(Composite: {result.report.final_composite_score:.3f})\n"
                f"Tools: {[t.name for t in result.agent.tools] if result.agent else []}\n"
                f"Artifacts: {result.out_dir}",
                title="ForgeAI Summary",
            )
        )


@app.command()
def chat(
    agent_dir: Path = typer.Argument(..., exists=True, help="Directory containing agent_definition.yaml."),
    question: Optional[str] = typer.Option(None, "--question", "-q", help="Run single-turn question and exit."),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="Override provider (ollama|gemini)."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Override model name."),
    docs_dir: Optional[Path] = typer.Option(None, "--docs", help="Optional override for docs directory."),
) -> None:
    """Generic runnable chat interface for any ForgeAI agent bundle."""
    runtime = AgentRuntime.from_directory(
        agent_dir,
        provider=provider,
        model_name=model,
        docs_dir_override=docs_dir,
    )
    console.print(
        Panel.fit(
            f"[bold]{runtime.agent.name}[/bold] (model: `{runtime.llm.model_name}`)\n"
            f"Tools: {[t.name for t in runtime.agent.tools] or ['none']}\n"
            f"Type 'exit' or 'quit' to stop.",
            title="ForgeAI Chat Runner",
        )
    )

    def _handle_turn(q: str, history: list[dict[str, str]]) -> None:
        trace = runtime.ask(q, chat_history=history)
        r = trace.response
        status_tag = "[red]REFUSED[/red]" if r.refused else "[green]ANSWERED[/green]"
        console.print(f"\n{status_tag} (confidence={r.confidence}, latency={trace.latency_ms:.0f}ms)")
        console.print(r.answer)
        if r.citations:
            cits = "; ".join(f"{c.get('source')}: \"{c.get('excerpt','')[:70]}\"" for c in r.citations)
            console.print(f"[dim]Citations: {cits}[/dim]")
        if r.tools_used:
            console.print(f"[dim]Tools used: {r.tools_used}[/dim]")
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": r.answer})

    history: list[dict[str, str]] = []
    if question:
        _handle_turn(question, history)
        return

    while True:
        q = Prompt.ask("\n[bold cyan]You[/bold cyan]").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit", "q"}:
            break
        _handle_turn(q, history)


@app.command(name="eval")
def eval_cmd(
    agent_dir: Path = typer.Argument(..., exists=True, help="Agent definition file or directory."),
    eval_file: Path = typer.Option(..., "--eval-set", "-e", exists=True, help="Path to eval_set.json."),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Optional output JSON scorecard path."),
    provider: Optional[str] = typer.Option(None, "--provider", "-p"),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    judge_model: Optional[str] = typer.Option(None, "--judge-model"),
) -> None:
    """Evaluate any agent definition against an eval_set.json."""
    cfg = get_settings()
    runtime = AgentRuntime.from_directory(agent_dir, provider=provider, model_name=model, settings=cfg)
    judge = build_provider(
        provider=provider or runtime.llm.provider_name,
        model_name=judge_model or runtime.agent.judge_model_name,
        settings=cfg,
    )
    eval_set = EvalSet.model_validate_json(eval_file.read_text(encoding="utf-8"))
    card = evaluate_agent_on_cases(runtime, eval_set.cases, judge, iteration=0, label="Standalone Eval")
    console.print(
        f"Scorecard: {card.passed_cases}/{card.total_cases} ({round(card.pass_rate*100,1)}%) "
        f"| composite={card.mean_composite_score:.3f} | groundedness={card.mean_groundedness:.3f}"
    )
    if out:
        out.write_text(json.dumps(card.model_dump(mode="json"), indent=2), encoding="utf-8")


@app.command()
def regress(
    agent_dir: Path = typer.Option(..., "--agent", "-a", exists=True, help="Existing agent output folder."),
    docs: Path = typer.Option(..., "--docs", "-d", exists=True, help="Updated document folder."),
    out: Path = typer.Option(..., "--out", "-o", help="Output folder for regression report."),
    provider: Optional[str] = typer.Option(None, "--provider", "-p"),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
) -> None:
    """Bonus 5: Run regression mode after document corpus updates."""
    diff, card = run_regression_check(
        agent_dir=agent_dir,
        new_docs_dir=docs,
        out_dir=out,
        provider=provider,
        model_name=model,
    )
    console.print(
        Panel.fit(
            f"Added: {diff.added_files} | Modified: {diff.modified_files} | Removed: {diff.removed_files}\n"
            f"Pass Rate: {round(diff.old_pass_rate*100,1)}% -> {round(diff.new_pass_rate*100,1)}%\n"
            f"Newly Broken: {len(diff.newly_broken)} | Now Ungroundable: {len(diff.now_ungroundable)} | Newly Fixed: {len(diff.newly_fixed)}\n"
            f"Report saved to: {out / 'regression_report.md'}",
            title="ForgeAI Regression Result",
        )
    )


@app.command()
def baseline(
    human_yaml: Path = typer.Option(..., "--human", exists=True, help="Hand-written baseline YAML."),
    forge_dir: Path = typer.Option(..., "--forge", exists=True, help="ForgeAI generated output folder."),
    out_md: Path = typer.Option(..., "--out", help="Output markdown comparison report path."),
    provider: Optional[str] = typer.Option(None, "--provider", "-p"),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    judge_model: Optional[str] = typer.Option(None, "--judge-model"),
) -> None:
    """Bonus 4: Compare hand-written human baseline against ForgeAI generated agent."""
    import sys as _sys

    root_dir = Path(__file__).resolve().parents[3]
    if str(root_dir) not in _sys.path:
        _sys.path.insert(0, str(root_dir))
    from baselines.compare_baseline import compare_factory_vs_human

    summary = compare_factory_vs_human(
        human_yaml=human_yaml,
        forge_dir=forge_dir,
        out_md=out_md,
        provider=provider,
        model_name=model,
        judge_model_name=judge_model,
    )
    console.print(
        f"Human pass_rate={summary['human']['pass_rate']:.3f} vs "
        f"ForgeAI pass_rate={summary['forgeai']['pass_rate']:.3f}. Saved: {out_md}"
    )


@app.command()
def ui(
    port: int = typer.Option(8501, "--port", help="Streamlit server port."),
) -> None:
    """Launch the ForgeAI Streamlit Factory & Chat Runner web interface."""
    app_path = Path(__file__).resolve().parent.parent / "ui" / "app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port)],
        check=True,
    )


if __name__ == "__main__":
    app()
