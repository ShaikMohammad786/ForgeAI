"""Streamlit Web UI (Bonus 6) — Dual/Hybrid Provider Factory, Interactive Checkpoint, Evidence Inspector & Regression Mode."""

from __future__ import annotations

import json
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import yaml

# Ensure src/ is on sys.path when launched directly on Streamlit Cloud / HF Spaces
_ROOT_DIR = Path(__file__).resolve().parents[3]
_SRC_DIR = _ROOT_DIR / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from forgeai.config import ForgeSettings, get_settings
from forgeai.factory.eval_gen import generate_eval_set
from forgeai.factory.generator import generate_agent_candidates
from forgeai.factory.intake import generate_clarifying_questions, run_intake_interview
from forgeai.factory.judge import run_prompt_tournament
from forgeai.factory.optimizer import run_optimization_loop
from forgeai.factory.pipeline import run_forge_pipeline, save_bundle_artifacts
from forgeai.factory.regression import run_regression_check
from forgeai.factory.reporter import build_final_report, render_report_markdown
from forgeai.ingestion.loader import SUPPORTED_EXTENSIONS, load_corpus
from forgeai.llm.base import BudgetExceededError, BudgetTracker
from forgeai.llm.router import build_factory_providers, discover_backends
from forgeai.runtime.executor import AgentRuntime
from forgeai.schemas.agent import AgentDefinition
from forgeai.schemas.requirements import RequirementsDoc
from forgeai.tools.registry import TOOL_CATALOG

UPLOAD_TYPES = sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS)


def _discover_agent_bundles() -> dict[str, Path]:
    """Locate valid saved AgentDefinition bundles in demos, agents, and session builds."""
    candidates: dict[str, Path] = {}
    search_roots = [
        _ROOT_DIR / "demos",
        _ROOT_DIR / "agents",
        _ROOT_DIR / ".forgeai_ui_builds",
    ]
    for base in search_roots:
        if not base.exists():
            continue
        for def_file in sorted(base.rglob("agent_definition.yaml")):
            # Exclude nested regression output folders from top-level unless needed
            if "regression_demo" in def_file.parts:
                continue
            bundle_dir = def_file.parent
            try:
                rel = bundle_dir.relative_to(_ROOT_DIR).as_posix()
            except ValueError:
                rel = str(bundle_dir)
            label = rel.replace("/output", "").replace("demos/", "Demo: ").replace(".forgeai_ui_builds/", "UI Build: ")
            candidates[label] = bundle_dir
    return candidates


def _save_uploaded_files(uploaded_files: list[Any], prefix: str) -> Path:
    tmp_dir = Path(tempfile.mkdtemp(prefix=prefix))
    for uf in uploaded_files:
        safe_name = Path(uf.name).name
        (tmp_dir / safe_name).write_bytes(uf.getvalue())
    return tmp_dir


def _load_bundle_summary_files(out_dir: Path) -> dict[str, Any]:
    """Load persisted bundle artifacts safely for display and downloads."""
    loaded: dict[str, Any] = {}
    for fname in (
        "agent_definition.yaml",
        "requirements.yaml",
        "eval_set.json",
        "scorecard.json",
        "iteration_log.json",
        "tournament_results.json",
        "report.md",
    ):
        fpath = out_dir / fname
        if fpath.exists():
            loaded[fname] = fpath.read_text(encoding="utf-8")
    return loaded


def _render_bundle_results(out_dir: Path) -> None:
    """Render scorecard KPIs, tools & custom macros, charts, and downloads for a finished build."""
    files = _load_bundle_summary_files(out_dir)
    if "report.md" not in files:
        st.warning("No final report found in this bundle.")
        return

    agent_data = yaml.safe_load(files["agent_definition.yaml"]) if "agent_definition.yaml" in files else None
    scorecards = json.loads(files["scorecard.json"]) if "scorecard.json" in files else []
    iterations = json.loads(files["iteration_log.json"]) if "iteration_log.json" in files else []
    req_data = yaml.safe_load(files["requirements.yaml"]) if "requirements.yaml" in files else None

    if agent_data and scorecards:
        accepted = [sc for sc in scorecards if "[ROLLED BACK]" not in str(sc.get("label", ""))]
        final_sc = accepted[-1] if accepted else scorecards[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Final Pass Rate",
            f"{round(final_sc['pass_rate'] * 100, 1)}%",
            f"{final_sc['passed_cases']}/{final_sc['total_cases']} cases",
        )
        c2.metric("Composite Score", f"{final_sc['mean_composite_score']:.3f}", f"Refusal Acc: {final_sc['refusal_accuracy']:.2f}")
        c3.metric("Standard Tools", len(agent_data.get("tools", [])), f"Out of {len(TOOL_CATALOG)} available")
        c4.metric("Custom Macros (AST)", len(agent_data.get("custom_tools", [])), f"{sum(1 for it in iterations if it.get('accepted'))} accepted iters")

    # Tool & Custom Macro breakdown
    if agent_data:
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("#### Attached Tools & Synthesized Custom Macros")
            for t in agent_data.get("tools", []):
                st.success(f"**{t['name']}** — {t.get('description', '')}")
            for ct in agent_data.get("custom_tools", []):
                detail = (
                    f"Formula: `{ct.get('expression')}` (params: `{ct.get('parameters')}`)"
                    if ct.get("tool_type") == "formula"
                    else f"Tabular Filter on `{ct.get('target_file')}`: `{ct.get('preset_filters')}`"
                )
                st.info(f"**Custom Macro `{ct['name']}`** ({ct.get('tool_type')}) — {ct.get('description', '')}\n\n{detail}")
            if not agent_data.get("tools") and not agent_data.get("custom_tools"):
                st.caption("No external tools attached (pure grounded RAG policy mode).")
        with col_r:
            st.markdown("#### Excluded Tools (Capability Gate)")
            rejected = (req_data or {}).get("rejected_tools", {})
            if rejected:
                rej_rows = [{"Tool": k, "Exclusion Reason": v} for k, v in rejected.items()]
                st.dataframe(pd.DataFrame(rej_rows), use_container_width=True, hide_index=True)
            else:
                st.caption("No excluded tool logs recorded.")

    # Scorecard progression chart
    if len(scorecards) >= 1:
        st.markdown("#### Evaluation Scorecard Progression")
        chart_df = pd.DataFrame(
            [
                {
                    "Stage": f"Iter {sc['iteration']}: {sc['label'][:24]}",
                    "Pass Rate (%)": round(sc["pass_rate"] * 100, 1),
                    "Composite (%)": round(sc["mean_composite_score"] * 100, 1),
                    "Groundedness (%)": round(sc["mean_groundedness"] * 100, 1),
                }
                for sc in scorecards
            ]
        ).set_index("Stage")
        st.line_chart(chart_df[["Pass Rate (%)", "Composite (%)", "Groundedness (%)"]])

    # Download bundle artifacts
    st.markdown("#### Download Auditable Artifacts")
    dl_cols = st.columns(4)
    dl_specs = [
        ("agent_definition.yaml", "Agent YAML", "text/yaml"),
        ("eval_set.json", "Eval Set JSON", "application/json"),
        ("requirements.yaml", "Requirements YAML", "text/yaml"),
        ("report.md", "Quality Report MD", "text/markdown"),
    ]
    for col, (fname, label, mime) in zip(dl_cols, dl_specs):
        if fname in files:
            col.download_button(label=f"Download {label}", data=files[fname], file_name=fname, mime=mime, use_container_width=True)

    with st.expander("Full Quality Report (`report.md`)", expanded=True):
        st.markdown(files["report.md"])


def _render_sidebar(cfg: ForgeSettings) -> dict[str, Any]:
    """Render dual/hybrid execution settings & architecture summary in sidebar."""
    status = discover_backends(cfg)

    with st.sidebar:
        st.markdown("## ForgeAI Engine Setup")
        st.caption("Zero-server Streamlit runtime: run locally with Ollama, in cloud via Gemini API, or Hybrid.")

        execution_mode = st.radio(
            "Execution Mode",
            options=["Ollama (Local)", "Gemini API (Cloud)", "Hybrid (Mix Generator & Judge)"],
            index=1 if (not status.ollama_available and status.gemini_available) else 0,
        )

        gemini_key_override = st.text_input(
            "Gemini API Key (optional override)",
            value="",
            type="password",
            help="Leave blank to use GEMINI_API_KEY from .env or Streamlit Secrets.",
        )
        effective_gemini_ok = bool(gemini_key_override.strip() or status.gemini_available)

        ollama_choices = status.ollama_models or [cfg.ollama_model, cfg.ollama_judge_model]
        gemini_choices = status.gemini_models or [cfg.gemini_model, cfg.gemini_judge_model]

        if execution_mode == "Ollama (Local)":
            gen_provider = "ollama"
            judge_provider = "ollama"
            default_gen = ollama_choices.index(cfg.ollama_model) if cfg.ollama_model in ollama_choices else 0
            default_j = ollama_choices.index(cfg.ollama_judge_model) if cfg.ollama_judge_model in ollama_choices else 0
            gen_model = st.selectbox("Generator / Agent Model", options=ollama_choices, index=default_gen)
            judge_model = st.selectbox("Cross-Judge Model", options=ollama_choices, index=default_j)
        elif execution_mode == "Gemini API (Cloud)":
            gen_provider = "gemini"
            judge_provider = "gemini"
            default_gen = gemini_choices.index(cfg.gemini_model) if cfg.gemini_model in gemini_choices else 0
            default_j = gemini_choices.index(cfg.gemini_judge_model) if cfg.gemini_judge_model in gemini_choices else 0
            gen_model = st.selectbox("Generator / Agent Model", options=gemini_choices, index=default_gen)
            judge_model = st.selectbox("Cross-Judge Model", options=gemini_choices, index=default_j)
        else:
            col_gp, col_jp = st.columns(2)
            gen_provider = col_gp.selectbox("Gen Backend", options=["gemini", "ollama"], index=0 if effective_gemini_ok else 1)
            judge_provider = col_jp.selectbox("Judge Backend", options=["ollama", "gemini"], index=0 if status.ollama_available else 1)
            gen_pool = gemini_choices if gen_provider == "gemini" else ollama_choices
            judge_pool = gemini_choices if judge_provider == "gemini" else ollama_choices
            gen_model = st.selectbox("Generator / Agent Model", options=gen_pool, index=0)
            judge_model = st.selectbox("Cross-Judge Model", options=judge_pool, index=0)

        st.markdown("---")
        b1, b2 = st.columns(2)
        b1.metric("Ollama", "Online" if status.ollama_available else "Offline", f"{len(status.ollama_models)} models")
        b2.metric("Gemini API", "Configured" if effective_gemini_ok else "No Key", "v1beta" if effective_gemini_ok else "Set key")

        with st.expander("Factory Budgets & Optimizer"):
            max_iters = st.slider("Max Critic Iterations", min_value=1, max_value=3, value=3)
            eval_cases = st.slider("Eval Cases to Generate", min_value=20, max_value=30, value=24)
            max_tokens = st.number_input("Token Budget Cap", min_value=30_000, max_value=600_000, value=250_000, step=10_000)
            use_llm_judge = st.checkbox("Hybrid LLM + Deterministic Judge", value=True)
            run_tournament = st.checkbox("Run 3-Prompt Tournament", value=True)

        with st.expander(f"Standard Tool Registry ({len(TOOL_CATALOG)} tools)"):
            for spec in TOOL_CATALOG.values():
                st.caption(f"**{spec.name}**: {spec.description}")

    return {
        "execution_mode": execution_mode,
        "gen_provider": gen_provider,
        "judge_provider": judge_provider,
        "gen_model": gen_model,
        "judge_model": judge_model,
        "gemini_key_override": gemini_key_override.strip() or None,
        "max_iters": max_iters,
        "eval_cases": eval_cases,
        "max_tokens": int(max_tokens),
        "use_llm_judge": use_llm_judge,
        "run_tournament": run_tournament,
    }


def _render_factory_tab(cfg: ForgeSettings, sb: dict[str, Any]) -> None:
    """Tab 1: Brief, 17-format Document Upload, Human Checkpoint, Optimization & Reporting."""
    st.markdown("### Build a Verified Domain Agent")
    st.caption(
        "Workflow: **1. Brief & Corpus Inspection** -> **2. Intake & Groundability Guard** -> "
        "**3. Agent, Custom Macros & Eval Synthesis** -> **4. Critic-Optimizer & Honest Report**"
    )

    pcol1, pcol2, pcol3 = st.columns(3)
    hr_brief_path = _ROOT_DIR / "demos" / "hr_policy" / "brief.txt"
    ship_brief_path = _ROOT_DIR / "demos" / "ship_spares" / "brief.txt"

    if pcol1.button("Load Demo 1: Shore HR Policy", use_container_width=True):
        st.session_state["brief_input"] = hr_brief_path.read_text(encoding="utf-8") if hr_brief_path.exists() else (
            "Build an HR policy assistant for shore staff answering annual leave, sick leave, and domestic travel meal caps, while refusing medical or legal advice."
        )
        st.session_state["docs_folder_input"] = str((_ROOT_DIR / "demos" / "hr_policy" / "docs").resolve())
        st.session_state.pop("intake_state", None)
    if pcol2.button("Load Demo 2: Ship Spares Procurement", use_container_width=True):
        st.session_state["brief_input"] = ship_brief_path.read_text(encoding="utf-8") if ship_brief_path.exists() else (
            "Build a ship-spares procurement assistant over our spares catalogue CSV and supplier/freight PDFs for landed cost quotes and lead times."
        )
        st.session_state["docs_folder_input"] = str((_ROOT_DIR / "demos" / "ship_spares" / "docs").resolve())
        st.session_state.pop("intake_state", None)
    if pcol3.button("Clear / Upload Custom Docs", use_container_width=True):
        st.session_state["brief_input"] = ""
        st.session_state["docs_folder_input"] = ""
        st.session_state.pop("intake_state", None)

    brief = st.text_area(
        "1. Agent Brief (what should it do, who uses it, what must it refuse?)",
        value=st.session_state.get("brief_input", ""),
        height=95,
        placeholder="Example: Build a maritime procurement assistant over spares_catalogue.csv and SLA PDFs that computes landed costs and refuses speculative ungrounded answers.",
    )

    col_up, col_dir = st.columns(2)
    uploaded_files = col_up.file_uploader(
        f"2a. Upload Corpus Documents ({len(UPLOAD_TYPES)} formats: PDF, DOCX, XLSX, CSV, TSV, MD, TXT, JSON/L, YAML, HTML, XML, LOG)",
        type=UPLOAD_TYPES,
        accept_multiple_files=True,
    )
    folder_path = col_dir.text_input(
        "2b. Or Local Corpus Folder Path",
        value=st.session_state.get("docs_folder_input", ""),
        placeholder="e.g. demos/hr_policy/docs",
    )

    # Resolve & preview corpus cleanly before building
    docs_dir: Path | None = None
    if uploaded_files:
        sig = tuple((uf.name, uf.size) for uf in uploaded_files)
        if st.session_state.get("uploaded_sig") != sig:
            st.session_state["uploaded_dir"] = str(_save_uploaded_files(uploaded_files, "forgeai_docs_"))
            st.session_state["uploaded_sig"] = sig
        docs_dir = Path(st.session_state["uploaded_dir"])
    elif folder_path.strip():
        cand = Path(folder_path.strip())
        if cand.exists() and cand.is_dir():
            docs_dir = cand.resolve()

    if docs_dir is not None:
        try:
            preview_corpus = load_corpus(docs_dir)
            if preview_corpus.documents:
                inv_rows = [
                    {
                        "Filename": d.summary.filename,
                        "Type": d.summary.doc_type.upper(),
                        "Sections": d.summary.num_sections,
                        "Characters": d.summary.total_chars,
                        "Key Headings / Schema": ", ".join(d.summary.key_headings[:3]) or "-",
                    }
                    for d in preview_corpus.documents
                ]
                with st.expander(f"Corpus Inventory Preview ({len(inv_rows)} files loaded)", expanded=True):
                    st.dataframe(pd.DataFrame(inv_rows), use_container_width=True, hide_index=True)
            else:
                st.warning("The specified folder contains no supported files.")
                docs_dir = None
        except Exception as exc:
            st.error(f"Could not inspect corpus: {exc}")
            docs_dir = None

    b_col1, b_col2 = st.columns(2)
    start_interactive = b_col1.button("Step 1 -> 2: Start Intake & Human Checkpoint", type="primary", use_container_width=True)
    run_unattended = b_col2.button("Run Full Factory Unattended (1-Click)", use_container_width=True)

    # Interactive path
    if start_interactive:
        if not brief.strip() or docs_dir is None:
            st.error("Provide both a brief and a valid document folder or uploaded files.")
        else:
            with st.spinner("Generating <=5 clarifying questions from brief and corpus..."):
                corpus = load_corpus(docs_dir)
                budget = BudgetTracker(max_tokens=sb["max_tokens"], max_seconds=cfg.max_seconds_budget)
                factory_llm, judge_llm = build_factory_providers(
                    provider=sb["gen_provider"],
                    model_name=sb["gen_model"],
                    judge_provider=sb["judge_provider"],
                    judge_model_name=sb["judge_model"],
                    gemini_api_key=sb["gemini_key_override"],
                    budget=budget,
                    settings=cfg,
                )
                questions = generate_clarifying_questions(brief, corpus, factory_llm, max_questions=5)
                st.session_state["intake_state"] = {
                    "brief": brief,
                    "docs_dir": str(docs_dir),
                    "questions": [q.model_dump() for q in questions],
                    "requirements": None,
                }

    # Unattended 1-click path
    if run_unattended:
        if not brief.strip() or docs_dir is None:
            st.error("Provide both a brief and a valid document folder or uploaded files.")
        else:
            run_id = f"build_{datetime.now(timezone.utc).strftime('%H%M%S')}_{uuid.uuid4().hex[:4]}"
            out_dir = _ROOT_DIR / ".forgeai_ui_builds" / run_id
            prog = st.progress(0.05, text="Starting ForgeAI Factory...")
            result = run_forge_pipeline(
                brief=brief,
                docs_dir=docs_dir,
                out_dir=out_dir,
                provider=sb["gen_provider"],
                model_name=sb["gen_model"],
                judge_provider=sb["judge_provider"],
                judge_model_name=sb["judge_model"],
                gemini_api_key=sb["gemini_key_override"],
                unattended=True,
                run_tournament=sb["run_tournament"],
                max_iterations=sb["max_iters"],
                max_tokens_budget=sb["max_tokens"],
                eval_target_cases=sb["eval_cases"],
                use_llm_judge=sb["use_llm_judge"],
                progress_callback=lambda msg, pct: prog.progress(min(max(pct, 0.0), 1.0), text=msg),
                settings=cfg,
            )
            st.session_state["last_build_dir"] = str(out_dir)
            if result.halted_ungroundable:
                st.error("ForgeAI Groundability Guard HALTED build: corpus cannot support this brief without hallucination.")
                st.markdown((out_dir / "report.md").read_text(encoding="utf-8"))
            else:
                st.success(f"Factory build completed! Bundle saved to `{out_dir.relative_to(_ROOT_DIR).as_posix()}`.")

    # Interactive Step 2 dialogue + Requirements Checkpoint
    intake = st.session_state.get("intake_state")
    if intake:
        st.markdown("---")
        st.markdown("### Step 2: Clarifying Dialogue & Requirements Checkpoint")
        with st.form("clarifying_qa_form"):
            answers: list[str] = []
            for idx, q in enumerate(intake["questions"], 1):
                st.markdown(f"**Q{idx}. {q['question']}**")
                if q.get("rationale"):
                    st.caption(q["rationale"])
                answers.append(
                    st.text_input(
                        f"Answer {idx}",
                        value=q["default_answer"],
                        key=f"qa_ans_{idx}",
                        label_visibility="collapsed",
                    )
                )
            synth_clicked = st.form_submit_button("Synthesize Requirements & Run Groundability Guard")

        if synth_clicked:
            with st.spinner("Running Groundability Guard & Requirements Synthesis..."):
                corpus = load_corpus(Path(intake["docs_dir"]))
                budget = BudgetTracker(max_tokens=sb["max_tokens"], max_seconds=cfg.max_seconds_budget)
                factory_llm, _ = build_factory_providers(
                    provider=sb["gen_provider"],
                    model_name=sb["gen_model"],
                    judge_provider=sb["judge_provider"],
                    judge_model_name=sb["judge_model"],
                    gemini_api_key=sb["gemini_key_override"],
                    budget=budget,
                    settings=cfg,
                )
                req = run_intake_interview(
                    intake["brief"],
                    corpus,
                    factory_llm,
                    embed_model_name=cfg.embed_model,
                    pre_supplied_answers=answers,
                )
                intake["requirements"] = req.model_dump(mode="json")

        if intake.get("requirements"):
            req = RequirementsDoc.model_validate(intake["requirements"])
            g = req.groundability
            if g.overall_status == "ungroundable":
                st.error(f"GROUNDABILITY GUARD BLOCKED (ratio={g.groundability_ratio}): Brief is unsupported by documents.")
            elif g.overall_status == "partially_groundable":
                st.warning(f"Partially Groundable (ratio={g.groundability_ratio}): Unsupported topics moved to Must-Refuse.")
            else:
                st.success(f"Fully Groundable (ratio={g.groundability_ratio}): Evidence verified across corpus.")

            if g.items:
                g_rows = [
                    {
                        "Requested Topic": it.requested_topic,
                        "Status": it.status,
                        "Evidence Score": round(it.evidence_score, 3),
                        "Supporting Sources": ", ".join(it.matching_sources) or "-",
                    }
                    for it in g.items
                ]
                st.dataframe(pd.DataFrame(g_rows), use_container_width=True, hide_index=True)

            with st.form("human_checkpoint_form"):
                c1, c2 = st.columns(2)
                scope_edit = c1.text_area("Scope", value=req.scope, height=90)
                tone_edit = c1.text_input("Tone", value=req.tone)
                must_ans_edit = c1.text_area("Must-Answer Topics (one per line)", value="\n".join(req.must_answer_topics), height=110)
                must_ref_edit = c2.text_area("Must-Refuse Topics (one per line)", value="\n".join(req.must_refuse_topics), height=110)
                c2.info(f"Eligible Tools Selected: `{req.recommended_tools or ['none']}`")
                if req.rejected_tools:
                    c2.caption(f"Rejected Tools ({len(req.rejected_tools)}): {', '.join(req.rejected_tools.keys())}")
                approve_btn = st.form_submit_button(
                    "Approve Requirements & Generate Agent",
                    type="primary",
                    disabled=(g.overall_status == "ungroundable"),
                )

            if approve_btn:
                req.scope = scope_edit.strip()
                req.tone = tone_edit.strip()
                req.must_answer_topics = [x.strip() for x in must_ans_edit.splitlines() if x.strip()]
                req.must_refuse_topics = [x.strip() for x in must_ref_edit.splitlines() if x.strip()]
                req.human_checkpoint_status = "edited"

                run_id = f"build_{datetime.now(timezone.utc).strftime('%H%M%S')}_{uuid.uuid4().hex[:4]}"
                out_dir = _ROOT_DIR / ".forgeai_ui_builds" / run_id
                prog = st.progress(0.10, text="Synthesizing Agent Definition & Custom Macros...")
                corpus = load_corpus(Path(intake["docs_dir"]))
                budget = BudgetTracker(max_tokens=sb["max_tokens"], max_seconds=cfg.max_seconds_budget)
                factory_llm, judge_llm = build_factory_providers(
                    provider=sb["gen_provider"],
                    model_name=sb["gen_model"],
                    judge_provider=sb["judge_provider"],
                    judge_model_name=sb["judge_model"],
                    gemini_api_key=sb["gemini_key_override"],
                    budget=budget,
                    settings=cfg,
                )
                base_agent, candidates = generate_agent_candidates(
                    req,
                    corpus,
                    factory_llm,
                    judge_model_name=judge_llm.model_name,
                    num_candidates=3 if sb["run_tournament"] else 1,
                )
                prog.progress(0.35, text="Generating >=20 grounded evaluation cases...")
                eval_set = generate_eval_set(req, base_agent, corpus, factory_llm, target_cases=sb["eval_cases"])
                rt = AgentRuntime(base_agent, factory_llm, corpus=corpus, embed_model_name=cfg.embed_model)
                if sb["run_tournament"] and len(candidates) > 1:
                    prog.progress(0.55, text="Running position-debiased Prompt Tournament...")
                    try:
                        base_agent, candidates = run_prompt_tournament(base_agent, candidates, eval_set.cases, rt, judge_llm)
                    except BudgetExceededError as exc:
                        budget.stopped_early_reason = str(exc)

                prog.progress(0.75, text="Running Critic-Optimizer Loop...")
                final_agent, scorecards, iterations = run_optimization_loop(
                    base_agent,
                    eval_set.cases,
                    rt,
                    factory_llm,
                    judge_llm,
                    budget,
                    max_iterations=sb["max_iters"],
                    min_gain_threshold=cfg.min_gain_threshold,
                    use_llm_judge=sb["use_llm_judge"],
                )
                report = build_final_report(final_agent, req, scorecards, iterations, candidates, budget)
                save_bundle_artifacts(
                    out_dir,
                    corpus=corpus,
                    requirements=req,
                    agent=final_agent,
                    eval_set=eval_set,
                    scorecards=scorecards,
                    iterations=iterations,
                    candidates=candidates,
                    report=report,
                )
                prog.progress(1.0, text="Completed!")
                st.session_state["last_build_dir"] = str(out_dir)
                st.session_state.pop("intake_state", None)
                st.success(f"Build finished! Bundle saved to `{out_dir.relative_to(_ROOT_DIR).as_posix()}`.")

    if st.session_state.get("last_build_dir"):
        bdir = Path(st.session_state["last_build_dir"])
        if bdir.exists():
            st.markdown("---")
            st.markdown("### Latest Factory Build Dashboard")
            _render_bundle_results(bdir)


def _render_chat_tab(cfg: ForgeSettings, sb: dict[str, Any]) -> None:
    """Tab 2: Generic Chat Runner with Starter Chips & Grounded Evidence/Tool Trace Inspector."""
    st.markdown("### Generic Agent Chat Runner & Trace Inspector")
    bundles = _discover_agent_bundles()
    if not bundles:
        st.info("No built agent bundles found yet. Build one in Tab 1 first.")
        return

    c_sel, c_ov = st.columns([2, 1])
    selected_label = c_sel.selectbox("Select Generated Agent Bundle", options=list(bundles.keys()))
    bundle_dir = bundles[selected_label]
    ov_choice = c_ov.selectbox(
        "Runtime Inference Backend",
        options=["Sidebar Selected Engine", "Agent Saved Default"],
        index=0,
    )

    agent_yaml = yaml.safe_load((bundle_dir / "agent_definition.yaml").read_text(encoding="utf-8"))
    agent_def = AgentDefinition.model_validate(agent_yaml)

    # Agent Profile Card
    with st.expander(f"Agent Blueprint: {agent_def.name} (v{agent_def.version})", expanded=False):
        p1, p2, p3 = st.columns(3)
        p1.markdown(f"**Built With:** `{agent_def.provider}` / `{agent_def.model_name}`")
        p1.markdown(f"**Judge:** `{agent_def.judge_model_name}`")
        r = agent_def.retrieval
        p2.markdown(f"**Retrieval:** `top_k={r.top_k}`, `alpha={r.hybrid_alpha}`, `threshold={r.min_score_threshold}`")
        p2.markdown(f"**Chunking:** `{r.chunking_strategy}` (`{r.chunk_size}` / `{r.chunk_overlap}`)")
        std_names = [t.name for t in agent_def.tools] or ["none"]
        macro_names = [ct.name for ct in agent_def.custom_tools] or ["none"]
        p3.markdown(f"**Standard Tools:** `{std_names}`")
        p3.markdown(f"**Custom Macros:** `{macro_names}`")
        st.caption(f"**Must Answer:** {'; '.join(agent_def.must_answer_topics[:4])}")
        st.caption(f"**Must Refuse:** {'; '.join(agent_def.must_refuse_topics[:4])}")

    # Chat history state scoped per bundle
    hist_key = f"chat_hist::{bundle_dir}"
    if hist_key not in st.session_state:
        st.session_state[hist_key] = []

    col_clr, _ = st.columns([1, 4])
    if col_clr.button("Reset Conversation"):
        st.session_state[hist_key] = []
        st.rerun()

    # Starter Question Quick Chips
    starters: list[str] = []
    for topic in agent_def.must_answer_topics[:2]:
        starters.append(f"What are the exact rules or details regarding {topic}?")
    if agent_def.must_refuse_topics:
        starters.append(f"[Refusal Test] Can you give advice on {agent_def.must_refuse_topics[0]}?")
    else:
        starters.append("[Refusal Test] What is the CEO's personal salary and stock allocation?")

    st.caption("Quick Test Prompts:")
    qcols = st.columns(len(starters))
    queued_prompt: str | None = None
    for idx, (qcol, starter) in enumerate(zip(qcols, starters)):
        short_lbl = starter[:46] + ("..." if len(starter) > 46 else "")
        if qcol.button(short_lbl, key=f"starter_{bundle_dir}_{idx}", use_container_width=True):
            queued_prompt = starter.replace("[Refusal Test] ", "")

    # Render history
    for item in st.session_state[hist_key]:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            if item["role"] == "assistant" and "meta" in item:
                meta = item["meta"]
                m1, m2, m3, m4 = st.columns(4)
                m1.caption(f"**Status:** {'Refused' if meta['refused'] else 'Answered'}")
                m2.caption(f"**Confidence:** `{meta['confidence']}`")
                m3.caption(f"**Latency:** `{meta['latency_ms']} ms`")
                m4.caption(f"**Tools:** `{meta['tools_used'] or ['none']}`")
                if meta.get("citations"):
                    cit_lines = "\n".join(
                        f"- `{c.get('source', '?')}`: \"{c.get('excerpt', '')[:160]}\""
                        for c in meta["citations"]
                    )
                    st.info(f"**Citations:**\n{cit_lines}")
                with st.expander("Evidence & Execution Trace"):
                    if meta.get("tool_outputs"):
                        st.markdown("**Tool Calls & Outputs:**")
                        st.json(meta["tool_outputs"])
                    st.markdown("**Retrieved Context Chunks:**")
                    for ch in meta.get("chunks", []):
                        st.caption(
                            f"`{ch['source']}` ({ch['location']}) | hybrid={ch['score']:.3f} "
                            f"(bm25={ch['bm25']:.3f}, dense={ch['dense']:.3f})"
                        )
                        st.text(ch["text"][:320])

    user_q = st.chat_input(f"Ask {agent_def.name} anything grounded in its corpus...")
    active_query = queued_prompt or user_q

    if active_query:
        st.session_state[hist_key].append({"role": "user", "content": active_query})
        with st.spinner("Retrieving context and executing tools..."):
            use_sidebar = ov_choice == "Sidebar Selected Engine"
            rt = AgentRuntime.from_directory(
                bundle_dir,
                provider=sb["gen_provider"] if use_sidebar else None,
                model_name=sb["gen_model"] if use_sidebar else None,
                gemini_api_key=sb["gemini_key_override"],
                settings=cfg,
            )
            trace = rt.ask(active_query, chat_history=st.session_state[hist_key][:-1])
            resp = trace.response

        st.session_state[hist_key].append(
            {
                "role": "assistant",
                "content": resp.answer,
                "meta": {
                    "refused": resp.refused,
                    "confidence": resp.confidence,
                    "latency_ms": trace.latency_ms,
                    "tools_used": resp.tools_used,
                    "citations": resp.citations,
                    "tool_outputs": trace.tool_outputs,
                    "chunks": [
                        {
                            "source": h.chunk.source,
                            "location": h.chunk.page_or_row,
                            "score": h.combined_score,
                            "bm25": h.bm25_score,
                            "dense": h.dense_score,
                            "text": h.chunk.text,
                        }
                        for h in trace.retrieved_chunks
                    ],
                },
            }
        )
        st.rerun()


def _render_regression_tab(cfg: ForgeSettings, sb: dict[str, Any]) -> None:
    """Tab 3: Bonus 5 Regression Mode — Compare old eval set against updated documents."""
    st.markdown("### Regression Mode: Document Update Impact Audit")
    st.caption(
        "Detect added/modified/removed files via SHA-256 manifests and re-run the baseline eval suite "
        "to surface broken or newly ungroundable questions."
    )

    bundles = _discover_agent_bundles()
    if not bundles:
        st.info("No built agent bundles found yet.")
        return

    sel_label = st.selectbox("Baseline Agent Bundle", options=list(bundles.keys()), key="reg_bundle_select")
    sel_dir = bundles[sel_label]

    col_u, col_p = st.columns(2)
    reg_uploads = col_u.file_uploader(
        "Upload Updated Documents Folder Contents",
        type=UPLOAD_TYPES,
        accept_multiple_files=True,
        key="reg_uploader",
    )
    reg_folder = col_p.text_input(
        "Or Updated Corpus Folder Path",
        value="",
        placeholder="e.g. demos/hr_policy/docs_v2",
        key="reg_folder_input",
    )

    if st.button("Run Regression Audit", type="primary"):
        reg_docs: Path | None = None
        if reg_uploads:
            reg_docs = _save_uploaded_files(reg_uploads, "forgeai_reg_")
        elif reg_folder.strip() and Path(reg_folder.strip()).is_dir():
            reg_docs = Path(reg_folder.strip()).resolve()

        if reg_docs is None:
            st.error("Upload updated documents or enter a valid folder path.")
            return

        reg_out = _ROOT_DIR / ".forgeai_ui_builds" / f"reg_{uuid.uuid4().hex[:6]}"
        with st.spinner("Re-indexing updated corpus, auditing groundability, and re-scoring eval cases..."):
            diff, _ = run_regression_check(
                agent_dir=sel_dir,
                new_docs_dir=reg_docs,
                out_dir=reg_out,
                provider=sb["gen_provider"],
                model_name=sb["gen_model"],
                judge_provider=sb["judge_provider"],
                judge_model_name=sb["judge_model"],
                gemini_api_key=sb["gemini_key_override"],
                use_llm_judge=sb["use_llm_judge"],
                settings=cfg,
            )

        delta_pp = round((diff.new_pass_rate - diff.old_pass_rate) * 100, 1)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Files Added / Modified", f"+{len(diff.added_files)} / ~{len(diff.modified_files)}")
        m2.metric("Files Removed", len(diff.removed_files))
        m3.metric("Newly Broken Cases", len(diff.newly_broken), f"{len(diff.now_ungroundable)} ungroundable")
        m4.metric("Pass Rate Delta", f"{delta_pp:+.1f} pp", f"{round(diff.old_pass_rate * 100, 1)}% -> {round(diff.new_pass_rate * 100, 1)}%")

        rep_md = (reg_out / "regression_report.md").read_text(encoding="utf-8")
        st.download_button(
            "Download Regression Report MD",
            data=rep_md,
            file_name="regression_report.md",
            mime="text/markdown",
        )
        st.markdown(rep_md)


def main() -> None:
    st.set_page_config(
        page_title="ForgeAI — Autonomous Agent Factory",
        page_icon="🛠️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    cfg = get_settings()

    st.title("ForgeAI — Autonomous Agent Factory")
    st.markdown(
        "**17 Document Formats** · **10 Sandboxed Standard Tools + AST Custom Macros** · "
        "**Hybrid BM25 + Dense Retrieval** · **Debiased Judge & Critic-Optimizer**"
    )

    sidebar_cfg = _render_sidebar(cfg)

    tab1, tab2, tab3 = st.tabs(
        [
            "1. Agent Factory (Build & Optimize)",
            "2. Chat Runner & Evidence Inspector",
            "3. Regression Mode (Doc Diffs)",
        ]
    )

    with tab1:
        _render_factory_tab(cfg, sidebar_cfg)
    with tab2:
        _render_chat_tab(cfg, sidebar_cfg)
    with tab3:
        _render_regression_tab(cfg, sidebar_cfg)


if __name__ == "__main__":
    main()
