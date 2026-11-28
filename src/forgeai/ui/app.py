"""Streamlit Web UI (Bonus 6) — Dual Provider Selector, Factory Runner, Checkpoint & Chat Runner."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
import yaml

# Ensure src/ is on sys.path when launched directly on Streamlit Cloud / HF Spaces
_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from forgeai.config import get_settings
from forgeai.factory.eval_gen import generate_eval_set
from forgeai.factory.generator import generate_agent_candidates
from forgeai.factory.intake import generate_clarifying_questions, run_intake_interview
from forgeai.factory.judge import run_prompt_tournament
from forgeai.factory.optimizer import run_optimization_loop
from forgeai.factory.pipeline import save_bundle_artifacts
from forgeai.factory.regression import run_regression_check
from forgeai.factory.reporter import build_final_report, render_report_markdown
from forgeai.ingestion.loader import load_corpus
from forgeai.llm.base import BudgetTracker
from forgeai.llm.router import build_factory_providers, discover_backends
from forgeai.runtime.executor import AgentRuntime
from forgeai.schemas.requirements import RequirementsDoc


REPO_ROOT = Path(__file__).resolve().parents[3]
DEMOS_DIR = REPO_ROOT / "demos"


def _get_writable_runs_dir() -> Path:
    preferred = REPO_ROOT / "runs"
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        probe = preferred / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return preferred
    except Exception:
        fallback = Path(tempfile.gettempdir()) / "forgeai_runs"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


RUNS_DIR = _get_writable_runs_dir()


def _sync_streamlit_secrets_to_env() -> None:
    try:
        for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OLLAMA_HOST", "FORGEAI_PROVIDER"):
            if k in st.secrets and not os.environ.get(k):
                os.environ[k] = str(st.secrets[k])
    except Exception:
        pass


def _discover_built_agents() -> list[Path]:
    found: list[Path] = []
    for base in (DEMOS_DIR, RUNS_DIR):
        if not base.exists():
            continue
        for p in sorted(base.rglob("agent_definition.yaml")):
            found.append(p.parent)
    return found


def main() -> None:
    st.set_page_config(
        page_title="ForgeAI — The Agent That Builds Agents",
        page_icon="⚒️",
        layout="wide",
    )
    _sync_streamlit_secrets_to_env()
    cfg = get_settings()
    backends = discover_backends(cfg)

    st.title("⚒️ ForgeAI — Autonomous Agent Factory")
    st.caption(
        "Brief + Documents -> Intake & Groundability Guard -> Declarative Agent + Prompt Tournament -> >=20 Verified Evals -> Debiased Judge -> Targeted Optimizer -> Runnable Agent"
    )

    # Sidebar: 3-Mode Provider Selector (Ollama Local vs Gemini API vs Hybrid Cross-Provider)
    with st.sidebar:
        st.header("Model Backend")
        default_is_gemini = (
            cfg.provider == "gemini"
            or (not backends.ollama_available and backends.gemini_available)
        )
        mode_options = [
            "Ollama (Local Open Models)",
            "Gemini API (Interactions API / .env)",
            "Hybrid (Cross-Provider Agent + Judge)",
        ]
        provider_label = st.radio(
            "Select LLM Mode",
            options=mode_options,
            index=1 if default_is_gemini else 0,
        )

        gemini_key_override = ""
        env_key_present = bool(cfg.effective_gemini_key)
        ollama_choices = backends.ollama_models or [cfg.ollama_model, cfg.ollama_judge_model]
        gemini_choices = backends.gemini_models

        def _pick_provider_ui(role_label: str, default_prov: str, key_prefix: str) -> tuple[str, str]:
            prov = st.selectbox(
                f"{role_label} Provider",
                options=["ollama", "gemini"],
                index=0 if default_prov == "ollama" else 1,
                key=f"{key_prefix}_prov",
            )
            if prov == "ollama":
                if backends.ollama_available and backends.ollama_models:
                    default_idx = 0 if key_prefix == "gen" else (
                        backends.ollama_models.index(cfg.ollama_judge_model)
                        if cfg.ollama_judge_model in backends.ollama_models
                        else (1 if len(backends.ollama_models) > 1 else 0)
                    )
                    m_name = st.selectbox(
                        f"{role_label} Model",
                        options=ollama_choices,
                        index=default_idx,
                        key=f"{key_prefix}_model_ol",
                    )
                else:
                    m_name = st.text_input(
                        f"{role_label} Ollama Model",
                        value=cfg.ollama_model if key_prefix == "gen" else cfg.ollama_judge_model,
                        key=f"{key_prefix}_model_ol_txt",
                    )
            else:
                default_idx = 0 if key_prefix == "gen" else (1 if len(gemini_choices) > 1 else 0)
                m_name = st.selectbox(
                    f"{role_label} Model",
                    options=gemini_choices,
                    index=default_idx,
                    key=f"{key_prefix}_model_gm",
                )
            return prov, m_name

        if provider_label.startswith("Ollama"):
            selected_provider = "ollama"
            judge_provider = "ollama"
            if backends.ollama_available and backends.ollama_models:
                st.success(f"Ollama online ({len(backends.ollama_models)} models)")
                gen_model = st.selectbox("Factory & Agent Model", options=ollama_choices, index=0)
                default_j = (
                    ollama_choices.index(cfg.ollama_judge_model)
                    if cfg.ollama_judge_model in ollama_choices
                    else (1 if len(ollama_choices) > 1 else 0)
                )
                judge_model = st.selectbox("Judge Model (Cross-Judge)", options=ollama_choices, index=default_j)
            else:
                st.warning("Ollama server not reachable on localhost:11434. Start `ollama serve` or switch to Gemini/Hybrid.")
                gen_model = st.text_input("Ollama Model", value=cfg.ollama_model)
                judge_model = st.text_input("Ollama Judge Model", value=cfg.ollama_judge_model)
        elif provider_label.startswith("Gemini"):
            selected_provider = "gemini"
            judge_provider = "gemini"
            if env_key_present:
                st.success("GEMINI_API_KEY loaded from .env")
            else:
                st.info("GEMINI_API_KEY not found in .env — enter key below.")
            gemini_key_override = st.text_input(
                "Gemini API Key (optional override)",
                value="",
                type="password",
                placeholder="Using .env GEMINI_API_KEY" if env_key_present else "AIza...",
            )
            gen_model = st.selectbox("Factory & Agent Model", options=gemini_choices, index=0)
            judge_model = st.selectbox(
                "Judge Model (Cross-Judge)",
                options=gemini_choices,
                index=1 if len(gemini_choices) > 1 else 0,
            )
        else:
            st.caption("Run agent generation & runtime on one provider and blind judging on another.")
            if backends.ollama_available:
                st.success(f"Ollama online ({len(backends.ollama_models)} models)")
            else:
                st.warning("Ollama offline on localhost:11434")
            if env_key_present:
                st.success("GEMINI_API_KEY loaded from .env")
            selected_provider, gen_model = _pick_provider_ui("Factory & Agent", "ollama", "gen")
            judge_provider, judge_model = _pick_provider_ui("Judge", "gemini", "jdg")
            if selected_provider == "gemini" or judge_provider == "gemini":
                gemini_key_override = st.text_input(
                    "Gemini API Key (optional override)",
                    value="",
                    type="password",
                    placeholder="Using .env GEMINI_API_KEY" if env_key_present else "AIza...",
                )

        st.divider()
        st.header("Factory Guardrails")
        max_iters = st.slider("Max Optimizer Iterations", 0, 3, 3)
        eval_target = st.slider("Target Auto-Eval Cases", 20, 32, 24, step=2)
        run_tournament = st.checkbox("Run Prompt Tournament (3 candidates)", value=True)
        use_llm_judge = st.checkbox("Enable Atomic NLI LLM Judge", value=True)
        max_tokens = st.number_input("Max Token Budget", min_value=10000, max_value=500000, value=200000, step=10000)

    tab_factory, tab_chat, tab_inspect = st.tabs(
        ["1. Agent Factory", "2. Chat Runner", "3. Regression & Artifacts"]
    )

    # =========================================================================
    # TAB 1: AGENT FACTORY (Intake -> Human Checkpoint -> Build & Optimize)
    # =========================================================================
    with tab_factory:
        col_left, col_right = st.columns([1, 1])
        with col_left:
            source_mode = st.radio(
                "Corpus Source",
                ["Upload Documents", "Use Committed Demo Corpus"],
                horizontal=True,
            )
            docs_path_str = ""
            if source_mode == "Use Committed Demo Corpus":
                demo_choice = st.selectbox(
                    "Select Demo",
                    ["HR Policy Assistant (demos/hr_policy)", "Ship-Spares Procurement (demos/ship_spares)"],
                )
                demo_folder = DEMOS_DIR / ("hr_policy" if "HR" in demo_choice else "ship_spares")
                default_brief = (
                    (demo_folder / "brief.txt").read_text(encoding="utf-8").strip()
                    if (demo_folder / "brief.txt").exists()
                    else ""
                )
                docs_path_str = str(demo_folder / "docs")
                brief_input = st.text_area("One-Paragraph Brief", value=default_brief, height=110)
            else:
                uploaded = st.file_uploader(
                    "Upload PDF, CSV, Markdown, TXT, or JSON files",
                    accept_multiple_files=True,
                    type=["pdf", "csv", "md", "txt", "json"],
                )
                brief_input = st.text_area(
                    "One-Paragraph Brief",
                    placeholder="e.g., I need an assistant that answers HR policy questions for shore staff...",
                    height=110,
                )
                if uploaded:
                    tmp_dir = Path(tempfile.mkdtemp(prefix="forgeai_docs_"))
                    for uf in uploaded:
                        (tmp_dir / uf.name).write_bytes(uf.getvalue())
                    docs_path_str = str(tmp_dir)

            run_name = st.text_input("Output Folder Name", value="custom_agent")

        with col_right:
            st.markdown("#### Step 1: Intake & Groundability Audit")
            if st.button("Run Intake & Groundability Check", type="secondary", use_container_width=True):
                if not brief_input.strip() or not docs_path_str:
                    st.error("Please provide both a brief and documents.")
                else:
                    with st.spinner("Loading documents, generating clarifying questions, and auditing groundability..."):
                        budget = BudgetTracker(max_tokens=int(max_tokens))
                        f_llm, j_llm = build_factory_providers(
                            provider=selected_provider,
                            model_name=gen_model,
                            judge_provider=judge_provider,
                            judge_model_name=judge_model,
                            gemini_api_key=gemini_key_override or None,
                            budget=budget,
                            settings=cfg,
                        )
                        corpus = load_corpus(docs_path_str)
                        qs = generate_clarifying_questions(brief_input.strip(), corpus, f_llm)
                        st.session_state["intake_brief"] = brief_input.strip()
                        st.session_state["intake_docs"] = docs_path_str
                        st.session_state["intake_questions"] = [q.model_dump() for q in qs]
                        st.session_state["intake_answers"] = [q.default_answer for q in qs]
                        # Run groundability + initial requirements
                        req = run_intake_interview(
                            brief_input.strip(),
                            corpus,
                            f_llm,
                            embed_model_name=cfg.embed_model,
                        )
                        st.session_state["draft_req"] = req.model_dump(mode="json")

        # Human Checkpoint UI (Bonus 3)
        if "draft_req" in st.session_state:
            st.divider()
            st.subheader("Human Checkpoint — Clarify & Approve Requirements")
            qs_data = st.session_state.get("intake_questions", [])
            edited_answers = []
            for idx, q in enumerate(qs_data):
                ans = st.text_input(
                    f"Q{idx+1}: {q['question']}",
                    value=st.session_state["intake_answers"][idx],
                    key=f"q_ans_{idx}",
                )
                edited_answers.append(ans)

            req_obj = RequirementsDoc.model_validate(st.session_state["draft_req"])
            g = req_obj.groundability
            if g.overall_status == "ungroundable":
                st.error(f"GROUNDABILITY BLOCK ({g.groundability_ratio:.2f}): Corpus cannot support this brief.")
            elif g.overall_status == "partially_grounded":
                st.warning(f"PARTIAL GROUNDABILITY ({g.groundability_ratio:.2f}): Unsupported items moved to must-refuse.")
            else:
                st.success(f"GROUNDED ({g.groundability_ratio:.2f}): All requested topics backed by corpus.")

            for w in g.warnings:
                st.caption(f"⚠️ {w}")

            c1, c2 = st.columns(2)
            with c1:
                must_ans_txt = st.text_area(
                    "Must-Answer Topics (one per line)",
                    value="\n".join(req_obj.must_answer_topics),
                    height=130,
                )
            with c2:
                must_ref_txt = st.text_area(
                    "Must-Refuse Topics (one per line)",
                    value="\n".join(req_obj.must_refuse_topics),
                    height=130,
                )

            st.write(f"**Selected Tools:** `{req_obj.recommended_tools}` | **Rejected Tools:** `{req_obj.rejected_tools}`")

            if st.button("Approve Checkpoint & Build Agent", type="primary", use_container_width=True):
                if g.overall_status == "ungroundable":
                    st.error("Blocked: ForgeAI will not generate an agent for an ungroundable brief.")
                else:
                    req_obj.must_answer_topics = [x.strip() for x in must_ans_txt.splitlines() if x.strip()]
                    req_obj.must_refuse_topics = [x.strip() for x in must_ref_txt.splitlines() if x.strip()]
                    req_obj.human_checkpoint_status = "approved"
                    for i, qa in enumerate(req_obj.clarifying_qa):
                        if i < len(edited_answers):
                            qa.answer = edited_answers[i]

                    out_dir = RUNS_DIR / (run_name.strip() or "custom_agent")
                    prog = st.progress(0.1, text="Starting Agent Generation...")
                    budget = BudgetTracker(max_tokens=int(max_tokens))
                    f_llm, j_llm = build_factory_providers(
                        provider=selected_provider,
                        model_name=gen_model,
                        judge_provider=judge_provider,
                        judge_model_name=judge_model,
                        gemini_api_key=gemini_key_override or None,
                        budget=budget,
                        settings=cfg,
                    )
                    corpus = load_corpus(st.session_state["intake_docs"])

                    prog.progress(0.3, text="Generating Declarative Agent & Prompt Candidates...")
                    agent, candidates = generate_agent_candidates(
                        req_obj, corpus, f_llm, judge_model_name=j_llm.model_name, num_candidates=3 if run_tournament else 1
                    )
                    prog.progress(0.45, text=f"Generating >= {eval_target} Document-Grounded Eval Cases...")
                    eval_set = generate_eval_set(req_obj, agent, corpus, f_llm, target_cases=int(eval_target))
                    runtime = AgentRuntime(agent, f_llm, corpus=corpus, embed_model_name=cfg.embed_model)

                    if run_tournament and len(candidates) > 1:
                        prog.progress(0.60, text="Running Position-Debiased Prompt Tournament...")
                        agent, candidates = run_prompt_tournament(agent, candidates, eval_set.cases, runtime, j_llm)

                    prog.progress(0.75, text="Running Hybrid Judge & Critic-Optimizer Loop...")
                    final_agent, scorecards, iterations = run_optimization_loop(
                        agent,
                        eval_set.cases,
                        runtime,
                        f_llm,
                        j_llm,
                        budget,
                        max_iterations=int(max_iters),
                        use_llm_judge=use_llm_judge,
                    )
                    report = build_final_report(final_agent, req_obj, scorecards, iterations, candidates, budget)
                    save_bundle_artifacts(
                        out_dir,
                        corpus=corpus,
                        requirements=req_obj,
                        agent=final_agent,
                        eval_set=eval_set,
                        scorecards=scorecards,
                        iterations=iterations,
                        candidates=candidates,
                        report=report,
                    )
                    prog.progress(1.0, text=f"Finished! Saved to {out_dir}")
                    st.success(f"Built {final_agent.name}! Pass rate: {round(report.final_pass_rate*100,1)}%")
                    st.markdown(render_report_markdown(report, final_agent, req_obj))

    # =========================================================================
    # TAB 2: GENERIC CHAT RUNNER (Req 8)
    # =========================================================================
    with tab_chat:
        st.subheader("Generic Runnable Agent Chat")
        bundles = _discover_built_agents()
        if not bundles:
            st.info("No built agent bundles found yet in demos/ or runs/. Build an agent in Tab 1 first.")
        else:
            bundle_labels = [str(p.relative_to(REPO_ROOT)) for p in bundles]
            chosen_idx = st.selectbox(
                "Load Agent Definition Bundle",
                options=list(range(len(bundle_labels))),
                format_func=lambda i: bundle_labels[i],
            )
            chosen_dir = bundles[chosen_idx]
            agent_data = yaml.safe_load((chosen_dir / "agent_definition.yaml").read_text(encoding="utf-8"))
            with st.expander("Inspect Loaded Agent Definition", expanded=False):
                st.json(agent_data)

            if "chat_msgs" not in st.session_state or st.session_state.get("chat_dir") != str(chosen_dir):
                st.session_state["chat_msgs"] = []
                st.session_state["chat_dir"] = str(chosen_dir)

            for msg in st.session_state["chat_msgs"]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg.get("meta"):
                        st.caption(msg["meta"])

            user_q = st.chat_input("Ask the loaded ForgeAI agent...")
            if user_q:
                st.session_state["chat_msgs"].append({"role": "user", "content": user_q})
                with st.chat_message("user"):
                    st.markdown(user_q)

                with st.chat_message("assistant"):
                    with st.spinner("Retrieving context & running agent..."):
                        # Resolve relative docs_dir cleanly if needed
                        docs_override = None
                        raw_docs = Path(agent_data.get("docs_dir", ""))
                        if not raw_docs.exists():
                            candidate_docs = chosen_dir.parent / "docs"
                            if candidate_docs.exists():
                                docs_override = candidate_docs

                        rt = AgentRuntime.from_directory(
                            chosen_dir,
                            provider=selected_provider,
                            model_name=gen_model,
                            gemini_api_key=gemini_key_override or None,
                            docs_dir_override=docs_override,
                            settings=cfg,
                        )
                        trace = rt.ask(user_q, chat_history=st.session_state["chat_msgs"][:-1])
                        r = trace.response
                        badge = "🚫 REFUSED" if r.refused else "✅ ANSWERED"
                        st.markdown(f"**[{badge} | confidence: {r.confidence}]**\n\n{r.answer}")
                        cits_str = ", ".join(f"`{c.get('source')}`" for c in r.citations) if r.citations else "none"
                        meta = f"Citations: {cits_str} | Tools: {r.tools_used or ['none']} | Latency: {trace.latency_ms:.0f}ms"
                        st.caption(meta)
                        st.session_state["chat_msgs"].append(
                            {"role": "assistant", "content": r.answer, "meta": meta}
                        )

    # =========================================================================
    # TAB 3: REGRESSION & ARTIFACTS INSPECTOR (Bonus 4 & 5)
    # =========================================================================
    with tab_inspect:
        st.subheader("Inspect Reports, Baseline Comparisons & Regression Mode")
        bundles = _discover_built_agents()
        if bundles:
            bundle_labels = [str(p.relative_to(REPO_ROOT)) for p in bundles]
            sel_i = st.selectbox(
                "Select Agent Bundle to Inspect",
                options=list(range(len(bundle_labels))),
                format_func=lambda i: bundle_labels[i],
                key="inspect_sel",
            )
            sel_dir = bundles[sel_i]
            for fname in ("report.md", "scorecard.md", "iteration_log.md", "baseline_comparison.md"):
                fp = sel_dir / fname
                if fp.exists():
                    with st.expander(fname, expanded=(fname == "report.md")):
                        st.markdown(fp.read_text(encoding="utf-8"))

            st.divider()
            st.markdown("#### Run Regression Mode on Updated Documents")
            reg_uploads = st.file_uploader(
                "Upload updated document set to test against the existing eval_set.json",
                accept_multiple_files=True,
                type=["pdf", "csv", "md", "txt", "json"],
                key="reg_upload",
            )
            if reg_uploads and st.button("Run Regression Check"):
                reg_docs = Path(tempfile.mkdtemp(prefix="forgeai_reg_docs_"))
                for uf in reg_uploads:
                    (reg_docs / uf.name).write_bytes(uf.getvalue())
                reg_out = RUNS_DIR / f"regression_{sel_dir.name}"
                with st.spinner("Running regression check..."):
                    diff, new_card = run_regression_check(
                        agent_dir=sel_dir,
                        new_docs_dir=reg_docs,
                        out_dir=reg_out,
                        provider=selected_provider,
                        model_name=gen_model,
                        judge_provider=judge_provider,
                        judge_model_name=judge_model,
                        gemini_api_key=gemini_key_override or None,
                        use_llm_judge=use_llm_judge,
                        settings=cfg,
                    )
                st.markdown((reg_out / "regression_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
