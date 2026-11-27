# ForgeAI Complete Specification & Master Prompt (A–Z)

This document serves two purposes:
1. Complete engineering specification verifying every core & bonus requirement of the OceanAI GenAI Assignment.
2. A self-contained master prompt specification that reproduces or extends ForgeAI end-to-end.

---

## Master Prompt Specification

> **Role & Objective**: Build **ForgeAI**, a production-grade autonomous Agent Factory in Python that converts a one-paragraph brief plus a folder of unstructured/structured documents into a declarative, evaluated, iteratively optimized, and runnable AI agent — while operating reliably on small open models (`1.5B–3B` via Ollama) as well as cloud APIs (Google Gemini Interactions API).

### Mandatory Core Capabilities (All 9)

1. **Intake Interviewer (`forgeai/factory/intake.py`)**
   - Reads the one-paragraph brief and scans the document directory (PDF, CSV, Markdown, TXT, JSON).
   - Generates up to **5** high-signal clarifying questions targeting real ambiguities (audience, refusal edge cases, citation verbosity, numeric precision, tool intent).
   - Supports **both** interactive CLI/UI dialogue AND unattended mode (`--unattended`, self-answering conservatively from corpus context while logging assumptions transparently).
   - Outputs `requirements.yaml` containing: `scope`, `audience`, `tone`, `must_answer_topics`, `must_refuse_topics`, `clarifying_qa`, `recommended_tools`, and `groundability_gaps`.

2. **Groundability Guard (`forgeai/factory/groundability.py`)**
   - Extracts requested capabilities/topics from the brief and verifies each against the indexed document corpus (BM25 + dense vector scores + CSV column inspection + LLM evidence check).
   - Any requested topic lacking sufficient document evidence is flagged with `supported: false`, evidence checked, and moved into `must_refuse_topics` so the generated agent never hallucinates missing knowledge.
   - If the entire brief is ungroundable (`groundability_ratio == 0`), flags critical block cleanly instead of building a hallucinating agent.

3. **Agent Generator (`forgeai/factory/generator.py`)**
   - Emits `agent_definition.yaml` / `.json` containing:
     - `system_prompt` (role, strict grounding rules, refusal rules, citation mandate, output contract rules)
     - `retrieval` (`chunk_size`, `chunk_overlap`, `top_k`, `hybrid_alpha`, `min_score_threshold`, `chunking_strategy`)
     - `tools` (chosen from `ToolRegistry`: `csv_lookup`, `calculator`, `open_meteo_weather`, `current_date`)
     - `output_contract` (JSON schema: `answer`, `citations`, `refused`, `refusal_reason`, `confidence`, `tools_used`)
   - **Strict Tool Pruning**: Deterministic capability filters block irrelevant tools before selection; an HR assistant never attaches `open_meteo_weather`, and a text-only corpus never attaches `csv_lookup`.

4. **Eval Generator (`forgeai/factory/eval_gen.py`)**
   - Dynamically generates **>= 20** test cases from actual document chunks and groundability gaps (never hand-written), storing all generation prompts under `src/forgeai/prompts/`.
   - Balanced categories:
     - `answerable` (grounded in specific source doc + verified `key_facts`)
     - `out_of_scope` (explicitly unanswerable / groundability gaps that must be refused)
     - `edge_case` (ambiguous phrasing, multi-document synthesis spanning >=2 docs, numeric/tool queries)
   - Runs a grounding verification pass over generated `key_facts` so bogus/unverified auto-eval cases are filtered or regenerated.

5. **Hybrid Debiased Judge (`forgeai/factory/judge.py`)**
   - Combines deterministic automated checks (contract validity, citation doc existence, chunk lexical overlap, exact numeric/SKU accuracy, refusal polarity, verbosity ratio) with atomic fact-by-fact NLI LLM evaluation.
   - Explicitly defends against:
     - **Position Bias**: Forward + swapped `(A, B)` & `(B, A)` evaluation in pairwise tournament comparisons.
     - **Self-Preference Bias**: Strips prompt/identity metadata and allows separate judge model (`judge_model != agent_model`).
     - **Verbosity Bias**: Atomic checklist scoring + fluff/unsupported-claim penalties rather than length-correlated rubric scores.

6. **Critic–Optimizer Loop (`forgeai/factory/optimizer.py`)**
   - Evaluates current `AgentDefinition`, classifies every failed case into `{retrieval_miss, prompt_defect, hallucination, wrong_refusal}`.
   - Applies a **targeted patch** (no full prompt/config wipe) to address the highest-impact failure cluster:
     - `retrieval_miss` -> adjust `top_k`, `chunk_size`/`overlap`, `hybrid_alpha`, `min_score_threshold`.
     - `prompt_defect` -> targeted surgical amendment to `system_prompt` instructions or output formatting rules.
     - `hallucination` -> add strict negative constraint, reduce `temperature`, tighten `min_score_threshold`.
     - `wrong_refusal` -> refine `must_answer` vs `must_refuse` boundary rules and abstention threshold.
   - Maximum **3 iterations**. Logs every patch (`what_changed`, `why`, `failure_ids_addressed`, `score_before`, `score_after`, `accepted/rolled_back`).

7. **Honest Reporting (`forgeai/factory/reporter.py`)**
   - Produces `scorecard.json/.md`, `iteration_log.json/.md`, and `report.md`.
   - Explicitly enumerates **what still fails and why**, category-by-category metrics, groundability caveats, tournament comparison, and budget utilization.

8. **Runnable Product (`forgeai/runtime/`)**
   - Generic agent executor (`AgentRuntime`) capable of loading ANY produced `agent_definition` directory, building/loading its hybrid document index, binding only its authorized tools, executing multi-step tool calls, and validating/repairing output against `output_contract`.
   - Exposed via CLI (`forgeai chat <agent_dir>`) and Streamlit UI.

9. **Two Full Demonstrations (`demos/hr_policy/`, `demos/ship_spares/`)**
   - Demo 1: HR Policy Assistant over 3 multi-page HR PDFs.
   - Demo 2: Ship-Spares Procurement Assistant over `spares_catalogue.csv` + 3 supplier/spec/freight PDFs.
   - All synthesis prompts, source docs, and complete output artifacts committed in repository.

### Bonus Capabilities (All 6 Included)

1. **Prompt Tournament**: Generates 3 distinct system-prompt strategies, benchmarks each with debiased pairwise + rubric scoring, selects winner, records table in `report.md`.
2. **Budget Enforcement**: Tracks token usage + wall-clock latency + LLM calls; stops early if budget ceiling hit or marginal iteration gain `< min_gain_threshold`.
3. **Human Checkpoint**: Interactive CLI and UI gate allowing requester to review/edit `requirements.yaml` before generation proceeds.
4. **Factory vs Human Baseline (`baselines/`)**: Hand-authored HR Policy agent evaluated under the exact same Judge + Eval set with head-to-head comparison report.
5. **Regression Mode (`forgeai regress`)**: Detects document SHA-256 diffs, updates index/agent, re-runs original eval set, and reports `newly_broken`, `newly_fixed`, `still_failing`, and `now_ungroundable`.
6. **UI & Hosting (`src/forgeai/ui/app.py` + `Dockerfile` + `requirements.txt`)**: Streamlit multipage interface for Factory Pipeline + Generic Chat Runner + Model Provider Switcher (Ollama / Gemini API from `.env`), ready for Hugging Face Spaces or Streamlit Community Cloud.
