# ForgeAI — The Agent That Builds Agents

**ForgeAI** is an autonomous Agent Factory that takes a one-paragraph brief and a folder of documents, interviews the requester, verifies groundability, generates a declarative agent definition (system prompt, hybrid retrieval parameters, pruned tool bindings, output contract), synthesizes a >=20-case evaluation suite from the documents, runs a debiased hybrid Judge, iteratively optimizes failures via targeted patches (max 3 iterations), and ships a runnable product with an honest quality report.

## Features

### Core Architecture (All 9 Mandatory Requirements)
- **Intake Interviewer**: Reads brief + 17-format corpus inventory (`.pdf`, `.docx`, `.xlsx`, `.csv`, `.tsv`, `.md`, `.txt`, `.log`, `.rst`, `.json`, `.jsonl`, `.yaml`/`.yml`, `.html`/`.htm`, `.xml`), asks <=5 clarifying questions (interactive CLI/UI or `--unattended`), outputs structured `requirements.yaml`.
- **Groundability Guard**: Verifies requested scope against indexed corpus evidence; moves unsupported asks into `must_refuse` (or safely halts 0%-grounded briefs) so the generated agent will not hallucinate missing coverage.
- **Declarative Agent Generator & Custom Macros**: Emits `agent_definition.yaml/.json` with prompt, chunking/hybrid-retrieval config, strictly pruned tool selection from a **10-tool sandboxed registry** (`csv_lookup`, `json_record_lookup`, `calculator`, `statistics_calculator`, `unit_converter`, `currency_converter`, `open_meteo_weather`, `http_public_country_info`, `text_stats_and_regex`, `current_date`), **AST-verified declarative Custom Macro Tools** (`formula` / `tabular_filter`, zero `exec`), and JSON output contract.
- **Document-Grounded Eval Generator**: Auto-generates >=20 verified test cases (answerable, out-of-scope refusals, multi-doc/ambiguous/tool edge cases) using committed Jinja2 prompt templates (`src/forgeai/prompts/`).
- **Debiased Hybrid Judge**: Combines deterministic contract/citation-excerpt/numeric/refusal gates with atomic NLI fact verification, cross-model option, position-swapped tournament checks, and verbosity penalties.
- **Targeted Critic–Optimizer**: Classifies failures (`retrieval_miss`, `prompt_defect`, `hallucination`, `wrong_refusal`), applies minimal diff patches (<=3 iterations) with rollback protection, and logs every before/after change.
- **Honest Quality Reporting**: Generates `scorecard.md`, `iteration_log.md`, and `report.md` detailing exact passing rates and *what still fails and why*.
- **Generic Runnable Agent**: CLI (`forgeai chat`), zero-server Streamlit UI (`forgeai ui`, 3-mode Ollama/Gemini/Hybrid), and optional FastAPI REST backend (`forgeai serve`) capable of loading any produced `agent_definition` folder with live tool execution and citations.
- **Two Committed End-to-End Demos** (run locally on `qwen2.5:3b` generator/agent + `qwen2.5:1.5b` cross-judge):
  1. `demos/hr_policy/` — **HR Policy Assistant (`ShoreHRPolicyAssistant`)** over 3 multi-page HR PDFs: achieved **`79.2% (19/24)`** pass rate (`0.854` composite, `1.000` contract adherence; `calculator` & `current_date` attached, all 8 off-domain/ungrounded tools rejected), matching the hand-written Human Baseline (`79.2%` vs `79.2%`, ForgeAI beating Human on `eval_04` and Human beating ForgeAI on `eval_19` in [baseline_comparison.md](demos/hr_policy/output/baseline_comparison.md)), plus committed Regression Mode verification (`79.2% -> 33.3%`, 11 newly broken / 4 now ungroundable in [regression_report.md](demos/hr_policy/output/regression_demo/regression_report.md)). See [HR Report](demos/hr_policy/output/report.md).
  2. `demos/ship_spares/` — **Ship-Spares Procurement Assistant (`Ship_Spares_Agent`)** over `spares_catalogue.csv` + 3 supplier/spec/freight PDFs: improved from `58.3% (14/24)` (`0.791`) -> **`70.8% (17/24)`** (`0.810` composite) via accepted Iteration 1 optimization (`csv_lookup`, `calculator`, `unit_converter`, `current_date` attached + 2 AST-verified custom macros `lookup_critical_spares` & `calculate_total_landed_cost`; 6 off-domain tools rejected). See [Ship-Spares Report](demos/ship_spares/output/report.md).

### Bonus Capabilities Included
1. **Prompt Tournament**: Competes 3 candidate prompt strategies using debiased evaluation before optimization.
2. **Budget Enforcement**: Token + latency budget ceilings and marginal-gain early stopping.
3. **Human Checkpoint**: Interactive approval/editing of `requirements.yaml` before generation.
4. **Factory vs Human Baseline**: Objective head-to-head benchmark in `baselines/`.
5. **Regression Mode**: `forgeai regress` detects SHA-256 document changes and evaluates what broke on the prior eval suite.
6. **Streamlit Web UI & Cloud Deployment**: 3-mode Execution Selector (`Ollama`, `Gemini API`, `Hybrid`), 17-format upload, Custom Macro display, scorecard charts, and chat trace inspector packaged for Hugging Face Spaces / Streamlit Community Cloud (`forgeai ui`) plus optional REST API (`forgeai serve`).

---

## Quickstart

### 1. Prerequisites
- Python 3.11–3.13 (`uv` recommended)
- **Local Open Model (Recommended offline/review)**: [Ollama](https://ollama.com/) with `qwen2.5:3b` and/or `qwen2.5:1.5b`
- **Cloud API (Optional)**: `GEMINI_API_KEY` in `.env` (uses `google-genai` Interactions API)

### 2. Installation
```bash
git clone https://github.com/ShaikMohammad786/ForgeAI.git
cd ForgeAI

uv venv --python 3.12 .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
# source .venv/bin/activate

uv pip install -e ".[dev]"
cp .env.example .env
```

### 3. CLI Usage
```bash
# Run unattended on an unseen brief + docs folder (e.g. using local Ollama)
forgeai build --brief "I need an assistant that answers HR policy questions for shore staff" \
              --docs ./demos/hr_policy/docs \
              --out ./runs/hr_test \
              --provider ollama \
              --model qwen2.5:3b \
              --judge-model qwen2.5:1.5b \
              --unattended

# Interactive mode (asks <=5 clarifying questions + human checkpoint)
forgeai build --brief-file ./demos/ship_spares/brief.txt --docs ./demos/ship_spares/docs --out ./runs/spares

# Run using Gemini Interactions API (.env GEMINI_API_KEY)
forgeai build --brief-file ./demos/hr_policy/brief.txt --docs ./demos/hr_policy/docs --out ./runs/hr_gemini --provider gemini --unattended

# Chat with any generated agent
forgeai chat ./demos/hr_policy/output

# Run regression check when documents change
forgeai regress --agent ./demos/hr_policy/output --docs ./updated_docs --out ./runs/hr_regression

# Launch Streamlit Factory + Chat Runner UI (zero separate server needed)
forgeai ui

# Optional: Launch FastAPI REST backend with OpenAPI docs at /docs
forgeai serve --port 8000
```

### 4. Hosting (Hugging Face Spaces / Streamlit Community Cloud)
- **Streamlit Community Cloud**: Point app file to `src/forgeai/ui/app.py`, install `requirements.txt`, add `GEMINI_API_KEY` in Secrets (or run against remote `OLLAMA_HOST`).
- **Hugging Face Spaces (Docker)**: Select Docker SDK; the included `Dockerfile` launches Streamlit on port `7860`.

---

## Project Structure
See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/forgeai_architecture.png](docs/forgeai_architecture.png) for architecture details and the end-to-end flowchart.
