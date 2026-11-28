"""Tests for AgentRuntime portable loading, end-to-end persistence, and Regression Mode."""

from __future__ import annotations

import json
from pathlib import Path

from forgeai.factory.pipeline import run_forge_pipeline
from forgeai.factory.regression import run_regression_check
from forgeai.llm.base import LLMProvider, LLMResponse
from forgeai.runtime.executor import AgentRuntime


class _DeterministicFactoryLLM(LLMProvider):
    provider_name = "ollama"

    def complete(self, messages, *, temperature=0.1, max_tokens=1200, json_mode=False, json_schema=None):
        user = messages[-1]["content"]
        if "RETRIEVED CONTEXT:" in user:
            if "cryptocurrency" in user.lower() or "ceo" in user.lower() or "stock" in user.lower() or "official rules" in user.lower() or "recommend" in user.lower() or "diagnose" in user.lower() or "london" in user.lower() or "crude oil" in user.lower() or "wire transfer" in user.lower():
                out = {"action": "final", "answer": "I cannot answer this based on the provided documents.", "citations": [], "refused": True, "refusal_reason": "Out of scope", "confidence": "high", "tools_used": []}
            elif "30 days" in user.lower() or "leave" in user.lower() or "hr.md" in user.lower():
                if "[NO RETRIEVED CHUNKS" in user:
                    out = {"action": "final", "answer": "I cannot answer this based on the provided documents.", "citations": [], "refused": True, "refusal_reason": "Missing from updated docs", "confidence": "high", "tools_used": []}
                else:
                    out = {"action": "final", "answer": "Shore staff receive 24 days of annual leave and 12 sick days.", "citations": [{"source": "hr.md", "excerpt": "24 days"}], "refused": False, "confidence": "high", "tools_used": []}
            else:
                out = {"action": "final", "answer": "Domestic meal allowance is USD 55 per day.", "citations": [{"source": "hr.md", "excerpt": "USD 55"}], "refused": False, "confidence": "high", "tools_used": []}
            return LLMResponse(text=json.dumps(out), model=self.model_name, provider="ollama", prompt_tokens=25, completion_tokens=20, total_tokens=45)
        return LLMResponse(text="{}", model=self.model_name, provider="ollama", prompt_tokens=10, completion_tokens=5, total_tokens=15)


def test_pipeline_and_regression_mode(monkeypatch, tmp_path: Path) -> None:
    docs_v1 = tmp_path / "docs_v1"
    docs_v1.mkdir()
    (docs_v1 / "hr.md").write_text(
        "# Annual Leave\nShore staff receive 24 paid annual leave days per year.\n"
        "# Sick Leave\nEmployees get 12 paid sick leave days annually.\n"
        "# Meal Cap\nDomestic travel meal reimbursement is capped at USD 55 per day.\n",
        encoding="utf-8",
    )
    out_v1 = tmp_path / "agent_v1"
    dummy_llm = _DeterministicFactoryLLM("qwen2.5:1.5b")

    monkeypatch.setattr(
        "forgeai.factory.pipeline.build_factory_providers",
        lambda **kwargs: (dummy_llm, dummy_llm),
    )
    monkeypatch.setattr(
        "forgeai.factory.regression.build_factory_providers",
        lambda **kwargs: (dummy_llm, dummy_llm),
    )
    monkeypatch.setattr(
        "forgeai.runtime.executor.build_provider",
        lambda **kwargs: dummy_llm,
    )

    res = run_forge_pipeline(
        brief="Answer HR annual leave, sick leave, and meal allowance questions for shore staff.",
        docs_dir=docs_v1,
        out_dir=out_v1,
        unattended=True,
        run_tournament=False,
        max_iterations=1,
        eval_target_cases=20,
        use_llm_judge=False,
    )
    assert res.eval_set is not None
    assert res.eval_set.total_cases >= 20
    assert (out_v1 / "agent_definition.yaml").exists()
    assert (out_v1 / "report.md").exists()

    # Load generically via AgentRuntime.from_directory
    rt = AgentRuntime.from_directory(out_v1)
    tr = rt.ask("How many annual leave days do shore staff receive?")
    assert not tr.response.refused

    # Regression mode: remove Annual Leave from updated docs -> flags now_ungroundable / broken!
    docs_v2 = tmp_path / "docs_v2"
    docs_v2.mkdir()
    (docs_v2 / "hr.md").write_text(
        "# Meal Cap\nDomestic travel meal reimbursement is capped at USD 55 per day.\n",
        encoding="utf-8",
    )
    reg_out = tmp_path / "reg_out"
    diff, _ = run_regression_check(
        agent_dir=out_v1,
        new_docs_dir=docs_v2,
        out_dir=reg_out,
        use_llm_judge=False,
    )
    assert diff.modified_files == ["hr.md"]
    assert len(diff.now_ungroundable) >= 1
    assert (reg_out / "regression_report.md").exists()


def test_runtime_executes_declarative_custom_macro_end_to_end(monkeypatch, tmp_path: Path) -> None:
    from forgeai.schemas.agent import AgentDefinition, CustomToolSpec, RetrievalConfig

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "pricing.md").write_text(
        "# Bulk Order Rule\nBulk orders get a discount percentage off unit price plus shipping.\n",
        encoding="utf-8",
    )

    class _MacroRuntimeLLM(LLMProvider):
        provider_name = "ollama"

        def complete(self, messages, *, temperature=0.1, max_tokens=1200, json_mode=False, json_schema=None):
            last = messages[-1]["content"]
            if "TOOL OBSERVATION (calc_bulk_quote):" in last:
                out = {
                    "action": "final",
                    "answer": "The bulk quote total is USD 980 based on pricing.md.",
                    "citations": [{"source": "pricing.md", "excerpt": "Bulk orders get a discount percentage"}],
                    "refused": False,
                    "confidence": "high",
                    "tools_used": ["calc_bulk_quote"],
                }
            else:
                out = {
                    "action": "tool_call",
                    "tool": "calc_bulk_quote",
                    "args": {"unit_price": 100, "qty": 10, "discount_pct": 10, "shipping": 80},
                }
            return LLMResponse(text=json.dumps(out), model="dummy", provider="ollama", prompt_tokens=15, completion_tokens=15, total_tokens=30)

    agent = AgentDefinition(
        agent_id="macro-test",
        name="Macro Runtime Agent",
        brief="Calculate bulk order quotes",
        provider="ollama",
        model_name="dummy",
        judge_model_name="dummy",
        system_prompt="Use calc_bulk_quote when asked for bulk quotes.",
        retrieval=RetrievalConfig(top_k=3, min_score_threshold=0.0),
        custom_tools=[
            CustomToolSpec(
                name="calc_bulk_quote",
                description="Calculate bulk order landed quote.",
                tool_type="formula",
                parameters=["unit_price", "qty", "discount_pct", "shipping"],
                expression="unit_price * qty * (1 - discount_pct / 100) + shipping",
                source_document="pricing.md",
            )
        ],
        docs_dir=str(docs_dir),
        created_at="2026-11-28T00:00:00Z",
    )

    rt = AgentRuntime(agent, _MacroRuntimeLLM("dummy"), docs_dir_override=docs_dir)
    trace = rt.ask("What is the bulk quote for 10 units at 100 USD each with 10% discount and 80 USD shipping?")
    assert not trace.response.refused
    assert "calc_bulk_quote" in trace.response.tools_used
    assert any(t["tool"] == "calc_bulk_quote" and "980" in t["output"] for t in trace.tool_outputs)

