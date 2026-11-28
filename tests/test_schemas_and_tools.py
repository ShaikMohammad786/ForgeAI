"""Unit tests for schemas, JSON repair, BudgetTracker, and sandboxed tools."""

from __future__ import annotations

from pathlib import Path

import pytest

from forgeai.llm.base import BudgetExceededError, BudgetTracker, LLMResponse, parse_json_dict
from forgeai.schemas.agent import CustomToolSpec, ToolBinding
from forgeai.tools import (
    TOOL_CATALOG,
    ToolAuthorizationError,
    execute_tool,
    validate_custom_tool_spec,
)


def test_parse_json_dict_repairs_fenced_and_trailing_commas() -> None:
    noisy = 'Sure, here is the JSON:\n```json\n{"answer": "24 days", "refused": false,}\n```\nHope this helps!'
    parsed = parse_json_dict(noisy)
    assert parsed["answer"] == "24 days"
    assert parsed["refused"] is False


def test_budget_tracker_enforces_token_cap() -> None:
    budget = BudgetTracker(max_tokens=100, max_seconds=60.0)
    budget.record(
        LLMResponse(
            text="hello",
            model="m",
            provider="p",
            prompt_tokens=60,
            completion_tokens=45,
            total_tokens=105,
        )
    )
    with pytest.raises(BudgetExceededError):
        budget.check_before_call()


def test_calculator_ast_safety_and_math(tmp_path: Path) -> None:
    binding = [ToolBinding(name="calculator", description="calc", justification="test")]
    ok = execute_tool(
        "calculator",
        {"expression": "(640 * 10) * 0.95 + 140"},
        authorized_tools=binding,
        docs_dir=tmp_path,
    )
    assert "6220" in ok

    bad = execute_tool(
        "calculator",
        {"expression": "__import__('os').system('echo pwned')"},
        authorized_tools=binding,
        docs_dir=tmp_path,
    )
    assert "error" in bad.lower()


def test_tool_authorization_blocks_unattached_tools(tmp_path: Path) -> None:
    only_date = [ToolBinding(name="current_date", description="date", justification="test")]
    out = execute_tool("current_date", {"add_days": 7}, authorized_tools=only_date, docs_dir=tmp_path)
    assert "target_date" in out

    with pytest.raises(ToolAuthorizationError):
        execute_tool(
            "open_meteo_weather",
            {"location": "Singapore"},
            authorized_tools=only_date,
            docs_dir=tmp_path,
        )


def test_csv_lookup_filters_and_aggregates(tmp_path: Path) -> None:
    (tmp_path / "parts.csv").write_text(
        "part_id,supplier_id,unit_price_usd,criticality\n"
        "SP-1,SUP-A,100,CRITICAL\n"
        "SP-2,SUP-B,250,STANDARD\n"
        "SP-3,SUP-A,400,CRITICAL\n",
        encoding="utf-8",
    )
    binding = [ToolBinding(name="csv_lookup", description="csv", justification="test")]
    res = execute_tool(
        "csv_lookup",
        {"filters": {"supplier_id": "SUP-A"}, "aggregate": "sum", "agg_column": "unit_price_usd"},
        authorized_tools=binding,
        docs_dir=tmp_path,
    )
    assert "500" in res


def test_unit_converter_and_json_record_lookup(tmp_path: Path) -> None:
    (tmp_path / "vessels.json").write_text(
        '[{"imo": "9312456", "name": "MV Oceanic Star", "port": "Singapore"}]',
        encoding="utf-8",
    )
    bindings = [
        ToolBinding(name="unit_converter", description="units", justification="test"),
        ToolBinding(name="json_record_lookup", description="json", justification="test"),
    ]
    u_out = execute_tool(
        "unit_converter",
        {"value": 100, "from_unit": "bar", "to_unit": "psi"},
        authorized_tools=bindings,
        docs_dir=tmp_path,
    )
    assert "1450.37" in u_out

    j_out = execute_tool(
        "json_record_lookup",
        {"query": "Oceanic"},
        authorized_tools=bindings,
        docs_dir=tmp_path,
    )
    assert "9312456" in j_out


def test_extended_tabular_and_structured_record_lookups(tmp_path: Path) -> None:
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["sku", "price"])
    ws.append(["MP-77", 350])
    wb.save(str(tmp_path / "catalog.xlsx"))

    (tmp_path / "ports.tsv").write_text("port\tfee\nRotterdam\t620\n", encoding="utf-8")
    (tmp_path / "events.jsonl").write_text('{"id": 10, "msg": "Drydock inspection"}\n', encoding="utf-8")
    (tmp_path / "rules.yaml").write_text("sla:\n  max_hours: 48\n", encoding="utf-8")

    bindings = [
        ToolBinding(name="csv_lookup", description="tabular", justification="test"),
        ToolBinding(name="json_record_lookup", description="records", justification="test"),
    ]
    x_out = execute_tool("csv_lookup", {"filename": "catalog.xlsx", "query": "MP-77"}, authorized_tools=bindings, docs_dir=tmp_path)
    assert "350" in x_out

    t_out = execute_tool("csv_lookup", {"filename": "ports.tsv", "query": "Rotterdam"}, authorized_tools=bindings, docs_dir=tmp_path)
    assert "620" in t_out

    l_out = execute_tool("json_record_lookup", {"filename": "events.jsonl", "query": "Drydock"}, authorized_tools=bindings, docs_dir=tmp_path)
    assert "Drydock inspection" in l_out

    y_out = execute_tool("json_record_lookup", {"filename": "rules.yaml", "query": "48"}, authorized_tools=bindings, docs_dir=tmp_path)
    assert "48" in y_out


def test_statistics_calculator_and_text_stats_regex(tmp_path: Path) -> None:
    (tmp_path / "engine.log").write_text(
        "2026-11-20 INFO Startup complete\n"
        "2026-11-21 ERROR Pump pressure low\n"
        "2026-11-22 ERROR Oil temp high\n",
        encoding="utf-8",
    )
    bindings = [
        ToolBinding(name="statistics_calculator", description="stats", justification="test"),
        ToolBinding(name="text_stats_and_regex", description="text", justification="test"),
    ]
    s_out = execute_tool(
        "statistics_calculator",
        {"values": [10, 20, 30], "weights": [1, 2, 1]},
        authorized_tools=bindings,
        docs_dir=tmp_path,
    )
    assert '"mean": 20.0' in s_out
    assert '"weighted_avg": 20.0' in s_out

    pct_out = execute_tool(
        "statistics_calculator",
        {"old_value": 100, "new_value": 125},
        authorized_tools=bindings,
        docs_dir=tmp_path,
    )
    assert '"percent_change": 25.0' in pct_out

    r_out = execute_tool(
        "text_stats_and_regex",
        {"pattern": "ERROR", "filename": "engine.log"},
        authorized_tools=bindings,
        docs_dir=tmp_path,
    )
    assert '"total_matches": 2' in r_out


def test_custom_macro_ast_validation_and_execution(tmp_path: Path) -> None:
    (tmp_path / "catalog.csv").write_text("sku,criticality,price\nA1,CRITICAL,900\nB2,LOW,50\n", encoding="utf-8")
    formula_macro = CustomToolSpec(
        name="calc_landed_cost",
        description="Compute bulk quote with discount and shipping.",
        tool_type="formula",
        parameters=["unit_price", "qty", "discount_pct", "shipping"],
        expression="unit_price * qty * (1 - discount_pct / 100) + shipping",
    )
    ok, msg = validate_custom_tool_spec(
        formula_macro,
        docs_dir=tmp_path,
        reserved_names=set(TOOL_CATALOG.keys()),
        has_tabular=True,
    )
    assert ok, msg

    out = execute_tool(
        "calc_landed_cost",
        {"unit_price": 500, "qty": 4, "discount_pct": 10, "shipping": 120},
        authorized_tools=[],
        custom_tools=[formula_macro],
        docs_dir=tmp_path,
    )
    assert "1920" in out

    # Reject unsafe AST, undeclared variables, and reserved name collision
    evil = CustomToolSpec(
        name="bad_exec",
        description="bad",
        tool_type="formula",
        parameters=["x"],
        expression="__import__('os').system('ls')",
    )
    ok_evil, _ = validate_custom_tool_spec(evil, docs_dir=tmp_path, reserved_names=set(TOOL_CATALOG.keys()), has_tabular=False)
    assert ok_evil is False

    collision = CustomToolSpec(
        name="calculator",
        description="collision",
        tool_type="formula",
        parameters=["x"],
        expression="x + 1",
    )
    ok_col, _ = validate_custom_tool_spec(collision, docs_dir=tmp_path, reserved_names=set(TOOL_CATALOG.keys()), has_tabular=False)
    assert ok_col is False

    # Valid tabular filter preset
    tab_macro = CustomToolSpec(
        name="critical_spares_only",
        description="List only CRITICAL spares",
        tool_type="tabular_filter",
        target_file="catalog.csv",
        preset_filters={"criticality": "CRITICAL"},
    )
    ok_tab, msg_tab = validate_custom_tool_spec(
        tab_macro,
        docs_dir=tmp_path,
        reserved_names=set(TOOL_CATALOG.keys()),
        has_tabular=True,
    )
    assert ok_tab, msg_tab
    tab_out = execute_tool(
        "critical_spares_only",
        {},
        authorized_tools=[],
        custom_tools=[tab_macro],
        docs_dir=tmp_path,
    )
    assert "A1" in tab_out and "B2" not in tab_out


def test_custom_ollama_tunnel_host_propagation(monkeypatch) -> None:
    from forgeai.llm.ollama_client import OllamaProvider
    from forgeai.llm.router import build_factory_providers, discover_backends

    seen_hosts: list[str] = []

    def fake_is_avail(host: str = "") -> bool:
        seen_hosts.append(host)
        return True

    def fake_list_models(host: str = "") -> list[str]:
        seen_hosts.append(host)
        return ["llama3.2:3b", "qwen2.5:1.5b"]

    monkeypatch.setattr(OllamaProvider, "is_available", staticmethod(fake_is_avail))
    monkeypatch.setattr(OllamaProvider, "list_local_models", staticmethod(fake_list_models))

    tunnel = "https://user-gpu.trycloudflare.com/"
    status = discover_backends(ollama_host=tunnel)
    assert status.ollama_available is True
    assert status.ollama_models == ["llama3.2:3b", "qwen2.5:1.5b"]
    assert all(h == "https://user-gpu.trycloudflare.com" for h in seen_hosts)

    gen, judge = build_factory_providers(provider="ollama", ollama_host=tunnel)
    assert gen.host == "https://user-gpu.trycloudflare.com"
    assert judge.host == "https://user-gpu.trycloudflare.com"


