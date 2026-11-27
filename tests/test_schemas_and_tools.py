"""Unit tests for schemas, JSON repair, BudgetTracker, and sandboxed tools."""

from __future__ import annotations

from pathlib import Path

import pytest

from forgeai.llm.base import BudgetExceededError, BudgetTracker, LLMResponse, parse_json_dict
from forgeai.schemas.agent import ToolBinding
from forgeai.tools.registry import ToolAuthorizationError, execute_tool


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
