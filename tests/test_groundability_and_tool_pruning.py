"""Tests for Groundability Guard and strict capability-based tool pruning."""

from __future__ import annotations

from pathlib import Path

from forgeai.factory.groundability import audit_groundability
from forgeai.factory.intake import run_intake_interview
from forgeai.ingestion import load_corpus
from forgeai.llm.base import LLMProvider, LLMResponse
from forgeai.tools.registry import evaluate_tool_eligibility


class _NeutralLLM(LLMProvider):
    provider_name = "test"

    def complete(self, messages, *, temperature=0.1, max_tokens=1200, json_mode=False, json_schema=None):
        return LLMResponse(text="{}", model=self.model_name, provider="test", prompt_tokens=5, completion_tokens=5, total_tokens=10)


def test_hr_brief_never_attaches_weather_or_csv_tools(tmp_path: Path) -> None:
    (tmp_path / "policy.txt").write_text(
        "Shore staff get 24 paid annual leave days and USD 55 daily domestic meal allowance.",
        encoding="utf-8",
    )
    corpus = load_corpus(tmp_path)
    elig, rejected = evaluate_tool_eligibility(
        "I need an assistant that answers HR leave and expense questions for shore staff.",
        corpus,
    )
    elig_names = {t.name for t in elig}
    assert "open_meteo_weather" not in elig_names
    assert "csv_lookup" not in elig_names
    assert "json_record_lookup" not in elig_names
    assert "currency_converter" not in elig_names
    assert "unit_converter" not in elig_names
    assert "open_meteo_weather" in rejected
    assert "csv_lookup" in rejected
    assert "currency_converter" in rejected


def test_groundability_guard_blocks_ungroundable_brief_and_flags_partial_gaps(tmp_path: Path) -> None:
    (tmp_path / "policy.txt").write_text(
        "Shore staff receive 24 paid annual leave days and 12 paid sick leave days annually.",
        encoding="utf-8",
    )
    corpus = load_corpus(tmp_path)
    llm = _NeutralLLM("dummy")

    # Completely ungroundable brief
    bad_rep = audit_groundability(
        "Provide Bitcoin futures trading strategies and maritime piracy ransom negotiation playbooks.",
        corpus,
        llm,
    )
    assert bad_rep.overall_status == "ungroundable"
    assert bad_rep.unsupported_requests

    # Partially grounded brief moves unsupported ask into must_refuse
    req = run_intake_interview(
        "Answer shore staff annual leave policies AND explain cryptocurrency stock option vesting.",
        corpus,
        llm,
    )
    assert any("cryptocurrency" in t.lower() or "stock" in t.lower() for t in req.must_refuse_topics)
    assert not any("cryptocurrency" in t.lower() for t in req.must_answer_topics)
