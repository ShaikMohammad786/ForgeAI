"""Tests for the optional FastAPI REST backend."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from forgeai.api.server import app
from forgeai.llm.base import LLMProvider, LLMResponse
from forgeai.schemas.agent import AgentDefinition, RetrievalConfig


class _DummyAPIChatLLM(LLMProvider):
    provider_name = "ollama"

    def complete(self, messages, *, temperature=0.1, max_tokens=1200, json_mode=False, json_schema=None):
        return LLMResponse(
            text=json.dumps(
                {
                    "action": "final",
                    "answer": "Shore staff receive 24 paid annual leave days.",
                    "citations": [{"source": "hr.md", "excerpt": "24 paid annual leave days"}],
                    "refused": False,
                    "refusal_reason": None,
                    "confidence": "high",
                    "tools_used": [],
                }
            ),
            model="dummy",
            provider="ollama",
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
        )


def test_health_and_models_endpoints() -> None:
    client = TestClient(app)
    h = client.get("/health")
    assert h.status_code == 200
    assert h.json()["status"] == "ok"

    m = client.get("/models")
    assert m.status_code == 200
    assert "gemini_models" in m.json()

    a = client.get("/agents")
    assert a.status_code == 200
    assert "agents" in a.json()


def test_chat_endpoint_loads_agent_and_answers(monkeypatch, tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "hr.md").write_text("# Leave\nShore staff receive 24 paid annual leave days.\n", encoding="utf-8")

    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    agent = AgentDefinition(
        agent_id="api-test-1",
        name="API Test Agent",
        brief="HR",
        provider="ollama",
        model_name="qwen2.5:3b",
        judge_model_name="qwen2.5:1.5b",
        system_prompt="Answer HR questions.",
        retrieval=RetrievalConfig(),
        docs_dir=str(docs_dir),
        created_at="2026-01-01T00:00:00Z",
    )
    (agent_dir / "agent_definition.json").write_text(agent.model_dump_json(indent=2), encoding="utf-8")

    monkeypatch.setattr("forgeai.runtime.executor.build_provider", lambda **kwargs: _DummyAPIChatLLM("dummy"))

    client = TestClient(app)
    resp = client.post("/chat", json={"agent_dir": str(agent_dir), "question": "How many annual leave days?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["response"]["refused"] is False
    assert "24" in body["response"]["answer"]


def test_ui_helpers_match_live_corpus_and_trace_schemas(tmp_path: Path) -> None:
    from forgeai.ingestion.loader import load_corpus
    from forgeai.runtime.executor import AgentRuntime
    from forgeai.ui.app import _corpus_to_inventory_rows, _trace_to_chat_meta

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "hr.md").write_text("# Leave\nShore staff receive 24 paid annual leave days.\n", encoding="utf-8")

    corpus = load_corpus(docs_dir)
    rows = _corpus_to_inventory_rows(corpus)
    assert len(rows) == 1
    assert rows[0]["Filename"] == "hr.md"
    assert rows[0]["Sections"] >= 1
    assert rows[0]["Characters"] >= 10

    agent = AgentDefinition(
        agent_id="ui-helper-test",
        name="UI Helper Agent",
        brief="HR",
        provider="ollama",
        model_name="dummy",
        judge_model_name="dummy",
        system_prompt="Answer HR questions.",
        retrieval=RetrievalConfig(min_score_threshold=0.0),
        docs_dir=str(docs_dir),
        created_at="2026-01-01T00:00:00Z",
    )
    rt = AgentRuntime(agent, _DummyAPIChatLLM("dummy"), corpus=corpus, docs_dir_override=docs_dir)
    trace = rt.ask("How many annual leave days?")
    meta = _trace_to_chat_meta(trace)
    assert meta["refused"] is False
    assert meta["chunks"]
    assert isinstance(meta["chunks"][0]["score"], float)
    assert isinstance(meta["chunks"][0]["bm25"], float)
    assert isinstance(meta["chunks"][0]["dense"], float)

