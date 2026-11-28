"""Optional FastAPI REST backend exposing ForgeAI Factory, Chat Runtime, and Regression endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from forgeai import __version__
from forgeai.config import get_settings
from forgeai.factory.pipeline import run_forge_pipeline
from forgeai.factory.regression import run_regression_check
from forgeai.llm.router import discover_backends
from forgeai.runtime.executor import AgentRuntime


REPO_ROOT = Path(__file__).resolve().parents[3]


class BuildRequest(BaseModel):
    """Request body for POST /build."""

    brief: str = Field(..., min_length=10)
    docs_dir: str
    out_dir: str
    provider: str | None = None
    model_name: str | None = None
    judge_provider: str | None = None
    judge_model_name: str | None = None
    max_iterations: int = Field(default=3, ge=0, le=3)
    eval_target_cases: int = Field(default=24, ge=20, le=50)
    run_tournament: bool = True
    use_llm_judge: bool = True
    max_tokens_budget: int = 250_000
    max_seconds_budget: float = 900.0


class ChatRequest(BaseModel):
    """Request body for POST /chat."""

    agent_dir: str
    question: str = Field(..., min_length=1)
    chat_history: list[dict[str, str]] = Field(default_factory=list)
    provider: str | None = None
    model_name: str | None = None
    docs_dir_override: str | None = None


class RegressRequest(BaseModel):
    """Request body for POST /regress."""

    agent_dir: str
    new_docs_dir: str
    out_dir: str
    provider: str | None = None
    model_name: str | None = None
    judge_provider: str | None = None
    judge_model_name: str | None = None
    use_llm_judge: bool = True


def create_app() -> FastAPI:
    """Create configured FastAPI application with OpenAPI Swagger docs at /docs."""
    api = FastAPI(
        title="ForgeAI Agent Factory API",
        version=__version__,
        description=(
            "Optional REST API for ForgeAI: build grounded agents from brief + documents, "
            "run generic agent inference, inspect bundles, and run document regression checks."
        ),
    )

    @api.get("/health")
    def health() -> dict[str, Any]:
        cfg = get_settings()
        status = discover_backends(cfg)
        return {
            "status": "ok",
            "version": __version__,
            "ollama_available": status.ollama_available,
            "gemini_configured": status.gemini_available,
        }

    @api.get("/models")
    def models() -> dict[str, Any]:
        cfg = get_settings()
        status = discover_backends(cfg)
        return {
            "ollama_available": status.ollama_available,
            "ollama_models": status.ollama_models,
            "gemini_available": status.gemini_available,
            "gemini_models": status.gemini_models,
        }

    @api.get("/agents")
    def list_agents() -> dict[str, Any]:
        bundles: list[dict[str, Any]] = []
        for base_name in ("demos", "runs"):
            base = REPO_ROOT / base_name
            if not base.exists():
                continue
            for p in sorted(base.rglob("agent_definition.yaml")):
                try:
                    data = yaml.safe_load(p.read_text(encoding="utf-8"))
                    bundles.append(
                        {
                            "name": data.get("name"),
                            "version": data.get("version"),
                            "provider": data.get("provider"),
                            "model_name": data.get("model_name"),
                            "tools": [t.get("name") for t in data.get("tools", [])],
                            "custom_tools": [ct.get("name") for ct in data.get("custom_tools", [])],
                            "path": str(p.parent),
                        }
                    )
                except Exception:
                    continue
        return {"count": len(bundles), "agents": bundles}

    @api.post("/build")
    def build_agent(req: BuildRequest) -> dict[str, Any]:
        docs_path = Path(req.docs_dir)
        if not docs_path.exists() or not docs_path.is_dir():
            raise HTTPException(status_code=404, detail=f"docs_dir not found: {req.docs_dir}")
        try:
            res = run_forge_pipeline(
                brief=req.brief,
                docs_dir=docs_path,
                out_dir=req.out_dir,
                provider=req.provider,
                model_name=req.model_name,
                judge_provider=req.judge_provider,
                judge_model_name=req.judge_model_name,
                unattended=True,
                run_tournament=req.run_tournament,
                max_iterations=req.max_iterations,
                max_tokens_budget=req.max_tokens_budget,
                max_seconds_budget=req.max_seconds_budget,
                eval_target_cases=req.eval_target_cases,
                use_llm_judge=req.use_llm_judge,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return {
            "halted_ungroundable": res.halted_ungroundable,
            "out_dir": str(res.out_dir),
            "groundability_status": res.requirements.groundability.overall_status,
            "agent_name": res.agent.name if res.agent else None,
            "tools": [t.name for t in res.agent.tools] if res.agent else [],
            "final_pass_rate": res.report.final_pass_rate if res.report else None,
            "final_composite_score": res.report.final_composite_score if res.report else None,
            "report_path": str(res.out_dir / "report.md"),
        }

    @api.post("/chat")
    def chat_agent(req: ChatRequest) -> dict[str, Any]:
        agent_path = Path(req.agent_dir)
        if not agent_path.exists():
            raise HTTPException(status_code=404, detail=f"agent_dir not found: {req.agent_dir}")
        try:
            rt = AgentRuntime.from_directory(
                agent_path,
                provider=req.provider,
                model_name=req.model_name,
                docs_dir_override=Path(req.docs_dir_override) if req.docs_dir_override else None,
            )
            trace = rt.ask(req.question, chat_history=req.chat_history)
            return {
                "agent_name": rt.agent.name,
                "model": rt.llm.model_name,
                "response": trace.response.model_dump(mode="json"),
                "retrieved_sources": list(dict.fromkeys(h.chunk.source for h in trace.retrieved_chunks)),
                "latency_ms": trace.latency_ms,
                "raw_contract_valid": trace.raw_contract_valid,
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @api.post("/regress")
    def regress_agent(req: RegressRequest) -> dict[str, Any]:
        if not Path(req.agent_dir).exists() or not Path(req.new_docs_dir).exists():
            raise HTTPException(status_code=404, detail="agent_dir or new_docs_dir not found")
        try:
            diff, card = run_regression_check(
                agent_dir=req.agent_dir,
                new_docs_dir=req.new_docs_dir,
                out_dir=req.out_dir,
                provider=req.provider,
                model_name=req.model_name,
                judge_provider=req.judge_provider,
                judge_model_name=req.judge_model_name,
                use_llm_judge=req.use_llm_judge,
            )
            return {
                "diff": diff.model_dump(mode="json"),
                "new_pass_rate": card.pass_rate,
                "report_path": str(Path(req.out_dir).resolve() / "regression_report.md"),
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return api


app = create_app()
