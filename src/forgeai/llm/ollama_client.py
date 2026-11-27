"""Local Ollama LLM backend optimized for small open models."""

from __future__ import annotations

import time
from typing import Any

import httpx
import ollama

from forgeai.llm.base import BudgetTracker, LLMProvider, LLMResponse, LLMUnavailableError


class OllamaProvider(LLMProvider):
    """Ollama client supporting any locally pulled open model."""

    provider_name = "ollama"

    def __init__(
        self,
        model_name: str,
        host: str = "http://127.0.0.1:11434",
        budget: BudgetTracker | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        super().__init__(model_name=model_name, budget=budget)
        self.host = host.rstrip("/")
        self.client = ollama.Client(host=self.host, timeout=timeout_seconds)

    @staticmethod
    def list_local_models(host: str = "http://127.0.0.1:11434") -> list[str]:
        """Return installed Ollama model tags or [] if Ollama server is down."""
        try:
            resp = httpx.get(f"{host.rstrip('/')}/api/tags", timeout=3.0)
            resp.raise_for_status()
            data = resp.json()
            return [m["name"] for m in data.get("models", []) if m.get("name")]
        except Exception:
            return []

    @staticmethod
    def is_available(host: str = "http://127.0.0.1:11434") -> bool:
        """Check whether the local Ollama HTTP daemon responds."""
        try:
            resp = httpx.get(f"{host.rstrip('/')}/api/tags", timeout=2.5)
            return resp.status_code == 200
        except Exception:
            return False

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.1,
        max_tokens: int = 1200,
        json_mode: bool = False,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        if self.budget is not None:
            self.budget.check_before_call()

        t0 = time.perf_counter()
        fmt: Any = json_schema if json_schema is not None else ("json" if json_mode else None)
        options = {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": 8192,
        }

        try:
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                format=fmt,
                options=options,
            )
        except Exception as exc:
            raise LLMUnavailableError(
                f"Ollama call failed for model '{self.model_name}' at '{self.host}': {exc}"
            ) from exc

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        text = ""
        if getattr(response, "message", None) is not None:
            text = response.message.content or ""

        prompt_toks = int(getattr(response, "prompt_eval_count", None) or 0)
        comp_toks = int(getattr(response, "eval_count", None) or 0)
        # Estimate tokens if Ollama omitted eval_count
        if prompt_toks == 0:
            prompt_toks = max(1, sum(len(m.get("content", "")) for m in messages) // 4)
        if comp_toks == 0 and text:
            comp_toks = max(1, len(text) // 4)

        result = LLMResponse(
            text=text,
            model=self.model_name,
            provider=self.provider_name,
            prompt_tokens=prompt_toks,
            completion_tokens=comp_toks,
            total_tokens=prompt_toks + comp_toks,
            latency_ms=latency_ms,
            raw_metadata={"done_reason": getattr(response, "done_reason", None)},
        )
        if self.budget is not None:
            self.budget.record(result)
        return result
