"""Unified OpenAI-Compatible REST client for local servers (LM Studio, vLLM, llama.cpp) and cloud providers (OpenAI, Groq, OpenRouter, HF)."""

from __future__ import annotations

import os
import re
import time
from typing import Any

import httpx

from forgeai.llm.base import BudgetTracker, LLMProvider, LLMResponse, LLMUnavailableError

_TUNNEL_HEADERS = {"ngrok-skip-browser-warning": "true"}

OPENAI_COMPAT_PRESETS: dict[str, tuple[str, str, bool]] = {
    "Groq": ("https://api.groq.com/openai/v1", "llama-3.3-70b-versatile", True),
    "OpenAI": ("https://api.openai.com/v1", "gpt-4o-mini", True),
    "OpenRouter": ("https://openrouter.ai/api/v1", "qwen/qwen-2.5-7b-instruct", True),
    "Hugging Face Inference": ("https://router.huggingface.co/v1", "Qwen/Qwen2.5-7B-Instruct", True),
    "Together AI": ("https://api.together.xyz/v1", "Qwen/Qwen2.5-7B-Instruct-Turbo", True),
    "LM Studio (Local / Tunnel)": ("http://127.0.0.1:1234/v1", "local-model", False),
    "vLLM / llama.cpp (Local / Tunnel)": ("http://127.0.0.1:8000/v1", "default", False),
    "Custom OpenAI-Compatible": ("", "", False),
}


def _normalize_base_url(base_url: str) -> str:
    url = base_url.strip().rstrip("/")
    if url.endswith("/chat/completions"):
        url = url[: -len("/chat/completions")].rstrip("/")
    return url


class OpenAICompatProvider(LLMProvider):
    """HTTP Chat Completions client compatible with OpenAI, Groq, OpenRouter, HF, LM Studio, vLLM, and llama.cpp."""

    provider_name = "openai_compatible"

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str | None = None,
        budget: BudgetTracker | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        super().__init__(model_name=model_name, budget=budget)
        self.base_url = _normalize_base_url(base_url)
        self.api_key = (
            api_key
            or os.getenv("OPENAI_COMPAT_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("GROQ_API_KEY")
            or os.getenv("HF_TOKEN")
            or ""
        ).strip()
        self.timeout_seconds = timeout_seconds
        if not self.base_url:
            raise LLMUnavailableError(
                "OpenAI-compatible Base URL is required (e.g. https://api.groq.com/openai/v1 or http://127.0.0.1:1234/v1)."
            )

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            **_TUNNEL_HEADERS,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    def list_models(base_url: str, api_key: str | None = None) -> list[str]:
        """Fetch available model IDs from GET {base_url}/models cleanly."""
        norm = _normalize_base_url(base_url)
        if not norm:
            return []
        headers = dict(_TUNNEL_HEADERS)
        if api_key and api_key.strip():
            headers["Authorization"] = f"Bearer {api_key.strip()}"
        try:
            resp = httpx.get(f"{norm}/models", headers=headers, timeout=5.0)
            resp.raise_for_status()
            payload = resp.json()
            data = payload.get("data") if isinstance(payload, dict) else payload
            if not isinstance(data, list):
                return []
            ids = [str(item["id"]) for item in data if isinstance(item, dict) and item.get("id")]
            return sorted(dict.fromkeys(ids))[:80]
        except Exception:
            return []

    @staticmethod
    def is_available(base_url: str, api_key: str | None = None) -> bool:
        """Check whether GET {base_url}/models responds with HTTP 2xx."""
        norm = _normalize_base_url(base_url)
        if not norm:
            return False
        headers = dict(_TUNNEL_HEADERS)
        if api_key and api_key.strip():
            headers["Authorization"] = f"Bearer {api_key.strip()}"
        try:
            resp = httpx.get(f"{norm}/models", headers=headers, timeout=4.0)
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
        endpoint = f"{self.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode or json_schema is not None:
            payload["response_format"] = {"type": "json_object"}

        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                resp = client.post(endpoint, headers=self._headers(), json=payload)
                # Retry cleanly without response_format if server/model does not support json_object mode
                if resp.status_code == 400 and "response_format" in payload:
                    payload.pop("response_format", None)
                    resp = client.post(endpoint, headers=self._headers(), json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:240] if exc.response is not None else str(exc)
            raise LLMUnavailableError(
                f"OpenAI-compatible endpoint '{self.base_url}' returned HTTP {exc.response.status_code}: {detail}"
            ) from exc
        except Exception as exc:
            raise LLMUnavailableError(
                f"OpenAI-compatible call failed for '{self.model_name}' at '{self.base_url}': {exc}"
            ) from exc

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        text = ""
        choices = data.get("choices") or []
        if choices and isinstance(choices[0], dict):
            msg = choices[0].get("message") or {}
            text = str(msg.get("content") or "")
            # Strip reasoning tags if reasoning models emit <think>...</think> before JSON
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        usage = data.get("usage") or {}
        prompt_toks = int(usage.get("prompt_tokens") or 0)
        comp_toks = int(usage.get("completion_tokens") or 0)
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
            total_tokens=int(usage.get("total_tokens") or (prompt_toks + comp_toks)),
            latency_ms=latency_ms,
        )
        if self.budget is not None:
            self.budget.record(result)
        return result
