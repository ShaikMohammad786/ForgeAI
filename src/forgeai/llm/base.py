"""LLM client abstractions, JSON repair, and token/latency budget enforcement."""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError


T = TypeVar("T", bound=BaseModel)


class BudgetExceededError(RuntimeError):
    """Raised when a factory run exceeds its token or wall-clock budget."""


class LLMUnavailableError(RuntimeError):
    """Raised when the configured LLM backend cannot be reached or authenticated."""


@dataclass
class LLMResponse:
    """Normalized LLM completion response across Ollama and Gemini."""

    text: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    raw_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BudgetTracker:
    """Tracks cumulative tokens, calls, and elapsed time for budget enforcement."""

    max_tokens: int = 250_000
    max_seconds: float = 900.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    started_at: float = field(default_factory=time.monotonic)
    stopped_early_reason: str | None = None

    @property
    def elapsed_seconds(self) -> float:
        return round(time.monotonic() - self.started_at, 2)

    def remaining_tokens(self) -> int:
        return max(0, self.max_tokens - self.total_tokens)

    def remaining_seconds(self) -> float:
        return max(0.0, self.max_seconds - (time.monotonic() - self.started_at))

    def check_before_call(self) -> None:
        if self.total_tokens >= self.max_tokens:
            self.stopped_early_reason = (
                f"Token budget reached ({self.total_tokens}/{self.max_tokens})"
            )
            raise BudgetExceededError(self.stopped_early_reason)
        if self.elapsed_seconds >= self.max_seconds:
            self.stopped_early_reason = (
                f"Latency budget reached ({self.elapsed_seconds:.1f}s/{self.max_seconds:.1f}s)"
            )
            raise BudgetExceededError(self.stopped_early_reason)

    def record(self, response: LLMResponse) -> None:
        p = max(0, response.prompt_tokens)
        c = max(0, response.completion_tokens)
        t = max(p + c, response.total_tokens)
        # Fallback estimate if provider didn't return token counts
        if t == 0 and response.text:
            c = max(1, len(response.text) // 4)
            t = c
        self.prompt_tokens += p
        self.completion_tokens += c
        self.total_tokens += t
        self.llm_calls += 1


def extract_json_block(raw_text: str) -> str:
    """Extract and clean a JSON object/array from potentially noisy small-model output."""
    text = raw_text.strip()
    if not text:
        raise ValueError("Empty response when JSON was expected")

    # Strip fenced markdown blocks first
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fence_match:
        candidate = fence_match.group(1).strip()
        if candidate.startswith("{") or candidate.startswith("["):
            text = candidate

    # Locate first opening brace/bracket and balance respecting strings
    start_idx = None
    open_char = "{"
    close_char = "}"
    for i, ch in enumerate(text):
        if ch in "{[":
            start_idx = i
            open_char = ch
            close_char = "}" if ch == "{" else "]"
            break

    if start_idx is None:
        raise ValueError(f"No JSON object or array found in response: {text[:180]}")

    depth = 0
    in_string = False
    escape = False
    end_idx = None
    for i in range(start_idx, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == open_char:
            depth += 1
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                end_idx = i + 1
                break

    extracted = text[start_idx:end_idx] if end_idx is not None else text[start_idx:]
    # Remove trailing commas before } or ]
    cleaned = re.sub(r",\s*([}\]])", r"\1", extracted)
    return cleaned


def parse_json_dict(raw_text: str) -> dict[str, Any]:
    """Parse JSON object from raw model text with repair heuristics."""
    cleaned = extract_json_block(raw_text)
    parsed = json.loads(cleaned)
    if isinstance(parsed, list):
        return {"items": parsed}
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")
    return parsed


class LLMProvider(ABC):
    """Abstract base class for unified Ollama and Gemini model backends."""

    provider_name: str
    model_name: str

    def __init__(self, model_name: str, budget: BudgetTracker | None = None) -> None:
        self.model_name = model_name
        self.budget = budget

    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.1,
        max_tokens: int = 1200,
        json_mode: bool = False,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Execute a single chat/text completion."""

    def complete_structured(
        self,
        messages: list[dict[str, str]],
        response_model: type[T],
        *,
        temperature: float = 0.05,
        max_tokens: int = 1400,
        retries: int = 2,
    ) -> tuple[T, LLMResponse]:
        """Execute completion and parse/validate into a Pydantic model with retry repair."""
        working_messages = [dict(m) for m in messages]
        last_error: Exception | None = None
        last_response: LLMResponse | None = None

        for attempt in range(retries + 1):
            resp = self.complete(
                working_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=True,
                json_schema=None,
            )
            last_response = resp
            try:
                data = parse_json_dict(resp.text)
                validated = response_model.model_validate(data)
                return validated, resp
            except (ValueError, json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                if attempt >= retries:
                    break
                working_messages.append({"role": "assistant", "content": resp.text[:1200]})
                working_messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your previous output was invalid JSON/schema: {exc}.\n"
                            "Return ONLY a valid JSON object matching the exact keys requested. "
                            "No markdown prose outside JSON."
                        ),
                    }
                )

        raise ValueError(
            f"Model '{self.model_name}' failed structured parsing for {response_model.__name__} "
            f"after {retries + 1} attempts: {last_error}\nRaw output: {(last_response.text[:300] if last_response else '')}"
        )
