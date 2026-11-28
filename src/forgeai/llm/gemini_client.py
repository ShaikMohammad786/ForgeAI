"""Google Gemini LLM backend using the Interactions API (with GenerateContent fallback)."""

from __future__ import annotations

import time
from typing import Any

from google import genai
from google.genai import types as genai_types

from forgeai.llm.base import BudgetTracker, LLMProvider, LLMResponse, LLMUnavailableError


class GeminiProvider(LLMProvider):
    """Gemini client powered by google-genai Interactions API."""

    provider_name = "gemini"

    def __init__(
        self,
        model_name: str,
        api_key: str,
        use_interactions: bool = True,
        budget: BudgetTracker | None = None,
    ) -> None:
        super().__init__(model_name=model_name, budget=budget)
        if not api_key.strip():
            raise LLMUnavailableError(
                "GEMINI_API_KEY is not configured. Set it in .env or switch to Ollama."
            )
        self.api_key = api_key.strip()
        self.use_interactions = use_interactions
        self.client = genai.Client(
            api_key=self.api_key,
            http_options=genai_types.HttpOptions(api_version="v1beta"),
        )

    @staticmethod
    def list_recommended_models() -> list[str]:
        """Return supported Gemini models selectable in ForgeAI."""
        return [
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.5-pro",
        ]

    @staticmethod
    def _split_messages(messages: list[dict[str, str]]) -> tuple[str | None, str]:
        system_parts: list[str] = []
        convo_parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user").lower()
            content = msg.get("content", "")
            if role == "system":
                system_parts.append(content)
            elif role == "assistant":
                convo_parts.append(f"Assistant:\n{content}")
            else:
                convo_parts.append(f"User:\n{content}")

        system_instruction = "\n\n".join(system_parts).strip() or None
        prompt_text = "\n\n".join(convo_parts).strip()
        return system_instruction, prompt_text

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

        system_instruction, prompt_text = self._split_messages(messages)
        if json_mode and "JSON" not in prompt_text[-180:]:
            prompt_text += "\n\nRespond ONLY with a valid JSON object."

        t0 = time.perf_counter()
        if self.use_interactions:
            try:
                return self._call_interactions(
                    prompt_text=prompt_text,
                    system_instruction=system_instruction,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                    json_schema=json_schema,
                    t0=t0,
                )
            except Exception as exc:
                # Fall back cleanly to standard generate_content if Interactions API
                # rejects a parameter or transient route issue occurs.
                return self._call_generate_content(
                    prompt_text=prompt_text,
                    system_instruction=system_instruction,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                    t0=t0,
                    fallback_from=str(exc),
                )

        return self._call_generate_content(
            prompt_text=prompt_text,
            system_instruction=system_instruction,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            t0=t0,
        )

    def _call_interactions(
        self,
        *,
        prompt_text: str,
        system_instruction: str | None,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        json_schema: dict[str, Any] | None,
        t0: float,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "input": prompt_text,
            "generation_config": {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        }
        if system_instruction:
            kwargs["system_instruction"] = system_instruction
        if json_schema is not None:
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_format"] = json_schema
        elif json_mode:
            kwargs["response_mime_type"] = "application/json"

        interaction = self.client.interactions.create(**kwargs)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        text = getattr(interaction, "output_text", None) or ""
        usage = getattr(interaction, "usage", None)
        prompt_toks = int(getattr(usage, "total_input_tokens", None) or 0)
        comp_toks = int(getattr(usage, "total_output_tokens", None) or 0)
        total_toks = int(getattr(usage, "total_tokens", None) or (prompt_toks + comp_toks))

        result = LLMResponse(
            text=text,
            model=self.model_name,
            provider="gemini_interactions",
            prompt_tokens=prompt_toks,
            completion_tokens=comp_toks,
            total_tokens=total_toks,
            latency_ms=latency_ms,
            raw_metadata={"interaction_id": getattr(interaction, "id", None)},
        )
        if self.budget is not None:
            self.budget.record(result)
        return result

    def _call_generate_content(
        self,
        *,
        prompt_text: str,
        system_instruction: str | None,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        t0: float,
        fallback_from: str | None = None,
    ) -> LLMResponse:
        config_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"

        try:
            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt_text,
                config=genai_types.GenerateContentConfig(**config_kwargs),
            )
        except Exception as exc:
            raise LLMUnavailableError(
                f"Gemini API call failed for '{self.model_name}': {exc}"
                + (f" (after Interactions error: {fallback_from})" if fallback_from else "")
            ) from exc

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        text = getattr(resp, "text", None) or ""
        meta = getattr(resp, "usage_metadata", None)
        prompt_toks = int(getattr(meta, "prompt_token_count", None) or 0)
        comp_toks = int(getattr(meta, "candidates_token_count", None) or 0)
        total_toks = int(getattr(meta, "total_token_count", None) or (prompt_toks + comp_toks))

        result = LLMResponse(
            text=text,
            model=self.model_name,
            provider="gemini_generate_content",
            prompt_tokens=prompt_toks,
            completion_tokens=comp_toks,
            total_tokens=total_toks,
            latency_ms=latency_ms,
            raw_metadata={"fallback_from": fallback_from} if fallback_from else {},
        )
        if self.budget is not None:
            self.budget.record(result)
        return result
