"""Provider router & discovery supporting both local Ollama and Gemini API."""

from __future__ import annotations

from dataclasses import dataclass

from forgeai.config import ForgeSettings, get_settings
from forgeai.llm.base import BudgetTracker, LLMProvider, LLMUnavailableError
from forgeai.llm.gemini_client import GeminiProvider
from forgeai.llm.ollama_client import OllamaProvider


@dataclass
class BackendStatus:
    """Live availability report used by CLI and Streamlit UI."""

    ollama_available: bool
    ollama_models: list[str]
    gemini_available: bool
    gemini_models: list[str]


def discover_backends(settings: ForgeSettings | None = None) -> BackendStatus:
    """Detect available local Ollama models and configured Gemini credentials."""
    cfg = settings or get_settings()
    ollama_ok = OllamaProvider.is_available(cfg.ollama_host)
    ollama_models = OllamaProvider.list_local_models(cfg.ollama_host) if ollama_ok else []
    gemini_ok = bool(cfg.effective_gemini_key)
    gemini_models = GeminiProvider.list_recommended_models()
    return BackendStatus(
        ollama_available=ollama_ok,
        ollama_models=ollama_models,
        gemini_available=gemini_ok,
        gemini_models=gemini_models,
    )


def resolve_provider_choice(
    provider: str | None = None,
    settings: ForgeSettings | None = None,
) -> str:
    """Resolve 'auto', 'ollama', or 'gemini' into a concrete backend."""
    cfg = settings or get_settings()
    requested = (provider or cfg.provider or "auto").lower()
    if requested in {"ollama", "gemini"}:
        return requested

    status = discover_backends(cfg)
    if status.ollama_available and status.ollama_models:
        return "ollama"
    if status.gemini_available:
        return "gemini"
    if status.ollama_available:
        return "ollama"
    raise LLMUnavailableError(
        "No LLM backend available. Either start Ollama (`ollama serve`) or set GEMINI_API_KEY in .env."
    )


def build_provider(
    *,
    provider: str | None = None,
    model_name: str | None = None,
    gemini_api_key: str | None = None,
    budget: BudgetTracker | None = None,
    settings: ForgeSettings | None = None,
) -> LLMProvider:
    """Instantiate a single LLMProvider for either Ollama or Gemini."""
    cfg = settings or get_settings()
    chosen = resolve_provider_choice(provider, cfg)

    if chosen == "gemini":
        key = (gemini_api_key or cfg.effective_gemini_key).strip()
        model = model_name or cfg.gemini_model
        return GeminiProvider(
            model_name=model,
            api_key=key,
            use_interactions=cfg.gemini_use_interactions,
            budget=budget,
        )

    # Local Ollama
    local_models = OllamaProvider.list_local_models(cfg.ollama_host)
    chosen_model = model_name or cfg.ollama_model
    if local_models and chosen_model not in local_models:
        # Prefer qwen2.5:3b -> qwen2.5:1.5b -> first available local model
        for fallback in (cfg.ollama_model, cfg.ollama_judge_model, "qwen2.5:1.5b"):
            if fallback in local_models:
                chosen_model = fallback
                break
        else:
            chosen_model = local_models[0]

    return OllamaProvider(
        model_name=chosen_model,
        host=cfg.ollama_host,
        budget=budget,
    )


def build_factory_providers(
    *,
    provider: str | None = None,
    model_name: str | None = None,
    judge_provider: str | None = None,
    judge_model_name: str | None = None,
    gemini_api_key: str | None = None,
    budget: BudgetTracker | None = None,
    settings: ForgeSettings | None = None,
) -> tuple[LLMProvider, LLMProvider]:
    """Build distinct generator and judge LLMProviders (reduces self-preference bias)."""
    cfg = settings or get_settings()
    main_prov = build_provider(
        provider=provider,
        model_name=model_name,
        gemini_api_key=gemini_api_key,
        budget=budget,
        settings=cfg,
    )

    effective_judge_provider = judge_provider or main_prov.provider_name
    if effective_judge_provider.startswith("gemini"):
        effective_judge_provider = "gemini"

    if judge_model_name:
        j_model = judge_model_name
    elif effective_judge_provider == "gemini":
        j_model = cfg.gemini_judge_model
    else:
        local = OllamaProvider.list_local_models(cfg.ollama_host)
        # Use a distinct small model for cross-judging if available
        other_models = [m for m in local if m != main_prov.model_name]
        if cfg.ollama_judge_model in local and cfg.ollama_judge_model != main_prov.model_name:
            j_model = cfg.ollama_judge_model
        elif other_models:
            j_model = other_models[0]
        else:
            j_model = main_prov.model_name

    judge_prov = build_provider(
        provider=effective_judge_provider,
        model_name=j_model,
        gemini_api_key=gemini_api_key,
        budget=budget,
        settings=cfg,
    )
    return main_prov, judge_prov
