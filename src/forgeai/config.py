"""Global configuration settings for ForgeAI."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


load_dotenv()


class ForgeSettings(BaseSettings):
    """Runtime settings loaded from environment variables or .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    provider: Literal["auto", "ollama", "gemini"] = Field(
        default="ollama", alias="FORGEAI_PROVIDER"
    )

    # Local Ollama
    ollama_host: str = Field(default="http://127.0.0.1:11434", alias="OLLAMA_HOST")
    ollama_model: str = Field(default="qwen2.5:3b", alias="FORGEAI_OLLAMA_MODEL")
    ollama_judge_model: str = Field(
        default="qwen2.5:1.5b", alias="FORGEAI_OLLAMA_JUDGE_MODEL"
    )

    # Google Gemini (Interactions API + GenerateContent)
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-flash", alias="FORGEAI_GEMINI_MODEL")
    gemini_judge_model: str = Field(
        default="gemini-2.5-flash-lite", alias="FORGEAI_GEMINI_JUDGE_MODEL"
    )
    gemini_use_interactions: bool = Field(
        default=True, alias="FORGEAI_GEMINI_USE_INTERACTIONS"
    )

    # Embeddings
    embed_model: str = Field(
        default="BAAI/bge-small-en-v1.5", alias="FORGEAI_EMBED_MODEL"
    )

    # Budget & optimizer defaults
    max_iterations: int = Field(default=3, alias="FORGEAI_MAX_ITERATIONS")
    max_tokens_budget: int = Field(default=250_000, alias="FORGEAI_MAX_TOKENS_BUDGET")
    max_seconds_budget: float = Field(default=900.0, alias="FORGEAI_MAX_SECONDS_BUDGET")
    min_gain_threshold: float = Field(default=0.02, alias="FORGEAI_MIN_GAIN_THRESHOLD")
    eval_case_target: int = Field(default=24, alias="FORGEAI_EVAL_CASE_TARGET")

    @property
    def effective_gemini_key(self) -> str:
        """Return configured Gemini/Google API key if present."""
        return (self.gemini_api_key or self.google_api_key or "").strip()

    @property
    def prompts_dir(self) -> Path:
        """Return path to committed Jinja2 prompt templates."""
        return Path(__file__).resolve().parent / "prompts"


def get_settings() -> ForgeSettings:
    """Reload settings cleanly from current environment and .env."""
    load_dotenv(override=False)
    return ForgeSettings()
