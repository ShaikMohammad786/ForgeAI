"""Committed prompt template loader for ForgeAI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined


_PROMPTS_DIR = Path(__file__).resolve().parent
_ENV = Environment(
    loader=FileSystemLoader(str(_PROMPTS_DIR)),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_prompt(template_name: str, **context: Any) -> str:
    """Render a committed Jinja2 prompt template from src/forgeai/prompts."""
    template = _ENV.get_template(template_name)
    return template.render(**context)
