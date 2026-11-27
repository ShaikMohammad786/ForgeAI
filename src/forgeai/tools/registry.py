"""Central Tool Registry with deterministic capability filtering & runtime execution guard."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from forgeai.ingestion.loader import LoadedCorpus
from forgeai.schemas.agent import ToolBinding
from forgeai.tools.calculator import run_calculator
from forgeai.tools.csv_lookup import run_csv_lookup
from forgeai.tools.current_date import run_current_date
from forgeai.tools.http_weather import run_open_meteo_weather


class ToolAuthorizationError(PermissionError):
    """Raised when an agent attempts to invoke a tool not bound to its definition."""


@dataclass(frozen=True)
class ToolSpec:
    """Metadata and implementation hook for a ForgeAI tool."""

    name: str
    description: str
    args_schema_hint: str
    handler: Callable[[dict[str, Any], Path], str]
    requires_csv: bool = False
    weather_intent_required: bool = False


TOOL_CATALOG: dict[str, ToolSpec] = {
    "csv_lookup": ToolSpec(
        name="csv_lookup",
        description="Search, filter, sort, or aggregate rows in CSV files inside the corpus.",
        args_schema_hint='{"filename": "optional.csv", "query": "search terms", "filters": {"col": "val"}, "aggregate": "count|sum|mean|min|max", "agg_column": "col", "limit": 8}',
        handler=run_csv_lookup,
        requires_csv=True,
    ),
    "calculator": ToolSpec(
        name="calculator",
        description="Evaluate exact arithmetic expressions, totals, percentages, discounts, or unit math.",
        args_schema_hint='{"expression": "(1250 * 4) * 1.15"}',
        handler=run_calculator,
    ),
    "open_meteo_weather": ToolSpec(
        name="open_meteo_weather",
        description="Fetch live weather & 3-day wind/temperature/precipitation forecast for a city or port via Open-Meteo.",
        args_schema_hint='{"location": "Singapore"}',
        handler=run_open_meteo_weather,
        weather_intent_required=True,
    ),
    "current_date": ToolSpec(
        name="current_date",
        description="Get today's date, day of week, or compute date offsets (add_days) and day differences (start_date, end_date).",
        args_schema_hint='{"add_days": 14, "base_date": "optional YYYY-MM-DD"}',
        handler=run_current_date,
    ),
}


_WEATHER_PATTERNS = re.compile(
    r"\b(weather|forecast|temperature|rain|precipitation|wind|storm|cyclone|typhoon|sea\s*state|meteorolog|climate|port\s*weather)\b",
    re.IGNORECASE,
)

_NUMERIC_PATTERNS = re.compile(
    r"\b(calculate|calculation|total|cost|price|pricing|budget|discount|tax|freight|quote|lead\s*time|quantity|qty|salary|allowance|reimburse|expense|days|percentage|percent|ratio|sum|margin|invoice|procurement|order|spares|parts|fee)\b",
    re.IGNORECASE,
)

_DATE_PATTERNS = re.compile(
    r"\b(date|today|deadline|schedule|delivery|lead\s*time|tenure|probation|notice\s*period|expiry|effective|calendar|day|week|month|year|eligibility|leave|holiday|sla|valid\s*until)\b",
    re.IGNORECASE,
)


def evaluate_tool_eligibility(
    brief: str,
    corpus: LoadedCorpus,
) -> tuple[list[ToolSpec], dict[str, str]]:
    """Deterministically exclude impossible/off-domain tools before LLM selection."""
    eligible: list[ToolSpec] = []
    rejected: dict[str, str] = {}

    combined_hints = brief + "\n" + " ".join(corpus.filenames) + "\n" + " ".join(
        d.summary.preview for d in corpus.documents
    )

    for name, spec in TOOL_CATALOG.items():
        if spec.requires_csv and not corpus.has_csv:
            rejected[name] = "Excluded deterministically: no CSV files exist in the document corpus."
            continue

        if spec.weather_intent_required and not _WEATHER_PATTERNS.search(brief):
            # Also require explicit weather/forecast intent in the brief itself
            # so unrelated docs mentioning 'rain check' or ports without weather intent don't attach it.
            rejected[name] = (
                "Excluded deterministically: user brief does not request weather or meteorological data."
            )
            continue

        if name == "calculator" and not (_NUMERIC_PATTERNS.search(combined_hints) or corpus.has_csv):
            rejected[name] = "Excluded deterministically: no numeric/pricing/calculation domain signals."
            continue

        if name == "current_date" and not _DATE_PATTERNS.search(combined_hints):
            rejected[name] = "Excluded deterministically: no schedule, deadline, or date-sensitive signals."
            continue

        eligible.append(spec)

    return eligible, rejected


def format_tools_for_prompt(tools: list[ToolSpec | ToolBinding]) -> str:
    """Format eligible or authorized tools cleanly for LLM prompts."""
    if not tools:
        return "NONE (Do not call external tools; answer solely from retrieved text.)"
    lines: list[str] = []
    for t in tools:
        spec = TOOL_CATALOG.get(t.name)
        hint = spec.args_schema_hint if spec else "{}"
        lines.append(f"- {t.name}: {t.description}\n  Args JSON: {hint}")
    return "\n".join(lines)


def execute_tool(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    authorized_tools: list[ToolBinding],
    docs_dir: Path,
) -> str:
    """Execute a tool ONLY if it is explicitly attached to the agent's ToolBindings."""
    allowed_names = {t.name for t in authorized_tools}
    if tool_name not in allowed_names:
        raise ToolAuthorizationError(
            f"Tool '{tool_name}' is not authorized for this agent. Allowed: {sorted(allowed_names)}"
        )
    spec = TOOL_CATALOG.get(tool_name)
    if spec is None:
        raise ToolAuthorizationError(f"Unknown tool '{tool_name}'")
    return spec.handler(arguments or {}, docs_dir)
