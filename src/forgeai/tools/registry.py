"""Central Tool Registry (10 standard tools + declarative AST custom macros) with strict runtime guard."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from forgeai.ingestion.loader import LoadedCorpus
from forgeai.schemas.agent import CustomToolSpec, ToolBinding
from forgeai.tools.calculator import run_calculator
from forgeai.tools.country_lookup import run_country_info
from forgeai.tools.csv_lookup import run_csv_lookup
from forgeai.tools.currency_converter import run_currency_converter
from forgeai.tools.current_date import run_current_date
from forgeai.tools.custom_macro import execute_custom_tool
from forgeai.tools.http_weather import run_open_meteo_weather
from forgeai.tools.json_lookup import run_json_record_lookup
from forgeai.tools.statistics_tool import run_statistics_calculator
from forgeai.tools.text_analyzer import run_text_stats_and_regex
from forgeai.tools.unit_converter import run_unit_converter


class ToolAuthorizationError(PermissionError):
    """Raised when an agent attempts to invoke a tool not bound to its definition."""


@dataclass(frozen=True)
class ToolSpec:
    """Metadata and implementation hook for a ForgeAI standard tool."""

    name: str
    description: str
    args_schema_hint: str
    handler: Callable[[dict[str, Any], Path], str]
    requires_csv: bool = False
    requires_json: bool = False
    weather_intent_required: bool = False
    fx_intent_required: bool = False
    unit_intent_required: bool = False
    country_intent_required: bool = False
    stats_intent_required: bool = False
    text_analysis_intent_required: bool = False


TOOL_CATALOG: dict[str, ToolSpec] = {
    "csv_lookup": ToolSpec(
        name="csv_lookup",
        description="Search, filter, sort, or aggregate rows in CSV, TSV, and XLSX spreadsheets inside the corpus.",
        args_schema_hint='{"filename": "optional.csv", "query": "search terms", "filters": {"col": "val"}, "aggregate": "count|sum|mean|min|max", "agg_column": "col", "limit": 8}',
        handler=run_csv_lookup,
        requires_csv=True,
    ),
    "json_record_lookup": ToolSpec(
        name="json_record_lookup",
        description="Search and filter structured JSON, JSONL, and YAML record files inside the document corpus.",
        args_schema_hint='{"filename": "optional.json", "query": "search terms", "limit": 10}',
        handler=run_json_record_lookup,
        requires_json=True,
    ),
    "calculator": ToolSpec(
        name="calculator",
        description="Evaluate exact arithmetic expressions, totals, percentages, discounts, or unit math.",
        args_schema_hint='{"expression": "(1250 * 4) * 1.15"}',
        handler=run_calculator,
    ),
    "statistics_calculator": ToolSpec(
        name="statistics_calculator",
        description="Compute descriptive statistics (mean, median, stdev, min, max, weighted_avg) or percentage changes.",
        args_schema_hint='{"values": [120, 150, 180], "weights": [1, 2, 1]} OR {"old_value": 100, "new_value": 125}',
        handler=run_statistics_calculator,
        stats_intent_required=True,
    ),
    "unit_converter": ToolSpec(
        name="unit_converter",
        description="Convert engineering, logistics, and maritime units (kg/lb/mt, km/mi/nm, bar/psi, C/F, knots/kmh, L/gal).",
        args_schema_hint='{"value": 22.5, "from_unit": "kg", "to_unit": "lb"}',
        handler=run_unit_converter,
        unit_intent_required=True,
    ),
    "currency_converter": ToolSpec(
        name="currency_converter",
        description="Convert amounts across ISO currencies using live keyless Frankfurter ECB FX rates.",
        args_schema_hint='{"amount": 1500, "from_currency": "USD", "to_currency": "EUR"}',
        handler=run_currency_converter,
        fx_intent_required=True,
    ),
    "open_meteo_weather": ToolSpec(
        name="open_meteo_weather",
        description="Fetch live weather & 3-day wind/temperature/precipitation forecast for a city or port via Open-Meteo.",
        args_schema_hint='{"location": "Singapore"}',
        handler=run_open_meteo_weather,
        weather_intent_required=True,
    ),
    "http_public_country_info": ToolSpec(
        name="http_public_country_info",
        description="Fetch ISO codes, capital, currencies, calling code, and timezones for a country via keyless RestCountries API.",
        args_schema_hint='{"country": "Singapore"}',
        handler=run_country_info,
        country_intent_required=True,
    ),
    "text_stats_and_regex": ToolSpec(
        name="text_stats_and_regex",
        description="Compute document word/line stats or search exact regex/keyword occurrences across files in the corpus.",
        args_schema_hint='{"pattern": "ERROR|WARN", "filename": "optional.log", "limit": 12}',
        handler=run_text_stats_and_regex,
        text_analysis_intent_required=True,
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

_FX_PATTERNS = re.compile(
    r"\b(currency|exchange\s*rate|fx|forex|convert\s*(usd|eur|gbp|sgd|jpy|aed)|eur\b|gbp\b|sgd\b|jpy\b|multi-currency)\b",
    re.IGNORECASE,
)

_UNIT_PATTERNS = re.compile(
    r"\b(unit\s*conver|convert\s*(kg|lb|bar|psi|km|nm|celsius|fahrenheit|knots|liters)|nautical\s*mile|psi\b|bar\b|weight_kg|tonnes?|knots?)\b",
    re.IGNORECASE,
)

_COUNTRY_PATTERNS = re.compile(
    r"\b(country\s*info|country\s*code|iso\s*code|calling\s*code|capital\s*city|country\s*currency|timezone\s*of|port\s*country|restcountries|nation\s*metadata)\b",
    re.IGNORECASE,
)

_STATS_PATTERNS = re.compile(
    r"\b(statistics|statistical|mean|median|standard\s*deviation|stdev|variance|weighted\s*average|percent\s*change|distribution|benchmark|kpi\s*summary)\b",
    re.IGNORECASE,
)

_TEXT_ANALYSIS_PATTERNS = re.compile(
    r"\b(regex|pattern\s*match|word\s*count|line\s*count|occurrence|frequency|log\s*audit|audit\s*logs?|log\s*errors?|scan\s*documents?)\b",
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


def _positive_brief_only(brief: str) -> str:
    """Strip explicit refusal boundary phrases ('while refusing ...') before checking tool intent."""
    return re.sub(
        r"\b(?:while\s+refusing|refus(?:e|ing)\s+(?:to\s+answer\s+)?|must\s+refuse|do\s+not\s+answer|except\s+for)\b.+$",
        "",
        brief,
        flags=re.IGNORECASE,
    ).strip()


def evaluate_tool_eligibility(
    brief: str,
    corpus: LoadedCorpus,
) -> tuple[list[ToolSpec], dict[str, str]]:
    """Deterministically exclude impossible/off-domain tools before LLM selection."""
    eligible: list[ToolSpec] = []
    rejected: dict[str, str] = {}

    positive_brief = _positive_brief_only(brief)
    has_structured = corpus.has_structured_records
    has_logs = any(d.summary.doc_type == "log" for d in corpus.documents)
    combined_hints = (
        positive_brief
        + "\n"
        + " ".join(corpus.filenames)
        + "\n"
        + " ".join(d.summary.preview for d in corpus.documents)
    )

    for name, spec in TOOL_CATALOG.items():
        if spec.requires_csv and not corpus.has_csv:
            rejected[name] = "Excluded deterministically: no CSV/TSV/XLSX files exist in the document corpus."
            continue

        if spec.requires_json and not has_structured:
            rejected[name] = "Excluded deterministically: no JSON/JSONL/YAML files exist in the document corpus."
            continue

        if spec.weather_intent_required and not _WEATHER_PATTERNS.search(positive_brief):
            rejected[name] = (
                "Excluded deterministically: user brief does not request weather or meteorological data."
            )
            continue

        if spec.fx_intent_required and not _FX_PATTERNS.search(positive_brief):
            rejected[name] = (
                "Excluded deterministically: user brief does not request currency conversion or FX rates."
            )
            continue

        if spec.unit_intent_required and not (
            _UNIT_PATTERNS.search(positive_brief) or _UNIT_PATTERNS.search(combined_hints)
        ):
            rejected[name] = (
                "Excluded deterministically: no engineering/logistics unit conversion intent detected."
            )
            continue

        if spec.country_intent_required and not _COUNTRY_PATTERNS.search(positive_brief):
            rejected[name] = (
                "Excluded deterministically: user brief does not request country metadata, ISO codes, or calling codes."
            )
            continue

        if spec.stats_intent_required and not _STATS_PATTERNS.search(positive_brief):
            rejected[name] = (
                "Excluded deterministically: user brief does not request statistical/variance/KPI calculations."
            )
            continue

        if spec.text_analysis_intent_required and not (
            _TEXT_ANALYSIS_PATTERNS.search(positive_brief) or has_logs
        ):
            rejected[name] = (
                "Excluded deterministically: no regex/frequency text audit intent or log corpus detected."
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


def format_tools_for_prompt(
    tools: list[ToolSpec | ToolBinding],
    custom_tools: list[CustomToolSpec] | None = None,
) -> str:
    """Format eligible standard tools and declarative custom macro tools for LLM prompts."""
    custom_list = custom_tools or []
    if not tools and not custom_list:
        return "NONE (Do not call external tools; answer solely from retrieved text.)"
    lines: list[str] = []
    for t in tools:
        spec = TOOL_CATALOG.get(t.name)
        hint = spec.args_schema_hint if spec else "{}"
        lines.append(f"- {t.name}: {t.description}\n  Args JSON: {hint}")
    for ct in custom_list:
        if ct.tool_type == "formula":
            arg_hint = json.dumps({p: 0.0 for p in ct.parameters})
            lines.append(
                f"- {ct.name} [CUSTOM MACRO formula: {ct.expression}]: {ct.description}\n  Args JSON: {arg_hint}"
            )
        else:
            hint = json.dumps({"query": "optional keywords", "limit": 8})
            lines.append(
                f"- {ct.name} [CUSTOM MACRO preset lookup on {ct.target_file or 'tabular file'}]: {ct.description}\n  Args JSON: {hint}"
            )
    return "\n".join(lines)


def execute_tool(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    authorized_tools: list[ToolBinding],
    docs_dir: Path,
    custom_tools: list[CustomToolSpec] | None = None,
) -> str:
    """Execute a standard or declarative custom tool ONLY if bound to the agent definition."""
    custom_map = {ct.name: ct for ct in (custom_tools or [])}
    allowed_names = {t.name for t in authorized_tools} | set(custom_map.keys())
    if tool_name not in allowed_names:
        raise ToolAuthorizationError(
            f"Tool '{tool_name}' is not authorized for this agent. Allowed: {sorted(allowed_names)}"
        )
    if tool_name in custom_map:
        return execute_custom_tool(custom_map[tool_name], arguments or {}, docs_dir)
    spec = TOOL_CATALOG.get(tool_name)
    if spec is None:
        raise ToolAuthorizationError(f"Unknown tool '{tool_name}'")
    return spec.handler(arguments or {}, docs_dir)
