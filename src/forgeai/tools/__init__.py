"""Tool registry package."""

from forgeai.tools.custom_macro import execute_custom_tool, validate_custom_tool_spec
from forgeai.tools.registry import (
    TOOL_CATALOG,
    ToolAuthorizationError,
    ToolSpec,
    evaluate_tool_eligibility,
    execute_tool,
    format_tools_for_prompt,
)

__all__ = [
    "TOOL_CATALOG",
    "ToolAuthorizationError",
    "ToolSpec",
    "evaluate_tool_eligibility",
    "execute_custom_tool",
    "execute_tool",
    "format_tools_for_prompt",
    "validate_custom_tool_spec",
]
