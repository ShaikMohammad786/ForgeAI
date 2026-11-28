"""AST-verified declarative Custom Macro Tool validation and execution (zero exec/eval)."""

from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path
from typing import Any

from forgeai.schemas.agent import CustomToolSpec
from forgeai.tools.calculator import _BIN_OPS, _SAFE_FUNCS, _UNARY_OPS
from forgeai.tools.csv_lookup import _resolve_tabular_files, run_csv_lookup

_SAFE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{2,40}$")
_SAFE_CONSTANTS = {"pi": math.pi, "e": math.e}


def _collect_and_validate_ast(
    node: ast.AST,
    allowed_params: set[str],
) -> None:
    """Verify every AST node is a pure mathematical expression over allowed parameters."""
    if isinstance(node, ast.Expression):
        _collect_and_validate_ast(node.body, allowed_params)
        return
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return
    if isinstance(node, ast.Name):
        if node.id not in allowed_params and node.id not in _SAFE_CONSTANTS:
            raise ValueError(f"Undeclared variable '{node.id}' in custom formula")
        return
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        _collect_and_validate_ast(node.left, allowed_params)
        _collect_and_validate_ast(node.right, allowed_params)
        return
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        _collect_and_validate_ast(node.operand, allowed_params)
        return
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id not in _SAFE_FUNCS or node.keywords:
            raise ValueError(f"Unsupported or keyword function '{node.func.id}'")
        for arg in node.args:
            _collect_and_validate_ast(arg, allowed_params)
        return
    raise ValueError(f"Unsafe or unsupported syntax in custom formula: {ast.dump(node)}")


def _eval_ast_with_env(node: ast.AST, env: dict[str, float]) -> float:
    """Evaluate validated mathematical AST against strictly numeric parameter bindings."""
    if isinstance(node, ast.Expression):
        return _eval_ast_with_env(node.body, env)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id in env:
            return float(env[node.id])
        if node.id in _SAFE_CONSTANTS:
            return float(_SAFE_CONSTANTS[node.id])
        raise ValueError(f"Missing required parameter '{node.id}'")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_ast_with_env(node.left, env)
        right = _eval_ast_with_env(node.right, env)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent too large")
        return float(_BIN_OPS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return float(_UNARY_OPS[type(node.op)](_eval_ast_with_env(node.operand, env)))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fn = _SAFE_FUNCS[node.func.id]
        args = [_eval_ast_with_env(a, env) for a in node.args]
        return float(fn(*args))
    raise ValueError(f"Invalid AST node: {ast.dump(node)}")


def validate_custom_tool_spec(
    spec: CustomToolSpec,
    *,
    docs_dir: Path,
    reserved_names: set[str],
    has_tabular: bool,
) -> tuple[bool, str]:
    """Validate declarative CustomToolSpec safety and grounding constraints."""
    name = spec.name.strip().lower()
    if not _SAFE_NAME_RE.match(name):
        return False, f"Invalid tool identifier '{spec.name}'"
    if name in reserved_names:
        return False, f"Custom tool '{name}' collides with standard registry tool"

    if spec.tool_type == "formula":
        cleaned = (spec.expression or "").strip().replace("^", "**")
        if not cleaned or len(cleaned) > 220:
            return False, "Formula expression is empty or exceeds 220 chars"
        params = [p.strip() for p in spec.parameters if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", p.strip())]
        if not params:
            return False, "Formula must declare at least one named parameter"
        try:
            tree = ast.parse(cleaned, mode="eval")
            _collect_and_validate_ast(tree, set(params))
        except Exception as exc:
            return False, f"Formula AST rejected: {exc}"
        return True, "ok"

    if spec.tool_type == "tabular_filter":
        if not has_tabular:
            return False, "Corpus contains no CSV/TSV/XLSX files for tabular_filter macro"
        tabular = _resolve_tabular_files(docs_dir)
        if not tabular:
            return False, "No tabular files resolved inside docs_dir"
        if spec.target_file:
            names = {p.name.lower() for p in tabular}
            if Path(spec.target_file).name.lower() not in names:
                return False, f"Target file '{spec.target_file}' not found in corpus"
        return True, "ok"

    return False, f"Unsupported tool_type '{spec.tool_type}'"


def execute_custom_tool(
    spec: CustomToolSpec,
    args: dict[str, Any],
    docs_dir: Path,
) -> str:
    """Safely execute an AST-validated declarative CustomToolSpec."""
    if spec.tool_type == "formula":
        cleaned = spec.expression.strip().replace("^", "**")
        allowed = set(spec.parameters)
        tree = ast.parse(cleaned, mode="eval")
        _collect_and_validate_ast(tree, allowed)

        env: dict[str, float] = {}
        missing: list[str] = []
        for param in spec.parameters:
            if param not in args or args[param] is None or args[param] == "":
                missing.append(param)
                continue
            try:
                raw_num = str(args[param]).replace(",", "").strip()
                env[param] = float(raw_num)
            except (TypeError, ValueError):
                return json.dumps(
                    {
                        "custom_tool": spec.name,
                        "error": f"Parameter '{param}' must be numeric, got {args[param]!r}",
                    }
                )
        if missing:
            return json.dumps(
                {
                    "custom_tool": spec.name,
                    "error": f"Missing required numeric parameters: {missing}",
                    "required_parameters": spec.parameters,
                }
            )
        try:
            val = _eval_ast_with_env(tree, env)
            display_val: int | float = int(val) if val.is_integer() else round(val, 6)
            return json.dumps(
                {
                    "custom_tool": spec.name,
                    "expression": spec.expression,
                    "inputs": env,
                    "result": display_val,
                }
            )
        except Exception as exc:
            return json.dumps({"custom_tool": spec.name, "error": f"Formula evaluation failed: {exc}"})

    if spec.tool_type == "tabular_filter":
        merged_filters = dict(spec.preset_filters or {})
        if isinstance(args.get("filters"), dict):
            merged_filters.update({str(k): v for k, v in args["filters"].items()})
        lookup_args: dict[str, Any] = {
            "filename": spec.target_file or args.get("filename") or "",
            "query": args.get("query") or "",
            "filters": merged_filters,
            "sort_by": args.get("sort_by") or "",
            "ascending": args.get("ascending", True),
            "aggregate": args.get("aggregate") or "",
            "agg_column": args.get("agg_column") or "",
            "limit": args.get("limit", 8),
        }
        return run_csv_lookup(lookup_args, docs_dir)

    return json.dumps({"custom_tool": spec.name, "error": f"Unsupported custom tool type '{spec.tool_type}'"})
