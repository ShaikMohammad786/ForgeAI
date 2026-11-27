"""Safe AST-based arithmetic and numeric calculator tool (no eval)."""

from __future__ import annotations

import ast
import json
import math
import operator
from typing import Any


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_SAFE_FUNCS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "ceil": math.ceil,
    "floor": math.floor,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent too large")
        return float(_BIN_OPS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return float(_UNARY_OPS[type(node.op)](_eval_node(node.operand)))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fn_name = node.func.id
        if fn_name not in _SAFE_FUNCS or node.keywords:
            raise ValueError(f"Unsupported function '{fn_name}'")
        args = [_eval_node(a) for a in node.args]
        return float(_SAFE_FUNCS[fn_name](*args))
    raise ValueError(f"Unsupported expression syntax: {ast.dump(node)}")


def run_calculator(args: dict[str, Any], _docs_dir: Path | None = None) -> str:
    """Safely evaluate arithmetic expressions."""
    expr = str(args.get("expression") or args.get("expr") or "").strip()
    if not expr:
        return json.dumps({"error": "Missing 'expression' argument."})
    # Normalize common symbols
    cleaned = expr.replace(",", "").replace("^", "**")
    try:
        tree = ast.parse(cleaned, mode="eval")
        value = _eval_node(tree)
        # Format cleanly as int if exact
        display_val: int | float = int(value) if value.is_integer() else round(value, 6)
        return json.dumps({"expression": expr, "result": display_val})
    except Exception as exc:
        return json.dumps({"expression": expr, "error": f"Calculation failed: {exc}"})
