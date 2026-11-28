"""Deterministic descriptive statistics and financial KPI calculator tool."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any


def _to_float_list(raw: Any) -> list[float]:
    if not isinstance(raw, list):
        return []
    out: list[float] = []
    for item in raw:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            continue
    return out


def run_statistics_calculator(args: dict[str, Any], _docs_dir: Path | None = None) -> str:
    """Compute exact summary statistics, weighted averages, or percentage changes."""
    old_val = args.get("old_value")
    new_val = args.get("new_value")
    if old_val is not None and new_val is not None:
        try:
            o = float(old_val)
            n = float(new_val)
            if o == 0:
                return json.dumps({"error": "old_value cannot be 0 for percent_change."})
            pct = ((n - o) / abs(o)) * 100.0
            return json.dumps(
                {
                    "operation": "percent_change",
                    "old_value": o,
                    "new_value": n,
                    "absolute_delta": round(n - o, 6),
                    "percent_change": round(pct, 4),
                }
            )
        except (TypeError, ValueError) as exc:
            return json.dumps({"error": f"Invalid numeric values: {exc}"})

    values = _to_float_list(args.get("values"))
    if not values:
        return json.dumps({"error": "Provide a non-empty 'values' list or 'old_value'/'new_value'."})

    weights = _to_float_list(args.get("weights"))
    weighted_avg: float | None = None
    if weights:
        if len(weights) != len(values):
            return json.dumps({"error": "Length of 'weights' must match length of 'values'."})
        total_w = sum(weights)
        if total_w == 0:
            return json.dumps({"error": "Sum of 'weights' cannot be zero."})
        weighted_avg = round(sum(v * w for v, w in zip(values, weights)) / total_w, 6)

    result: dict[str, Any] = {
        "count": len(values),
        "sum": round(sum(values), 6),
        "mean": round(statistics.mean(values), 6),
        "median": round(statistics.median(values), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "range": round(max(values) - min(values), 6),
        "stdev": round(statistics.stdev(values), 6) if len(values) >= 2 else 0.0,
    }
    if weighted_avg is not None:
        result["weighted_avg"] = weighted_avg
    return json.dumps(result)
