"""Sandboxed JSON record lookup tool scoped strictly to the agent's docs folder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def run_json_record_lookup(args: dict[str, Any], docs_dir: Path) -> str:
    """Search and filter JSON objects/arrays inside docs_dir safely."""
    json_files = sorted(p for p in docs_dir.rglob("*.json") if p.is_file())
    if not json_files:
        return json.dumps({"error": "No JSON files exist in the document corpus."})

    requested = str(args.get("filename") or "").strip()
    target = json_files[0]
    if requested:
        matches = [p for p in json_files if p.name.lower() == Path(requested).name.lower()]
        if not matches:
            return json.dumps(
                {"error": f"JSON file '{requested}' not found.", "available_json_files": [p.name for p in json_files]}
            )
        target = matches[0]

    try:
        data = json.loads(target.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:
        return json.dumps({"error": f"Failed to parse {target.name}: {exc}"})

    query = str(args.get("query") or "").strip().lower()
    limit = min(max(1, int(args.get("limit", 10))), 25)
    records = data if isinstance(data, list) else [data]

    if query:
        tokens = [t for t in query.split() if len(t) >= 2]
        scored = []
        for rec in records:
            blob = json.dumps(rec, default=str).lower()
            hits = sum(1 for t in tokens if t in blob)
            if hits > 0:
                scored.append((hits, rec))
        scored.sort(key=lambda x: x[0], reverse=True)
        matched = [r for _, r in scored[:limit]]
    else:
        matched = records[:limit]

    return json.dumps(
        {
            "file": target.name,
            "total_records": len(records),
            "matches_returned": len(matched),
            "records": matched,
        },
        default=str,
    )
