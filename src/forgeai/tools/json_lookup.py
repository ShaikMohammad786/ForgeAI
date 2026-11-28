"""Sandboxed structured record lookup tool (JSON, JSONL, YAML) scoped strictly to docs_dir."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

RECORD_EXTENSIONS = {".json", ".jsonl", ".yaml", ".yml"}


def _load_records(path: Path) -> list[Any]:
    ext = path.suffix.lower()
    raw = path.read_text(encoding="utf-8", errors="replace")
    if ext == ".jsonl":
        records = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                records.append({"raw_line": line})
        return records
    if ext in {".yaml", ".yml"}:
        data = yaml.safe_load(raw)
        if data is None:
            return []
        return data if isinstance(data, list) else [data]
    data = json.loads(raw)
    return data if isinstance(data, list) else [data]


def run_json_record_lookup(args: dict[str, Any], docs_dir: Path) -> str:
    """Search and filter JSON, JSONL, and YAML record files inside docs_dir safely."""
    record_files = sorted(
        p
        for p in docs_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in RECORD_EXTENSIONS
    )
    if not record_files:
        return json.dumps({"error": "No JSON, JSONL, or YAML files exist in the document corpus."})

    requested = str(args.get("filename") or "").strip()
    target = record_files[0]
    if requested:
        matches = [p for p in record_files if p.name.lower() == Path(requested).name.lower()]
        if not matches:
            return json.dumps(
                {
                    "error": f"Structured file '{requested}' not found.",
                    "available_record_files": [p.name for p in record_files],
                }
            )
        target = matches[0]

    try:
        records = _load_records(target)
    except Exception as exc:
        return json.dumps({"error": f"Failed to parse {target.name}: {exc}"})

    query = str(args.get("query") or "").strip().lower()
    limit = min(max(1, int(args.get("limit", 10))), 25)

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
