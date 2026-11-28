"""Sandboxed document text analysis and regex search tool scoped strictly to docs_dir."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from forgeai.ingestion.loader import load_corpus


def run_text_stats_and_regex(args: dict[str, Any], docs_dir: Path) -> str:
    """Compute document word/line stats or safely search regex patterns across corpus documents."""
    pattern = str(args.get("pattern") or args.get("keyword") or "").strip()
    requested_file = str(args.get("filename") or "").strip()
    max_matches = min(max(1, int(args.get("limit", 12))), 25)
    case_sensitive = bool(args.get("case_sensitive", False))

    if len(pattern) > 120:
        return json.dumps({"error": "Regex/keyword pattern exceeds 120-character safety limit."})

    corpus = load_corpus(docs_dir)
    docs = corpus.documents
    if requested_file:
        docs = [d for d in docs if d.summary.filename.lower() == Path(requested_file).name.lower()]
        if not docs:
            return json.dumps(
                {
                    "error": f"Document '{requested_file}' not found.",
                    "available_files": corpus.filenames,
                }
            )

    if not pattern:
        stats = [
            {
                "filename": d.summary.filename,
                "doc_type": d.summary.doc_type,
                "word_count": len(d.full_text.split()),
                "line_count": len(d.full_text.splitlines()),
                "char_count": d.summary.total_chars,
                "sections": len(d.sections),
            }
            for d in docs
        ]
        return json.dumps({"documents_analyzed": len(stats), "stats": stats})

    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        regex = re.compile(pattern, flags)
    except re.error as exc:
        # Fallback to escaped literal search if user supplied invalid regex syntax
        regex = re.compile(re.escape(pattern), flags)
        fallback_note = f"Invalid regex compiled as literal ({exc})"
    else:
        fallback_note = ""

    matches: list[dict[str, Any]] = []
    counts_by_file: dict[str, int] = {}

    for doc in docs:
        file_hits = 0
        for line_no, line in enumerate(doc.full_text.splitlines(), start=1):
            clipped_line = line[:1000]
            found = list(regex.finditer(clipped_line))
            if not found:
                continue
            file_hits += len(found)
            if len(matches) < max_matches:
                matches.append(
                    {
                        "filename": doc.summary.filename,
                        "line_number": line_no,
                        "matched_text": found[0].group(0)[:120],
                        "line_snippet": clipped_line[:240],
                    }
                )
        if file_hits > 0:
            counts_by_file[doc.summary.filename] = file_hits

    payload: dict[str, Any] = {
        "pattern": pattern,
        "total_matches": sum(counts_by_file.values()),
        "counts_by_file": counts_by_file,
        "matches": matches,
    }
    if fallback_note:
        payload["note"] = fallback_note
    return json.dumps(payload)
