"""Safe sandboxed tabular lookup tool (CSV, TSV, XLSX) scoped strictly to docs_dir."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

TABULAR_EXTENSIONS = {".csv", ".tsv", ".xlsx"}


def _resolve_tabular_files(docs_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in docs_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in TABULAR_EXTENSIONS
    )


def _read_tabular_df(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".tsv":
        return pd.read_csv(path, sep="\t")
    if ext == ".xlsx":
        excel = pd.ExcelFile(path, engine="openpyxl")
        frames = []
        for sheet in excel.sheet_names:
            sdf = pd.read_excel(excel, sheet_name=sheet)
            if not sdf.empty:
                if len(excel.sheet_names) > 1:
                    sdf = sdf.copy()
                    sdf.insert(0, "_sheet", sheet)
                frames.append(sdf)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return pd.DataFrame()


def run_csv_lookup(args: dict[str, Any], docs_dir: Path) -> str:
    """Search, filter, sort, or aggregate CSV, TSV, and XLSX files inside docs_dir safely."""
    tabular_files = _resolve_tabular_files(docs_dir)
    if not tabular_files:
        return json.dumps({"error": "No CSV, TSV, or XLSX files exist in the document corpus."})

    requested_file = str(args.get("filename") or "").strip()
    target_path = tabular_files[0]
    if requested_file:
        matches = [p for p in tabular_files if p.name.lower() == Path(requested_file).name.lower()]
        if not matches:
            return json.dumps(
                {
                    "error": f"Tabular file '{requested_file}' not found.",
                    "available_tabular_files": [p.name for p in tabular_files],
                }
            )
        target_path = matches[0]

    df = _read_tabular_df(target_path)
    if df.empty:
        return json.dumps({"file": target_path.name, "columns": [], "total_matches": 0, "rows": []})

    query = str(args.get("query") or "").strip()
    filters = args.get("filters") or {}
    sort_by = str(args.get("sort_by") or "").strip()
    ascending = bool(args.get("ascending", True))
    aggregate = str(args.get("aggregate") or "").strip().lower()
    agg_column = str(args.get("agg_column") or "").strip()
    limit = min(max(1, int(args.get("limit", 8))), 20)

    filtered = df.copy()

    # Apply structured column filters if provided
    if isinstance(filters, dict):
        for col, val in filters.items():
            matched_cols = [c for c in filtered.columns if str(c).lower() == str(col).lower()]
            if not matched_cols:
                continue
            c_name = matched_cols[0]
            if isinstance(val, dict):
                if "min" in val:
                    filtered = filtered[pd.to_numeric(filtered[c_name], errors="coerce") >= float(val["min"])]
                if "max" in val:
                    filtered = filtered[pd.to_numeric(filtered[c_name], errors="coerce") <= float(val["max"])]
                if "contains" in val:
                    filtered = filtered[
                        filtered[c_name].astype(str).str.contains(str(val["contains"]), case=False, na=False)
                    ]
            else:
                filtered = filtered[
                    filtered[c_name].astype(str).str.contains(str(val), case=False, na=False)
                ]

    # Apply keyword query across all columns (OR over tokens if multi-word)
    if query:
        row_strings = filtered.astype(str).agg(" | ".join, axis=1)
        tokens = [t for t in query.lower().split() if len(t) >= 2]
        if tokens:
            scores = row_strings.str.lower().apply(
                lambda s: sum(1 for t in tokens if t in s)
            )
            matched = filtered[scores > 0].copy()
            if not matched.empty:
                matched["_match_score"] = scores[scores > 0]
                filtered = matched.sort_values("_match_score", ascending=False).drop(
                    columns=["_match_score"]
                )

    if sort_by and sort_by in filtered.columns and not filtered.empty:
        filtered = filtered.sort_values(sort_by, ascending=ascending)

    # Optional aggregation
    if aggregate:
        if aggregate == "count":
            return json.dumps({"file": target_path.name, "count": int(len(filtered))})
        if agg_column and agg_column in filtered.columns:
            series = pd.to_numeric(filtered[agg_column], errors="coerce").dropna()
            if series.empty:
                return json.dumps({"file": target_path.name, "error": f"No numeric data in '{agg_column}'."})
            ops = {
                "sum": float(series.sum()),
                "mean": float(series.mean()),
                "avg": float(series.mean()),
                "min": float(series.min()),
                "max": float(series.max()),
            }
            if aggregate in ops:
                return json.dumps(
                    {
                        "file": target_path.name,
                        "aggregate": aggregate,
                        "column": agg_column,
                        "value": round(ops[aggregate], 4),
                        "row_count": int(len(series)),
                    }
                )

    rows = filtered.head(limit).fillna("").to_dict(orient="records")
    return json.dumps(
        {
            "file": target_path.name,
            "columns": [str(c) for c in df.columns],
            "total_matches": int(len(filtered)),
            "rows": rows,
        },
        default=str,
    )
