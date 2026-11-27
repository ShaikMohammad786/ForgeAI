"""Multi-format document loader (PDF, CSV, Markdown, TXT, JSON) with SHA-256 manifest."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from pypdf import PdfReader

from forgeai.schemas.requirements import DocumentSummary


SUPPORTED_EXTENSIONS = {".pdf", ".csv", ".md", ".markdown", ".txt", ".json"}


@dataclass
class RawSection:
    """Logical section (PDF page/section, Markdown heading block, or CSV row group)."""

    source: str
    relative_path: str
    doc_type: str
    page_or_row: str
    heading: str
    text: str


@dataclass
class LoadedDocument:
    """Ingested source document plus structured sections and summary."""

    summary: DocumentSummary
    sections: list[RawSection] = field(default_factory=list)
    full_text: str = ""
    csv_path: Path | None = None


@dataclass
class LoadedCorpus:
    """All documents loaded from a folder plus SHA-256 manifest for regression diffing."""

    docs_dir: Path
    documents: list[LoadedDocument] = field(default_factory=list)
    manifest: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def filenames(self) -> list[str]:
        return [d.summary.filename for d in self.documents]

    @property
    def has_csv(self) -> bool:
        return any(d.summary.doc_type == "csv" for d in self.documents)

    def format_inventory(self) -> str:
        """Render concise deterministic summary of all files in the corpus."""
        lines: list[str] = []
        for doc in self.documents:
            s = doc.summary
            meta = f"type={s.doc_type}, count={s.num_pages_or_rows}"
            if s.columns:
                meta += f", columns={s.columns}"
            elif s.key_headings:
                meta += f", headings={s.key_headings[:6]}"
            lines.append(f"- {s.filename} ({meta})\n  Preview: {s.preview[:220]}")
        return "\n".join(lines)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _extract_headings(text: str, limit: int = 12) -> list[str]:
    headings: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or len(line) > 95:
            continue
        if line.startswith("#"):
            headings.append(line.lstrip("# ").strip())
        elif re.match(r"^(\d+(\.\d+)*|Section\s+\d+|Article\s+\d+)[\s.:-]+.+", line, re.I):
            headings.append(line)
        elif line.isupper() and 4 <= len(line) <= 65 and any(c.isalpha() for c in line):
            headings.append(line.title())
        if len(headings) >= limit:
            break
    # Preserve order while deduplicating
    return list(dict.fromkeys(headings))


def _load_pdf(path: Path, rel_path: str, sha: str) -> LoadedDocument:
    reader = PdfReader(str(path))
    sections: list[RawSection] = []
    all_pages: list[str] = []
    all_headings: list[str] = []

    for idx, page in enumerate(reader.pages, start=1):
        txt = (page.extract_text() or "").strip()
        if not txt:
            continue
        all_pages.append(txt)
        page_headings = _extract_headings(txt, limit=6)
        all_headings.extend(page_headings)
        heading = page_headings[0] if page_headings else f"Page {idx}"
        sections.append(
            RawSection(
                source=path.name,
                relative_path=rel_path,
                doc_type="pdf",
                page_or_row=f"Page {idx}",
                heading=heading,
                text=txt,
            )
        )

    full_text = "\n\n".join(all_pages)
    dedup_headings = list(dict.fromkeys(all_headings))[:12]
    preview = re.sub(r"\s+", " ", full_text)[:300]
    summary = DocumentSummary(
        filename=path.name,
        relative_path=rel_path,
        doc_type="pdf",
        sha256=sha,
        num_pages_or_rows=len(reader.pages),
        num_chunks=0,
        key_headings=dedup_headings,
        preview=preview,
    )
    return LoadedDocument(summary=summary, sections=sections, full_text=full_text)


def _load_csv(path: Path, rel_path: str, sha: str) -> LoadedDocument:
    df = pd.read_csv(path)
    cols = [str(c) for c in df.columns]
    sections: list[RawSection] = []

    schema_lines = [
        f"CSV Table: {path.name}",
        f"Total Rows: {len(df)}",
        f"Columns ({len(cols)}): {', '.join(cols)}",
    ]
    for col in cols:
        sample_vals = df[col].dropna().astype(str).head(4).tolist()
        schema_lines.append(f"  - {col}: examples={sample_vals}")
    schema_text = "\n".join(schema_lines)
    sections.append(
        RawSection(
            source=path.name,
            relative_path=rel_path,
            doc_type="csv",
            page_or_row="Schema",
            heading="CSV Schema & Columns",
            text=schema_text,
        )
    )

    batch_size = 6
    row_texts: list[str] = []
    for start in range(0, len(df), batch_size):
        batch = df.iloc[start : start + batch_size]
        lines = []
        for r_idx, row in batch.iterrows():
            pairs = [f"{c}: {row[c]}" for c in cols if pd.notna(row[c])]
            line = f"Row {int(r_idx) + 1} -> " + " | ".join(pairs)
            lines.append(line)
        block = "\n".join(lines)
        row_texts.append(block)
        sections.append(
            RawSection(
                source=path.name,
                relative_path=rel_path,
                doc_type="csv",
                page_or_row=f"Rows {start + 1}-{min(start + batch_size, len(df))}",
                heading=f"Catalogue Rows {start + 1}-{min(start + batch_size, len(df))}",
                text=block,
            )
        )

    full_text = schema_text + "\n\n" + "\n".join(row_texts)
    preview = f"Columns: {', '.join(cols)}. " + (row_texts[0][:180] if row_texts else "")
    summary = DocumentSummary(
        filename=path.name,
        relative_path=rel_path,
        doc_type="csv",
        sha256=sha,
        num_pages_or_rows=len(df),
        num_chunks=0,
        columns=cols,
        key_headings=["CSV Schema", f"{len(df)} data rows"],
        preview=re.sub(r"\s+", " ", preview)[:300],
    )
    return LoadedDocument(
        summary=summary, sections=sections, full_text=full_text, csv_path=path
    )


def _load_text_like(path: Path, rel_path: str, sha: str) -> LoadedDocument:
    ext = path.suffix.lower()
    raw = path.read_text(encoding="utf-8", errors="replace").strip()
    if ext == ".json":
        try:
            parsed = json.loads(raw)
            raw = json.dumps(parsed, indent=2)
        except Exception:
            pass
        doc_type = "json"
    elif ext in {".md", ".markdown"}:
        doc_type = "markdown"
    else:
        doc_type = "text"

    sections: list[RawSection] = []
    current_heading = "Overview"
    current_lines: list[str] = []
    sec_idx = 1

    for line in raw.splitlines():
        stripped = line.strip()
        is_heading = stripped.startswith("#") or bool(
            re.match(r"^(\d+(\.\d+)*|Section\s+\d+)[\s.:-]+\S+", stripped, re.I)
            and len(stripped) < 90
        )
        if is_heading and current_lines:
            sections.append(
                RawSection(
                    source=path.name,
                    relative_path=rel_path,
                    doc_type=doc_type,
                    page_or_row=f"Section {sec_idx}",
                    heading=current_heading,
                    text="\n".join(current_lines).strip(),
                )
            )
            sec_idx += 1
            current_heading = stripped.lstrip("# ").strip()
            current_lines = [stripped]
        else:
            if is_heading:
                current_heading = stripped.lstrip("# ").strip()
            current_lines.append(line)

    if current_lines:
        sections.append(
            RawSection(
                source=path.name,
                relative_path=rel_path,
                doc_type=doc_type,
                page_or_row=f"Section {sec_idx}",
                heading=current_heading,
                text="\n".join(current_lines).strip(),
            )
        )

    headings = _extract_headings(raw)
    preview = re.sub(r"\s+", " ", raw)[:300]
    summary = DocumentSummary(
        filename=path.name,
        relative_path=rel_path,
        doc_type=doc_type,
        sha256=sha,
        num_pages_or_rows=max(1, len(raw.splitlines())),
        num_chunks=0,
        key_headings=headings,
        preview=preview,
    )
    return LoadedDocument(summary=summary, sections=sections, full_text=raw)


def load_corpus(docs_dir: str | Path) -> LoadedCorpus:
    """Recursively load all supported documents from docs_dir."""
    root = Path(docs_dir).resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Document folder not found: {root}")

    files = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        raise ValueError(
            f"No supported documents ({', '.join(sorted(SUPPORTED_EXTENSIONS))}) found in {root}"
        )

    documents: list[LoadedDocument] = []
    manifest: dict[str, dict[str, Any]] = {}

    for file_path in files:
        rel_path = file_path.relative_to(root).as_posix()
        sha = _sha256_file(file_path)
        ext = file_path.suffix.lower()

        if ext == ".pdf":
            doc = _load_pdf(file_path, rel_path, sha)
        elif ext == ".csv":
            doc = _load_csv(file_path, rel_path, sha)
        else:
            doc = _load_text_like(file_path, rel_path, sha)

        documents.append(doc)
        manifest[rel_path] = {
            "filename": file_path.name,
            "sha256": sha,
            "size_bytes": file_path.stat().st_size,
            "doc_type": doc.summary.doc_type,
        }

    return LoadedCorpus(docs_dir=root, documents=documents, manifest=manifest)


def diff_manifests(
    old_manifest: dict[str, dict[str, Any]],
    new_manifest: dict[str, dict[str, Any]],
) -> tuple[list[str], list[str], list[str]]:
    """Return (added_files, modified_files, removed_files) between two corpus manifests."""
    old_keys = set(old_manifest.keys())
    new_keys = set(new_manifest.keys())
    added = sorted(new_keys - old_keys)
    removed = sorted(old_keys - new_keys)
    modified = sorted(
        k
        for k in (old_keys & new_keys)
        if old_manifest[k].get("sha256") != new_manifest[k].get("sha256")
    )
    return added, modified, removed
