"""Multi-format document loader (PDF, DOCX, XLSX, CSV, TSV, JSON, JSONL, YAML, HTML, XML, MD, TXT) with SHA-256 manifest."""

from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pypdf import PdfReader

from forgeai.schemas.requirements import DocumentSummary


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".xlsx",
    ".csv",
    ".tsv",
    ".md",
    ".markdown",
    ".txt",
    ".log",
    ".rst",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".html",
    ".htm",
    ".xml",
}


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
        return any(d.summary.doc_type in {"csv", "tsv", "xlsx"} for d in self.documents)

    @property
    def has_structured_records(self) -> bool:
        return any(d.summary.doc_type in {"json", "jsonl", "yaml"} for d in self.documents)

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


class _HTMLTextExtractor(HTMLParser):
    """Strip scripts/styles and extract clean text blocks from HTML files."""

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript"}:
            self._skip_depth += 1
        elif tag.lower() in {"h1", "h2", "h3", "h4", "p", "li", "tr", "div", "section"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript"} and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data.strip():
            self.parts.append(data.strip() + " ")


def _dataframe_to_doc(
    df: pd.DataFrame,
    path: Path,
    rel_path: str,
    sha: str,
    doc_type: str,
    sheet_label: str = "",
) -> LoadedDocument:
    cols = [str(c) for c in df.columns]
    sections: list[RawSection] = []
    prefix = f"{path.name} ({sheet_label})" if sheet_label else path.name

    schema_lines = [
        f"Tabular File: {prefix}",
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
            doc_type=doc_type,
            page_or_row="Schema",
            heading=f"{doc_type.upper()} Schema & Columns",
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
                doc_type=doc_type,
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
        doc_type=doc_type,
        sha256=sha,
        num_pages_or_rows=len(df),
        num_chunks=0,
        columns=cols,
        key_headings=[f"{doc_type.upper()} Schema", f"{len(df)} data rows"],
        preview=re.sub(r"\s+", " ", preview)[:300],
    )
    return LoadedDocument(
        summary=summary, sections=sections, full_text=full_text, csv_path=path
    )


def _load_csv(path: Path, rel_path: str, sha: str, sep: str = ",", doc_type: str = "csv") -> LoadedDocument:
    df = pd.read_csv(path, sep=sep)
    return _dataframe_to_doc(df, path, rel_path, sha, doc_type)


def _load_xlsx(path: Path, rel_path: str, sha: str) -> LoadedDocument:
    excel = pd.ExcelFile(path, engine="openpyxl")
    frames = []
    for sheet in excel.sheet_names:
        sdf = pd.read_excel(excel, sheet_name=sheet)
        if not sdf.empty:
            if len(excel.sheet_names) > 1:
                sdf = sdf.copy()
                sdf.insert(0, "_sheet", sheet)
            frames.append(sdf)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return _dataframe_to_doc(
        combined,
        path,
        rel_path,
        sha,
        "xlsx",
        sheet_label=", ".join(excel.sheet_names),
    )


def _load_docx(path: Path, rel_path: str, sha: str) -> LoadedDocument:
    from docx import Document as DocxDocument

    doc = DocxDocument(str(path))
    sections: list[RawSection] = []
    current_heading = "Overview"
    current_lines: list[str] = []
    sec_idx = 1

    for para in doc.paragraphs:
        txt = para.text.strip()
        if not txt:
            continue
        style_name = (para.style.name or "").lower() if para.style else ""
        is_heading = "heading" in style_name or (
            len(txt) < 90
            and bool(re.match(r"^(\d+(\.\d+)*|Section\s+\d+)[\s.:-]+\S+", txt, re.I))
        )
        if is_heading and current_lines:
            sections.append(
                RawSection(
                    source=path.name,
                    relative_path=rel_path,
                    doc_type="docx",
                    page_or_row=f"Section {sec_idx}",
                    heading=current_heading,
                    text="\n".join(current_lines).strip(),
                )
            )
            sec_idx += 1
            current_heading = txt
            current_lines = [txt]
        else:
            if is_heading:
                current_heading = txt
            current_lines.append(txt)

    if current_lines:
        sections.append(
            RawSection(
                source=path.name,
                relative_path=rel_path,
                doc_type="docx",
                page_or_row=f"Section {sec_idx}",
                heading=current_heading,
                text="\n".join(current_lines).strip(),
            )
        )

    for t_idx, tbl in enumerate(doc.tables, start=1):
        tbl_lines = []
        for row in tbl.rows:
            cells = [c.text.strip().replace("\n", " ") for c in row.cells]
            if any(cells):
                tbl_lines.append(" | ".join(cells))
        if tbl_lines:
            sections.append(
                RawSection(
                    source=path.name,
                    relative_path=rel_path,
                    doc_type="docx",
                    page_or_row=f"Table {t_idx}",
                    heading=f"Table {t_idx}",
                    text="\n".join(tbl_lines),
                )
            )

    full_text = "\n\n".join(s.text for s in sections)
    headings = [s.heading for s in sections[:12]]
    summary = DocumentSummary(
        filename=path.name,
        relative_path=rel_path,
        doc_type="docx",
        sha256=sha,
        num_pages_or_rows=max(1, len(sections)),
        num_chunks=0,
        key_headings=list(dict.fromkeys(headings)),
        preview=re.sub(r"\s+", " ", full_text)[:300],
    )
    return LoadedDocument(summary=summary, sections=sections, full_text=full_text)


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
    elif ext == ".jsonl":
        parsed_rows = []
        for ln in raw.splitlines():
            if ln.strip():
                try:
                    parsed_rows.append(json.dumps(json.loads(ln)))
                except Exception:
                    parsed_rows.append(ln.strip())
        raw = "\n".join(parsed_rows)
        doc_type = "jsonl"
    elif ext in {".yaml", ".yml"}:
        try:
            parsed = yaml.safe_load(raw)
            raw = yaml.safe_dump(parsed, sort_keys=False)
        except Exception:
            pass
        doc_type = "yaml"
    elif ext in {".html", ".htm"}:
        parser = _HTMLTextExtractor()
        parser.feed(raw)
        raw = re.sub(r"\n{3,}", "\n\n", "".join(parser.parts)).strip()
        doc_type = "html"
    elif ext == ".xml":
        try:
            root_el = ET.fromstring(raw)
            xml_lines: list[str] = []
            for elem in root_el.iter():
                val = (elem.text or "").strip()
                attrs = " ".join(f"{k}={v}" for k, v in elem.attrib.items())
                if val or attrs:
                    xml_lines.append(f"{elem.tag} ({attrs}): {val}".strip(" ():"))
            raw = "\n".join(xml_lines) or raw
        except Exception:
            pass
        doc_type = "xml"
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
        elif ext == ".docx":
            doc = _load_docx(file_path, rel_path, sha)
        elif ext == ".xlsx":
            doc = _load_xlsx(file_path, rel_path, sha)
        elif ext == ".csv":
            doc = _load_csv(file_path, rel_path, sha, sep=",", doc_type="csv")
        elif ext == ".tsv":
            doc = _load_csv(file_path, rel_path, sha, sep="\t", doc_type="tsv")
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
