"""Heading-aware and paragraph-aware recursive chunking."""

from __future__ import annotations

import re
from dataclasses import dataclass

from forgeai.ingestion.loader import LoadedCorpus, RawSection


@dataclass
class DocumentChunk:
    """Single indexed unit with provenance metadata."""

    chunk_id: str
    source: str
    relative_path: str
    doc_type: str
    page_or_row: str
    heading: str
    text: str

    def format_for_prompt(self) -> str:
        return (
            f"[Source: {self.source} | {self.page_or_row} | Section: {self.heading}]\n"
            f"{self.text}"
        )


def _split_recursive(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    clean = text.strip()
    if not clean:
        return []
    if len(clean) <= chunk_size:
        return [clean]

    step = max(80, chunk_size - min(chunk_overlap, chunk_size // 2))
    chunks: list[str] = []
    start = 0
    n = len(clean)

    while start < n:
        end = min(n, start + chunk_size)
        if end < n:
            # Prefer breaking cleanly on paragraph, newline, or sentence boundary
            window = clean[start:end]
            cut = -1
            for sep in ("\n\n", "\n", ". ", "; ", " "):
                pos = window.rfind(sep)
                if pos >= int(chunk_size * 0.45):
                    cut = pos + len(sep)
                    break
            if cut > 0:
                end = start + cut

        piece = clean[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        next_start = max(start + 1, end - chunk_overlap)
        if next_start <= start:
            next_start = start + step
        start = next_start

    return chunks


def _chunk_sections(
    sections: list[RawSection],
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[RawSection, str]]:
    out: list[tuple[RawSection, str]] = []
    for sec in sections:
        # Keep CSV schema/row batches compact and intact if reasonably sized
        limit = max(chunk_size, 850) if sec.doc_type == "csv" else chunk_size
        parts = _split_recursive(sec.text, limit, chunk_overlap)
        for p in parts:
            out.append((sec, p))
    return out


def chunk_corpus(
    corpus: LoadedCorpus,
    *,
    chunk_size: int = 600,
    chunk_overlap: int = 100,
    strategy: str = "heading_recursive",
) -> list[DocumentChunk]:
    """Chunk every document in LoadedCorpus according to RetrievalConfig."""
    chunks: list[DocumentChunk] = []
    counter = 1

    for doc in corpus.documents:
        doc_chunks_start = len(chunks)
        if strategy == "paragraph_recursive" and doc.summary.doc_type != "csv":
            normalized = re.sub(r"\n{3,}", "\n\n", doc.full_text)
            pseudo = RawSection(
                source=doc.summary.filename,
                relative_path=doc.summary.relative_path,
                doc_type=doc.summary.doc_type,
                page_or_row="Document",
                heading=doc.summary.key_headings[0] if doc.summary.key_headings else "Body",
                text=normalized,
            )
            pairs = _chunk_sections([pseudo], chunk_size, chunk_overlap)
        else:
            pairs = _chunk_sections(doc.sections, chunk_size, chunk_overlap)

        for sec, piece in pairs:
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{doc.summary.filename}#c{counter}",
                    source=sec.source,
                    relative_path=sec.relative_path,
                    doc_type=sec.doc_type,
                    page_or_row=sec.page_or_row,
                    heading=sec.heading,
                    text=piece,
                )
            )
            counter += 1

        doc.summary.num_chunks = len(chunks) - doc_chunks_start

    return chunks
