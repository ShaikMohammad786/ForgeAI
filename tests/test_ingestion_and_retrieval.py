"""Unit tests for document loading, SHA-256 manifest diffing, chunking, and hybrid retrieval."""

from __future__ import annotations

from pathlib import Path

from forgeai.ingestion import chunk_corpus, diff_manifests, load_corpus
from forgeai.retrieval import HybridRetriever
from forgeai.schemas.agent import RetrievalConfig


def test_load_corpus_and_manifest_diff(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# Leave\nShore staff get 24 paid leave days.\n", encoding="utf-8")
    (tmp_path / "b.csv").write_text("sku,price\nSP-1,500\n", encoding="utf-8")
    c1 = load_corpus(tmp_path)
    assert len(c1.documents) == 2
    assert c1.has_csv is True

    # Modify a.md, add c.txt, remove b.csv
    (tmp_path / "a.md").write_text("# Leave\nShore staff get 26 paid leave days.\n", encoding="utf-8")
    (tmp_path / "c.txt").write_text("Travel meals are capped at USD 55 per day.", encoding="utf-8")
    (tmp_path / "b.csv").unlink()
    c2 = load_corpus(tmp_path)
    added, modified, removed = diff_manifests(c1.manifest, c2.manifest)
    assert added == ["c.txt"]
    assert modified == ["a.md"]
    assert removed == ["b.csv"]


def test_hybrid_retriever_ranks_relevant_above_off_topic(tmp_path: Path) -> None:
    (tmp_path / "hr.md").write_text(
        "# Annual Leave\nShore staff receive 24 days of paid annual leave.\n\n"
        "# Meal Cap\nDomestic travel meals are capped at USD 55 per day.\n",
        encoding="utf-8",
    )
    corpus = load_corpus(tmp_path)
    chunks = chunk_corpus(corpus, chunk_size=120, chunk_overlap=20)
    ret = HybridRetriever(chunks, use_dense=False)  # Fast deterministic BM25+lexical unit check
    hits = ret.search("How many days of paid annual leave do shore staff receive?", RetrievalConfig(top_k=2, min_score_threshold=0.0))
    assert hits
    assert "24 days" in hits[0].chunk.text

    miss = ret.search("What is the price of crude oil futures?", RetrievalConfig(top_k=2, min_score_threshold=0.25))
    assert miss == []


def test_subquery_decomposition_surfaces_multiple_documents(tmp_path: Path) -> None:
    (tmp_path / "leave.md").write_text(
        "# Leave Rules\nShore staff receive 24 days annual leave and 12 days sick leave.\n",
        encoding="utf-8",
    )
    (tmp_path / "expenses.md").write_text(
        "# Travel Meals\nDomestic travel meals are capped at USD 55 per day with receipts.\n",
        encoding="utf-8",
    )
    corpus = load_corpus(tmp_path)
    chunks = chunk_corpus(corpus, chunk_size=200, chunk_overlap=20)
    ret = HybridRetriever(chunks, use_dense=False)
    hits = ret.search_with_decomposition(
        "What is the annual leave entitlement and what is the domestic travel meal cap?",
        RetrievalConfig(top_k=2, min_score_threshold=0.05),
    )
    sources = {h.chunk.source for h in hits}
    assert "leave.md" in sources
    assert "expenses.md" in sources


def test_extended_document_formats_loading(tmp_path: Path) -> None:
    import docx
    import openpyxl

    # DOCX with paragraph + table
    doc = docx.Document()
    doc.add_heading("Safety Protocol", level=1)
    doc.add_paragraph("Crew must inspect lifeboats every Monday.")
    tbl = doc.add_table(rows=2, cols=2)
    tbl.cell(0, 0).text = "Zone"
    tbl.cell(0, 1).text = "Interval"
    tbl.cell(1, 0).text = "Engine Room"
    tbl.cell(1, 1).text = "Weekly"
    doc.save(str(tmp_path / "safety.docx"))

    # XLSX workbook
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Inventory"
    ws.append(["sku", "part_name", "qty"])
    ws.append(["MP-101", "Fuel Injector", 12])
    wb.save(str(tmp_path / "parts.xlsx"))

    # TSV, JSONL, YAML, HTML, XML, LOG
    (tmp_path / "rates.tsv").write_text("port\tfee_usd\nSingapore\t450\n", encoding="utf-8")
    (tmp_path / "events.jsonl").write_text('{"id": 1, "event": "drydock"}\n{"id": 2, "event": "sea_trial"}\n', encoding="utf-8")
    (tmp_path / "policy.yaml").write_text("policy:\n  retention_years: 7\n", encoding="utf-8")
    (tmp_path / "notice.html").write_text("<html><body><h1>Port Notice</h1><p>Berth 4 closed.</p></body></html>", encoding="utf-8")
    (tmp_path / "manifest.xml").write_text("<manifest><vessel>MV Ocean</vessel></manifest>", encoding="utf-8")
    (tmp_path / "bridge.log").write_text("2026-11-28 INFO Radar calibrated.\n", encoding="utf-8")

    corpus = load_corpus(tmp_path)
    names = {d.summary.filename: d for d in corpus.documents}
    assert set(names.keys()) == {
        "safety.docx",
        "parts.xlsx",
        "rates.tsv",
        "events.jsonl",
        "policy.yaml",
        "notice.html",
        "manifest.xml",
        "bridge.log",
    }
    assert corpus.has_csv is True
    assert corpus.has_structured_records is True
    assert "lifeboats" in names["safety.docx"].full_text
    assert "Engine Room | Weekly" in names["safety.docx"].full_text
    assert "Fuel Injector" in names["parts.xlsx"].full_text
    assert "Singapore" in names["rates.tsv"].full_text
    assert "sea_trial" in names["events.jsonl"].full_text
    assert "retention_years: 7" in names["policy.yaml"].full_text
    assert "Berth 4 closed." in names["notice.html"].full_text
    assert "MV Ocean" in names["manifest.xml"].full_text

    chunks = chunk_corpus(corpus, chunk_size=300, chunk_overlap=40)
    assert len(chunks) >= 8

