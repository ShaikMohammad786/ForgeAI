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
