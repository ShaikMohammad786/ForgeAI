"""Stage 2: Groundability Guard — verifies requested brief topics against corpus evidence."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from forgeai.ingestion.chunker import chunk_corpus
from forgeai.ingestion.loader import LoadedCorpus
from forgeai.llm.base import LLMProvider
from forgeai.prompts import render_prompt
from forgeai.retrieval.hybrid import HybridRetriever
from forgeai.schemas.agent import RetrievalConfig
from forgeai.schemas.requirements import GroundabilityCheckItem, GroundabilityReport


_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "about", "what",
    "when", "where", "which", "would", "could", "should", "need", "assistant",
    "agent", "question", "questions", "answer", "answers", "help", "staff", "users",
    "document", "documents", "policy", "policies", "information", "based", "using",
}


class _LLMGroundabilityItem(BaseModel):
    requested_topic: str
    status: str
    matching_sources: list[str] = Field(default_factory=list)
    rationale: str = ""


class _LLMGroundabilityOutput(BaseModel):
    items: list[_LLMGroundabilityItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


_REFUSAL_CLAUSE_RE = re.compile(
    r"\b(?:while\s+refusing|refus(?:e|ing)\s+(?:to\s+answer\s+)?|must\s+refuse|do\s+not\s+answer|except\s+for)\b(.+)$",
    re.IGNORECASE,
)


def split_brief_capabilities_and_refusals(brief: str) -> tuple[str, list[str]]:
    """Separate positive requested capabilities from explicit refusal boundary clauses."""
    m = _REFUSAL_CLAUSE_RE.search(brief)
    if not m:
        return brief, []
    positive_part = brief[: m.start()].strip(" ,.;:-") or brief
    refusal_raw = m.group(1).strip(" .")
    refusal_items = [
        re.sub(r"\s+", " ", p).strip(" ,:-")
        for p in re.split(r",|\bor\b|\band\b", refusal_raw, flags=re.I)
        if len(p.strip(" ,:-")) >= 5
    ]
    return positive_part, list(dict.fromkeys(refusal_items))


def _extract_candidate_topics(brief: str) -> list[str]:
    """Deterministically split positive brief capabilities into testable clauses."""
    positive_brief, _ = split_brief_capabilities_and_refusals(brief)
    raw_parts = re.split(r"[.;\n]+|\band\b|\bplus\b|\balso\b", positive_brief, flags=re.I)
    topics: list[str] = []
    for part in raw_parts:
        clean = re.sub(r"\s+", " ", part).strip(" ,:-")
        clean = re.sub(
            r"^(i\s+need\s+(an?\s+)?(ai\s+)?(assistant|agent|bot)\s+(that|to|for)\s+|build\s+(an?\s+)?(ai\s+)?(assistant|agent|bot)\s+(that|to|for)\s+)",
            "",
            clean,
            flags=re.I,
        ).strip()
        if len(clean) >= 10:
            topics.append(clean[:140])

    if not topics:
        topics = [positive_brief.strip()[:160]]
    return list(dict.fromkeys(topics))[:8]


def _lexical_overlap(topic: str, text: str) -> float:
    t_words = {
        w.lower()
        for w in re.findall(r"[a-zA-Z0-9_-]{3,}", topic)
        if w.lower() not in _STOPWORDS
    }
    if not t_words:
        return 0.0
    corpus_words = set(re.findall(r"[a-zA-Z0-9_-]{3,}", text.lower()))
    hits = sum(1 for w in t_words if w in corpus_words)
    return hits / len(t_words)


def audit_groundability(
    brief: str,
    corpus: LoadedCorpus,
    llm: LLMProvider,
    *,
    embed_model_name: str = "BAAI/bge-small-en-v1.5",
) -> GroundabilityReport:
    """Verify every requested capability in the brief against indexed corpus chunks."""
    chunks = chunk_corpus(corpus, chunk_size=600, chunk_overlap=100)
    retriever = HybridRetriever(chunks, embed_model_name=embed_model_name, use_dense=True)
    topics = _extract_candidate_topics(brief)

    evidence_blocks: list[str] = []
    deterministic_stats: dict[str, dict] = {}
    corpus_full_lower = "\n".join(d.full_text.lower() for d in corpus.documents)

    for topic in topics:
        hits = retriever.search(
            topic,
            RetrievalConfig(top_k=4, hybrid_alpha=0.5, min_score_threshold=0.0),
            apply_threshold=False,
        )
        max_score = hits[0].score if hits else 0.0
        sources = list(dict.fromkeys(h.chunk.source for h in hits if h.score >= 0.12))
        snippets = [
            f"[{h.chunk.source} | score={h.score:.2f}] {h.chunk.text[:220]}"
            for h in hits[:3]
        ]
        overlap = _lexical_overlap(
            topic,
            " ".join(h.chunk.text for h in hits[:3]) + " " + corpus_full_lower[:8000],
        )
        deterministic_stats[topic] = {
            "max_score": max_score,
            "overlap": round(overlap, 3),
            "sources": sources,
            "snippets": snippets,
        }
        evidence_blocks.append(
            f"Topic: {topic}\n"
            f"Max Retrieval Score: {max_score:.3f}, Keyword Overlap: {overlap:.2f}\n"
            + ("\n".join(snippets) if snippets else "No relevant snippets found.")
        )

    prompt = render_prompt(
        "groundability_check.j2",
        brief=brief,
        corpus_inventory=corpus.format_inventory(),
        evidence_summary="\n\n---\n\n".join(evidence_blocks),
    )

    llm_items_by_topic: dict[str, _LLMGroundabilityItem] = {}
    llm_warnings: list[str] = []
    try:
        parsed, _ = llm.complete_structured(
            [
                {"role": "system", "content": "You output strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            _LLMGroundabilityOutput,
            temperature=0.0,
            max_tokens=1000,
        )
        llm_warnings = parsed.warnings
        for item in parsed.items:
            llm_items_by_topic[item.requested_topic.lower()] = item
    except Exception:
        # Proceed cleanly using deterministic verification below
        pass

    final_items: list[GroundabilityCheckItem] = []
    unsupported_requests: list[str] = []
    warnings: list[str] = list(llm_warnings)

    for topic in topics:
        stats = deterministic_stats[topic]
        max_score = float(stats["max_score"])
        overlap = float(stats["overlap"])
        matched_sources: list[str] = stats["sources"]
        snippets: list[str] = stats["snippets"]

        # Find best matching LLM verdict if present
        llm_item = llm_items_by_topic.get(topic.lower())
        if llm_item is None:
            for k, v in llm_items_by_topic.items():
                if k in topic.lower() or topic.lower() in k:
                    llm_item = v
                    break

        llm_status = (llm_item.status.lower().strip() if llm_item else "partial")
        if llm_status not in {"supported", "partial", "unsupported"}:
            llm_status = "partial"

        # Deterministic anti-hallucination safety net:
        # Small models often over-claim support; override when retrieval evidence is absent.
        if max_score < 0.18 and overlap < 0.18:
            status = "unsupported"
            rationale = (
                f"Insufficient corpus evidence (retrieval={max_score:.2f}, lexical_overlap={overlap:.2f}). "
                + (llm_item.rationale if llm_item and llm_item.rationale else "No supporting rules or records located.")
            )
        elif llm_status == "unsupported" or (max_score < 0.25 and overlap < 0.25):
            status = "unsupported" if overlap < 0.2 else "partial"
            rationale = (
                llm_item.rationale
                if llm_item and llm_item.rationale
                else f"Limited evidence found (score={max_score:.2f}, overlap={overlap:.2f})."
            )
        elif llm_status == "partial" or overlap < 0.35:
            status = "partial"
            rationale = (
                llm_item.rationale
                if llm_item and llm_item.rationale
                else f"Partial coverage found in {', '.join(matched_sources) or 'corpus'}."
            )
        else:
            status = "supported"
            rationale = (
                llm_item.rationale
                if llm_item and llm_item.rationale
                else f"Directly supported in {', '.join(matched_sources) or 'corpus'}."
            )

        if status == "unsupported":
            unsupported_requests.append(topic)
            warnings.append(
                f"GROUNDABILITY GAP: Requested topic '{topic}' is NOT supported by the documents "
                "and has been moved to must-refuse rules to prevent hallucination."
            )
        elif status == "partial":
            warnings.append(
                f"PARTIAL COVERAGE: '{topic}' is only partially documented ({', '.join(matched_sources) or 'weak match'})."
            )

        final_items.append(
            GroundabilityCheckItem(
                requested_topic=topic,
                status=status,  # type: ignore[arg-type]
                evidence_score=round(min(1.0, max(max_score, overlap)), 3),
                matching_sources=matched_sources,
                matching_snippets=snippets,
                rationale=rationale.strip(),
            )
        )

    score_sum = sum(
        1.0 if it.status == "supported" else (0.5 if it.status == "partial" else 0.0)
        for it in final_items
    )
    ratio = round(score_sum / max(1, len(final_items)), 3)

    if ratio < 0.2:
        overall = "ungroundable"
        warnings.insert(
            0,
            "CRITICAL GROUNDABILITY BLOCK: The document corpus cannot support the primary brief requirements.",
        )
    elif unsupported_requests or any(it.status == "partial" for it in final_items):
        overall = "partially_grounded"
    else:
        overall = "grounded"

    dedup_warnings = list(dict.fromkeys(w.strip() for w in warnings if w.strip()))
    return GroundabilityReport(
        overall_status=overall,  # type: ignore[arg-type]
        groundability_ratio=ratio,
        items=final_items,
        unsupported_requests=unsupported_requests,
        warnings=dedup_warnings,
    )
