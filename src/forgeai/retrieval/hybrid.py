"""Hybrid BM25 + FastEmbed dense retriever with calibrated gating."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi

from forgeai.ingestion.chunker import DocumentChunk
from forgeai.schemas.agent import RetrievalConfig


_TOKEN_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._-]*")


def _tokenize(text: str) -> list[str]:
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    # Also add split tokens on hyphens/underscores for SKUs like SP-MV-104
    expanded: list[str] = []
    for tok in tokens:
        expanded.append(tok)
        if "-" in tok or "_" in tok:
            for part in re.split(r"[-_]", tok):
                if len(part) >= 2:
                    expanded.append(part)
    return expanded


@lru_cache(maxsize=2)
def _get_fastembed_model(model_name: str):
    """Lazily initialize FastEmbed ONNX model on CPU (preserves GPU VRAM for Ollama)."""
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=model_name)


@dataclass
class RetrievedChunk:
    """Single retrieved chunk with calibrated hybrid score and component breakdown."""

    chunk: DocumentChunk
    score: float
    bm25_score: float
    dense_score: float


class HybridRetriever:
    """Combines BM25Okapi lexical precision and BGE-small dense vector recall."""

    def __init__(
        self,
        chunks: list[DocumentChunk],
        *,
        embed_model_name: str = "BAAI/bge-small-en-v1.5",
        use_dense: bool = True,
    ) -> None:
        self.chunks = chunks
        self.embed_model_name = embed_model_name
        self.use_dense = use_dense

        corpus_tokens = [_tokenize(f"{c.heading} {c.source} {c.text}") for c in chunks]
        self._bm25 = BM25Okapi(corpus_tokens) if chunks else None

        self._dense_matrix: np.ndarray | None = None
        self._tfidf_fallback = False

        if chunks and use_dense:
            try:
                model = _get_fastembed_model(embed_model_name)
                passages = [
                    f"passage: {c.source} | {c.heading}\n{c.text}" for c in chunks
                ]
                vecs = np.vstack(list(model.embed(passages))).astype(np.float32)
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)
                self._dense_matrix = vecs / norms
            except Exception:
                # Deterministic mathematical TF-IDF fallback if ONNX model unavailable
                self._tfidf_fallback = True
                self._build_tfidf_fallback(corpus_tokens)

    def _build_tfidf_fallback(self, corpus_tokens: list[list[str]]) -> None:
        vocab: dict[str, int] = {}
        df: Counter[str] = Counter()
        for toks in corpus_tokens:
            for t in set(toks):
                df[t] += 1
                if t not in vocab:
                    vocab[t] = len(vocab)
        self._vocab = vocab
        n_docs = max(1, len(corpus_tokens))
        self._idf = {
            t: math.log((1 + n_docs) / (1 + cnt)) + 1.0 for t, cnt in df.items()
        }
        mat = np.zeros((len(corpus_tokens), max(1, len(vocab))), dtype=np.float32)
        for i, toks in enumerate(corpus_tokens):
            counts = Counter(toks)
            for t, c in counts.items():
                mat[i, vocab[t]] = (1 + math.log(c)) * self._idf[t]
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self._dense_matrix = mat / norms

    def _embed_query(self, query: str) -> np.ndarray | None:
        if self._dense_matrix is None:
            return None
        if self._tfidf_fallback:
            toks = _tokenize(query)
            vec = np.zeros((len(self._vocab),), dtype=np.float32)
            counts = Counter(toks)
            for t, c in counts.items():
                if t in self._vocab:
                    vec[self._vocab[t]] = (1 + math.log(c)) * self._idf[t]
            norm = float(np.linalg.norm(vec))
            return vec / norm if norm > 0 else vec

        model = _get_fastembed_model(self.embed_model_name)
        vec = list(model.embed([f"query: {query}"]))[0].astype(np.float32)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec

    @staticmethod
    def _expand_query(query: str) -> str:
        # Lightweight deterministic expansion for acronyms/plurals/units
        cleaned = re.sub(r"[?!.,;]+", " ", query).strip()
        return cleaned

    def search(
        self,
        query: str,
        config: RetrievalConfig | None = None,
        *,
        apply_threshold: bool = True,
    ) -> list[RetrievedChunk]:
        """Return ranked RetrievedChunk results using calibrated BM25 + Dense fusion."""
        if not self.chunks or self._bm25 is None:
            return []

        cfg = config or RetrievalConfig()
        effective_query = (
            self._expand_query(query) if cfg.use_query_expansion else query
        )
        q_tokens = _tokenize(effective_query)

        bm25_raw = (
            np.asarray(self._bm25.get_scores(q_tokens), dtype=np.float32)
            if q_tokens
            else np.zeros((len(self.chunks),), dtype=np.float32)
        )
        # Smooth saturation into [0, 1); also include direct lexical coverage
        # so small corpora (where BM25 IDF can be <=0) still preserve keyword precision.
        q_set = {t for t in q_tokens if len(t) >= 3}
        lex_cov = np.zeros((len(self.chunks),), dtype=np.float32)
        if q_set:
            for i, c in enumerate(self.chunks):
                c_toks = set(_tokenize(f"{c.heading} {c.source} {c.text}"))
                lex_cov[i] = sum(1 for t in q_set if t in c_toks) / len(q_set)
        bm25_sat = np.where(bm25_raw > 0, bm25_raw / (bm25_raw + 3.5), 0.0)
        bm25_cal = np.maximum(bm25_sat, lex_cov * 0.85)

        dense_raw = np.zeros((len(self.chunks),), dtype=np.float32)
        q_vec = self._embed_query(effective_query)
        if q_vec is not None and self._dense_matrix is not None:
            dense_raw = self._dense_matrix @ q_vec

        if self._tfidf_fallback:
            dense_cal = np.clip(dense_raw, 0.0, 1.0)
        else:
            # BGE-small cosine for unrelated queries typically sits around 0.40-0.55;
            # shift & stretch so genuine semantic hits separate cleanly from noise.
            dense_cal = np.clip((dense_raw - 0.50) / 0.35, 0.0, 1.0)

        alpha = cfg.hybrid_alpha if self._dense_matrix is not None else 0.0
        hybrid_scores = (1.0 - alpha) * bm25_cal + alpha * dense_cal

        # Add small RRF boost for items ranking strongly in BOTH sparse and dense
        bm25_order = np.argsort(-bm25_raw)
        dense_order = np.argsort(-dense_raw)
        rrf_bonus = np.zeros((len(self.chunks),), dtype=np.float32)
        k_rrf = 30.0
        for rank, idx in enumerate(bm25_order[:20]):
            if bm25_cal[idx] > 0.15:
                rrf_bonus[idx] += 1.0 / (k_rrf + rank + 1)
        for rank, idx in enumerate(dense_order[:20]):
            if dense_raw[idx] > 0.56:
                rrf_bonus[idx] += 1.0 / (k_rrf + rank + 1)

        final_scores = np.clip(hybrid_scores + 1.5 * rrf_bonus, 0.0, 1.0)
        top_indices = np.argsort(-final_scores)[: cfg.top_k]

        results: list[RetrievedChunk] = []
        for idx in top_indices:
            score_val = float(final_scores[idx])
            if apply_threshold and score_val < cfg.min_score_threshold:
                continue
            results.append(
                RetrievedChunk(
                    chunk=self.chunks[int(idx)],
                    score=round(score_val, 4),
                    bm25_score=round(float(bm25_cal[idx]), 4),
                    dense_score=round(float(dense_cal[idx]), 4),
                )
            )
        return results
