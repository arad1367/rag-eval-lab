"""Three ways to find the chunks that best match a question.

1. BM25 (keyword / "sparse" search): scores chunks by shared words. Rare words count more
   (IDF), and repeating a word has diminishing returns (term-frequency saturation).
   Strong for exact names, acronyms, numbers. Weak for synonyms/paraphrases.
2. Dense (vector / semantic search): compares embeddings with cosine similarity.
   Strong for paraphrases ("car" ~ "automobile"). Weak for rare exact terms.
3. Hybrid: combines both rankings with Reciprocal Rank Fusion (RRF):
       score(chunk) = sum over retrievers of 1 / (k + rank)
   RRF only uses ranks, so we never have to compare BM25 scores with cosine scores
   (they are on totally different scales).
"""

import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from rag_lab.data import Chunk
from rag_lab.embeddings import Embedder

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(-scores)[:k]
        return [SearchResult(self.chunks[i], float(scores[i])) for i in top]


class DenseRetriever:
    name = "dense"

    def __init__(self, chunks: list[Chunk], embedder: Embedder, matrix: np.ndarray | None = None):
        self.chunks = chunks
        self.embedder = embedder
        # One row per chunk. This matrix is our tiny "vector database".
        # It can be passed in pre-computed (see index.py) or computed here.
        if matrix is None:
            matrix = embedder.embed_documents([c.text for c in chunks])
        if len(matrix) != len(chunks):
            raise ValueError("matrix must have one row per chunk")
        self.matrix = matrix

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        q = self.embedder.embed_query(query)
        scores = self.matrix @ q  # cosine similarity, because all vectors are normalized
        top = np.argsort(-scores)[:k]
        return [SearchResult(self.chunks[i], float(scores[i])) for i in top]


class HybridRetriever:
    name = "hybrid"

    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever, rrf_k: int = 60):
        self.bm25 = bm25
        self.dense = dense
        self.rrf_k = rrf_k  # 60 is the value from the original RRF paper

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        depth = max(k * 4, 20)  # look deeper in each list before fusing
        fused: dict[str, float] = {}
        by_id: dict[str, Chunk] = {}
        for retriever in (self.bm25, self.dense):
            for rank, result in enumerate(retriever.search(query, depth), start=1):
                cid = result.chunk.chunk_id
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
                by_id[cid] = result.chunk
        best = sorted(fused, key=fused.get, reverse=True)[:k]
        return [SearchResult(by_id[cid], fused[cid]) for cid in best]


def build_retrievers(chunks: list[Chunk], embedder: Embedder, matrix=None) -> dict:
    bm25 = BM25Retriever(chunks)
    dense = DenseRetriever(chunks, embedder, matrix)
    return {"bm25": bm25, "dense": dense, "hybrid": HybridRetriever(bm25, dense)}


def unique_papers(results: list[SearchResult]) -> list[str]:
    """Several chunks can come from the same paper; keep each paper once, in rank order."""
    seen: list[str] = []
    for r in results:
        if r.chunk.paper_id not in seen:
            seen.append(r.chunk.paper_id)
    return seen
