from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

try:
    import faiss
except Exception:  # pragma: no cover - optional
    faiss = None


@dataclass
class SearchResult:
    indices: List[int]
    scores: List[float]
    backend: str


class InMemoryVectorStore:
    def __init__(self, use_faiss: bool = False) -> None:
        self.use_faiss = use_faiss and faiss is not None

    def search(self, corpus_matrix: np.ndarray, query_matrix: np.ndarray, top_k: int = 5) -> SearchResult:
        if corpus_matrix.size == 0:
            return SearchResult(indices=[], scores=[], backend="empty")

        top_k = min(top_k, len(corpus_matrix))

        if self.use_faiss:
            dim = int(corpus_matrix.shape[1])
            index = faiss.IndexFlatIP(dim)
            index.add(corpus_matrix.astype("float32"))
            scores, indices = index.search(query_matrix.astype("float32"), top_k)
            return SearchResult(
                indices=[int(i) for i in indices[0].tolist()],
                scores=[float(s) for s in scores[0].tolist()],
                backend="faiss",
            )

        scores = cosine_similarity(query_matrix, corpus_matrix)[0]
        ranked = np.argsort(scores)[::-1][:top_k]
        return SearchResult(
            indices=[int(i) for i in ranked.tolist()],
            scores=[float(scores[i]) for i in ranked.tolist()],
            backend="cosine",
        )
