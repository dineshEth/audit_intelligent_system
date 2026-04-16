from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

try:  # optional
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional
    SentenceTransformer = None


@dataclass
class CorpusEmbeddings:
    matrix: np.ndarray
    texts: List[str]
    backend_name: str


class TfidfEmbedder:
    name = "tfidf"

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=4096)

    def fit_corpus(self, texts: List[str]) -> CorpusEmbeddings:
        matrix = self.vectorizer.fit_transform(texts).astype("float32")
        dense = matrix.toarray()
        dense = normalize(dense)
        return CorpusEmbeddings(matrix=dense, texts=texts, backend_name=self.name)

    def encode_query(self, query: str) -> np.ndarray:
        vector = self.vectorizer.transform([query]).astype("float32").toarray()
        return normalize(vector)


class SentenceTransformerEmbedder:
    name = "sentence-transformer"

    def __init__(self, model_path: str) -> None:
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is not installed")
        self.model = SentenceTransformer(model_path, local_files_only=True)

    def fit_corpus(self, texts: List[str]) -> CorpusEmbeddings:
        matrix = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return CorpusEmbeddings(matrix=np.asarray(matrix, dtype="float32"), texts=texts, backend_name=self.name)

    def encode_query(self, query: str) -> np.ndarray:
        vector = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vector, dtype="float32")


def build_embedder(local_embedding_model_path: str):
    if local_embedding_model_path and Path(local_embedding_model_path).exists():
        try:
            return SentenceTransformerEmbedder(local_embedding_model_path)
        except Exception:
            pass
    return TfidfEmbedder()
