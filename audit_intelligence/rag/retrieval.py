from __future__ import annotations

from typing import List, Optional

from ..ingestion.chunking import chunk_text
from ..schemas import ChunkRecord, RetrievedChunk
from ..utils.text import keyword_overlap_score
from .embeddings import build_embedder
from .vector_store import InMemoryVectorStore


class RetrievalEngine:
    def __init__(self, repositories, settings):
        self.repositories = repositories
        self.settings = settings
        self.embedder = build_embedder(settings.local_embedding_model_path)
        self.vector_store = InMemoryVectorStore(settings.use_faiss)

    def index_document(self, document_record) -> List[ChunkRecord]:
        existing = self.repositories.chunks.find_many({"document_id": document_record.id}, limit=10000, sort=("chunk_index", 1))
        if existing:
            return existing

        chunks = chunk_text(
            document_record.content_text,
            chunk_size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
        )
        records = [
            ChunkRecord(
                document_id=document_record.id,
                chunk_index=index,
                text=chunk,
                metadata={"file_name": document_record.file_name, "doc_type": document_record.doc_type},
            )
            for index, chunk in enumerate(chunks)
        ]
        self.repositories.chunks.insert_many(records)
        return records

    def retrieve(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        top_k = top_k or self.settings.top_k
        filter_dict = {"document_id": {"$in": document_ids}} if document_ids else {}
        chunks = self.repositories.chunks.find_many(filter_dict, limit=10000, sort=("chunk_index", 1))
        if not chunks:
            return []

        texts = [chunk.text for chunk in chunks]
        corpus = self.embedder.fit_corpus(texts)
        query_vec = self.embedder.encode_query(query)
        vector_results = self.vector_store.search(corpus.matrix, query_vec, top_k=top_k)

        retrieved: List[RetrievedChunk] = []
        for idx, raw_score in zip(vector_results.indices, vector_results.scores):
            chunk = chunks[idx]
            lexical = keyword_overlap_score(query, chunk.text)
            semantic = max(min((raw_score + 1.0) / 2.0, 1.0), 0.0) if raw_score < 0 else min(raw_score, 1.0)
            combined = round((0.75 * semantic) + (0.25 * lexical), 4)
            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    text=chunk.text,
                    score=combined,
                    lexical_score=round(lexical, 4),
                    semantic_score=round(semantic, 4),
                    metadata={
                        "vector_backend": vector_results.backend,
                        "embedding_backend": corpus.backend_name,
                        **chunk.metadata,
                    },
                )
            )

        return sorted(retrieved, key=lambda item: item.score, reverse=True)
