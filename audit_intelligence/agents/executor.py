from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from ..analysis.analyzer import FinancialAnalyzer
from ..ingestion.bank_statement import parse_bank_statement
from ..schemas import RetrievedChunk
from ..utils.text import compact_reference
from .base import BaseAgent


class ExecutorAgent(BaseAgent):
    name = "executor"

    def __init__(self, repositories, settings, retrieval_engine, llm) -> None:
        super().__init__(repositories, settings)
        self.retrieval_engine = retrieval_engine
        self.llm = llm
        self.analyzer = FinancialAnalyzer()

    def run(self, query: str, documents: List[Any], retries: int = 0) -> Dict[str, Any]:
        combined_text_parts: List[str] = []
        bank_frames: List[pd.DataFrame] = []

        for doc in documents:
            self.retrieval_engine.index_document(doc)
            combined_text_parts.append(doc.content_text)
            if getattr(doc, "doc_type", "generic") == "bank_statement":
                try:
                    bank_frames.append(parse_bank_statement(doc.file_path))
                except Exception as exc:
                    self.log(
                        "parse_bank_statement",
                        status="warning",
                        message=str(exc),
                        input_payload={"document_id": doc.id, "file_path": doc.file_path},
                    )

        top_k = self.settings.top_k + retries
        retrieved = self.retrieval_engine.retrieve(query, [doc.id for doc in documents], top_k=top_k)
        combined_text = "\n\n".join(part for part in combined_text_parts if part)

        references = [
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "score": chunk.score,
                "text": compact_reference(chunk.text),
                "metadata": chunk.metadata,
            }
            for chunk in retrieved[:top_k]
        ]

        analysis: Dict[str, Any] = {}
        bank_df = None
        if bank_frames:
            bank_df = pd.concat(bank_frames, ignore_index=True)
            analysis = self.analyzer.analyze_transactions(bank_df)
            summary = self.llm.summarize(query, [chunk.text for chunk in retrieved], analysis=analysis)
        else:
            analysis = self.analyzer.summarize_generic_document(combined_text, [item.model_dump() for item in retrieved])
            summary = self.llm.summarize(query, [chunk.text for chunk in retrieved]) or analysis.get("summary", "")

        output = {
            "summary_text": summary,
            "analysis": analysis,
            "references": references,
            "retrieved_chunk_ids": [chunk.chunk_id for chunk in retrieved],
            "retrieved_chunks": [item.model_dump() for item in retrieved],
            "combined_text": combined_text,
            "bank_df": bank_df,
        }
        self.log(
            "execute",
            status="success",
            input_payload={"query": query, "document_ids": [doc.id for doc in documents], "retries": retries},
            output_payload={"reference_count": len(references), "has_bank_df": bank_df is not None},
        )
        return output
