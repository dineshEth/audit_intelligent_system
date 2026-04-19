from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..agents.orchestrator import AuditOrchestrator
from ..config import get_settings
from ..db import MongoManager
from ..ingestion.parsers import parse_document
from ..llm.gemini_llm import GeminiLLM
from ..rag.retrieval import RetrievalEngine
from ..repositories import AuditRepositories
from ..schemas import DocumentRecord
from ..utils.files import save_bytes
from ..utils.hashing import sha256_file


class PipelineService:
    def __init__(self, settings=None) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_directories()
        self.mongo = MongoManager(self.settings.mongodb_uri, self.settings.mongodb_db_name)
        self.mongo.ensure_indexes()
        self.repositories = AuditRepositories(self.mongo.db)
        self.llm = GeminiLLM(api_key=self.settings.gemini_api_key, model_name=self.settings.gemini_model)
        self.retrieval_engine = RetrievalEngine(self.repositories, self.settings)
        self.orchestrator = AuditOrchestrator(self.repositories, self.settings, self.retrieval_engine, self.llm)

    def ingest_file(self, file_path: str | Path) -> DocumentRecord:
        path = Path(file_path)
        checksum = sha256_file(path)
        existing = self.repositories.documents.find_one({"checksum": checksum})
        if existing:
            self.repositories.log(
                agent="pipeline",
                action="ingest_file",
                status="info",
                message="Duplicate file detected; returning existing document.",
                input_payload={"file_path": str(path), "checksum": checksum},
                output_payload={"document_id": existing.id},
            )
            return existing

        parsed = parse_document(path)
        metadata = parsed["metadata"]
        record = DocumentRecord(
            file_name=path.name,
            file_path=str(path),
            mime_type=metadata.get("mime_type", "text/plain"),
            doc_type=metadata.get("doc_type", "generic"),
            content_text=parsed["text"],
            metadata={k: v for k, v in metadata.items() if k not in {"file_name", "file_path", "mime_type", "doc_type"}},
            checksum=checksum,
            status="ingested",
        )
        self.repositories.documents.insert(record)
        self.retrieval_engine.index_document(record)
        self.repositories.log(
            agent="pipeline",
            action="ingest_file",
            status="success",
            input_payload={"file_path": str(path)},
            output_payload={"document_id": record.id, "doc_type": record.doc_type},
        )
        return record

    def process_file(self, file_path: str | Path, user_query: Optional[str] = None) -> Dict[str, Any]:
        document = self.ingest_file(file_path)
        query = user_query or "Analyze this document for audit review, generate labels if applicable, and produce a report."
        state = self.orchestrator.invoke(query, [document.id])
        return {
            "document": document.model_dump(),
            "plan": state.get("plan", {}),
            "execution": state.get("execution", {}),
            "review": state.get("review", {}),
            "response": state.get("response", {}),
            "trace": state.get("trace", []),
        }

    def process_upload(self, file_name: str, data: bytes, user_query: Optional[str] = None) -> Dict[str, Any]:
        saved_path = save_bytes(file_name, data, self.settings.upload_dir)
        return self.process_file(saved_path, user_query=user_query)

    def answer_query(self, query: str, document_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        if not document_ids:
            docs = self.repositories.documents.find_many({}, limit=1000, sort=("created_at", -1))
            document_ids = [doc.id for doc in docs]
        state = self.orchestrator.invoke(query, document_ids)
        return {
            "query": query,
            "document_ids": document_ids,
            "response": state.get("response", {}),
            "review": state.get("review", {}),
            "trace": state.get("trace", []),
        }

    def latest_documents(self, limit: int = 20):
        return self.repositories.documents.find_many({}, limit=limit, sort=("created_at", -1))
