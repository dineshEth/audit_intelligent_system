from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid4())


class BaseRecord(BaseModel):
    id: str = Field(default_factory=new_id)
    version: int = 1
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class DocumentRecord(BaseRecord):
    file_name: str
    file_path: str
    mime_type: str
    doc_type: str = "generic"
    content_text: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    checksum: str
    status: str = "ingested"


class ChunkRecord(BaseRecord):
    document_id: str
    chunk_index: int
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    score: float
    lexical_score: float = 0.0
    semantic_score: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryRecord(BaseRecord):
    query_text: str
    document_ids: List[str] = Field(default_factory=list)
    response_text: str = ""
    retrieved_chunk_ids: List[str] = Field(default_factory=list)
    orchestrator_trace: List[Dict[str, Any]] = Field(default_factory=list)
    status: str = "completed"


class LabelRow(BaseModel):
    DATE: str
    DESCRIPTION: str
    DEBIT: float = 0.0
    CREDIT: float = 0.0
    BALANCE: float = 0.0
    CATEGORY: str = "UNCATEGORIZED"
    LABEL_SOURCE: str = "rule"
    CONFIDENCE: float = 0.0
    RAW_ROW: Dict[str, Any] = Field(default_factory=dict)


class LabelRecord(BaseRecord):
    document_id: str
    source_file: str
    labels: List[LabelRow] = Field(default_factory=list)
    export_paths: Dict[str, str] = Field(default_factory=dict)


class QAPairRecord(BaseRecord):
    document_id: Optional[str] = None
    question: str
    answer: str
    source_type: Literal["generated", "manual", "label-derived"] = "generated"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelRunRecord(BaseRecord):
    base_model_path: str = ""
    adapter_path: str = ""
    status: Literal["completed", "skipped", "failed", "running"] = "running"
    dataset_size: int = 0
    train_loss: Optional[float] = None
    eval_loss: Optional[float] = None
    accuracy: Optional[float] = None
    duration_seconds: float = 0.0
    notes: str = ""


class LogRecord(BaseRecord):
    agent: str
    action: str
    status: Literal["success", "error", "warning", "info"] = "info"
    message: str = ""
    input_payload: Dict[str, Any] = Field(default_factory=dict)
    output_payload: Dict[str, Any] = Field(default_factory=dict)


class AnalysisSummary(BaseModel):
    summary_text: str
    key_metrics: Dict[str, Any] = Field(default_factory=dict)
    references: List[Dict[str, Any]] = Field(default_factory=list)
    chart_paths: List[str] = Field(default_factory=list)
    report_path: Optional[str] = None


class ReviewDecision(BaseModel):
    approved: bool
    confidence: float
    reasons: List[str] = Field(default_factory=list)
    retry_advice: Optional[str] = None
