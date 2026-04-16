from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Generic, Iterable, List, Optional, Type, TypeVar

from pydantic import BaseModel

from .schemas import (
    ChunkRecord,
    DocumentRecord,
    LabelRecord,
    LogRecord,
    ModelRunRecord,
    QAPairRecord,
    QueryRecord,
)


T = TypeVar("T", bound=BaseModel)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class MongoRepository(Generic[T]):
    def __init__(self, collection, model_cls: Type[T]):
        self.collection = collection
        self.model_cls = model_cls

    def insert(self, record: T) -> T:
        self.collection.insert_one(record.model_dump(mode="python"))
        return record

    def insert_many(self, records: Iterable[T]) -> List[T]:
        payload = [record.model_dump(mode="python") for record in records]
        if payload:
            self.collection.insert_many(payload)
        return list(records)

    def upsert(self, record_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        payload["updated_at"] = utcnow()
        self.collection.update_one({"id": record_id}, {"$set": payload}, upsert=True)
        return self.collection.find_one({"id": record_id}) or {}

    def update(self, record_id: str, payload: Dict[str, Any]) -> Optional[T]:
        payload["updated_at"] = utcnow()
        self.collection.update_one({"id": record_id}, {"$set": payload}, upsert=False)
        doc = self.collection.find_one({"id": record_id})
        return self.model_cls(**doc) if doc else None

    def find_one(self, filter_dict: Dict[str, Any]) -> Optional[T]:
        doc = self.collection.find_one(filter_dict)
        return self.model_cls(**doc) if doc else None

    def find_many(
        self,
        filter_dict: Dict[str, Any] | None = None,
        limit: int = 100,
        sort: tuple[str, int] = ("created_at", -1),
    ) -> List[T]:
        filter_dict = filter_dict or {}
        cursor = self.collection.find(filter_dict).sort(*sort).limit(limit)
        return [self.model_cls(**doc) for doc in cursor]

    def count(self, filter_dict: Dict[str, Any] | None = None) -> int:
        return self.collection.count_documents(filter_dict or {})

    def delete_many(self, filter_dict: Dict[str, Any]) -> int:
        return self.collection.delete_many(filter_dict).deleted_count


class AuditRepositories:
    def __init__(self, db):
        self.documents = MongoRepository(db.documents, DocumentRecord)
        self.chunks = MongoRepository(db.chunks, ChunkRecord)
        self.queries = MongoRepository(db.queries, QueryRecord)
        self.labels = MongoRepository(db.labels, LabelRecord)
        self.qa_pairs = MongoRepository(db.qa_pairs, QAPairRecord)
        self.model_runs = MongoRepository(db.model_runs, ModelRunRecord)
        self.logs = MongoRepository(db.logs, LogRecord)

    def log(
        self,
        *,
        agent: str,
        action: str,
        status: str = "info",
        message: str = "",
        input_payload: Dict[str, Any] | None = None,
        output_payload: Dict[str, Any] | None = None,
    ) -> LogRecord:
        record = LogRecord(
            agent=agent,
            action=action,
            status=status,
            message=message,
            input_payload=input_payload or {},
            output_payload=output_payload or {},
        )
        self.logs.insert(record)
        return record
