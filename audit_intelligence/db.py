from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

try:  # optional dependency
    from pymongo import ASCENDING, DESCENDING, MongoClient
    PYMONGO_AVAILABLE = True
except Exception:  # pragma: no cover - optional
    ASCENDING = 1
    DESCENDING = -1
    MongoClient = None
    PYMONGO_AVAILABLE = False

try:  # optional dependency
    import mongomock
    MONGOMOCK_AVAILABLE = True
except Exception:  # pragma: no cover - optional
    mongomock = None
    MONGOMOCK_AVAILABLE = False


def _matches(document: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
    for key, value in (filter_dict or {}).items():
        if isinstance(value, dict) and "$in" in value:
            if document.get(key) not in value["$in"]:
                return False
        else:
            if document.get(key) != value:
                return False
    return True


class _InMemoryCursor:
    def __init__(self, docs: List[Dict[str, Any]]):
        self.docs = list(docs)

    def sort(self, field: str, direction: int):
        reverse = direction == -1
        self.docs = sorted(self.docs, key=lambda item: item.get(field), reverse=reverse)
        return self

    def limit(self, n: int):
        self.docs = self.docs[:n]
        return self

    def __iter__(self):
        return iter(self.docs)


@dataclass
class _DeleteResult:
    deleted_count: int


class _InMemoryCollection:
    def __init__(self):
        self.docs: List[Dict[str, Any]] = []

    def create_index(self, *args, **kwargs):
        return None

    def insert_one(self, document: Dict[str, Any]):
        self.docs.append(deepcopy(document))
        return None

    def insert_many(self, documents: Iterable[Dict[str, Any]]):
        for document in documents:
            self.docs.append(deepcopy(document))
        return None

    def find_one(self, filter_dict: Dict[str, Any]):
        for doc in self.docs:
            if _matches(doc, filter_dict):
                return deepcopy(doc)
        return None

    def find(self, filter_dict: Dict[str, Any] | None = None):
        matched = [deepcopy(doc) for doc in self.docs if _matches(doc, filter_dict or {})]
        return _InMemoryCursor(matched)

    def update_one(self, filter_dict: Dict[str, Any], update_dict: Dict[str, Any], upsert: bool = False):
        set_values = update_dict.get("$set", {})
        for idx, doc in enumerate(self.docs):
            if _matches(doc, filter_dict):
                updated = deepcopy(doc)
                updated.update(deepcopy(set_values))
                self.docs[idx] = updated
                return None
        if upsert:
            new_doc = deepcopy(filter_dict)
            new_doc.update(deepcopy(set_values))
            self.docs.append(new_doc)
        return None

    def delete_many(self, filter_dict: Dict[str, Any]):
        before = len(self.docs)
        self.docs = [doc for doc in self.docs if not _matches(doc, filter_dict)]
        return _DeleteResult(deleted_count=before - len(self.docs))

    def count_documents(self, filter_dict: Dict[str, Any] | None = None):
        return sum(1 for doc in self.docs if _matches(doc, filter_dict or {}))


class _InMemoryDatabase:
    def __init__(self):
        self._collections: Dict[str, _InMemoryCollection] = {}

    def __getattr__(self, name: str):
        return self.collection(name)

    def __getitem__(self, name: str):
        return self.collection(name)

    def collection(self, name: str):
        if name not in self._collections:
            self._collections[name] = _InMemoryCollection()
        return self._collections[name]


class _InMemoryAdmin:
    def command(self, name: str):
        if name == "ping":
            return {"ok": 1}
        return {"ok": 0}


class _InMemoryClient:
    def __init__(self):
        self._databases: Dict[str, _InMemoryDatabase] = {}
        self.admin = _InMemoryAdmin()

    def __getitem__(self, name: str):
        if name not in self._databases:
            self._databases[name] = _InMemoryDatabase()
        return self._databases[name]


class MongoManager:
    def __init__(self, uri: str, db_name: str):
        self.uri = uri
        self.db_name = db_name

        if uri.startswith("mongomock://"):
            if MONGOMOCK_AVAILABLE:
                self.client = mongomock.MongoClient()
            else:
                self.client = _InMemoryClient()
        else:
            if PYMONGO_AVAILABLE:
                self.client = MongoClient(uri, serverSelectionTimeoutMS=3000)
            else:
                self.client = _InMemoryClient()

        self.db = self.client[db_name]

    def ping(self) -> bool:
        try:
            result = self.client.admin.command("ping")
            return bool(result)
        except Exception:
            return False

    def ensure_indexes(self) -> None:
        self.db.documents.create_index([("checksum", ASCENDING)], unique=True)
        self.db.documents.create_index([("doc_type", ASCENDING)])
        self.db.chunks.create_index([("document_id", ASCENDING), ("chunk_index", ASCENDING)])
        self.db.queries.create_index([("created_at", DESCENDING)])
        self.db.logs.create_index([("agent", ASCENDING), ("created_at", DESCENDING)])
        self.db.labels.create_index([("document_id", ASCENDING), ("created_at", DESCENDING)])
        self.db.qa_pairs.create_index([("document_id", ASCENDING), ("created_at", DESCENDING)])
        self.db.model_runs.create_index([("created_at", DESCENDING)])

    def collection(self, name: str) -> Any:
        return self.db[name]
