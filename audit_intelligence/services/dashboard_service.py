from __future__ import annotations

from typing import Any, Dict


class DashboardService:
    def __init__(self, repositories) -> None:
        self.repositories = repositories

    def snapshot(self) -> Dict[str, Any]:
        latest_model_run = self.repositories.model_runs.find_many({}, limit=1, sort=("created_at", -1))
        recent_logs = self.repositories.logs.find_many({}, limit=10, sort=("created_at", -1))
        return {
            "counts": {
                "documents": self.repositories.documents.count({}),
                "chunks": self.repositories.chunks.count({}),
                "labels": self.repositories.labels.count({}),
                "qa_pairs": self.repositories.qa_pairs.count({}),
                "queries": self.repositories.queries.count({}),
                "model_runs": self.repositories.model_runs.count({}),
                "logs": self.repositories.logs.count({}),
            },
            "latest_model_run": latest_model_run[0].model_dump() if latest_model_run else None,
            "recent_logs": [item.model_dump() for item in recent_logs],
        }
