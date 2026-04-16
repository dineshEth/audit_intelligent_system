from __future__ import annotations

from typing import Dict, List, Optional

from .pipeline_service import PipelineService


class QueryService:
    def __init__(self, pipeline_service: PipelineService) -> None:
        self.pipeline_service = pipeline_service

    def ask(self, query: str, document_ids: Optional[List[str]] = None) -> Dict:
        return self.pipeline_service.answer_query(query, document_ids=document_ids)
