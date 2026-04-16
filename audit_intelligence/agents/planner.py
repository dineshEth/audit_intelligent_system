from __future__ import annotations

from typing import Any, Dict, List

from .base import BaseAgent


class PlannerAgent(BaseAgent):
    name = "planner"

    def run(self, query: str, documents: List[Any]) -> Dict[str, Any]:
        doc_types = sorted({getattr(doc, "doc_type", "generic") for doc in documents}) or ["generic"]
        tasks = [
            {"name": "load_documents", "goal": "Load document text and metadata from local storage."},
            {"name": "retrieve_context", "goal": "Retrieve relevant chunks using hybrid RAG."},
        ]

        if "bank_statement" in doc_types:
            tasks.extend(
                [
                    {"name": "parse_transactions", "goal": "Parse DATE, DESCRIPTION, DEBIT, CREDIT, BALANCE fields."},
                    {"name": "analyze_transactions", "goal": "Compute totals, balances, anomalies, and category insights."},
                    {"name": "label_transactions", "goal": "Apply hybrid rule-based and LLM-based transaction categorization."},
                    {"name": "export_outputs", "goal": "Write labeled CSV/JSON/DOCX plus a report and charts."},
                ]
            )
        else:
            tasks.extend(
                [
                    {"name": "summarize_document", "goal": "Create a concise audit-facing summary with references."},
                    {"name": "generate_qa", "goal": "Create local training Q&A pairs from the document content."},
                ]
            )

        plan = {
            "document_types": doc_types,
            "task_count": len(tasks),
            "tasks": tasks,
            "shared_memory_keys": [
                "documents",
                "retrieved_chunks",
                "analysis",
                "review",
                "labels",
                "qa_pairs",
            ],
        }
        self.log("plan", status="success", input_payload={"query": query, "doc_types": doc_types}, output_payload=plan)
        return plan
