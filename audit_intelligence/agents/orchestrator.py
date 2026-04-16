from __future__ import annotations

from typing import Any, Dict, List, TypedDict

try:  # optional dependency
    from langgraph.graph import END, START, StateGraph
    LANGGRAPH_AVAILABLE = True
except Exception:  # pragma: no cover - optional
    END = START = None
    StateGraph = None
    LANGGRAPH_AVAILABLE = False

from ..schemas import QueryRecord
from .executor import ExecutorAgent
from .labeling_agent import LabelingAgent
from .planner import PlannerAgent
from .reviewer import ReviewerAgent


class AuditGraphState(TypedDict, total=False):
    query: str
    document_ids: List[str]
    documents: List[Any]
    plan: Dict[str, Any]
    execution: Dict[str, Any]
    review: Dict[str, Any]
    response: Dict[str, Any]
    retries: int
    trace: List[Dict[str, Any]]


class _SimpleCompiledGraph:
    def __init__(self, orchestrator: "AuditOrchestrator") -> None:
        self.orchestrator = orchestrator

    def invoke(self, state: AuditGraphState) -> AuditGraphState:
        merged = dict(state)
        merged.update(self.orchestrator._planner_node(merged))
        while True:
            merged.update(self.orchestrator._executor_node(merged))
            merged.update(self.orchestrator._reviewer_node(merged))
            route = self.orchestrator._route_after_review(merged)
            if route != "retry":
                break
        merged.update(self.orchestrator._labeler_node(merged))
        return merged


class AuditOrchestrator:
    def __init__(self, repositories, settings, retrieval_engine, llm) -> None:
        self.repositories = repositories
        self.settings = settings
        self.planner_agent = PlannerAgent(repositories, settings)
        self.executor_agent = ExecutorAgent(repositories, settings, retrieval_engine, llm)
        self.reviewer_agent = ReviewerAgent(repositories, settings)
        self.labeling_agent = LabelingAgent(repositories, settings, llm)
        self.graph = self._build_graph()

    def _build_graph(self):
        if not LANGGRAPH_AVAILABLE:
            return _SimpleCompiledGraph(self)

        graph = StateGraph(AuditGraphState)
        graph.add_node("planner", self._planner_node)
        graph.add_node("executor", self._executor_node)
        graph.add_node("reviewer", self._reviewer_node)
        graph.add_node("labeler", self._labeler_node)

        graph.add_edge(START, "planner")
        graph.add_edge("planner", "executor")
        graph.add_edge("executor", "reviewer")
        graph.add_conditional_edges(
            "reviewer",
            self._route_after_review,
            {"retry": "executor", "label": "labeler"},
        )
        graph.add_edge("labeler", END)
        return graph.compile()

    def _load_documents(self, document_ids: List[str]) -> List[Any]:
        documents = []
        for document_id in document_ids:
            doc = self.repositories.documents.find_one({"id": document_id})
            if doc:
                documents.append(doc)
        return documents

    def _append_trace(self, state: AuditGraphState, node: str, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        trace = list(state.get("trace", []))
        trace.append({"node": node, "payload": payload})
        return trace

    def _planner_node(self, state: AuditGraphState) -> AuditGraphState:
        documents = state.get("documents") or self._load_documents(state.get("document_ids", []))
        plan = self.planner_agent.run(state["query"], documents)
        return {
            "documents": documents,
            "plan": plan,
            "trace": self._append_trace(
                state,
                "planner",
                {"task_count": plan["task_count"], "document_types": plan["document_types"]},
            ),
        }

    def _executor_node(self, state: AuditGraphState) -> AuditGraphState:
        documents = state.get("documents") or self._load_documents(state.get("document_ids", []))
        retries = int(state.get("retries", 0))
        execution = self.executor_agent.run(state["query"], documents, retries=retries)
        return {
            "documents": documents,
            "execution": execution,
            "trace": self._append_trace(
                state,
                "executor",
                {
                    "reference_count": len(execution.get("references", [])),
                    "has_bank_df": execution.get("bank_df") is not None,
                    "retries": retries,
                },
            ),
        }

    def _reviewer_node(self, state: AuditGraphState) -> AuditGraphState:
        review = self.reviewer_agent.run(state["query"], state["execution"])
        prior_retries = int(state.get("retries", 0))
        new_retries = prior_retries if review["approved"] else prior_retries + 1
        return {
            "review": review,
            "retries": new_retries,
            "trace": self._append_trace(
                state,
                "reviewer",
                {"approved": review["approved"], "confidence": review["confidence"], "retries": new_retries},
            ),
        }

    def _route_after_review(self, state: AuditGraphState) -> str:
        review = state.get("review", {})
        retries = int(state.get("retries", 0))
        if not review.get("approved", False) and retries <= self.settings.max_retry:
            return "retry"
        return "label"

    def _labeler_node(self, state: AuditGraphState) -> AuditGraphState:
        output = self.labeling_agent.run(state.get("documents", []), state.get("execution", {}), state.get("review", {}))
        return {
            "response": output,
            "trace": self._append_trace(
                state,
                "labeler",
                {"report_path": output.get("report_path"), "qa_pair_count": output.get("qa_pair_count", 0)},
            ),
        }

    def invoke(self, query: str, document_ids: List[str]) -> Dict[str, Any]:
        state = self.graph.invoke(
            {
                "query": query,
                "document_ids": document_ids,
                "documents": self._load_documents(document_ids),
                "retries": 0,
                "trace": [],
            }
        )

        response = state.get("response", {})
        query_record = QueryRecord(
            query_text=query,
            document_ids=document_ids,
            response_text=response.get("summary_text", ""),
            retrieved_chunk_ids=state.get("execution", {}).get("retrieved_chunk_ids", []),
            orchestrator_trace=state.get("trace", []),
            status="completed",
        )
        self.repositories.queries.insert(query_record)
        return state
