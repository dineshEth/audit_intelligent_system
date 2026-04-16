from __future__ import annotations

from typing import Any, Dict, List

from ..schemas import ReviewDecision
from .base import BaseAgent


class ReviewerAgent(BaseAgent):
    name = "reviewer"

    def run(self, query: str, execution_output: Dict[str, Any]) -> Dict[str, Any]:
        reasons: List[str] = []
        score = 0.0

        references = execution_output.get("references", [])
        summary = (execution_output.get("summary_text") or "").strip()
        analysis = execution_output.get("analysis") or {}

        if references:
            score += min(0.35, 0.07 * len(references))
        else:
            reasons.append("No retrieval references were attached to the answer.")

        if len(summary.split()) >= 20:
            score += 0.25
        else:
            reasons.append("Summary is too short.")

        if analysis:
            score += 0.25
        else:
            reasons.append("No structured analysis payload was produced.")

        if analysis.get("transaction_count", 0) > 0:
            score += 0.15

        confidence = round(min(score, 0.99), 3)
        approved = confidence >= self.settings.confidence_threshold

        retry_advice = None
        if not approved:
            retry_advice = "Increase retrieval depth and rely more on source chunks before finalizing the answer."
            reasons.append("Confidence fell below the configured threshold.")

        decision = ReviewDecision(
            approved=approved,
            confidence=confidence,
            reasons=reasons,
            retry_advice=retry_advice,
        ).model_dump()

        self.log(
            "review",
            status="success" if approved else "warning",
            input_payload={"query": query},
            output_payload=decision,
        )
        return decision
