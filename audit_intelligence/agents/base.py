from __future__ import annotations

from typing import Any, Dict


class BaseAgent:
    name = "base"

    def __init__(self, repositories, settings) -> None:
        self.repositories = repositories
        self.settings = settings

    def log(self, action: str, status: str = "info", message: str = "", **payload: Dict[str, Any]) -> None:
        self.repositories.log(
            agent=self.name,
            action=action,
            status=status,
            message=message,
            input_payload=payload.get("input_payload", {}),
            output_payload=payload.get("output_payload", {}),
        )
