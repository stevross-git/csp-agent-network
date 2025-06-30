"""Data cleaner agent stub."""
from __future__ import annotations

from .base import BaseAgent
from ..protocols.csp import create_csp_message


class DataCleanerAgent(BaseAgent):
    """Simple data cleaning agent used in examples."""

    def receive_csp_message(self, message: dict) -> dict:
        task = message.get("task", {})
        intent = task.get("intent")
        if intent == "deduplicate_records":
            result = {
                "success": True,
                "deduplicated_count": 0,
                "stored_at": "memory://result"
            }
        else:
            result = {"error": "Unknown intent"}
        return create_csp_message(
            sender=self.agent_id,
            recipient=message.get("sender", ""),
            type_="TASK_RESULT",
            task=result,
            context_refs=[message.get("msg_id", "")]
        )
