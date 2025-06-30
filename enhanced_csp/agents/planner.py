"""Planner agent stub."""
from __future__ import annotations

from .base import BaseAgent
from ..protocols.csp import create_csp_message


class PlannerAgent(BaseAgent):
    """Simple planning agent used in examples."""

    def create_task_message(self, recipient: str, intent: str, resource_ref: str, parameters: dict) -> dict:
        task = {
            "intent": intent,
            "resource_ref": resource_ref,
            "parameters": parameters,
            "priority": "high",
        }
        return create_csp_message(
            sender=self.agent_id,
            recipient=recipient,
            type_="TASK_REQUEST",
            task=task,
        )
