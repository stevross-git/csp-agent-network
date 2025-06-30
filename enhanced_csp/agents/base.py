"""Base agent abstraction."""
from __future__ import annotations


class BaseAgent:
    """Minimal base agent used in legacy demos."""

    def __init__(self, agent_id: str, memory: object | None = None) -> None:
        self.agent_id = agent_id
        self.memory = memory

    def log(self, message: str) -> None:
        print(f"[{self.agent_id}] {message}")
