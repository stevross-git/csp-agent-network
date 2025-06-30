"""Simple in-memory log store used in legacy demos."""
from __future__ import annotations

from datetime import datetime
from typing import List, Dict


class CSPLogStore:
    """Minimal log storage."""

    def __init__(self, limit: int = 100) -> None:
        self.logs: List[Dict[str, str]] = []
        self.limit = limit

    def log(self, message: str) -> None:
        entry = {"time": datetime.utcnow().isoformat(), "msg": message}
        self.logs.append(entry)
        if len(self.logs) > self.limit:
            self.logs.pop(0)

    def get_logs(self) -> List[Dict[str, str]]:
        return list(reversed(self.logs))
