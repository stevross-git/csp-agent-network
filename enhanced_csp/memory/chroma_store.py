"""Stubbed Chroma vector store."""
from __future__ import annotations

from typing import Any, Dict


class ChromaVectorStore:
    """Minimal in-memory vector store as placeholder."""

    def __init__(self, collection_name: str = "csp_memory") -> None:
        self.collection_name = collection_name
        self._store: Dict[str, Dict[str, Any]] = {}

    def store(self, key: str, content: str, metadata: Dict[str, Any] | None = None) -> None:
        self._store[key] = {"content": content, "metadata": metadata or {}}

    def search(self, query_text: str, top_k: int = 3) -> Dict[str, Any]:
        # Very naive search returning first K items
        results = list(self._store.items())[:top_k]
        return {"ids": [k for k, _ in results], "documents": [v["content"] for _, v in results]}
