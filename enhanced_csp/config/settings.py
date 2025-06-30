from __future__ import annotations

from pydantic_settings import BaseSettings


class CSPSettings(BaseSettings):
    """Runtime configuration for Enhanced CSP."""

    enable_cognitive_comm: bool = True
    vector_store_backend: str = "inmemory"  # or "chroma"
    log_store_limit: int = 100


settings = CSPSettings()

__all__ = ["CSPSettings", "settings"]
