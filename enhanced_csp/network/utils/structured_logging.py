import json
import logging
import random
from contextlib import contextmanager
from typing import Any, Dict, Optional


class StructuredFormatter(logging.Formatter):
    """Format log records as JSON or human readable text."""

    def __init__(self, json_format: bool = False):
        super().__init__()
        self.json_format = json_format

    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - formatting
        data: Dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "component": getattr(record, "component", record.name),
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", None),
        }
        for field in ("peer_id", "message_type", "operation", "duration"):
            value = getattr(record, field, None)
            if value is not None:
                data[field] = value

        if self.json_format:
            return json.dumps(data)

        extras = " ".join(f"{k}={v}" for k, v in data.items() if k not in {"timestamp", "level", "component", "message"} and v is not None)
        return f"{data['timestamp']} {data['level']} {data['component']} {data['message']} {extras}".strip()


class SamplingFilter(logging.Filter):
    """Filter that randomly drops log records below WARNING level."""

    def __init__(self, rate: float = 1.0) -> None:
        super().__init__()
        self.rate = rate

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - nondeterministic
        if record.levelno >= logging.WARNING:
            return True
        return random.random() < self.rate


class StructuredAdapter(logging.LoggerAdapter):
    """Logger adapter with contextual support."""

    def __init__(self, logger: logging.Logger, extra: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(logger, extra or {})
        self._ctx_stack: list[Dict[str, Any]] = []

    def process(self, msg: str, kwargs: Dict[str, Any]):
        extra = kwargs.get("extra", {})
        context: Dict[str, Any] = {}
        for ctx in self._ctx_stack:
            context.update(ctx)
        context.update(self.extra)
        context.update(extra)
        kwargs["extra"] = context
        return msg, kwargs

    @contextmanager
    def context(self, **ctx: Any):
        self._ctx_stack.append(ctx)
        try:
            yield
        finally:
            self._ctx_stack.pop()


def get_logger(component: str, name: Optional[str] = None) -> StructuredAdapter:
    logger = logging.getLogger(name or component)
    return StructuredAdapter(logger, {"component": component})


def setup_logging(mode: str = "development", level: int = logging.INFO, sample_rate: float = 1.0) -> None:
    """Configure root logging handler."""
    handler = logging.StreamHandler()
    handler.setFormatter(StructuredFormatter(json_format=mode == "production"))
    if sample_rate < 1.0:
        handler.addFilter(SamplingFilter(sample_rate))
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)


# Predefined component loggers
NetworkLogger = get_logger("network")
SecurityLogger = get_logger("security")
PerformanceLogger = get_logger("performance")
AuditLogger = get_logger("audit")

__all__ = [
    "StructuredFormatter",
    "SamplingFilter",
    "StructuredAdapter",
    "get_logger",
    "setup_logging",
    "NetworkLogger",
    "SecurityLogger",
    "PerformanceLogger",
    "AuditLogger",
]
