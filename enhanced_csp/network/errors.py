from __future__ import annotations

"""Custom exceptions and error metrics for the Enhanced CSP Network."""

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict


class NetworkError(Exception):
    """Base class for network related errors."""


class ConnectionError(NetworkError):
    """Raised when a network connection fails."""


class TimeoutError(NetworkError):
    """Raised when an operation times out."""


class ProtocolError(NetworkError):
    """Raised when a protocol violation occurs."""


class SecurityError(NetworkError):
    """Raised on security related failures."""


class ValidationError(NetworkError):
    """Raised when configuration or data validation fails."""


class CircuitBreakerOpen(NetworkError):
    """Raised when a circuit breaker is open and rejects operations."""


@dataclass
class ErrorMetrics:
    """Simple container tracking error counts by type."""

    counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def record(self, exc: Exception) -> None:
        """Record an exception occurrence."""
        self.counts[exc.__class__.__name__] += 1
