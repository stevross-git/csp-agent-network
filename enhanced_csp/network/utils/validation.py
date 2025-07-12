import ipaddress
import json
import logging
import re
import threading
import time
from functools import wraps
from typing import Any, Callable, Dict
import inspect

from ..errors import ValidationError

logger = logging.getLogger(__name__)

MAX_MESSAGE_SIZE = 1024 * 1024  # 1 MB default


def validate_ip_address(value: str) -> str:
    """Validate an IPv4 or IPv6 address."""
    try:
        ipaddress.ip_address(value)
        return value
    except ValueError as exc:
        raise ValidationError(f"Invalid IP address: {value}") from exc


def validate_port_number(value: int) -> int:
    """Validate a TCP/UDP port number."""
    try:
        port = int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"Invalid port number: {value}") from exc
    if not (1 <= port <= 65535):
        raise ValidationError(f"Invalid port number: {value}")
    return port


NODE_ID_RE = re.compile(r"^Qm[1-9A-HJ-NP-Za-km-z]{44}$")


def validate_node_id(value: str) -> str:
    """Validate a NodeID string."""
    if not value or not NODE_ID_RE.fullmatch(value):
        raise ValidationError(f"Invalid node ID: {value}")
    return value


def sanitize_string_input(value: str) -> str:
    """Remove control characters from a string."""
    return re.sub(r"[\n\r\t\0]", "", str(value))


def validate_message_size(data: Any, max_size: int = MAX_MESSAGE_SIZE) -> None:
    """Ensure that message size does not exceed limits."""
    if isinstance(data, (bytes, bytearray)):
        size = len(data)
    elif isinstance(data, str):
        size = len(data.encode())
    else:
        try:
            serialized = json.dumps(data).encode()
            size = len(serialized)
        except Exception:
            size = 0
    if size > max_size:
        raise ValidationError(f"Message too large: {size} bytes")


def validate_input(**validators: Callable[[Any], Any]):
    """Decorator to validate function arguments."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            bound = inspect.signature(func).bind_partial(*args, **kwargs)
            for name, validator in validators.items():
                if name in bound.arguments:
                    bound.arguments[name] = validator(bound.arguments[name])
            return func(*bound.args, **bound.kwargs)

        return wrapper

    return decorator


class PeerRateLimiter:
    """Simple request rate limiter per peer."""

    def __init__(self, max_requests: int, interval: float) -> None:
        self.max_requests = max_requests
        self.interval = interval
        self._requests: Dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def is_allowed(self, peer_id: str) -> bool:
        now = time.time()
        with self._lock:
            times = [t for t in self._requests.get(peer_id, []) if now - t < self.interval]
            allowed = len(times) < self.max_requests
            if allowed:
                times.append(now)
            self._requests[peer_id] = times
            if not allowed:
                logger.warning("rate limit exceeded for peer %s", peer_id)
            return allowed
