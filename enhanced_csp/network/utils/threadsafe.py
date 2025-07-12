import threading
import time
import os
from typing import Any, Dict, Iterable, Optional

from .structured_logging import get_logger

logger = get_logger("threadsafe")

_DEADLOCK_TIMEOUT = 5.0 if os.getenv("CSP_DEVELOPMENT_MODE") else None


def _acquire(lock: threading.Lock, timeout: Optional[float] = _DEADLOCK_TIMEOUT):
    if timeout is None:
        lock.acquire()
        return True
    start = time.time()
    while True:
        acquired = lock.acquire(timeout=timeout)
        if acquired:
            return True
        # timed out
        if time.time() - start >= timeout:
            logger.warning("Potential deadlock detected after %.1fs", timeout)
            timeout *= 2  # back off for next check


class ThreadSafeCounter:
    """Atomic counter with optional deadlock detection."""

    def __init__(self, initial: int = 0) -> None:
        self._value = initial
        self._lock = threading.Lock()

    def increment(self, amount: int = 1) -> int:
        _acquire(self._lock)
        try:
            self._value += amount
            return self._value
        finally:
            self._lock.release()

    def get(self) -> int:
        _acquire(self._lock)
        try:
            return self._value
        finally:
            self._lock.release()

    def reset(self) -> None:
        _acquire(self._lock)
        try:
            self._value = 0
        finally:
            self._lock.release()


class ThreadSafeDict:
    """Thread-safe dictionary supporting basic operations."""

    def __init__(self, initial: Optional[Dict[Any, Any]] = None) -> None:
        self._data: Dict[Any, Any] = dict(initial or {})
        self._lock = threading.RLock()

    def __getitem__(self, key: Any) -> Any:
        _acquire(self._lock)
        try:
            return self._data[key]
        finally:
            self._lock.release()

    def __setitem__(self, key: Any, value: Any) -> None:
        _acquire(self._lock)
        try:
            self._data[key] = value
        finally:
            self._lock.release()

    def get(self, key: Any, default: Any = None) -> Any:
        _acquire(self._lock)
        try:
            return self._data.get(key, default)
        finally:
            self._lock.release()

    def update(self, other: Dict[Any, Any]) -> None:
        _acquire(self._lock)
        try:
            self._data.update(other)
        finally:
            self._lock.release()

    def increment(self, key: Any, amount: int = 1) -> int:
        _acquire(self._lock)
        try:
            self._data[key] = self._data.get(key, 0) + amount
            return self._data[key]
        finally:
            self._lock.release()

    def snapshot(self) -> Dict[Any, Any]:
        _acquire(self._lock)
        try:
            return dict(self._data)
        finally:
            self._lock.release()

    def clear(self) -> None:
        _acquire(self._lock)
        try:
            self._data.clear()
        finally:
            self._lock.release()


class ThreadSafeStats(ThreadSafeDict):
    """Thread-safe statistics container with convenience methods."""

    def reset(self, defaults: Optional[Dict[Any, Any]] = None) -> None:
        _acquire(self._lock)
        try:
            self._data.clear()
            if defaults:
                self._data.update(defaults)
        finally:
            self._lock.release()
