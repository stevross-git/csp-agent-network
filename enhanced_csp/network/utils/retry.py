from __future__ import annotations

"""Retry utilities and a simple circuit breaker."""

import asyncio
import time
from typing import Awaitable, Callable, Iterable, Type

from .structured_logging import get_logger

from ..errors import CircuitBreakerOpen

logger = get_logger("retry")


async def retry_async(
    func: Callable[..., Awaitable],
    *args,
    attempts: int = 3,
    initial_delay: float = 0.2,
    factor: float = 2.0,
    exceptions: Iterable[Type[Exception]] = (Exception,),
    **kwargs,
) -> any:
    """Retry an async function with exponential backoff."""
    delay = initial_delay
    for attempt in range(1, attempts + 1):
        try:
            return await func(*args, **kwargs)
        except tuple(exceptions) as exc:
            if attempt == attempts:
                raise
            logger.debug(
                "retry %d/%d for %s due to %s",
                attempt,
                attempts,
                getattr(func, "__name__", str(func)),
                exc,
            )
            await asyncio.sleep(delay)
            delay *= factor


class CircuitBreaker:
    """Minimal async circuit breaker."""

    def __init__(self, threshold: int = 5, reset_timeout: float = 30.0) -> None:
        self.threshold = threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.opened_at: float | None = None

    def _check(self) -> None:
        if self.opened_at is not None:
            if time.monotonic() - self.opened_at < self.reset_timeout:
                raise CircuitBreakerOpen("circuit breaker open")
            self.failure_count = 0
            self.opened_at = None

    async def call(self, func: Callable[..., Awaitable], *args, **kwargs):
        self._check()
        try:
            result = await func(*args, **kwargs)
        except Exception:
            self.failure_count += 1
            if self.failure_count >= self.threshold:
                self.opened_at = time.monotonic()
                logger.error("circuit breaker opened after %d failures", self.failure_count)
            raise
        else:
            self.failure_count = 0
            self.opened_at = None
            return result
