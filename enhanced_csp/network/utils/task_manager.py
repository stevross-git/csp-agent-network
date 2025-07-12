from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncIterator, Set

from .structured_logging import get_logger

logger = get_logger("task_manager")


class TaskManager:
    """Manage lifecycle of asyncio tasks."""

    def __init__(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self.loop = loop or asyncio.get_event_loop()
        self.tasks: Set[asyncio.Task] = set()

    def create_task(self, coro: asyncio.coroutines, name: str | None = None) -> asyncio.Task:
        """Create and track an asyncio task."""
        task = self.loop.create_task(coro, name=name)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

    @asynccontextmanager
    async def manage(self, coro: asyncio.coroutines, name: str | None = None) -> AsyncIterator[asyncio.Task]:
        """Context manager that runs a task and ensures cleanup."""
        task = self.create_task(coro, name=name)
        try:
            yield task
            await task
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    logger.debug("task %s cancelled", getattr(task, "get_name", lambda: "?")())

    async def cancel_all(self, timeout: float = 5.0) -> None:
        """Cancel all tracked tasks and wait for completion."""
        for task in list(self.tasks):
            if not task.done():
                task.cancel()
        if not self.tasks:
            return
        done, pending = await asyncio.wait(self.tasks, timeout=timeout)
        for t in pending:
            with logger.context(operation="cancel_all"):
                logger.warning(
                    "task %s did not finish before timeout",
                    getattr(t, "get_name", lambda: "?")(),
                )
        self.tasks.clear()


class ResourceManager:
    """Track resources to ensure they are closed properly."""

    def __init__(self) -> None:
        self.resources: Set[Any] = set()

    @contextmanager
    def manage(self, resource: Any) -> Any:
        """Context manager that registers a resource and closes it on exit."""
        self.resources.add(resource)
        try:
            yield resource
        finally:
            self.close(resource)

    def close(self, resource: Any) -> None:
        if resource not in self.resources:
            return
        try:
            close = getattr(resource, "close", None)
            if callable(close):
                close()
        except Exception as exc:  # pragma: no cover - runtime issues
            logger.warning("error closing resource %s: %s", resource, exc)
        finally:
            self.resources.discard(resource)

    async def close_all(self) -> None:
        for res in list(self.resources):
            self.close(res)
        if self.resources:
            logger.warning("potential resource leak: %d resources", len(self.resources))
            self.resources.clear()
