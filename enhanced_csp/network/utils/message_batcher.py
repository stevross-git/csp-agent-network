"""Simple async message batching utility."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Awaitable, List
import msgpack

from .task_manager import TaskManager


@dataclass
class BatchMetrics:
    total_batches: int = 0
    total_messages: int = 0
    priority_bypass_count: int = 0

    @property
    def average_batch_size(self) -> float:
        if self.total_batches == 0:
            return 0.0
        return self.total_messages / self.total_batches


@dataclass
class BatchConfig:
    max_messages: int = 20
    max_batch_bytes: int = 64 * 1024
    flush_interval: float = 0.05
    urgent_priority: int = 10


class MessageBatcher:
    """Batch NetworkMessages before sending to improve throughput."""

    def __init__(
        self,
        send_batch: Callable[[dict], Awaitable[bool]],
        config: BatchConfig | None = None,
        task_manager: TaskManager | None = None,
    ) -> None:
        self.send_batch = send_batch
        self.config = config or BatchConfig()
        self.task_manager = task_manager
        self._queue: List[dict] = []
        self._queue_bytes = 0
        self._lock = asyncio.Lock()
        self._task: asyncio.Task | None = None
        self._running = False
        self.metrics = BatchMetrics()

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        loop_task = self._flush_loop()
        if self.task_manager:
            self._task = self.task_manager.create_task(loop_task, name="batcher")
        else:
            self._task = asyncio.create_task(loop_task)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        await self.flush()

    async def add_message(self, message: dict, priority: int = 0) -> None:
        if priority >= self.config.urgent_priority:
            self.metrics.priority_bypass_count += 1
            await self._send([message])
            return
        encoded = msgpack.packb(message, use_bin_type=True)
        async with self._lock:
            self._queue.append(message)
            self._queue_bytes += len(encoded)
            if (
                len(self._queue) >= self.config.max_messages
                or self._queue_bytes >= self.config.max_batch_bytes
            ):
                batch = self._queue
                self._queue = []
                self._queue_bytes = 0
                await self._send(batch)

    async def flush(self) -> None:
        async with self._lock:
            if not self._queue:
                return
            batch = self._queue
            self._queue = []
            self._queue_bytes = 0
        await self._send(batch)

    async def _flush_loop(self) -> None:
        try:
            while self._running:
                await asyncio.sleep(self.config.flush_interval)
                await self.flush()
        except asyncio.CancelledError:
            pass

    async def _send(self, messages: List[dict]) -> None:
        batch_data = {
            "type": "batch",
            "count": len(messages),
            "messages": messages,
            "timestamp": datetime.utcnow().isoformat(),
        }
        await self.send_batch(batch_data)
        self.metrics.total_batches += 1
        self.metrics.total_messages += len(messages)

    def get_queue_size(self) -> int:
        return len(self._queue)
