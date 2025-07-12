# enhanced_csp/network/batching.py
import asyncio
import time
import msgpack
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from .utils.structured_logging import get_logger
from collections import deque
import heapq

logger = get_logger("batching")

@dataclass
class PendingMessage:
    """Message with deadline tracking"""
    message: Dict[str, Any]
    priority: int
    timestamp: float
    deadline: float
    size: int
    
    def __lt__(self, other):
        # For heap - earliest deadline first
        return self.deadline < other.deadline

# enhanced_csp/network/batching.py (key fix section)

class MessageBatcher:
    """Intelligent message batching with deadline-driven flushing"""
    
    def __init__(self, config: BatchConfig, send_callback: Optional[Callable] = None):
        self.config = config
        self.pending_messages: List[PendingMessage] = []
        self.pending_size = 0
        self.send_callback = send_callback
        self.metrics = BatchMetrics()
        
        # Bounded queue for backpressure
        self.message_queue: asyncio.Queue = asyncio.Queue(maxsize=config.queue_size)
        
        # Deadline management
        self._deadline_event = asyncio.Event()
        self._next_deadline: Optional[float] = None
        self._deadline_tasks: List[asyncio.Task] = []  # Track deadline tasks
        
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None
        
        # Lock for concurrent batch operations
        self._batch_lock = asyncio.Lock()
        
    async def stop(self):
        """Stop the batching worker and flush pending messages"""
        self._running = False
        
        # Cancel all deadline tasks
        for task in self._deadline_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._deadline_tasks.clear()
        
        # Flush remaining messages
        await self._flush_batch()
        
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
                
        # Clear queues
        while not self.message_queue.empty():
            try:
                self.message_queue.get_nowait()
                self.message_queue.task_done()
            except asyncio.QueueEmpty:
                break
                
        logger.info("Message batcher stopped")
        
    async def _schedule_deadline(self, deadline: float):
        """Schedule a deadline event"""
        wait_time = deadline - time.monotonic()
        if wait_time > 0:
            await asyncio.sleep(wait_time)
            self._deadline_event.set()
    
    async def _add_to_pending(self, item: Dict[str, Any]):
        """Add message to pending batch with deadline tracking"""
        async with self._batch_lock:
            # ... existing code ...
            
            # Update deadline tracking
            if self._next_deadline is None or deadline < self._next_deadline:
                self._next_deadline = deadline
                # Create and track deadline task
                task = asyncio.create_task(self._schedule_deadline(deadline))
                self._deadline_tasks.append(task)
                
                # Clean up completed tasks
                self._deadline_tasks = [t for t in self._deadline_tasks if not t.done()]