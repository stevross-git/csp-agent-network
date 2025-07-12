# enhanced_csp/network/batching_fixes.py
"""
Fixed version of MessageBatcher with proper task management and deadline handling.
Resolves critical memory leaks and improves reliability.
"""

import asyncio
import time
import heapq
import logging
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class DeadlineTask:
    """Wrapper for deadline management tasks"""
    deadline: float
    task: asyncio.Task
    message_count: int
    
    def __lt__(self, other):
        return self.deadline < other.deadline


class ImprovedMessageBatcher:
    """
    Fixed version of MessageBatcher with proper task management.
    
    Key fixes:
    - Proper task lifecycle management
    - Memory leak prevention
    - Better deadline handling
    - Thread-safe operations
    """
    
    def __init__(self, config, send_callback: Callable):
        self.config = config
        self.send_callback = send_callback
        self.pending_messages = []
        self.pending_size = 0
        
        # Fixed: Use heap for deadline management
        self._deadline_heap: List[DeadlineTask] = []
        self._running = False
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Locks for thread safety
        self._batch_lock = asyncio.Lock()
        self._deadline_lock = asyncio.Lock()
        
        # Performance metrics
        self.stats = {
            'messages_batched': 0,
            'batches_sent': 0,
            'deadline_violations': 0,
            'avg_batch_size': 0.0
        }
    
    async def start(self):
        """Start the batcher with proper task management"""
        if self._running:
            return
            
        self._running = True
        # Start cleanup task for completed deadline tasks
        self._cleanup_task = asyncio.create_task(self._cleanup_completed_tasks())
        logger.info("ImprovedMessageBatcher started")
    
    async def stop(self):
        """Stop batcher and clean up all tasks"""
        if not self._running:
            return
            
        self._running = False
        logger.info("Stopping ImprovedMessageBatcher...")
        
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Cancel all deadline tasks
        async with self._deadline_lock:
            while self._deadline_heap:
                deadline_task = heapq.heappop(self._deadline_heap)
                if not deadline_task.task.done():
                    deadline_task.task.cancel()
                    try:
                        await deadline_task.task
                    except asyncio.CancelledError:
                        pass
        
        # Flush remaining messages
        await self._flush_batch()
        logger.info("ImprovedMessageBatcher stopped")
    
    async def add_message(self, message, deadline_ms=None):
        """Add message with proper deadline scheduling"""
        if not self._running:
            return False
            
        now = time.time()
        
        # Calculate deadline
        if deadline_ms is None:
            deadline_ms = getattr(self.config, 'max_wait_time_ms', 100)
        deadline = now + (deadline_ms / 1000.0)
        
        async with self._batch_lock:
            self.pending_messages.append({
                'message': message,
                'deadline': deadline,
                'timestamp': now
            })
            self.pending_size += len(str(message))  # Rough size estimate
            
            # Schedule deadline task if this is the earliest deadline
            await self._maybe_schedule_deadline(deadline)
            
            # Check immediate flush conditions
            max_batch_size = getattr(self.config, 'max_batch_size', 50)
            max_batch_bytes = getattr(self.config, 'max_batch_bytes', 1024 * 1024)
            
            if (len(self.pending_messages) >= max_batch_size or
                self.pending_size >= max_batch_bytes):
                await self._flush_batch()
                
        return True
    
    async def _maybe_schedule_deadline(self, deadline: float):
        """Schedule deadline task if needed"""
        async with self._deadline_lock:
            # Check if we need a new deadline task
            if not self._deadline_heap or deadline < self._deadline_heap[0].deadline:
                delay = max(0, deadline - time.time())
                task = asyncio.create_task(self._deadline_flush(delay))
                
                deadline_task = DeadlineTask(
                    deadline=deadline,
                    task=task,
                    message_count=len(self.pending_messages)
                )
                
                heapq.heappush(self._deadline_heap, deadline_task)
    
    async def _deadline_flush(self, delay: float):
        """Flush batch after deadline delay"""
        try:
            await asyncio.sleep(delay)
            await self._flush_batch()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Deadline flush error: {e}")
    
    async def _flush_batch(self):
        """Flush current batch"""
        async with self._batch_lock:
            if not self.pending_messages:
                return
            
            batch = self.pending_messages.copy()
            self.pending_messages.clear()
            self.pending_size = 0
        
        # Update statistics
        self.stats['messages_batched'] += len(batch)
        self.stats['batches_sent'] += 1
        if self.stats['batches_sent'] > 0:
            self.stats['avg_batch_size'] = self.stats['messages_batched'] / self.stats['batches_sent']
        
        # Send batch
        if self.send_callback:
            try:
                await self.send_callback(batch)
            except Exception as e:
                logger.error(f"Batch send callback failed: {e}")
    
    async def _cleanup_completed_tasks(self):
        """Background task to clean up completed deadline tasks"""
        while self._running:
            try:
                async with self._deadline_lock:
                    # Remove completed tasks from heap
                    while (self._deadline_heap and 
                           self._deadline_heap[0].task.done()):
                        completed_task = heapq.heappop(self._deadline_heap)
                        # Task is already completed, no need to cancel
                
                await asyncio.sleep(1.0)  # Check every second
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(1.0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batcher statistics"""
        return {
            **self.stats,
            'pending_messages': len(self.pending_messages),
            'active_deadline_tasks': len(self._deadline_heap),
            'running': self._running
        }
