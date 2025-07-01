# enhanced_csp/network/optimized_channel.py
"""
Optimized Network Channel with Complete Integration
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, List
from enum import Enum

from enhanced_csp.ai_comm import AdvancedCommPattern
from .compression import MessageCompressor, CompressionConfig
from .batching import MessageBatcher, BatchConfig
from .connection_pool import ConnectionPool
from .protocol_optimizer import BinaryProtocol, MessageType
from .adaptive_optimizer import AdaptiveNetworkOptimizer

logger = logging.getLogger(__name__)

class OptimizedNetworkChannel:
    """Network-optimized communication channel with idempotent operations"""
    
    def __init__(self, 
                 channel_id: str, 
                 pattern: AdvancedCommPattern,
                 endpoint: str,
                 config: Optional[Dict[str, Any]] = None):
        self.channel_id = channel_id
        self.pattern = pattern
        self.endpoint = endpoint
        self.config = config or {}
        
        # Initialize optimization components (each channel gets its own)
        self.compressor = MessageCompressor(
            config=CompressionConfig(**self.config.get('compression', {}))
        )
        
        # Configure protocol with max size from config
        protocol_config = self.config.get('protocol', {})
        self.protocol = BinaryProtocol(
            version=protocol_config.get('version', 1),
            max_message_size=protocol_config.get('max_message_mb', 16) * 1024 * 1024
        )
        
        # Batcher with retry callback
        self.batcher = MessageBatcher(
            BatchConfig(**self.config.get('batching', {})),
            send_callback=self._send_batch
        )
        self.batcher.retry_callback = self._queue_batch_retry
        
        self.connection_pool = ConnectionPool(
            **self.config.get('connection_pool', {})
        )
        
        # Each channel gets its own optimizer (no shared state)
        self.optimizer = AdaptiveNetworkOptimizer(
            update_callback=self._apply_optimization_updates,
            create_new_compressor=True
        )
        
        # Concurrency control
        self._send_lock = asyncio.Lock()
        
        # Retry queue for failed batches (not individual messages)
        self.retry_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._retry_task: Optional[asyncio.Task] = None
        
        # Metrics
        self.send_success_count = 0
        self.send_failure_count = 0
        
        # Track if we're started
        self._started = False
        
        # Deadline tracking reference
        self._deadline_task = None
        
    async def start(self):
        """Start all optimization components (idempotent)"""
        if self._started:
            logger.debug(f"Channel {self.channel_id} already started")
            return
            
        # Clear any stale retry queue items from previous run
        while not self.retry_queue.empty():
            try:
                self.retry_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        # Clear batcher queue as well
        while not self.batcher.message_queue.empty():
            try:
                self.batcher.message_queue.get_nowait()
                self.batcher.message_queue.task_done()
            except asyncio.QueueEmpty:
                break
        
        await self.connection_pool.start()
        await self.batcher.start()
        await self.optimizer.start()
        self._retry_task = asyncio.create_task(self._retry_worker())
        
        self._started = True
        logger.info(f"Optimized channel {self.channel_id} started")
    
    async def stop(self):
        """Stop all optimization components (idempotent)"""
        if not self._started:
            logger.debug(f"Channel {self.channel_id} already stopped")
            return
            
        self._started = False
        
        # Cancel deadline task if exists
        if self._deadline_task and not self._deadline_task.done():
            self._deadline_task.cancel()
            try:
                await self._deadline_task
            except asyncio.CancelledError:
                pass
        
        await self.optimizer.stop()
        await self.batcher.stop()
        await self.connection_pool.stop()
        
        if self._retry_task:
            self._retry_task.cancel()
            try:
                await self._retry_task
            except asyncio.CancelledError:
                pass
                
        logger.info(f"Optimized channel {self.channel_id} stopped")
    
    async def send_optimized(self, message: Dict[str, Any], priority: int = 0) -> bool:
        """Send message with full optimization stack"""
        if not self._started:
            logger.error(f"Channel {self.channel_id} not started")
            return False
            
        # Record request
        self.optimizer.record_request()
        
        # Add to batch (handles priority internally)
        success = await self.batcher.add_message(message, priority)
        
        if not success:
            # Queue is full, try retry queue
            try:
                self.retry_queue.put_nowait({
                    "message": message,
                    "priority": priority,
                    "attempts": 0
                })
            except asyncio.QueueFull:
                logger.error("Both main and retry queues are full")
                self.send_failure_count += 1
                return False
                
        return True
    
    async def _send_batch(self, batch: Dict[str, Any]):
        """Send batch with compression and protocol optimization"""
        async with self._send_lock:
            try:
                # Encode with binary protocol
                msg_type = MessageType.BATCH if batch["type"] == "batch" else MessageType.COMPRESSED
                encoded = self.protocol.encode_message(
                    batch, 
                    msg_type,
                    compressed=batch.get("type") == "compressed_batch",
                    batched=True
                )
                
                # Get connection from pool
                async with self.connection_pool.get_connection(self.endpoint) as session:
                    # Send with timing
                    start_time = time.perf_counter()
                    
                    async with session.post(
                        f"{self.endpoint}/messages",
                        data=encoded,
                        headers={"Content-Type": "application/octet-stream"}
                    ) as resp:
                        resp.raise_for_status()
                        response_data = await resp.read()
                    
                    # Record metrics
                    latency = (time.perf_counter() - start_time) * 1000
                    self.optimizer.record_latency(latency)
                    
                    # Calculate throughput
                    bytes_sent = len(encoded)
                    bytes_received = len(response_data)
                    duration = latency / 1000.0
                    throughput = (bytes_sent + bytes_received) / duration if duration > 0 else 0
                    self.optimizer.record_throughput(throughput)
                    
                    self.send_success_count += 1
                    
            except Exception as e:
                logger.error(f"Failed to send batch: {e}")
                self.optimizer.record_error()
                self.send_failure_count += 1
                
                # Record negative throughput for congestion signal
                self.optimizer.record_throughput(-1000)
                
                # For retryable errors, queue the entire batch
                if hasattr(e, 'status') and e.status in (429, 502, 503, 504):
                    await self._queue_batch_retry(batch)
                elif not hasattr(e, 'status'):
                    # Network errors, timeouts, etc
                    await self._queue_batch_retry(batch)
                    
    async def _queue_batch_retry(self, batch: Dict[str, Any]):
        """Queue entire batch for retry (preserves batching benefits)"""
        try:
            self.retry_queue.put_nowait({
                "batch": batch,
                "attempts": 1,
                "first_attempt_time": time.time()
            })
        except asyncio.QueueFull:
            logger.error("Retry queue full, batch dropped")
    
    async def _retry_worker(self):
        """Worker to retry failed batches"""
        while self._started:
            try:
                item = await asyncio.wait_for(self.retry_queue.get(), timeout=5.0)
                
                # Exponential backoff based on attempts
                wait_time = min(60, 2 ** item["attempts"])
                await asyncio.sleep(wait_time)
                
                # Check if too old (give up after 5 minutes)
                if time.time() - item["first_attempt_time"] > 300:
                    logger.error(f"Batch dropped after {item['attempts']} attempts (timeout)")
                    continue
                
                # Retry the batch
                await self._send_batch(item["batch"])
                
            except asyncio.TimeoutError:
                # No items to retry
                continue
            except Exception as e:
                logger.error(f"Retry worker error: {e}")
                await asyncio.sleep(5)
                
    def _apply_optimization_updates(self, new_params: Dict[str, Any]):
        """Apply optimization parameter updates"""
        # Update batcher config
        if "batch_size" in new_params:
            self.batcher.config.max_batch_size = new_params["batch_size"]
        
        # Update compressor algorithm
        if "compression_algorithm" in new_params:
            from .compression import CompressionAlgorithm
            self.compressor.default_algorithm = CompressionAlgorithm(
                new_params["compression_algorithm"]
            )
        
        # Update connection pool size would require more complex logic
        # as we can't easily resize an active pool
        
        logger.info(f"Applied optimization updates: {new_params}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive channel statistics"""
        return {
            "channel_id": self.channel_id,
            "messages_sent": self.send_success_count,
            "messages_failed": self.send_failure_count,
            "success_rate": self.send_success_count / (self.send_success_count + self.send_failure_count) 
                           if (self.send_success_count + self.send_failure_count) > 0 else 0,
            "batch_queue_size": self.batcher.get_queue_size(),
            "retry_queue_size": self.retry_queue.qsize(),
            "compression_stats": self.compressor.export_stats(),
            "batch_metrics": {
                "total_batches": self.batcher.metrics.total_batches,
                "average_batch_size": self.batcher.metrics.average_batch_size,
                "priority_bypass_count": self.batcher.metrics.priority_bypass_count,
                "queue_full_count": self.batcher.metrics.queue_full_count
            },
            "connection_pool": self.connection_pool.get_stats(),
            "current_optimization": self.optimizer.optimization_params
        }