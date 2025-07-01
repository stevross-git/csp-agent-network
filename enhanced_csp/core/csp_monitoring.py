"""
CSP Engine monitoring instrumentation
"""
from typing import Optional
import time
import asyncio

try:
    from monitoring import get_default
    monitor = get_default()
    MONITORING_ENABLED = True
except ImportError:
    monitor = None
    MONITORING_ENABLED = False

class MonitoredProcess:
    """CSP Process with monitoring"""
    
    def __init__(self, process_type: str, process_id: str):
        self.process_type = process_type
        self.process_id = process_id
        self.start_time = time.time()
        
        if MONITORING_ENABLED:
            monitor.record_process_created(process_type)
    
    async def send(self, channel: 'MonitoredChannel', message: Any):
        """Send message with monitoring"""
        await channel.send(message)
        
        if MONITORING_ENABLED:
            monitor.record_message_exchanged(channel.channel_type)
    
    async def receive(self, channel: 'MonitoredChannel') -> Any:
        """Receive message with monitoring"""
        message = await channel.receive()
        
        if MONITORING_ENABLED:
            monitor.record_message_exchanged(channel.channel_type)
        
        return message
    
    def __del__(self):
        """Record process termination"""
        if MONITORING_ENABLED:
            lifetime = time.time() - self.start_time
            monitor.record_process_lifetime(self.process_type, lifetime)

class MonitoredChannel:
    """CSP Channel with monitoring"""
    
    def __init__(self, channel_type: str, capacity: int = 1):
        self.channel_type = channel_type
        self.capacity = capacity
        self.queue = asyncio.Queue(maxsize=capacity)
        self.message_count = 0
        
        if MONITORING_ENABLED:
            monitor.record_channel_created(channel_type)
    
    async def send(self, message: Any):
        """Send message through channel"""
        await self.queue.put(message)
        self.message_count += 1
    
    async def receive(self) -> Any:
        """Receive message from channel"""
        return await self.queue.get()
    
    def size(self) -> int:
        """Get current channel size"""
        return self.queue.qsize()

def monitor_csp_execution(execution_id: str):
    """Decorator to monitor CSP execution"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            if not MONITORING_ENABLED:
                return await func(*args, **kwargs)
            
            # Update active executions
            monitor.increment_active_executions()
            
            start_time = time.time()
            status = "success"
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "failure"
                raise
            finally:
                # Record execution metrics
                duration = time.time() - start_time
                monitor.record_execution(execution_id, duration, status)
                monitor.decrement_active_executions()
        
        return wrapper
    return decorator
