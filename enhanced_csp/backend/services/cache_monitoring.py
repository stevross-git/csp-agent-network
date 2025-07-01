"""
Cache monitoring instrumentation
"""
from functools import wraps
from typing import Callable, Any, Optional
import logging

try:
    from monitoring import get_default
    monitor = get_default()
    MONITORING_ENABLED = True
except ImportError:
    monitor = None
    MONITORING_ENABLED = False

logger = logging.getLogger(__name__)

class MonitoredCache:
    """Cache wrapper with monitoring"""
    
    def __init__(self, cache_backend):
        self.cache = cache_backend
        self.hits = 0
        self.misses = 0
        self.total_memory = 0
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with monitoring"""
        value = await self.cache.get(key)
        
        if MONITORING_ENABLED:
            if value is not None:
                self.hits += 1
                monitor.record_cache_operation("get", True)
            else:
                self.misses += 1
                monitor.record_cache_operation("get", False)
            
            # Update hit rate
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0
            monitor.update_cache_metrics(hit_rate, self.total_memory)
        
        return value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache with monitoring"""
        await self.cache.set(key, value, ttl)
        
        if MONITORING_ENABLED:
            monitor.record_cache_operation("set", True)
            
            # Estimate memory usage (simplified)
            import sys
            self.total_memory += sys.getsizeof(value)
            monitor.update_cache_metrics(
                self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0,
                self.total_memory
            )
    
    async def delete(self, key: str):
        """Delete from cache with monitoring"""
        await self.cache.delete(key)
        
        if MONITORING_ENABLED:
            monitor.record_cache_operation("delete", True)
