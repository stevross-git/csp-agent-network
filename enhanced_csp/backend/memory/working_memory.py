"""
Working Memory Layer Implementation
===================================

This module implements the Working Memory layer for the Enhanced-CSP system.
Working Memory provides fast, temporary storage for active agent processes.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import weakref
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


@dataclass
class MemoryItem:
    """Represents a single item in working memory"""
    key: str
    value: Any
    timestamp: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    ttl: Optional[timedelta] = None
    
    def is_expired(self) -> bool:
        """Check if this memory item has expired"""
        if self.ttl is None:
            return False
        return datetime.now() > self.timestamp + self.ttl
    
    def access(self):
        """Update access statistics"""
        self.access_count += 1
        self.last_accessed = datetime.now()


class WorkingCache:
    """High-performance cache for frequently accessed items"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: OrderedDict[str, MemoryItem] = OrderedDict()
        self._access_patterns: Dict[str, List[datetime]] = {}
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with LRU behavior"""
        if key in self._cache:
            item = self._cache[key]
            if not item.is_expired():
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                item.access()
                self._track_access(key)
                return item.value
            else:
                # Remove expired item
                del self._cache[key]
        return None
    
    def put(self, key: str, value: Any, ttl: Optional[timedelta] = None):
        """Add item to cache"""
        if key in self._cache:
            # Update existing
            self._cache[key].value = value
            self._cache[key].timestamp = datetime.now()
            self._cache[key].ttl = ttl
            self._cache.move_to_end(key)
        else:
            # Add new
            if len(self._cache) >= self.max_size:
                # Remove least recently used
                self._cache.popitem(last=False)
            self._cache[key] = MemoryItem(key, value, ttl=ttl)
    
    def _track_access(self, key: str):
        """Track access patterns for optimization"""
        if key not in self._access_patterns:
            self._access_patterns[key] = []
        self._access_patterns[key].append(datetime.now())
        # Keep only last 100 accesses
        if len(self._access_patterns[key]) > 100:
            self._access_patterns[key] = self._access_patterns[key][-100:]
    
    def get_hot_keys(self, threshold: int = 10) -> Set[str]:
        """Get frequently accessed keys"""
        hot_keys = set()
        now = datetime.now()
        for key, accesses in self._access_patterns.items():
            recent_accesses = [a for a in accesses if now - a < timedelta(minutes=5)]
            if len(recent_accesses) >= threshold:
                hot_keys.add(key)
        return hot_keys
    
    def clear_expired(self):
        """Remove all expired items"""
        expired_keys = [k for k, v in self._cache.items() if v.is_expired()]
        for key in expired_keys:
            del self._cache[key]


class WorkingMemory:
    """
    Working Memory implementation for CSP agents.
    Provides fast, temporary storage with automatic cleanup.
    """
    
    def __init__(self, agent_id: str, capacity_mb: float = 256):
        self.agent_id = agent_id
        self.capacity_bytes = capacity_mb * 1024 * 1024
        self.used_bytes = 0
        
        # Memory stores
        self._memory: Dict[str, MemoryItem] = {}
        self._cache = WorkingCache(max_size=1000)
        self._weak_refs: Dict[str, weakref.ref] = {}
        
        # Memory pressure handling
        self._pressure_threshold = 0.9  # 90% capacity
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.stats = {
            'total_writes': 0,
            'total_reads': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'evictions': 0,
            'last_cleanup': datetime.now()
        }
        
        logger.info(f"WorkingMemory initialized for agent {agent_id} with {capacity_mb}MB capacity")
    
    async def start(self):
        """Start background cleanup task"""
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
    
    async def stop(self):
        """Stop background tasks"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
    
    def store(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> bool:
        """Store item in working memory"""
        try:
            # Estimate memory usage (simplified)
            estimated_size = len(str(value).encode('utf-8'))
            
            # Check capacity
            if self.used_bytes + estimated_size > self.capacity_bytes:
                if not self._make_space(estimated_size):
                    logger.warning(f"WorkingMemory full for agent {self.agent_id}")
                    return False
            
            # Store in memory
            item = MemoryItem(key, value, ttl=ttl)
            self._memory[key] = item
            self.used_bytes += estimated_size
            
            # Add to cache if frequently accessed
            if key in self._cache.get_hot_keys():
                self._cache.put(key, value, ttl)
            
            # Create weak reference for large objects
            if estimated_size > 1024 * 100:  # 100KB
                self._weak_refs[key] = weakref.ref(value)
            
            self.stats['total_writes'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Error storing in WorkingMemory: {e}")
            return False
    
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve item from working memory"""
        # Check cache first
        cached_value = self._cache.get(key)
        if cached_value is not None:
            self.stats['cache_hits'] += 1
            return cached_value
        
        self.stats['cache_misses'] += 1
        
        # Check main memory
        if key in self._memory:
            item = self._memory[key]
            if not item.is_expired():
                item.access()
                # Add to cache for future access
                self._cache.put(key, item.value, item.ttl)
                self.stats['total_reads'] += 1
                return item.value
            else:
                # Remove expired item
                self.remove(key)
        
        # Check weak references
        if key in self._weak_refs:
            weak_ref = self._weak_refs[key]
            value = weak_ref()
            if value is not None:
                return value
            else:
                del self._weak_refs[key]
        
        return None
    
    def remove(self, key: str) -> bool:
        """Remove item from working memory"""
        if key in self._memory:
            item = self._memory[key]
            estimated_size = len(str(item.value).encode('utf-8'))
            self.used_bytes -= estimated_size
            del self._memory[key]
            
            # Remove from cache
            if key in self._cache._cache:
                del self._cache._cache[key]
            
            # Remove weak reference
            if key in self._weak_refs:
                del self._weak_refs[key]
            
            return True
        return False
    
    def clear(self):
        """Clear all working memory"""
        self._memory.clear()
        self._cache._cache.clear()
        self._weak_refs.clear()
        self.used_bytes = 0
        logger.info(f"WorkingMemory cleared for agent {self.agent_id}")
    
    def get_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        return {
            'agent_id': self.agent_id,
            'used_mb': self.used_bytes / (1024 * 1024),
            'capacity_mb': self.capacity_bytes / (1024 * 1024),
            'usage_percent': (self.used_bytes / self.capacity_bytes) * 100,
            'item_count': len(self._memory),
            'cache_size': len(self._cache._cache),
            'stats': self.stats
        }
    
    def _make_space(self, required_bytes: int) -> bool:
        """Try to free up memory space"""
        # Remove expired items first
        self._cleanup_expired()
        
        if self.used_bytes + required_bytes <= self.capacity_bytes:
            return True
        
        # Remove least recently used items
        items_by_access = sorted(
            self._memory.items(),
            key=lambda x: x[1].last_accessed
        )
        
        for key, item in items_by_access:
            if self.used_bytes + required_bytes <= self.capacity_bytes:
                return True
            self.remove(key)
            self.stats['evictions'] += 1
        
        return self.used_bytes + required_bytes <= self.capacity_bytes
    
    def _cleanup_expired(self):
        """Remove all expired items"""
        expired_keys = [k for k, v in self._memory.items() if v.is_expired()]
        for key in expired_keys:
            self.remove(key)
        self._cache.clear_expired()
    
    async def _periodic_cleanup(self):
        """Periodic cleanup task"""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                self._cleanup_expired()
                
                # Check memory pressure
                usage_percent = (self.used_bytes / self.capacity_bytes) * 100
                if usage_percent > self._pressure_threshold * 100:
                    logger.warning(f"High memory pressure for agent {self.agent_id}: {usage_percent:.1f}%")
                
                self.stats['last_cleanup'] = datetime.now()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")