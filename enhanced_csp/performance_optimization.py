#!/usr/bin/env python3
"""
Performance & Scalability Optimization for Enhanced CSP System
==============================================================

Complete performance optimization including:
- Database indexing and query optimization
- Multi-level caching strategies
- Connection pooling and resource management
- Auto-scaling policies and load balancing
- Performance monitoring and profiling
- Memory optimization and garbage collection
- Asynchronous processing optimization
"""

import asyncio
import logging
import time
import psutil
import threading
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json
import pickle
import hashlib
from functools import wraps, lru_cache
import weakref
import gc

# Database and caching imports
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import QueuePool, NullPool
from sqlalchemy import Index, text
import redis.asyncio as redis
import aioredis
from aiocache import Cache, cached
from aiocache.serializers import PickleSerializer

# Monitoring and profiling
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, Summary
import psutil
import resource
import tracemalloc

# Load balancing and scaling
from kubernetes import client as k8s_client, config as k8s_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# PERFORMANCE METRICS AND MONITORING
# ============================================================================

class PerformanceMetrics:
    """Comprehensive performance metrics collection"""
    
    def __init__(self):
        # Prometheus metrics
        self.request_duration = Histogram(
            'csp_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint', 'status']
        )
        
        self.active_connections = Gauge(
            'csp_active_connections',
            'Number of active connections'
        )
        
        self.cache_hits = Counter(
            'csp_cache_hits_total',
            'Total cache hits',
            ['cache_type']
        )
        
        self.cache_misses = Counter(
            'csp_cache_misses_total',
            'Total cache misses',
            ['cache_type']
        )
        
        self.database_queries = Counter(
            'csp_database_queries_total',
            'Total database queries',
            ['operation', 'table']
        )
        
        self.memory_usage = Gauge(
            'csp_memory_usage_bytes',
            'Memory usage in bytes'
        )
        
        self.cpu_usage = Gauge(
            'csp_cpu_usage_percent',
            'CPU usage percentage'
        )
        
        self.process_queue_size = Gauge(
            'csp_process_queue_size',
            'Size of process queue'
        )
        
        # Internal metrics tracking
        self.response_times = deque(maxlen=1000)
        self.error_rates = defaultdict(int)
        self.throughput_tracker = deque(maxlen=100)
        
    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record request metrics"""
        self.request_duration.labels(method=method, endpoint=endpoint, status=status).observe(duration)
        self.response_times.append(duration)
        
    def record_cache_hit(self, cache_type: str):
        """Record cache hit"""
        self.cache_hits.labels(cache_type=cache_type).inc()
        
    def record_cache_miss(self, cache_type: str):
        """Record cache miss"""
        self.cache_misses.labels(cache_type=cache_type).inc()
        
    def record_database_query(self, operation: str, table: str):
        """Record database query"""
        self.database_queries.labels(operation=operation, table=table).inc()
        
    def update_system_metrics(self):
        """Update system resource metrics"""
        # Memory usage
        process = psutil.Process()
        memory_info = process.memory_info()
        self.memory_usage.set(memory_info.rss)
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        self.cpu_usage.set(cpu_percent)
        
        # Connection count
        connections = len(process.connections())
        self.active_connections.set(connections)
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        if self.response_times:
            avg_response_time = sum(self.response_times) / len(self.response_times)
            p95_response_time = sorted(self.response_times)[int(len(self.response_times) * 0.95)]
        else:
            avg_response_time = 0
            p95_response_time = 0
            
        return {
            'average_response_time': avg_response_time,
            'p95_response_time': p95_response_time,
            'total_requests': len(self.response_times),
            'error_rate': sum(self.error_rates.values()) / max(len(self.response_times), 1),
            'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
            'cpu_usage_percent': psutil.cpu_percent(),
            'active_connections': len(psutil.Process().connections())
        }

# Global metrics instance
performance_metrics = PerformanceMetrics()

# ============================================================================
# DATABASE OPTIMIZATION
# ============================================================================

class DatabaseOptimizer:
    """Advanced database optimization and connection management"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = None
        self.session_factory = None
        self.connection_pool_size = 20
        self.max_overflow = 30
        self.pool_timeout = 30
        self.pool_recycle = 3600  # 1 hour
        
    async def initialize(self):
        """Initialize optimized database engine"""
        
        # Create engine with optimized pool settings
        self.engine = create_async_engine(
            self.database_url,
            poolclass=QueuePool,
            pool_size=self.connection_pool_size,
            max_overflow=self.max_overflow,
            pool_timeout=self.pool_timeout,
            pool_recycle=self.pool_recycle,
            pool_pre_ping=True,  # Validate connections
            echo=False,  # Disable SQL logging in production
            future=True,
            # Connection pool optimization
            connect_args={
                "server_settings": {
                    "application_name": "enhanced_csp_system",
                    "jit": "off"  # Disable JIT for better performance
                }
            }
        )
        
        # Create session factory
        self.session_factory = sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        logger.info(f"Database engine initialized with pool_size={self.connection_pool_size}")
        
    async def create_indexes(self):
        """Create performance-critical database indexes"""
        
        indexes = [
            # CSP Process indexes
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_processes_status ON csp_processes(status) WHERE status = 'active'",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_processes_created_at ON csp_processes(created_at DESC)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_processes_type_status ON csp_processes(process_type, status)",
            
            # Channel indexes
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_channels_type ON csp_channels(channel_type)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_channels_active ON csp_channels(is_active) WHERE is_active = TRUE",
            
            # Event indexes
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_events_timestamp ON csp_events(timestamp DESC)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_events_process_id ON csp_events(process_id)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_events_channel_id ON csp_events(channel_id)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_events_composite ON csp_events(process_id, channel_id, timestamp DESC)",
            
            # AI Agent indexes
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ai_agents_status ON ai_agents(status)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ai_agents_type ON ai_agents(agent_type)",
            
            # Quantum state indexes
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_quantum_states_agent_id ON quantum_states(agent_id)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_quantum_entanglements_active ON quantum_entanglements(is_active) WHERE is_active = TRUE",
            
            # Blockchain transaction indexes
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_blockchain_txs_hash ON blockchain_transactions(transaction_hash)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_blockchain_txs_timestamp ON blockchain_transactions(timestamp DESC)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_blockchain_txs_status ON blockchain_transactions(status)",
            
            # Metrics and monitoring indexes
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_metrics_timestamp ON performance_metrics(timestamp DESC)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_metrics_metric_name ON performance_metrics(metric_name, timestamp DESC)",
        ]
        
        async with self.engine.begin() as conn:
            for index_sql in indexes:
                try:
                    await conn.execute(text(index_sql))
                    logger.info(f"Created index: {index_sql.split('idx_')[1].split(' ')[0]}")
                except Exception as e:
                    logger.warning(f"Index creation failed: {e}")
                    
    async def optimize_tables(self):
        """Optimize table structure and perform maintenance"""
        
        optimization_queries = [
            # Update table statistics
            "ANALYZE csp_processes",
            "ANALYZE csp_channels", 
            "ANALYZE csp_events",
            "ANALYZE ai_agents",
            "ANALYZE quantum_states",
            "ANALYZE blockchain_transactions",
            
            # Vacuum tables to reclaim space
            "VACUUM ANALYZE csp_events",
            "VACUUM ANALYZE performance_metrics",
            
            # Update configuration for better performance
            "SET work_mem = '256MB'",
            "SET maintenance_work_mem = '512MB'",
            "SET checkpoint_completion_target = 0.9",
            "SET wal_buffers = '16MB'",
            "SET random_page_cost = 1.1",  # For SSD storage
        ]
        
        async with self.engine.begin() as conn:
            for query in optimization_queries:
                try:
                    await conn.execute(text(query))
                    logger.info(f"Executed optimization: {query}")
                except Exception as e:
                    logger.warning(f"Optimization failed: {e}")
                    
    async def get_pool_status(self) -> Dict[str, Any]:
        """Get connection pool status"""
        pool = self.engine.pool
        return {
            'pool_size': pool.size(),
            'checked_in': pool.checkedin(),
            'checked_out': pool.checkedout(),
            'overflow': pool.overflow(),
            'invalid': pool.invalid()
        }

# ============================================================================
# MULTI-LEVEL CACHING SYSTEM
# ============================================================================

class MultiLevelCache:
    """Advanced multi-level caching with Redis and in-memory layers"""
    
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.redis_client = None
        self.local_cache = {}
        self.cache_stats = defaultdict(int)
        self.cache_sizes = {
            'L1': 1000,  # In-memory cache size
            'L2': 10000,  # Redis cache size
        }
        
        # Cache configuration
        self.default_ttl = {
            'L1': 300,   # 5 minutes
            'L2': 3600,  # 1 hour
            'L3': 86400  # 24 hours (persistent)
        }
        
    async def initialize(self):
        """Initialize cache connections"""
        self.redis_client = redis.from_url(
            self.redis_url,
            encoding='utf-8',
            decode_responses=False,
            max_connections=20,
            retry_on_timeout=True,
            health_check_interval=30
        )
        
        # Test Redis connection
        await self.redis_client.ping()
        logger.info("Multi-level cache initialized")
        
    async def get(self, key: str, cache_level: str = 'auto') -> Optional[Any]:
        """Get value from multi-level cache"""
        
        # Try L1 cache (in-memory) first
        if key in self.local_cache:
            entry = self.local_cache[key]
            if entry['expires'] > time.time():
                self.cache_stats['L1_hits'] += 1
                performance_metrics.record_cache_hit('L1')
                return entry['value']
            else:
                del self.local_cache[key]
                
        # Try L2 cache (Redis)
        try:
            redis_value = await self.redis_client.get(f"csp:{key}")
            if redis_value:
                value = pickle.loads(redis_value)
                
                # Promote to L1 cache
                self.local_cache[key] = {
                    'value': value,
                    'expires': time.time() + self.default_ttl['L1']
                }
                
                # Manage L1 cache size
                if len(self.local_cache) > self.cache_sizes['L1']:
                    self._evict_lru_local()
                    
                self.cache_stats['L2_hits'] += 1
                performance_metrics.record_cache_hit('L2')
                return value
                
        except Exception as e:
            logger.warning(f"Redis cache error: {e}")
            
        # Cache miss
        self.cache_stats['misses'] += 1
        performance_metrics.record_cache_miss('multilevel')
        return None
        
    async def set(self, key: str, value: Any, ttl: Optional[int] = None, cache_level: str = 'auto'):
        """Set value in multi-level cache"""
        
        ttl = ttl or self.default_ttl['L2']
        
        # Set in L1 cache (in-memory)
        self.local_cache[key] = {
            'value': value,
            'expires': time.time() + min(ttl, self.default_ttl['L1'])
        }
        
        # Manage L1 cache size
        if len(self.local_cache) > self.cache_sizes['L1']:
            self._evict_lru_local()
            
        # Set in L2 cache (Redis)
        try:
            serialized_value = pickle.dumps(value)
            await self.redis_client.setex(f"csp:{key}", ttl, serialized_value)
        except Exception as e:
            logger.warning(f"Redis cache set error: {e}")
            
    async def delete(self, key: str):
        """Delete from all cache levels"""
        
        # Delete from L1
        self.local_cache.pop(key, None)
        
        # Delete from L2
        try:
            await self.redis_client.delete(f"csp:{key}")
        except Exception as e:
            logger.warning(f"Redis cache delete error: {e}")
            
    async def clear_cache(self, pattern: str = None):
        """Clear cache with optional pattern"""
        
        # Clear L1
        if pattern:
            keys_to_delete = [k for k in self.local_cache.keys() if pattern in k]
            for key in keys_to_delete:
                del self.local_cache[key]
        else:
            self.local_cache.clear()
            
        # Clear L2
        try:
            if pattern:
                keys = await self.redis_client.keys(f"csp:*{pattern}*")
                if keys:
                    await self.redis_client.delete(*keys)
            else:
                await self.redis_client.flushdb()
        except Exception as e:
            logger.warning(f"Redis cache clear error: {e}")
            
    def _evict_lru_local(self):
        """Evict least recently used items from L1 cache"""
        if len(self.local_cache) <= self.cache_sizes['L1']:
            return
            
        # Find expired entries first
        current_time = time.time()
        expired_keys = [k for k, v in self.local_cache.items() if v['expires'] <= current_time]
        
        for key in expired_keys:
            del self.local_cache[key]
            
        # If still over limit, remove oldest entries
        if len(self.local_cache) > self.cache_sizes['L1']:
            # Sort by expiration time and remove oldest
            sorted_items = sorted(self.local_cache.items(), key=lambda x: x[1]['expires'])
            items_to_remove = len(self.local_cache) - self.cache_sizes['L1']
            
            for i in range(items_to_remove):
                key = sorted_items[i][0]
                del self.local_cache[key]
                
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        total_requests = sum(self.cache_stats.values())
        l1_hit_rate = self.cache_stats['L1_hits'] / max(total_requests, 1)
        l2_hit_rate = self.cache_stats['L2_hits'] / max(total_requests, 1)
        miss_rate = self.cache_stats['misses'] / max(total_requests, 1)
        
        return {
            'L1_cache_size': len(self.local_cache),
            'L1_hits': self.cache_stats['L1_hits'],
            'L2_hits': self.cache_stats['L2_hits'],
            'cache_misses': self.cache_stats['misses'],
            'L1_hit_rate': l1_hit_rate,
            'L2_hit_rate': l2_hit_rate,
            'miss_rate': miss_rate,
            'total_hit_rate': l1_hit_rate + l2_hit_rate
        }

# Global cache instance
multi_level_cache = None

# ============================================================================
# PERFORMANCE DECORATORS AND UTILITIES
# ============================================================================

def performance_monitor(func):
    """Decorator to monitor function performance"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            performance_metrics.record_request(
                method=func.__name__,
                endpoint=func.__module__,
                status=200,
                duration=duration
            )
            return result
        except Exception as e:
            duration = time.time() - start_time
            performance_metrics.record_request(
                method=func.__name__,
                endpoint=func.__module__,
                status=500,
                duration=duration
            )
            raise
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            performance_metrics.record_request(
                method=func.__name__,
                endpoint=func.__module__,
                status=200,
                duration=duration
            )
            return result
        except Exception as e:
            duration = time.time() - start_time
            performance_metrics.record_request(
                method=func.__name__,
                endpoint=func.__module__,
                status=500,
                duration=duration
            )
            raise
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper

def cached_result(ttl: int = 300, cache_key_func: Optional[Callable] = None):
    """Decorator for caching function results"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__module__}.{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached_value = await multi_level_cache.get(cache_key)
            if cached_value is not None:
                return cached_value
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await multi_level_cache.set(cache_key, result, ttl)
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For synchronous functions, use simpler caching
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__module__}.{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Simple in-memory cache for sync functions
            if not hasattr(sync_wrapper, '_cache'):
                sync_wrapper._cache = {}
            
            if cache_key in sync_wrapper._cache:
                entry = sync_wrapper._cache[cache_key]
                if entry['expires'] > time.time():
                    return entry['value']
                else:
                    del sync_wrapper._cache[cache_key]
            
            result = func(*args, **kwargs)
            sync_wrapper._cache[cache_key] = {
                'value': result,
                'expires': time.time() + ttl
            }
            return result
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    return decorator

# ============================================================================
# RESOURCE MANAGEMENT AND OPTIMIZATION
# ============================================================================

class ResourceManager:
    """Advanced resource management and optimization"""
    
    def __init__(self):
        self.thread_pool = ThreadPoolExecutor(max_workers=20)
        self.process_pool = ProcessPoolExecutor(max_workers=4)
        self.semaphores = {}
        self.resource_limits = {
            'max_concurrent_requests': 1000,
            'max_memory_usage_mb': 4096,
            'max_cpu_usage_percent': 80,
            'max_database_connections': 50
        }
        
    def get_semaphore(self, name: str, limit: int) -> asyncio.Semaphore:
        """Get or create a named semaphore for resource limiting"""
        if name not in self.semaphores:
            self.semaphores[name] = asyncio.Semaphore(limit)
        return self.semaphores[name]
    
    async def execute_cpu_bound_task(self, func: Callable, *args, **kwargs):
        """Execute CPU-bound task in process pool"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.process_pool, func, *args, **kwargs)
    
    async def execute_io_bound_task(self, func: Callable, *args, **kwargs):
        """Execute I/O-bound task in thread pool"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, func, *args, **kwargs)
    
    def check_resource_limits(self) -> Dict[str, Any]:
        """Check current resource usage against limits"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        cpu_percent = psutil.cpu_percent()
        
        status = {
            'memory_usage_mb': memory_mb,
            'memory_limit_mb': self.resource_limits['max_memory_usage_mb'],
            'memory_ok': memory_mb < self.resource_limits['max_memory_usage_mb'],
            'cpu_usage_percent': cpu_percent,
            'cpu_limit_percent': self.resource_limits['max_cpu_usage_percent'],
            'cpu_ok': cpu_percent < self.resource_limits['max_cpu_usage_percent'],
            'connections': len(process.connections()),
        }
        
        return status
    
    async def optimize_memory(self):
        """Perform memory optimization"""
        # Force garbage collection
        collected = gc.collect()
        
        # Get memory statistics
        if hasattr(tracemalloc, 'is_tracing') and tracemalloc.is_tracing():
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            
            logger.info(f"Memory optimization: collected {collected} objects")
            logger.info(f"Top memory consumers: {len(top_stats)} allocations")
        
        return collected

# Global resource manager
resource_manager = ResourceManager()

# ============================================================================
# AUTO-SCALING AND LOAD BALANCING
# ============================================================================

class AutoScaler:
    """Kubernetes-based auto-scaling controller"""
    
    def __init__(self):
        self.k8s_apps_api = None
        self.k8s_metrics_api = None
        self.scaling_config = {
            'min_replicas': 2,
            'max_replicas': 20,
            'target_cpu_utilization': 70,
            'target_memory_utilization': 80,
            'scale_up_threshold': 85,
            'scale_down_threshold': 30,
            'cooldown_period': 300  # 5 minutes
        }
        self.last_scale_time = {}
        
    async def initialize(self):
        """Initialize Kubernetes API clients"""
        try:
            k8s_config.load_incluster_config()  # For running inside cluster
        except:
            try:
                k8s_config.load_kube_config()  # For development
            except:
                logger.warning("Kubernetes config not available - auto-scaling disabled")
                return
        
        self.k8s_apps_api = k8s_client.AppsV1Api()
        logger.info("Auto-scaler initialized")
    
    async def check_scaling_needs(self, deployment_name: str, namespace: str = 'enhanced-csp'):
        """Check if scaling is needed based on metrics"""
        
        if not self.k8s_apps_api:
            return
        
        try:
            # Get current deployment
            deployment = self.k8s_apps_api.read_namespaced_deployment(
                name=deployment_name,
                namespace=namespace
            )
            
            current_replicas = deployment.status.replicas or 0
            
            # Get performance metrics
            perf_summary = performance_metrics.get_performance_summary()
            cpu_usage = perf_summary['cpu_usage_percent']
            memory_usage_mb = perf_summary['memory_usage_mb']
            
            # Calculate memory usage percentage (assuming 2GB per replica)
            memory_usage_percent = (memory_usage_mb / 2048) * 100
            
            # Determine scaling action
            scale_action = None
            
            # Scale up conditions
            if (cpu_usage > self.scaling_config['scale_up_threshold'] or 
                memory_usage_percent > self.scaling_config['scale_up_threshold']):
                if current_replicas < self.scaling_config['max_replicas']:
                    scale_action = 'up'
            
            # Scale down conditions
            elif (cpu_usage < self.scaling_config['scale_down_threshold'] and 
                  memory_usage_percent < self.scaling_config['scale_down_threshold']):
                if current_replicas > self.scaling_config['min_replicas']:
                    scale_action = 'down'
            
            # Apply cooldown period
            if scale_action:
                last_scale = self.last_scale_time.get(deployment_name, 0)
                if time.time() - last_scale < self.scaling_config['cooldown_period']:
                    logger.info(f"Scaling {deployment_name} skipped due to cooldown")
                    return
                
                await self._perform_scaling(deployment_name, namespace, scale_action, current_replicas)
                
        except Exception as e:
            logger.error(f"Auto-scaling check failed: {e}")
    
    async def _perform_scaling(self, deployment_name: str, namespace: str, 
                             action: str, current_replicas: int):
        """Perform the actual scaling operation"""
        
        if action == 'up':
            new_replicas = min(current_replicas + 2, self.scaling_config['max_replicas'])
        else:  # scale down
            new_replicas = max(current_replicas - 1, self.scaling_config['min_replicas'])
        
        if new_replicas == current_replicas:
            return
        
        try:
            # Update deployment
            body = {'spec': {'replicas': new_replicas}}
            self.k8s_apps_api.patch_namespaced_deployment(
                name=deployment_name,
                namespace=namespace,
                body=body
            )
            
            self.last_scale_time[deployment_name] = time.time()
            logger.info(f"Scaled {deployment_name} from {current_replicas} to {new_replicas} replicas")
            
        except Exception as e:
            logger.error(f"Scaling operation failed: {e}")

# Global auto-scaler
auto_scaler = AutoScaler()

# ============================================================================
# PERFORMANCE OPTIMIZATION CONTROLLER
# ============================================================================

class PerformanceOptimizationController:
    """Main controller for all performance optimizations"""
    
    def __init__(self, database_url: str, redis_url: str):
        self.database_optimizer = DatabaseOptimizer(database_url)
        self.cache_system = MultiLevelCache(redis_url)
        self.resource_manager = ResourceManager()
        self.auto_scaler = AutoScaler()
        self.optimization_tasks = []
        
    async def initialize(self):
        """Initialize all performance optimization components"""
        
        logger.info("Initializing Performance Optimization System...")
        
        # Initialize database optimization
        await self.database_optimizer.initialize()
        await self.database_optimizer.create_indexes()
        await self.database_optimizer.optimize_tables()
        
        # Initialize cache system
        await self.cache_system.initialize()
        global multi_level_cache
        multi_level_cache = self.cache_system
        
        # Initialize auto-scaler
        await self.auto_scaler.initialize()
        
        # Start background optimization tasks
        await self._start_background_tasks()
        
        logger.info("Performance Optimization System initialized successfully")
    
    async def _start_background_tasks(self):
        """Start background optimization tasks"""
        
        # System metrics collection
        self.optimization_tasks.append(
            asyncio.create_task(self._metrics_collection_loop())
        )
        
        # Database maintenance
        self.optimization_tasks.append(
            asyncio.create_task(self._database_maintenance_loop())
        )
        
        # Cache optimization
        self.optimization_tasks.append(
            asyncio.create_task(self._cache_optimization_loop())
        )
        
        # Auto-scaling monitoring
        self.optimization_tasks.append(
            asyncio.create_task(self._auto_scaling_loop())
        )
        
        # Resource optimization
        self.optimization_tasks.append(
            asyncio.create_task(self._resource_optimization_loop())
        )
    
    async def _metrics_collection_loop(self):
        """Background task for metrics collection"""
        while True:
            try:
                performance_metrics.update_system_metrics()
                await asyncio.sleep(10)  # Update every 10 seconds
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(30)
    
    async def _database_maintenance_loop(self):
        """Background task for database maintenance"""
        while True:
            try:
                # Run maintenance every hour
                await asyncio.sleep(3600)
                await self.database_optimizer.optimize_tables()
                logger.info("Database maintenance completed")
            except Exception as e:
                logger.error(f"Database maintenance error: {e}")
    
    async def _cache_optimization_loop(self):
        """Background task for cache optimization"""
        while True:
            try:
                # Check cache performance every 5 minutes
                await asyncio.sleep(300)
                
                stats = self.cache_system.get_cache_stats()
                if stats['total_hit_rate'] < 0.7:  # Less than 70% hit rate
                    logger.warning(f"Cache hit rate low: {stats['total_hit_rate']:.2%}")
                    
                # Clear expired entries
                current_time = time.time()
                expired_keys = [
                    k for k, v in self.cache_system.local_cache.items()
                    if v['expires'] <= current_time
                ]
                for key in expired_keys:
                    del self.cache_system.local_cache[key]
                    
            except Exception as e:
                logger.error(f"Cache optimization error: {e}")
    
    async def _auto_scaling_loop(self):
        """Background task for auto-scaling"""
        while True:
            try:
                # Check scaling needs every 2 minutes
                await asyncio.sleep(120)
                await self.auto_scaler.check_scaling_needs('csp-core')
                await self.auto_scaler.check_scaling_needs('csp-quantum')
                await self.auto_scaler.check_scaling_needs('csp-blockchain')
            except Exception as e:
                logger.error(f"Auto-scaling error: {e}")
    
    async def _resource_optimization_loop(self):
        """Background task for resource optimization"""
        while True:
            try:
                # Check resources every 30 seconds
                await asyncio.sleep(30)
                
                resource_status = self.resource_manager.check_resource_limits()
                
                # If memory usage is high, trigger optimization
                if not resource_status['memory_ok']:
                    logger.warning(f"High memory usage: {resource_status['memory_usage_mb']:.1f}MB")
                    await self.resource_manager.optimize_memory()
                
                # Update metrics
                performance_metrics.memory_usage.set(resource_status['memory_usage_mb'] * 1024 * 1024)
                performance_metrics.cpu_usage.set(resource_status['cpu_usage_percent'])
                
            except Exception as e:
                logger.error(f"Resource optimization error: {e}")
    
    async def get_optimization_status(self) -> Dict[str, Any]:
        """Get comprehensive optimization status"""
        
        # Database status
        db_pool_status = await self.database_optimizer.get_pool_status()
        
        # Cache status
        cache_stats = self.cache_system.get_cache_stats()
        
        # Resource status
        resource_status = self.resource_manager.check_resource_limits()
        
        # Performance summary
        perf_summary = performance_metrics.get_performance_summary()
        
        return {
            'database': {
                'pool_status': db_pool_status,
                'optimization_enabled': True
            },
            'cache': cache_stats,
            'resources': resource_status,
            'performance': perf_summary,
            'auto_scaling': {
                'enabled': self.auto_scaler.k8s_apps_api is not None,
                'config': self.auto_scaler.scaling_config
            },
            'background_tasks': {
                'running': len([t for t in self.optimization_tasks if not t.done()]),
                'total': len(self.optimization_tasks)
            }
        }
    
    async def shutdown(self):
        """Shutdown optimization system"""
        logger.info("Shutting down Performance Optimization System...")
        
        # Cancel background tasks
        for task in self.optimization_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.optimization_tasks, return_exceptions=True)
        
        # Close connections
        if self.database_optimizer.engine:
            await self.database_optimizer.engine.dispose()
        
        if self.cache_system.redis_client:
            await self.cache_system.redis_client.close()
        
        logger.info("Performance Optimization System shutdown complete")

# ============================================================================
# MAIN INITIALIZATION
# ============================================================================

async def initialize_performance_optimization(database_url: str, redis_url: str):
    """Initialize the complete performance optimization system"""
    
    controller = PerformanceOptimizationController(database_url, redis_url)
    await controller.initialize()
    
    return controller

# Example usage
if __name__ == "__main__":
    async def main():
        # Initialize performance optimization
        controller = await initialize_performance_optimization(
            database_url="postgresql+asyncpg://csp_user:csp_pass@localhost:5432/csp_db",
            redis_url="redis://localhost:6379/0"
        )
        
        # Run for demo
        await asyncio.sleep(60)
        
        # Get status
        status = await controller.get_optimization_status()
        print(json.dumps(status, indent=2, default=str))
        
        # Shutdown
        await controller.shutdown()
    
    asyncio.run(main())
