# enhanced_csp/network/connection_pool.py
import asyncio
import time
import socket
import ssl
import aiohttp
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
import logging
from contextlib import asynccontextmanager

try:
    import aiodns
    AIODNS_AVAILABLE = True
except ImportError:
    AIODNS_AVAILABLE = False
    
logger = logging.getLogger(__name__)

class ConnectionPool:
    """Advanced connection pooling with DNS awareness and graceful shutdown"""
    
    def __init__(self, 
                 min_connections: int = 5,
                 max_connections: int = 100,
                 keepalive_timeout: int = 300,
                 enable_http2: bool = True,
                 health_check_interval: int = 30,
                 dns_ttl: int = 30):  # Short TTL for Kubernetes
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.keepalive_timeout = keepalive_timeout
        self.enable_http2 = enable_http2 and self._check_http2_support()
        self.health_check_interval = health_check_interval
        self.dns_ttl = dns_ttl
        
        self.connections: Dict[str, aiohttp.ClientSession] = {}
        self.connection_stats: Dict[str, ConnectionStats] = {}
        self.available_connections: asyncio.Queue = asyncio.Queue()
        self.connection_semaphore = asyncio.Semaphore(max_connections)
        
        # Track in-use connections for graceful shutdown
        self._in_use_connections: Set[str] = set()
        self._in_use_lock = asyncio.Lock()
        
        self._cleanup_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._running = False
        self._lock = asyncio.Lock()
        
    async def stop(self):
        """Stop connection pool with graceful shutdown"""
        logger.info("Starting graceful connection pool shutdown")
        self._running = False
        
        # Cancel maintenance tasks
        for task in [self._cleanup_task, self._health_check_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Wait for in-use connections to be returned (with timeout)
        wait_start = time.time()
        timeout = 30  # 30 second graceful shutdown timeout
        
        while self._in_use_connections and (time.time() - wait_start < timeout):
            logger.info(f"Waiting for {len(self._in_use_connections)} connections to be returned")
            await asyncio.sleep(0.5)
        
        if self._in_use_connections:
            logger.warning(f"Force closing {len(self._in_use_connections)} connections after timeout")
        
        # Close all connections
        async with self._lock:
            for conn_id, session in self.connections.items():
                if not session.closed:
                    await session.close()
                    
        self.connections.clear()
        self.connection_stats.clear()
        self._in_use_connections.clear()
        
        logger.info("Connection pool stopped")
    
    @asynccontextmanager
    async def get_connection(self, endpoint: str):
        """Get connection from pool with automatic tracking and return"""
        conn_id = None
        session = None
        
        try:
            async with self.connection_semaphore:
                # Short-circuit if shutting down
                if not self._running:
                    raise RuntimeError("Connection pool is shutting down")
                    
                # Try to get existing connection
                try:
                    conn_id = await asyncio.wait_for(
                        self.available_connections.get(), 
                        timeout=0.1
                    )
                    session = self.connections.get(conn_id)
                    
                    # Validate connection
                    if session and not session.closed:
                        # Track as in-use
                        async with self._in_use_lock:
                            self._in_use_connections.add(conn_id)
                            
                        self.connection_stats[conn_id].last_used = time.time()
                        yield session
                        return
                except asyncio.TimeoutError:
                    pass
                
                # Create new connection if still running
                if not self._running:
                    raise RuntimeError("Connection pool is shutting down")
                    
                conn_id, session = await self._create_connection(endpoint)
                
                # Track as in-use
                async with self._in_use_lock:
                    self._in_use_connections.add(conn_id)
                    
                yield session
                
        finally:
            # Return connection to pool
            if conn_id and session and not session.closed:
                # Remove from in-use set
                async with self._in_use_lock:
                    self._in_use_connections.discard(conn_id)
                
                # Only return to pool if still running
                if self._running:
                    try:
                        self.available_connections.put_nowait(conn_id)
                    except asyncio.QueueFull:
                        # Pool is full, close this connection
                        await session.close()
                else:
                    # Shutting down, close the connection
                    await session.close()
    
    async def _create_connection(self, endpoint: str = None) -> Tuple[str, aiohttp.ClientSession]:
        """Create optimized connection with DNS and HTTP/2 awareness"""
        connector_kwargs = {
            "limit": 0,  # No connection limit per host
            "ttl_dns_cache": self.dns_ttl,  # Short TTL for Kubernetes
            "enable_cleanup_closed": True,
            "keepalive_timeout": self.keepalive_timeout,
            "family": socket.AF_INET,  # IPv4 for consistent behavior
        }
        
        # Use aiodns if available for better DNS control
        if AIODNS_AVAILABLE:
            resolver = aiodns.DNSResolver()
            connector_kwargs["resolver"] = resolver
        
        # Configure for HTTP/2 if available and endpoint uses HTTPS
        if self.enable_http2 and endpoint and endpoint.startswith("https://"):
            try:
                ssl_context = ssl.create_default_context()
                ssl_context.set_alpn_protocols(['h2', 'http/1.1'])
                connector_kwargs["ssl"] = ssl_context
                connector_kwargs["force_close"] = False
                logger.debug("Configured connection for HTTP/2 with ALPN")
            except Exception as e:
                logger.warning(f"HTTP/2 configuration failed: {e}, using HTTP/1.1")
        
        connector = aiohttp.TCPConnector(**connector_kwargs)
        
        session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=30, connect=5),
            headers={
                "Connection": "keep-alive",
                "Keep-Alive": f"timeout={self.keepalive_timeout}"
            }
        )
        
        conn_id = f"conn_{int(time.time() * 1000000)}"
        
        async with self._lock:
            self.connections[conn_id] = session
            self.connection_stats[conn_id] = ConnectionStats(
                created_at=time.time(),
                last_used=time.time()
            )
        
        # Make available (but not if we're shutting down)
        if self._running:
            try:
                self.available_connections.put_nowait(conn_id)
            except asyncio.QueueFull:
                pass
                
        return conn_id, session