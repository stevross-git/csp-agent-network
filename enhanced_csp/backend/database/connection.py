# File: backend/database/connection.py
"""
Database Connection and Session Management
=========================================
Async SQLAlchemy setup with connection pooling
"""

import os
import logging
from typing import AsyncGenerator, Optional
import asyncio
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    create_async_engine, 
    AsyncSession, 
    async_sessionmaker,
    AsyncEngine
)
# Use the async-adapted queue pool for SQLAlchemy async engine
# Import the async-compatible queue pool. Older SQLAlchemy versions expose it
# from sqlalchemy.ext.asyncio while newer versions provide it in sqlalchemy.pool
try:
    from sqlalchemy.pool import AsyncAdaptedQueuePool, NullPool
except ImportError:  # pragma: no cover - fallback for older SQLAlchemy
    from sqlalchemy.ext.asyncio import AsyncAdaptedQueuePool
    from sqlalchemy.pool import NullPool
from sqlalchemy.orm import declarative_base
from sqlalchemy import event, text
import redis.asyncio as redis

# Configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql+asyncpg://csp_user:csp_password@localhost:5432/csp_visual_designer"
)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Global variables
engine: Optional[AsyncEngine] = None
SessionLocal: Optional[async_sessionmaker] = None
redis_client: Optional[redis.Redis] = None

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Database connection manager with health monitoring"""
    
    def __init__(self):
        self.engine: Optional[AsyncEngine] = None
        self.session_factory: Optional[async_sessionmaker] = None
        self.redis_client: Optional[redis.Redis] = None
        self._health_check_interval = 30  # seconds
        self._health_check_task: Optional[asyncio.Task] = None
    
    async def initialize(self, 
                        database_url: str = DATABASE_URL,
                        redis_url: str = REDIS_URL,
                        **engine_kwargs):
        """Initialize database connections"""
        try:
            # Database engine configuration
            engine_config = {
                "poolclass": QueuePool,
                "pool_size": 20,
                "max_overflow": 30,
                "pool_pre_ping": True,
                "pool_recycle": 3600,  # 1 hour
                "echo": os.getenv("SQL_DEBUG", "false").lower() == "true",
                **engine_kwargs
            }
            
            # Create async engine
            self.engine = create_async_engine(database_url, **engine_config)
            
            # Create session factory
            self.session_factory = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=True,
                autocommit=False
            )
            
            # Initialize Redis
            self.redis_client = redis.from_url(
                redis_url,
                decode_responses=True,
                max_connections=20,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={},
                health_check_interval=30
            )
            
            # Test connections
            await self._test_database_connection()
            await self._test_redis_connection()
            
            # Start health monitoring
            self._health_check_task = asyncio.create_task(self._health_monitor())
            
            logger.info("✅ Database connections initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize database connections: {e}")
            await self.cleanup()
            raise
    
    async def _test_database_connection(self):
        """Test database connectivity"""
        if not self.engine:
            raise RuntimeError("Database engine not initialized")
        
        async with self.engine.begin() as conn:
            result = await conn.execute(text("SELECT 1"))
            assert result.scalar() == 1
        
        logger.info("✅ Database connection test passed")
    
    async def _test_redis_connection(self):
        """Test Redis connectivity"""
        if not self.redis_client:
            raise RuntimeError("Redis client not initialized")
        
        await self.redis_client.ping()
        logger.info("✅ Redis connection test passed")
    
    async def _health_monitor(self):
        """Background health monitoring task"""
        while True:
            try:
                await asyncio.sleep(self._health_check_interval)
                
                # Check database health
                try:
                    await self._test_database_connection()
                except Exception as e:
                    logger.error(f"Database health check failed: {e}")
                
                # Check Redis health
                try:
                    await self._test_redis_connection()
                except Exception as e:
                    logger.error(f"Redis health check failed: {e}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
    
    async def get_session(self) -> AsyncSession:
        """Get a database session"""
        if not self.session_factory:
            raise RuntimeError("Database not initialized")
        
        return self.session_factory()
    
    async def get_redis(self) -> redis.Redis:
        """Get Redis client"""
        if not self.redis_client:
            raise RuntimeError("Redis not initialized")
        
        return self.redis_client
    
    async def cleanup(self):
        """Clean up database connections"""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Redis connection closed")
        
        if self.engine:
            await self.engine.dispose()
            logger.info("Database engine disposed")
        
        logger.info("✅ Database cleanup completed")

# Global database manager instance
db_manager = DatabaseManager()

# Dependency injection functions for FastAPI
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for database sessions"""
    session = await db_manager.get_session()
    try:
        yield session
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()

async def get_redis_client() -> redis.Redis:
    """FastAPI dependency for Redis client"""
    return await db_manager.get_redis()

# Transaction decorator
@asynccontextmanager
async def database_transaction():
    """Context manager for database transactions"""
    session = await db_manager.get_session()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()

# Database initialization and migration
async def create_tables():
    """Create all database tables"""
    from backend.models.database_models import Base
    
    if not db_manager.engine:
        raise RuntimeError("Database engine not initialized")
    
    async with db_manager.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    logger.info("✅ Database tables created")

async def init_default_data():
    """Initialize default data"""
    try:
        from backend.models.database_models import ComponentType
        
        # Default component types
        default_components = [
            {
                "component_type": "ai_agent",
                "category": "AI",
                "display_name": "AI Agent",
                "description": "Intelligent AI agent for processing and decision making",
                "icon": "robot",
                "color": "#4CAF50",
                "default_properties": {
                    "model": "gpt-4",
                    "temperature": 0.7,
                    "max_tokens": 1000
                },
                "input_ports": [
                    {"name": "input", "type": "text", "required": True}
                ],
                "output_ports": [
                    {"name": "output", "type": "text"}
                ],
                "implementation_class": "ai_extensions.csp_ai_extensions.AIAgent"
            },
            {
                "component_type": "data_processor",
                "category": "Data",
                "display_name": "Data Processor",
                "description": "Process and transform data",
                "icon": "database",
                "color": "#2196F3",
                "default_properties": {
                    "operation": "transform",
                    "format": "json"
                },
                "input_ports": [
                    {"name": "input", "type": "data", "required": True}
                ],
                "output_ports": [
                    {"name": "output", "type": "data"}
                ],
                "implementation_class": "components.data_processing.DataProcessor"
            },
            {
                "component_type": "input_validator",
                "category": "Security",
                "display_name": "Input Validator",
                "description": "Validate and sanitize input data",
                "icon": "shield",
                "color": "#FF9800",
                "default_properties": {
                    "strict_mode": True,
                    "max_length": 1000
                },
                "input_ports": [
                    {"name": "input", "type": "any", "required": True}
                ],
                "output_ports": [
                    {"name": "valid", "type": "any"},
                    {"name": "invalid", "type": "error"}
                ],
                "implementation_class": "components.security.InputValidator"
            },
            {
                "component_type": "metrics_collector",
                "category": "Monitoring",
                "display_name": "Metrics Collector",
                "description": "Collect and report performance metrics",
                "icon": "chart",
                "color": "#9C27B0",
                "default_properties": {
                    "sampling_rate": 1.0,
                    "metrics": ["cpu", "memory", "latency"]
                },
                "input_ports": [
                    {"name": "data", "type": "any", "required": False}
                ],
                "output_ports": [
                    {"name": "metrics", "type": "metrics"}
                ],
                "implementation_class": "components.monitoring.MetricsCollector"
            },
            {
                "component_type": "channel",
                "category": "Communication",
                "display_name": "Channel",
                "description": "Communication channel between processes",
                "icon": "arrow-right",
                "color": "#607D8B",
                "default_properties": {
                    "buffer_size": 100,
                    "blocking": True
                },
                "input_ports": [
                    {"name": "input", "type": "any", "required": True}
                ],
                "output_ports": [
                    {"name": "output", "type": "any"}
                ],
                "implementation_class": "core.advanced_csp_core.Channel"
            }
        ]
        
        async with database_transaction() as session:
            for comp_data in default_components:
                # Check if component already exists
                existing = await session.execute(
                    text("SELECT id FROM component_types WHERE component_type = :comp_type"),
                    {"comp_type": comp_data["component_type"]}
                )
                
                if not existing.scalar_one_or_none():
                    component = ComponentType(**comp_data)
                    session.add(component)
        
        logger.info("✅ Default component types initialized")
        
    except Exception as e:
        logger.error(f"Failed to initialize default data: {e}")
        raise

# Health check functions
async def check_database_health() -> dict:
    """Check database health status"""
    status = {
        "database": {"status": "unknown", "details": {}},
        "redis": {"status": "unknown", "details": {}}
    }
    
    # Check database
    try:
        if db_manager.engine:
            async with db_manager.engine.begin() as conn:
                result = await conn.execute(text("SELECT version()"))
                version = result.scalar()
                
                # Get connection pool status
                pool = db_manager.engine.pool
                status["database"] = {
                    "status": "healthy",
                    "details": {
                        "version": version,
                        "pool_size": pool.size(),
                        "checked_in": pool.checkedin(),
                        "checked_out": pool.checkedout(),
                        "overflow": pool.overflow(),
                        "invalid": pool.invalid()
                    }
                }
        else:
            status["database"]["status"] = "not_initialized"
    except Exception as e:
        status["database"] = {
            "status": "error",
            "details": {"error": str(e)}
        }
    
    # Check Redis
    try:
        if db_manager.redis_client:
            info = await db_manager.redis_client.info()
            status["redis"] = {
                "status": "healthy",
                "details": {
                    "version": info.get("redis_version"),
                    "connected_clients": info.get("connected_clients"),
                    "used_memory": info.get("used_memory_human"),
                    "uptime": info.get("uptime_in_seconds")
                }
            }
        else:
            status["redis"]["status"] = "not_initialized"
    except Exception as e:
        status["redis"] = {
            "status": "error",
            "details": {"error": str(e)}
        }
    
    return status

# Cache utilities
class CacheManager:
    """Redis-based cache manager"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.default_ttl = 3600  # 1 hour
    
    async def get(self, key: str, default=None):
        """Get value from cache"""
        try:
            value = await self.redis.get(key)
            if value is not None:
                import json
                return json.loads(value)
            return default
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return default
    
    async def set(self, key: str, value, ttl: Optional[int] = None):
        """Set value in cache"""
        try:
            import json
            ttl = ttl or self.default_ttl
            await self.redis.setex(key, ttl, json.dumps(value, default=str))
            return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    async def delete(self, key: str):
        """Delete key from cache"""
        try:
            await self.redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            return await self.redis.exists(key) > 0
        except Exception as e:
            logger.error(f"Cache exists error: {e}")
            return False
    
    async def get_or_set(self, key: str, factory_func, ttl: Optional[int] = None):
        """Get from cache or set using factory function"""
        value = await self.get(key)
        if value is not None:
            return value
        
        # Generate value using factory function
        if asyncio.iscoroutinefunction(factory_func):
            value = await factory_func()
        else:
            value = factory_func()
        
        await self.set(key, value, ttl)
        return value

async def get_cache_manager() -> CacheManager:
    """FastAPI dependency for cache manager"""
    redis_client = await get_redis_client()
    return CacheManager(redis_client)

# Startup and shutdown events
async def startup_database():
    """Startup database connections"""
    await db_manager.initialize()
    await create_tables()
    await init_default_data()

async def shutdown_database():
    """Shutdown database connections"""
    await db_manager.cleanup()
