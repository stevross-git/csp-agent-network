#!/usr/bin/env python3
"""
Enhanced CSP System - Main Application
======================================

Revolutionary AI-to-AI Communication Platform using Communicating Sequential Processes (CSP)
with quantum-inspired protocols, consciousness integration, and emergent behavior detection.

Features:
- FastAPI web server with WebSocket support
- Advanced CSP engine with quantum communication
- AI-powered protocol synthesis
- Real-time monitoring and metrics
- Distributed agent coordination
- Self-healing and optimization
- Visual development tools
- Production deployment support
"""

import time
import os
import sys
import asyncio
import logging
import json
import uuid
import sqlite3
import glob
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from contextlib import asynccontextmanager

# FastAPI and web dependencies
from fastapi import FastAPI, WebSocket, HTTPException, Depends, BackgroundTasks, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn

# Database and storage
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Optional dependencies with graceful fallbacks
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False
    logging.warning("Redis library not available; Redis features disabled")

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    Counter = Histogram = Gauge = generate_latest = None
    CONTENT_TYPE_LATEST = "text/plain"
    PROMETHEUS_AVAILABLE = False
    logging.warning("prometheus_client not available; metrics disabled")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available; system metrics disabled")

try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    aiofiles = None
    AIOFILES_AVAILABLE = False
    logging.warning("aiofiles not available; some features disabled")

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None
    AIOHTTP_AVAILABLE = False
    logging.warning("aiohttp not available; some features disabled")

try:
    import uvloop
    UVLOOP_AVAILABLE = True
except ImportError:
    uvloop = None
    UVLOOP_AVAILABLE = False
    logging.warning("uvloop not available; falling back to asyncio loop")

# Configuration and utilities
import yaml
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

# Enhanced CSP Core Components - with graceful fallbacks
CSP_CORE_AVAILABLE = False
AI_EXTENSIONS_AVAILABLE = False
RUNTIME_AVAILABLE = False
DEPLOYMENT_AVAILABLE = False
DEV_TOOLS_AVAILABLE = False
MONITORING_AVAILABLE = False

try:
    from core.advanced_csp_core import (
        AdvancedCSPEngine, Process, AtomicProcess, CompositeProcess,
        CompositionOperator, ChannelType, Event, ProcessSignature,
        ProcessContext, Channel, ProcessMatcher, ProtocolEvolution
    )
    CSP_CORE_AVAILABLE = True
    logging.info("✅ CSP Core components loaded")
except ImportError as e:
    logging.warning(f"CSP Core components not available: {e}")
    
    # Create fallback classes
    class Process:
        def __init__(self, name):
            self.name = name
            self.state = "initialized"
    
    class AtomicProcess(Process):
        def __init__(self, name, signature=None):
            super().__init__(name)
            self.signature = signature
    
    class CompositeProcess(Process):
        def __init__(self, name, processes=None):
            super().__init__(name)
            self.processes = processes or []
    
    class ProcessSignature:
        def __init__(self, inputs=None, outputs=None):
            self.inputs = inputs or []
            self.outputs = outputs or []
    
    class AdvancedCSPEngine:
        def __init__(self):
            self.processes = {}
        
        async def start_process(self, process):
            self.processes[process.name] = process
            return process
        
        async def stop_process(self, process):
            if process.name in self.processes:
                del self.processes[process.name]

try:
    from ai_extensions.csp_ai_extensions import (
        AdvancedCSPEngineWithAI, ProtocolSpec, ProtocolTemplate,
        EmergentBehaviorDetector, CausalityTracker
    )
    AI_EXTENSIONS_AVAILABLE = True
    logging.info("✅ AI Extensions loaded")
except ImportError as e:
    logging.warning(f"AI Extensions not available: {e}")
    
    class AdvancedCSPEngineWithAI(AdvancedCSPEngine):
        def __init__(self):
            super().__init__()
            self.ai_enabled = False

try:
    from ai_integration.csp_ai_integration import (
        AIAgent, LLMCapability, CollaborativeAIProcess,
        AdvancedAICSPDemo
    )
    AI_INTEGRATION_AVAILABLE = True
    logging.info("✅ AI Integration loaded")
except ImportError as e:
    logging.warning(f"AI Integration not available: {e}")
    
    class AIAgent:
        def __init__(self, name, capabilities=None):
            self.name = name
            self.capabilities = capabilities or []
    
    class LLMCapability:
        def __init__(self, model_name, specialized_domain=None):
            self.model_name = model_name
            self.specialized_domain = specialized_domain
    
    class MultiAgentReasoningCoordinator:
        def __init__(self, agents):
            self.agents = agents

try:
    from runtime.csp_runtime_environment import (
        CSPRuntimeOrchestrator, RuntimeConfig, ExecutionModel, 
        SchedulingPolicy
    )
    RUNTIME_AVAILABLE = True
    logging.info("✅ Runtime Environment loaded")
except ImportError as e:
    logging.warning(f"Runtime Environment not available: {e}")
    
    class RuntimeConfig:
        def __init__(self, **kwargs):
            self.execution_model = kwargs.get('execution_model', 'SINGLE_THREADED')
            self.scheduling_policy = kwargs.get('scheduling_policy', 'FIFO')
            self.max_workers = kwargs.get('max_workers', 4)
            self.memory_limit_gb = kwargs.get('memory_limit_gb', 8.0)
            self.enable_monitoring = kwargs.get('enable_monitoring', True)
            self.enable_optimization = kwargs.get('enable_optimization', True)
            self.debug_mode = kwargs.get('debug_mode', False)
    
    class ExecutionModel:
        SINGLE_THREADED = "SINGLE_THREADED"
        MULTI_THREADED = "MULTI_THREADED"
        ASYNC_CONCURRENT = "ASYNC_CONCURRENT"
    
    class SchedulingPolicy:
        FIFO = "FIFO"
        PRIORITY = "PRIORITY"
        ADAPTIVE = "ADAPTIVE"
    
    class CSPRuntimeOrchestrator:
        def __init__(self, config):
            self.config = config
            self.running = False
        
        async def start(self):
            self.running = True
            logging.info("Runtime orchestrator started (fallback mode)")
        
        async def stop(self):
            self.running = False
            logging.info("Runtime orchestrator stopped")

try:
    from deployment.csp_deployment_system import (
        CSPDeploymentOrchestrator, DeploymentConfig, DeploymentTarget,
        ScalingStrategy, HealthCheckConfig
    )
    DEPLOYMENT_AVAILABLE = True
    logging.info("✅ Deployment System loaded")
except ImportError as e:
    logging.warning(f"Deployment System not available: {e}")
    
    class CSPDeploymentOrchestrator:
        def __init__(self):
            pass

try:
    from dev_tools.csp_dev_tools import (
        CSPDevelopmentTools, CSPVisualDesigner, CSPDebugger,
        CSPCodeGenerator, CSPTestFramework
    )
    DEV_TOOLS_AVAILABLE = True
    logging.info("✅ Development Tools loaded")
except ImportError as e:
    logging.warning(f"Development Tools not available: {e}")
    
    class CSPDevelopmentTools:
        def __init__(self):
            self.visual_designer = CSPVisualDesigner()
            self.code_generator = CSPCodeGenerator()
    
    class CSPVisualDesigner:
        async def get_state(self):
            return {"status": "available", "components": []}
    
    class CSPCodeGenerator:
        async def generate_from_design(self, design_data):
            return "# Generated CSP code\nprint('Hello, CSP!')"

try:
    from monitoring.csp_monitoring import (
        CSPMonitor, MetricsCollector, PerformanceAnalyzer,
        AlertManager, SystemHealthChecker
    )
    MONITORING_AVAILABLE = True
    logging.info("✅ Monitoring System loaded")
except ImportError as e:
    logging.warning(f"Monitoring System not available: {e}")
    
    class CSPMonitor:
        def __init__(self):
            pass
        
        async def start(self):
            pass
        
        async def stop(self):
            pass

# Load environment variables
load_dotenv()

# Configure logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=getattr(logging, os.getenv('CSP_LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/enhanced_csp.log'),
        logging.FileHandler('logs/enhanced_csp_debug.log')
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================

class DatabaseConfig(BaseModel):
    """Database configuration"""
    url: str = "sqlite+aiosqlite:///./data/enhanced_csp.db"
    echo: bool = False
    pool_size: int = 20
    max_overflow: int = 30

class RedisConfig(BaseModel):
    """Redis configuration"""
    url: str = "redis://localhost:6379/0"
    max_connections: int = 20
    retry_on_timeout: bool = True

class MonitoringConfig(BaseModel):
    """Monitoring configuration"""
    enable_prometheus: bool = True
    enable_tracing: bool = True
    metrics_port: int = 9090
    health_check_interval: int = 30
    alert_threshold_cpu: float = 80.0
    alert_threshold_memory: float = 85.0

class AIConfig(BaseModel):
    """AI configuration"""
    enable_llm_integration: bool = True
    default_model: str = "gpt-4"
    max_agents_per_collaboration: int = 10
    reasoning_timeout: int = 300
    protocol_synthesis_enabled: bool = True

class RuntimeSettings(BaseModel):
    """Runtime configuration"""
    execution_model: str = "MULTI_THREADED"
    max_workers: int = os.cpu_count()
    memory_limit_gb: float = 8.0
    enable_optimization: bool = True
    enable_debugging: bool = False

class CSPConfig(BaseModel):
    """Enhanced CSP System Configuration"""
    
    # Core settings
    app_name: str = "Enhanced CSP System"
    version: str = "2.0.0"
    debug: bool = False
    environment: str = "development"
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    
    # Component configurations
    database: DatabaseConfig = DatabaseConfig()
    redis: RedisConfig = RedisConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    ai: AIConfig = AIConfig()
    runtime: RuntimeSettings = RuntimeSettings()
    
    # Security settings
    secret_key: str = os.getenv('CSP_SECRET_KEY', 'dev-secret-key')
    api_key_header: str = "X-CSP-API-Key"
    enable_auth: bool = True
    
    @field_validator('port')
    def validate_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError('Port must be between 1 and 65535')
        return v

# Load configuration
def load_config() -> CSPConfig:
    """Load configuration from environment and files"""
    config_path = os.getenv('CSP_CONFIG_PATH', 'config/system.yaml')
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        return CSPConfig(**config_data)
    else:
        logger.info(f"Config file not found at {config_path}, using defaults")
        return CSPConfig()

config = load_config()

# ============================================================================
# GLOBAL SYSTEM STATE
# ============================================================================

class SystemState:
    """Global system state manager"""
    
    def __init__(self):
        self.startup_time = datetime.now()
        self.csp_engine: Optional[AdvancedCSPEngine] = None
        self.ai_engine: Optional[AdvancedCSPEngineWithAI] = None
        self.runtime_orchestrator: Optional[CSPRuntimeOrchestrator] = None
        self.deployment_orchestrator: Optional[CSPDeploymentOrchestrator] = None
        self.dev_tools: Optional[CSPDevelopmentTools] = None
        self.monitor: Optional[CSPMonitor] = None
        self.redis_client: Optional[redis.Redis] = None
        self.db_engine = None
        self.active_processes: Dict[str, Process] = {}
        self.active_websockets: List[WebSocket] = []
        self.system_metrics: Dict[str, Any] = {}

system_state = SystemState()

# ============================================================================
# METRICS AND MONITORING
# ============================================================================

# Prometheus metrics
if PROMETHEUS_AVAILABLE:
    REQUEST_COUNT = Counter('csp_requests_total', 'Total requests', ['method', 'endpoint'])
    REQUEST_DURATION = Histogram('csp_request_duration_seconds', 'Request duration')
    ACTIVE_PROCESSES = Gauge('csp_active_processes', 'Number of active CSP processes')
    SYSTEM_HEALTH = Gauge('csp_system_health', 'System health score (0-1)')
    WEBSOCKET_CONNECTIONS = Gauge('csp_websocket_connections', 'Active WebSocket connections')
else:
    REQUEST_COUNT = REQUEST_DURATION = ACTIVE_PROCESSES = SYSTEM_HEALTH = WEBSOCKET_CONNECTIONS = None

class MetricsMiddleware:
    """Middleware for collecting metrics"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            start_time = time.time()
            
            async def send_wrapper(message):
                if message["type"] == "http.response.start" and REQUEST_COUNT and REQUEST_DURATION:
                    duration = time.time() - start_time
                    method = scope["method"]
                    path = scope["path"]
                    REQUEST_COUNT.labels(method=method, endpoint=path).inc()
                    REQUEST_DURATION.observe(duration)
                await send(message)
            
            await self.app(scope, receive, send_wrapper)
        else:
            await self.app(scope, receive, send)

# ============================================================================
# DATABASE SETUP
# ============================================================================

def run_migrations():
    """Simple database setup with SQLite"""
    try:
        os.makedirs('data', exist_ok=True)
        
        # Check if SQLite database exists
        db_path = Path("data/csp_system.db")
        if not db_path.exists():
            logger.info("💾 Setting up SQLite database...")
            
            # Create a simple SQLite connection to initialize the database
            conn = sqlite3.connect(str(db_path))
            
            # Create basic tables
            conn.execute('''CREATE TABLE IF NOT EXISTS system_info (
                id INTEGER PRIMARY KEY,
                key TEXT UNIQUE,
                value TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )''')
            
            conn.execute('''CREATE TABLE IF NOT EXISTS processes (
                id INTEGER PRIMARY KEY,
                process_id TEXT UNIQUE,
                name TEXT,
                type TEXT,
                status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )''')
            
            conn.execute('''CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                username TEXT UNIQUE,
                email TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )''')
            
            # Insert initial data
            conn.execute("INSERT OR IGNORE INTO system_info (key, value) VALUES ('initialized', 'true')")
            conn.execute("INSERT OR IGNORE INTO system_info (key, value) VALUES ('version', '2.0.0')")
            conn.commit()
            conn.close()
            
            logger.info(f"✅ SQLite database created at: {db_path}")
        else:
            logger.info(f"✅ SQLite database found at: {db_path}")
            
        logger.info("✅ SQLite migration completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Database setup error: {e}")

async def setup_database():
    """Initialize database connection and run migrations"""
    try:
        logger.info("🗄️ Starting database migrations...")
        
        # Check for PostgreSQL
        try:
            import psycopg2
            logger.info("✅ PostgreSQL driver available")
        except ImportError:
            logger.warning("⚠️ PostgreSQL driver not available, using SQLite only")
        
        # Always run SQLite setup as fallback
        run_migrations()
        
        # Create async engine for SQLite
        if not system_state.db_engine:
            system_state.db_engine = create_async_engine(
                config.database.url,
                echo=config.database.echo
            )
        
        logger.info("✅ Database setup completed")
        
    except Exception as e:
        logger.error(f"❌ Database setup failed: {e}")
        # Continue without database if needed

async def setup_redis():
    """Initialize Redis connection"""
    if not REDIS_AVAILABLE:
        logger.info("Redis not available, skipping Redis setup")
        return
        
    try:
        system_state.redis_client = redis.from_url(
            config.redis.url,
            max_connections=config.redis.max_connections,
            retry_on_timeout=config.redis.retry_on_timeout
        )
        
        # Test connection
        await system_state.redis_client.ping()
        logger.info("✅ Redis connection established")
    except Exception as e:
        logger.warning(f"⚠️ Redis setup failed: {e}")
        # Continue without Redis if it fails
        system_state.redis_client = None

# ============================================================================
# CSP SYSTEM INITIALIZATION
# ============================================================================

async def initialize_csp_system():
    """Initialize all CSP system components"""
    try:
        logger.info("🚀 Initializing Enhanced CSP System...")
        
        # 1. Initialize core CSP engine
        if CSP_CORE_AVAILABLE:
            system_state.csp_engine = AdvancedCSPEngine()
            logger.info("✅ Core CSP engine initialized")
        else:
            system_state.csp_engine = AdvancedCSPEngine()  # Use fallback
            logger.info("✅ Core CSP engine initialized (fallback mode)")
        
        # 2. Initialize AI-enhanced engine
        if config.ai.enable_llm_integration and AI_EXTENSIONS_AVAILABLE:
            system_state.ai_engine = AdvancedCSPEngineWithAI()
            logger.info("✅ AI-enhanced CSP engine initialized")
        elif config.ai.enable_llm_integration:
            system_state.ai_engine = AdvancedCSPEngineWithAI()  # Use fallback
            logger.info("✅ AI-enhanced CSP engine initialized (fallback mode)")
        
        # 3. Initialize runtime orchestrator
        if RUNTIME_AVAILABLE:
            runtime_config = RuntimeSettings(  # Use the fallback RuntimeConfig class
                execution_model=getattr(ExecutionModel, 'MULTI_THREADED', 'MULTI_THREADED'),
                scheduling_policy=getattr(SchedulingPolicy, 'ADAPTIVE', 'ADAPTIVE'),
                max_workers=config.runtime.max_workers,
                memory_limit_gb=config.runtime.memory_limit_gb,
                enable_monitoring=True,
                enable_optimization=config.runtime.enable_optimization,
                debug_mode=config.runtime.enable_debugging
            )
            system_state.runtime_orchestrator = CSPRuntimeOrchestrator(runtime_config)
            await system_state.runtime_orchestrator.start()
            logger.info("✅ Runtime orchestrator initialized")
        else:
            # Use fallback runtime orchestrator
            runtime_config = RuntimeSettings(
                execution_model="MULTI_THREADED",
                max_workers=config.runtime.max_workers
            )
            system_state.runtime_orchestrator = CSPRuntimeOrchestrator(runtime_config)
            await system_state.runtime_orchestrator.start()
            logger.info("✅ Runtime orchestrator initialized (fallback mode)")
        
        # 4. Initialize deployment orchestrator
        if DEPLOYMENT_AVAILABLE:
            system_state.deployment_orchestrator = CSPDeploymentOrchestrator()
            logger.info("✅ Deployment orchestrator initialized")
        
        # 5. Initialize development tools
        if DEV_TOOLS_AVAILABLE:
            system_state.dev_tools = CSPDevelopmentTools()
            logger.info("✅ Development tools initialized")
        else:
            system_state.dev_tools = CSPDevelopmentTools()  # Use fallback
            logger.info("✅ Development tools initialized (fallback mode)")
        
        # 6. Initialize monitoring
        if MONITORING_AVAILABLE:
            system_state.monitor = CSPMonitor()
            await system_state.monitor.start()
            logger.info("✅ Monitoring system initialized")
        
        logger.info("✅ Enhanced CSP System initialization completed")
        
    except Exception as e:
        logger.error(f"❌ CSP system initialization failed: {e}")
        # Continue anyway - basic functionality should still work
        logger.info("⚠️ Continuing with reduced functionality")

# ============================================================================
# DASHBOARD COMPONENTS
# ============================================================================

def create_dashboard_app():
    """Create dashboard sub-application"""
    dashboard = FastAPI(title="CSP Dashboard", version="2.0.0")
    
    @dashboard.get("/")
    async def dashboard_home():
        """Dashboard home page"""
        return {
            "message": "CSP Dashboard", 
            "status": "running",
            "timestamp": datetime.now().isoformat(),
            "features": {
                "csp_core": CSP_CORE_AVAILABLE,
                "ai_extensions": AI_EXTENSIONS_AVAILABLE,
                "runtime": RUNTIME_AVAILABLE,
                "monitoring": MONITORING_AVAILABLE
            }
        }
    
    @dashboard.get("/health")
    async def dashboard_health():
        """Dashboard health check"""
        return {
            "status": "healthy", 
            "service": "csp-dashboard",
            "components": {
                "database": system_state.db_engine is not None,
                "redis": system_state.redis_client is not None,
                "csp_engine": system_state.csp_engine is not None
            }
        }
    
    @dashboard.get("/metrics")
    async def dashboard_metrics():
        """Dashboard metrics"""
        return {
            "active_processes": len(system_state.active_processes),
            "websocket_connections": len(system_state.active_websockets),
            "uptime": str(datetime.now() - system_state.startup_time),
            "system": {
                "cpu_percent": psutil.cpu_percent() if PSUTIL_AVAILABLE else 0.0,
                "memory_percent": psutil.virtual_memory().percent if PSUTIL_AVAILABLE else 0.0
            }
        }
    
    return dashboard

# ============================================================================
# FASTAPI APPLICATION SETUP
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("🚀 Starting Enhanced CSP System...")
    
    try:
        await setup_database()
        await setup_redis()
        await initialize_csp_system()
        
        # Update system health
        if SYSTEM_HEALTH:
            SYSTEM_HEALTH.set(1.0)
        logger.info("✅ System startup completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        if SYSTEM_HEALTH:
            SYSTEM_HEALTH.set(0.5)  # Partial functionality
        # Don't raise - allow server to start with reduced functionality
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Enhanced CSP System...")
    
    try:
        # Close WebSocket connections
        for ws in list(system_state.active_websockets):
            try:
                await ws.close()
            except:
                pass
        
        # Shutdown components
        if system_state.runtime_orchestrator:
            await system_state.runtime_orchestrator.stop()
        
        if system_state.monitor:
            await system_state.monitor.stop()
        
        if system_state.redis_client:
            await system_state.redis_client.close()
        
        if system_state.db_engine:
            await system_state.db_engine.dispose()
        
        logger.info("✅ System shutdown completed")
        
    except Exception as e:
        logger.error(f"❌ Shutdown error: {e}")

# Create FastAPI application
app = FastAPI(
    title=config.app_name,
    description="Revolutionary AI-to-AI Communication Platform using CSP",
    version=config.version,
    debug=config.debug,
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
if PROMETHEUS_AVAILABLE:
    app.add_middleware(MetricsMiddleware)

# Security
security = HTTPBearer(auto_error=False)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Authentication dependency"""
    if not config.enable_auth:
        return {"user": "anonymous", "authenticated": False}
    
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    # Validate API key (simplified for demo)
    if credentials.credentials == config.secret_key:
        return {"user": "api_user", "authenticated": True}
    
    raise HTTPException(status_code=401, detail="Invalid authentication")

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "app": config.app_name,
        "version": config.version,
        "status": "running",
        "uptime": str(datetime.now() - system_state.startup_time),
        "features": {
            "csp_engine": system_state.csp_engine is not None,
            "ai_integration": system_state.ai_engine is not None,
            "runtime_orchestrator": system_state.runtime_orchestrator is not None,
            "deployment": system_state.deployment_orchestrator is not None,
            "development_tools": system_state.dev_tools is not None,
            "monitoring": system_state.monitor is not None
        },
        "components": {
            "csp_core": CSP_CORE_AVAILABLE,
            "ai_extensions": AI_EXTENSIONS_AVAILABLE,
            "runtime": RUNTIME_AVAILABLE,
            "deployment": DEPLOYMENT_AVAILABLE,
            "dev_tools": DEV_TOOLS_AVAILABLE,
            "monitoring": MONITORING_AVAILABLE,
            "redis": REDIS_AVAILABLE,
            "prometheus": PROMETHEUS_AVAILABLE,
            "psutil": PSUTIL_AVAILABLE
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_data = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system": {
            "cpu_percent": psutil.cpu_percent() if PSUTIL_AVAILABLE else 0.0,
            "memory_percent": psutil.virtual_memory().percent if PSUTIL_AVAILABLE else 0.0,
            "disk_percent": psutil.disk_usage('/').percent if PSUTIL_AVAILABLE else 0.0
        },
        "components": {
            "database": system_state.db_engine is not None,
            "redis": system_state.redis_client is not None,
            "csp_engine": system_state.csp_engine is not None,
            "ai_engine": system_state.ai_engine is not None
        },
        "active_processes": len(system_state.active_processes),
        "websocket_connections": len(system_state.active_websockets)
    }
    
    # Update metrics
    if ACTIVE_PROCESSES:
        ACTIVE_PROCESSES.set(len(system_state.active_processes))
    if WEBSOCKET_CONNECTIONS:
        WEBSOCKET_CONNECTIONS.set(len(system_state.active_websockets))
    
    return health_data

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    if generate_latest:
        return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
    return PlainTextResponse("metrics unavailable", media_type=CONTENT_TYPE_LATEST)

# CSP Process Management
@app.post("/api/processes")
async def create_process(
    process_data: dict,
    user: dict = Depends(get_current_user)
):
    """Create a new CSP process"""
    try:
        if not system_state.csp_engine:
            raise HTTPException(status_code=503, detail="CSP engine not available")
        
        # Create process based on type
        process_type = process_data.get("type", "atomic")
        process_id = str(uuid.uuid4())
        
        if process_type == "atomic":
            process = AtomicProcess(
                name=process_data.get("name", f"process_{process_id}"),
                signature=ProcessSignature(
                    inputs=process_data.get("inputs", []),
                    outputs=process_data.get("outputs", [])
                )
            )
        else:
            process = CompositeProcess(
                name=process_data.get("name", f"composite_{process_id}"),
                processes=[]
            )
        
        # Register process
        await system_state.csp_engine.start_process(process)
        system_state.active_processes[process_id] = process
        
        return {
            "process_id": process_id,
            "name": process.name,
            "type": process_type,
            "status": "created"
        }
        
    except Exception as e:
        logger.error(f"Process creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/processes")
async def list_processes(user: dict = Depends(get_current_user)):
    """List all active processes"""
    processes = []
    for process_id, process in system_state.active_processes.items():
        processes.append({
            "process_id": process_id,
            "name": process.name,
            "type": type(process).__name__,
            "status": getattr(process, 'state', 'unknown')
        })
    
    return {"processes": processes, "count": len(processes)}

@app.delete("/api/processes/{process_id}")
async def stop_process(
    process_id: str,
    user: dict = Depends(get_current_user)
):
    """Stop a specific process"""
    if process_id not in system_state.active_processes:
        raise HTTPException(status_code=404, detail="Process not found")
    
    try:
        process = system_state.active_processes[process_id]
        await system_state.csp_engine.stop_process(process)
        del system_state.active_processes[process_id]
        
        return {"process_id": process_id, "status": "stopped"}
    except Exception as e:
        logger.error(f"Process stop failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# AI Integration Endpoints
@app.post("/api/ai/collaborate")
async def create_ai_collaboration(
    collaboration_data: dict,
    user: dict = Depends(get_current_user)
):
    """Create an AI collaboration session"""
    if not system_state.ai_engine:
        raise HTTPException(status_code=503, detail="AI engine not available")
    
    try:
        # Create AI agents
        agents = []
        for agent_spec in collaboration_data.get("agents", []):
            capability = LLMCapability(
                model_name=agent_spec.get("model", config.ai.default_model),
                specialized_domain=agent_spec.get("domain")
            )
            agent = AIAgent(
                name=agent_spec.get("name"),
                capabilities=[capability]
            )
            agents.append(agent)
        
        # Create collaborative process
        coordinator = MultiAgentReasoningCoordinator(agents)
        collaboration_id = str(uuid.uuid4())
        
        # Store collaboration
        system_state.active_processes[collaboration_id] = coordinator
        
        return {
            "collaboration_id": collaboration_id,
            "agents": [{"name": a.name, "capabilities": len(a.capabilities)} for a in agents],
            "status": "created"
        }
        
    except Exception as e:
        logger.error(f"AI collaboration creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Development Tools Endpoints
@app.get("/api/dev/visual-designer")
async def get_visual_designer_state(user: dict = Depends(get_current_user)):
    """Get visual designer state"""
    if not system_state.dev_tools:
        raise HTTPException(status_code=503, detail="Development tools not available")
    
    designer_state = await system_state.dev_tools.visual_designer.get_state()
    return designer_state

@app.post("/api/dev/generate-code")
async def generate_code_from_design(
    design_data: dict,
    user: dict = Depends(get_current_user)
):
    """Generate CSP code from visual design"""
    if not system_state.dev_tools:
        raise HTTPException(status_code=503, detail="Development tools not available")
    
    try:
        code = await system_state.dev_tools.code_generator.generate_from_design(design_data)
        return {"generated_code": code, "status": "success"}
    except Exception as e:
        logger.error(f"Code generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# System Information Endpoints
@app.get("/api/system/info")
async def get_system_info():
    """Get detailed system information"""
    return {
        "version": config.version,
        "environment": config.environment,
        "startup_time": system_state.startup_time.isoformat(),
        "uptime": str(datetime.now() - system_state.startup_time),
        "features": {
            "csp_core": CSP_CORE_AVAILABLE,
            "ai_extensions": AI_EXTENSIONS_AVAILABLE,
            "runtime": RUNTIME_AVAILABLE,
            "deployment": DEPLOYMENT_AVAILABLE,
            "dev_tools": DEV_TOOLS_AVAILABLE,
            "monitoring": MONITORING_AVAILABLE,
        },
        "dependencies": {
            "redis": REDIS_AVAILABLE,
            "prometheus": PROMETHEUS_AVAILABLE,
            "psutil": PSUTIL_AVAILABLE,
            "aiofiles": AIOFILES_AVAILABLE,
            "aiohttp": AIOHTTP_AVAILABLE,
            "uvloop": UVLOOP_AVAILABLE
        }
    }

@app.get("/api/system/status")
async def get_system_status():
    """Get current system status"""
    return {
        "status": "operational",
        "active_processes": len(system_state.active_processes),
        "websocket_connections": len(system_state.active_websockets),
        "components": {
            "csp_engine": system_state.csp_engine is not None,
            "ai_engine": system_state.ai_engine is not None,
            "runtime_orchestrator": system_state.runtime_orchestrator is not None,
            "database": system_state.db_engine is not None,
            "redis": system_state.redis_client is not None
        }
    }

# ============================================================================
# WEBSOCKET ENDPOINTS
# ============================================================================

@app.websocket("/ws/system")
async def websocket_system_monitor(websocket: WebSocket):
    """WebSocket for real-time system monitoring"""
    await websocket.accept()
    system_state.active_websockets.append(websocket)
    
    try:
        while True:
            # Send system status
            status = {
                "timestamp": datetime.now().isoformat(),
                "active_processes": len(system_state.active_processes),
                "system_health": {
                    "cpu": psutil.cpu_percent() if PSUTIL_AVAILABLE else 0.0,
                    "memory": psutil.virtual_memory().percent if PSUTIL_AVAILABLE else 0.0,
                    "disk": psutil.disk_usage('/').percent if PSUTIL_AVAILABLE else 0.0
                },
                "websocket_connections": len(system_state.active_websockets),
                "components": {
                    "csp_engine": system_state.csp_engine is not None,
                    "ai_engine": system_state.ai_engine is not None,
                    "database": system_state.db_engine is not None,
                    "redis": system_state.redis_client is not None
                }
            }
            
            await websocket.send_json(status)
            await asyncio.sleep(5)  # Update every 5 seconds
            
    except Exception as e:
        logger.error(f"System WebSocket error: {e}")
    finally:
        if websocket in system_state.active_websockets:
            system_state.active_websockets.remove(websocket)

@app.websocket("/ws/processes/{process_id}")
async def websocket_process_monitor(websocket: WebSocket, process_id: str):
    """WebSocket for monitoring a specific process"""
    await websocket.accept()
    system_state.active_websockets.append(websocket)
    
    if process_id not in system_state.active_processes:
        await websocket.send_json({"error": "Process not found"})
        await websocket.close()
        return
    
    try:
        process = system_state.active_processes[process_id]
        
        while True:
            # Send process status
            status = {
                "timestamp": datetime.now().isoformat(),
                "process_id": process_id,
                "name": process.name,
                "type": type(process).__name__,
                "state": getattr(process, 'state', 'unknown'),
                "performance": getattr(process, 'performance_metrics', {})
            }
            
            await websocket.send_json(status)
            await asyncio.sleep(1)  # Update every second
            
    except Exception as e:
        logger.error(f"Process WebSocket error: {e}")
    finally:
        if websocket in system_state.active_websockets:
            system_state.active_websockets.remove(websocket)

# ============================================================================
# FRONTEND PAGES ROUTES - COMPLETE VERSION
# ============================================================================

# Mount frontend static files and assets
frontend_dir = Path("frontend")
if frontend_dir.exists():
    app.mount("/frontend", StaticFiles(directory=str(frontend_dir)), name="frontend")

# Helper function to serve HTML pages with fallback
def serve_html_page(page_name: str, page_title: str):
    """Serve HTML page with fallback if not found"""
    page_file = Path(f"frontend/pages/{page_name}.html")
    
    if page_file.exists():
        return FileResponse(str(page_file))
    else:
        # Create a fallback page with navigation
        return HTMLResponse(f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{page_title} - CSP System</title>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    margin: 0;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    min-height: 100vh;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    padding: 2rem;
                }}
                .container {{
                    background: rgba(255,255,255,0.1);
                    backdrop-filter: blur(10px);
                    border-radius: 15px;
                    padding: 2rem;
                    text-align: center;
                    max-width: 600px;
                    border: 1px solid rgba(255,255,255,0.2);
                }}
                h1 {{ color: #fff; margin-bottom: 1rem; }}
                .status {{ background: rgba(255,0,0,0.2); padding: 1rem; border-radius: 8px; margin: 1rem 0; }}
                .nav-links {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 2rem; }}
                .nav-link {{
                    background: rgba(255,255,255,0.1);
                    padding: 1rem;
                    border-radius: 8px;
                    text-decoration: none;
                    color: white;
                    border: 1px solid rgba(255,255,255,0.2);
                    transition: all 0.3s ease;
                }}
                .nav-link:hover {{
                    background: rgba(255,255,255,0.2);
                    transform: translateY(-2px);
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{page_title}</h1>
                <div class="status">
                    ⚠️ Page not found: <code>frontend/pages/{page_name}.html</code>
                    <br><small>This route is ready - just add the HTML file!</small>
                </div>
                
                <div class="nav-links">
                    <a href="/dashboard" class="nav-link">🏠 Dashboard</a>
                    <a href="/admin" class="nav-link">👨‍💼 Admin</a>
                    <a href="/designer" class="nav-link">🎨 Designer</a>
                    <a href="/monitoring" class="nav-link">📊 Monitoring</a>
                    <a href="/ai-agents" class="nav-link">🤖 AI Agents</a>
                    <a href="/security" class="nav-link">🔐 Security</a>
                    <a href="/pages" class="nav-link">📄 All Pages</a>
                    <a href="/docs" class="nav-link">📚 API Docs</a>
                </div>
                
                <p style="margin-top: 2rem; opacity: 0.8;">
                    <small>Enhanced CSP System - Ready for {page_name}.html</small>
                </p>
            </div>
        </body>
        </html>
        """)

def get_page_icon(page_name: str) -> str:
    """Get icon for page based on name"""
    icon_map = {
        'dashboard': '🏠', 'admin': '👨‍💼', 'designer': '🎨', 'developer-tools': '🔧',
        'ai-agents': '🤖', 'ai-models': '🧠', 'processes': '⚙️', 'channels': '📡',
        'quantum': '⚛️', 'blockchain': '🔗', 'neural': '🧬', 'neural-mesh': '🕸️',
        'monitoring': '📊', 'analytics': '📈', 'metrics': '📊', 'performance': '⚡',
        'security': '🔐', 'users': '👥', 'settings': '⚙️', 'config': '🔧',
        'database': '🗄️', 'storage': '💾', 'backups': '💿', 'testing': '🧪',
        'debugger': '🐛', 'logs': '📄', 'api-explorer': '🔍', 'deployment': '🚀',
        'infrastructure': '🏗️', 'containers': '📦', 'kubernetes': '☸️',
        'chat': '💬', 'collaboration': '🤝', 'notifications': '🔔'
    }
    return icon_map.get(page_name, '📄')

# Core System Pages
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page():
    """Main system dashboard"""
    return serve_html_page("dashboard", "🏠 Main Dashboard")

@app.get("/admin", response_class=HTMLResponse)
async def admin_page():
    """Admin portal and system management"""
    return serve_html_page("admin", "👨‍💼 Admin Portal")

@app.get("/designer", response_class=HTMLResponse)
async def designer_page():
    """Visual process designer"""
    return serve_html_page("designer", "🎨 Visual Designer")

@app.get("/developer-tools", response_class=HTMLResponse)
async def developer_tools_page():
    """Developer tools and debugging"""
    return serve_html_page("developer-tools", "🔧 Developer Tools")

# AI and Processing Pages  
@app.get("/ai-agents", response_class=HTMLResponse)
async def ai_agents_page():
    """AI agents management"""
    return serve_html_page("ai-agents", "🤖 AI Agents")

@app.get("/ai-models", response_class=HTMLResponse)
async def ai_models_page():
    """AI models configuration"""
    return serve_html_page("ai-models", "🧠 AI Models")

@app.get("/processes", response_class=HTMLResponse)
async def processes_page():
    """CSP processes management"""
    return serve_html_page("processes", "⚙️ CSP Processes")

@app.get("/channels", response_class=HTMLResponse)
async def channels_page():
    """Communication channels"""
    return serve_html_page("channels", "📡 Communication Channels")

# Advanced Technology Pages
@app.get("/quantum", response_class=HTMLResponse)
async def quantum_page():
    """Quantum computing interface"""
    return serve_html_page("quantum", "⚛️ Quantum Computing")

@app.get("/blockchain", response_class=HTMLResponse)
async def blockchain_page():
    """Blockchain network management"""
    return serve_html_page("blockchain", "🔗 Blockchain Network")

@app.get("/neural", response_class=HTMLResponse)
async def neural_page():
    """Neural network optimizer"""
    return serve_html_page("neural", "🧬 Neural Optimizer")

@app.get("/neural-mesh", response_class=HTMLResponse)
async def neural_mesh_page():
    """Neural mesh network"""
    return serve_html_page("neural-mesh", "🕸️ Neural Mesh")

# Monitoring and Analytics Pages
@app.get("/monitoring", response_class=HTMLResponse)
async def monitoring_page():
    """System monitoring and metrics"""
    return serve_html_page("monitoring", "📊 System Monitoring")

@app.get("/analytics", response_class=HTMLResponse)
async def analytics_page():
    """Performance analytics"""
    return serve_html_page("analytics", "📈 Analytics")

@app.get("/performance", response_class=HTMLResponse)
async def performance_page():
    """Performance monitoring"""
    return serve_html_page("performance", "⚡ Performance Monitor")

# Security and Management Pages
@app.get("/security", response_class=HTMLResponse)
async def security_page():
    """Security management"""
    return serve_html_page("security", "🔐 Security Center")

@app.get("/users", response_class=HTMLResponse)
async def users_page():
    """User management"""
    return serve_html_page("users", "👥 User Management")

@app.get("/settings", response_class=HTMLResponse)
async def settings_page():
    """System settings"""
    return serve_html_page("settings", "⚙️ System Settings")

@app.get("/config", response_class=HTMLResponse)
async def config_page():
    """Configuration management"""
    return serve_html_page("config", "🔧 Configuration")

# Data and Storage Pages
@app.get("/database", response_class=HTMLResponse)
async def database_page():
    """Database management"""
    return serve_html_page("database", "🗄️ Database Manager")

@app.get("/storage", response_class=HTMLResponse)
async def storage_page():
    """Storage management"""
    return serve_html_page("storage", "💾 Storage Manager")

@app.get("/backups", response_class=HTMLResponse)
async def backups_page():
    """Backup management"""
    return serve_html_page("backups", "💿 Backup Manager")

# Development and Testing Pages
@app.get("/testing", response_class=HTMLResponse)
async def testing_page():
    """Testing framework"""
    return serve_html_page("testing", "🧪 Testing Framework")

@app.get("/debugger", response_class=HTMLResponse)
async def debugger_page():
    """System debugger"""
    return serve_html_page("debugger", "🐛 System Debugger")

@app.get("/logs", response_class=HTMLResponse)
async def logs_page():
    """System logs viewer"""
    return serve_html_page("logs", "📄 System Logs")

@app.get("/api-explorer", response_class=HTMLResponse)
async def api_explorer_page():
    """API explorer and testing"""
    return serve_html_page("api-explorer", "🔍 API Explorer")

# Deployment and Infrastructure Pages
@app.get("/deployment", response_class=HTMLResponse)
async def deployment_page():
    """Deployment management"""
    return serve_html_page("deployment", "🚀 Deployment Manager")

@app.get("/infrastructure", response_class=HTMLResponse)
async def infrastructure_page():
    """Infrastructure management"""
    return serve_html_page("infrastructure", "🏗️ Infrastructure")

@app.get("/containers", response_class=HTMLResponse)
async def containers_page():
    """Container management"""
    return serve_html_page("containers", "📦 Container Manager")

@app.get("/kubernetes", response_class=HTMLResponse)
async def kubernetes_page():
    """Kubernetes management"""
    return serve_html_page("kubernetes", "☸️ Kubernetes")

# Communication and Collaboration Pages
@app.get("/chat", response_class=HTMLResponse)
async def chat_page():
    """Chat interface"""
    return serve_html_page("chat", "💬 Chat Interface")

@app.get("/collaboration", response_class=HTMLResponse)
async def collaboration_page():
    """Collaboration tools"""
    return serve_html_page("collaboration", "🤝 Collaboration")

@app.get("/notifications", response_class=HTMLResponse)
async def notifications_page():
    """Notifications center"""
    return serve_html_page("notifications", "🔔 Notifications")

# Dynamic route for any page in frontend/pages
@app.get("/page/{page_name}", response_class=HTMLResponse)
async def serve_frontend_page(page_name: str):
    """Serve any page from frontend/pages directory"""
    return serve_html_page(page_name, f"📄 {page_name.replace('-', ' ').title()}")

# List all available pages (scans the actual directory)
@app.get("/pages", response_class=HTMLResponse)
async def list_pages():
    """List all available frontend pages"""
    pages_dir = Path("frontend/pages")
    
    if not pages_dir.exists():
        return HTMLResponse("""
        <h1>❌ Frontend pages directory not found</h1>
        <p>Expected: <code>frontend/pages/</code></p>
        <p><a href="/">← Back to Home</a></p>
        """)
    
    # Get all HTML files
    html_files = list(pages_dir.glob("*.html"))
    
    if not html_files:
        return HTMLResponse("""
        <h1>📁 No HTML pages found</h1>
        <p>Directory: <code>frontend/pages/</code></p>
        <p>Add HTML files to this directory and refresh!</p>
        <p><a href="/">← Back to Home</a></p>
        """)
    
    # Create a styled page list
    page_links = ""
    for file in sorted(html_files):
        page_name = file.stem
        icon = get_page_icon(page_name)
        page_links += f'''
        <div class="page-item">
            <a href="/{page_name}" class="page-link">
                <span class="page-icon">{icon}</span>
                <span class="page-name">{page_name.replace('-', ' ').title()}</span>
                <span class="page-file">{file.name}</span>
            </a>
        </div>
        '''
    
    return HTMLResponse(f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>📄 Available Pages - CSP System</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                margin: 0;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                min-height: 100vh;
                padding: 2rem;
            }}
            .container {{
                max-width: 1000px;
                margin: 0 auto;
                background: rgba(255,255,255,0.1);
                backdrop-filter: blur(10px);
                border-radius: 15px;
                padding: 2rem;
                border: 1px solid rgba(255,255,255,0.2);
            }}
            h1 {{ text-align: center; margin-bottom: 2rem; }}
            .pages-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 1rem;
                margin-bottom: 2rem;
            }}
            .page-item {{
                background: rgba(255,255,255,0.1);
                border-radius: 8px;
                overflow: hidden;
                transition: all 0.3s ease;
            }}
            .page-item:hover {{
                background: rgba(255,255,255,0.2);
                transform: translateY(-2px);
            }}
            .page-link {{
                display: flex;
                align-items: center;
                padding: 1rem;
                text-decoration: none;
                color: white;
                gap: 1rem;
            }}
            .page-icon {{ font-size: 1.5rem; }}
            .page-name {{ flex: 1; font-weight: 600; }}
            .page-file {{ font-size: 0.8rem; opacity: 0.7; }}
            .stats {{
                text-align: center;
                padding: 1rem;
                background: rgba(255,255,255,0.1);
                border-radius: 8px;
                margin-bottom: 2rem;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📄 Available Pages</h1>
            
            <div class="stats">
                <strong>{len(html_files)} pages available</strong> in <code>frontend/pages/</code>
            </div>
            
            <div class="pages-grid">
                {page_links}
            </div>
            
            <div style="text-align: center;">
                <a href="/" style="color: white; text-decoration: none;">← Back to Home</a> |
                <a href="/docs" style="color: white; text-decoration: none;">📚 API Documentation</a>
            </div>
        </div>
    </body>
    </html>
    """)

# Updated root route to show main navigation
@app.get("/", response_class=HTMLResponse)
async def root_with_complete_navigation():
    """Root endpoint with complete navigation"""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🚀 Enhanced CSP System</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                margin: 0;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                min-height: 100vh;
                padding: 2rem;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: rgba(255,255,255,0.1);
                backdrop-filter: blur(10px);
                border-radius: 15px;
                padding: 2rem;
                border: 1px solid rgba(255,255,255,0.2);
            }
            h1 { text-align: center; font-size: 3rem; margin-bottom: 0.5rem; }
            .subtitle { text-align: center; opacity: 0.9; margin-bottom: 3rem; }
            .nav-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 1rem;
                margin-bottom: 2rem;
            }
            .nav-section {
                background: rgba(255,255,255,0.1);
                border-radius: 10px;
                padding: 1.5rem;
                border: 1px solid rgba(255,255,255,0.2);
            }
            .nav-section h3 {
                margin-top: 0;
                margin-bottom: 1rem;
                color: #fff;
                font-size: 1.1rem;
            }
            .nav-link {
                display: block;
                color: white;
                text-decoration: none;
                padding: 0.5rem;
                border-radius: 5px;
                margin-bottom: 0.5rem;
                transition: all 0.3s ease;
            }
            .nav-link:hover {
                background: rgba(255,255,255,0.1);
                transform: translateX(5px);
            }
            .status-bar {
                background: rgba(0,255,0,0.2);
                padding: 1rem;
                border-radius: 8px;
                text-align: center;
                margin-bottom: 2rem;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Enhanced CSP System</h1>
            <p class="subtitle">Advanced AI-to-AI Communication Platform</p>
            
            <div class="status-bar">
                ✅ System Online | 🔄 All Services Running | 📊 Ready for Operations
            </div>
            
            <div class="nav-grid">
                <div class="nav-section">
                    <h3>🏠 Core System</h3>
                    <a href="/dashboard" class="nav-link">🏠 Main Dashboard</a>
                    <a href="/admin" class="nav-link">👨‍💼 Admin Portal</a>
                    <a href="/designer" class="nav-link">🎨 Visual Designer</a>
                    <a href="/monitoring" class="nav-link">📊 System Monitoring</a>
                </div>
                
                <div class="nav-section">
                    <h3>🤖 AI & Processing</h3>
                    <a href="/ai-agents" class="nav-link">🤖 AI Agents</a>
                    <a href="/ai-models" class="nav-link">🧠 AI Models</a>
                    <a href="/processes" class="nav-link">⚙️ CSP Processes</a>
                    <a href="/channels" class="nav-link">📡 Communication</a>
                </div>
                
                <div class="nav-section">
                    <h3>⚛️ Advanced Tech</h3>
                    <a href="/quantum" class="nav-link">⚛️ Quantum Computing</a>
                    <a href="/blockchain" class="nav-link">🔗 Blockchain</a>
                    <a href="/neural" class="nav-link">🧬 Neural Networks</a>
                    <a href="/neural-mesh" class="nav-link">🕸️ Neural Mesh</a>
                </div>
                
                <div class="nav-section">
                    <h3>🔧 Development</h3>
                    <a href="/developer-tools" class="nav-link">🔧 Dev Tools</a>
                    <a href="/debugger" class="nav-link">🐛 Debugger</a>
                    <a href="/testing" class="nav-link">🧪 Testing</a>
                    <a href="/api-explorer" class="nav-link">🔍 API Explorer</a>
                </div>
                
                <div class="nav-section">
                    <h3>🔐 Security & Config</h3>
                    <a href="/security" class="nav-link">🔐 Security Center</a>
                    <a href="/users" class="nav-link">👥 User Management</a>
                    <a href="/settings" class="nav-link">⚙️ Settings</a>
                    <a href="/logs" class="nav-link">📄 System Logs</a>
                </div>
                
                <div class="nav-section">
                    <h3>📚 Resources</h3>
                    <a href="/pages" class="nav-link">📄 All Pages</a>
                    <a href="/docs" class="nav-link">📚 API Documentation</a>
                    <a href="/health" class="nav-link">❤️ Health Check</a>
                </div>
            </div>
        </div>
    </body>
    </html>
    """)

# ============================================================================
# STATIC FILES AND TEMPLATES
# ============================================================================

# Mount dashboard if available
try:
    dashboard_app = create_dashboard_app()
    app.mount("/dashboard-api", dashboard_app, name="dashboard")
    logger.info("Dashboard API mounted at /dashboard-api")
except Exception as e:
    logger.warning(f"Dashboard API not available: {e}")

# Mount static files
static_dir = Path("web_ui/static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Templates
templates_dir = Path("web_ui/templates")
if templates_dir.exists():
    templates = Jinja2Templates(directory=str(templates_dir))
    
    @app.get("/ui", response_class=HTMLResponse)
    async def web_ui(request: Request):
        """Web UI for the CSP system"""
        return templates.TemplateResponse("index.html", {"request": request})

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point"""
    logger.info(f"🚀 Starting {config.app_name} v{config.version}")
    logger.info(f"Environment: {config.environment}")
    logger.info(f"Host: {config.host}:{config.port}")
    
    # Set up event loop optimization
    if UVLOOP_AVAILABLE and sys.platform != 'win32':
        try:
            uvloop.install()
            logger.info("✅ uvloop installed for better performance")
        except Exception as e:
            logger.warning(f"uvloop installation failed: {e}")
    else:
        logger.warning("uvloop not available; using default event loop")
    
    try:
        uvicorn.run(
            app,
            host=config.host,
            port=config.port,
            workers=config.workers,
            reload=config.reload and config.debug,
            log_level="info" if not config.debug else "debug",
            access_log=True
        )
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()