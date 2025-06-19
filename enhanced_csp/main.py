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

import os
import sys
import asyncio
import logging
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from contextlib import asynccontextmanager

# FastAPI and web dependencies
from fastapi import FastAPI, WebSocket, HTTPException, Depends, BackgroundTasks, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn

# Database and storage
import sqlite3

try:
    import redis.asyncio as redis
except Exception:  # pragma: no cover - optional dependency
    redis = None
    logging.warning("Redis library not available; Redis features disabled")
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Monitoring and metrics
try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )
except Exception:  # pragma: no cover - optional dependency
    Counter = Histogram = Gauge = generate_latest = None
    CONTENT_TYPE_LATEST = "text/plain"
    logging.warning("prometheus_client not available; metrics disabled")

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency
    psutil = None
    logging.warning("psutil not available; system metrics disabled")

# Configuration and utilities
import yaml
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

try:
    import aiofiles
except Exception:  # pragma: no cover - optional dependency
    aiofiles = None
    logging.warning("aiofiles not available; some features disabled")

try:
    import aiohttp
except Exception:  # pragma: no cover - optional dependency
    aiohttp = None
    logging.warning("aiohttp not available; some features disabled")

# Enhanced CSP Core Components
try:
    from core.advanced_csp_core import (
        AdvancedCSPEngine, Process, AtomicProcess, CompositeProcess,
        CompositionOperator, ChannelType, Event, ProcessSignature,
        ProcessContext, Channel, ProcessMatcher, ProtocolEvolution
    )
    from ai_integration.csp_ai_extensions import (
        AdvancedCSPEngineWithAI, ProtocolSpec, ProtocolTemplate,
        EmergentBehaviorDetector, CausalityTracker, QuantumCSPChannel
    )
    from ai_integration.csp_ai_integration import (
        AIAgent, LLMCapability, CollaborativeAIProcess,
        MultiAgentReasoningCoordinator, AdvancedAICSPDemo
    )
    from runtime.csp_runtime_environment import (
        CSPRuntimeOrchestrator, RuntimeConfig, ExecutionModel, 
        SchedulingPolicy, HighPerformanceRuntimeExecutor
    )
    from deployment.csp_deployment_system import (
        CSPDeploymentOrchestrator, DeploymentConfig, DeploymentTarget,
        ScalingStrategy, HealthCheckConfig
    )
    from dev_tools.csp_dev_tools import (
        CSPDevelopmentTools, CSPVisualDesigner, CSPDebugger,
        CSPCodeGenerator, CSPTestFramework
    )
    from monitoring.csp_monitoring import (
        CSPMonitor, MetricsCollector, PerformanceAnalyzer,
        AlertManager, SystemHealthChecker
    )
    from web_ui.dashboard.app import create_dashboard_app
except ImportError as e:
    logging.error(f"Failed to import CSP components: {e}")
    logging.error("Some features may not be available")# Database migrations are required even if optional components fail to import
try:
    from database.migrate import migrate_main as run_migrations
except Exception as e:  # ensure a definition exists
    logging.error(f"Database migration module unavailable: {e}")

    def run_migrations() -> None:
        """Fallback migration runner when migrate module is missing."""
        logging.error("run_migrations placeholder invoked; no migrations run")



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

class RuntimeConfig(BaseModel):
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
    runtime: RuntimeConfig = RuntimeConfig()
    
    # Security settings
    secret_key: str = os.getenv('CSP_SECRET_KEY', 'dev-secret-key')
    api_key_header: str = "X-CSP-API-Key"
    enable_auth: bool = True
    
    @validator('port')
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
        logger.warning(f"Config file not found at {config_path}, using defaults")
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
REQUEST_COUNT = Counter('csp_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('csp_request_duration_seconds', 'Request duration')
ACTIVE_PROCESSES = Gauge('csp_active_processes', 'Number of active CSP processes')
SYSTEM_HEALTH = Gauge('csp_system_health', 'System health score (0-1)')
WEBSOCKET_CONNECTIONS = Gauge('csp_websocket_connections', 'Active WebSocket connections')

class MetricsMiddleware:
    """Middleware for collecting metrics"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            start_time = time.time()
            
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
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

async def setup_database():
    """Initialize database connection and run migrations"""
    try:
        # Create async engine
        system_state.db_engine = create_async_engine(
            config.database.url,
            echo=config.database.echo,
            pool_size=config.database.pool_size,
            max_overflow=config.database.max_overflow
        )
        
        # Run migrations
        logger.info("Running database migrations...")
        await run_migrations()
        
        logger.info("Database setup completed")
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        raise

async def setup_redis():
    """Initialize Redis connection"""
    try:
        system_state.redis_client = redis.from_url(
            config.redis.url,
            max_connections=config.redis.max_connections,
            retry_on_timeout=config.redis.retry_on_timeout
        )
        
        # Test connection
        await system_state.redis_client.ping()
        logger.info("Redis connection established")
    except Exception as e:
        logger.error(f"Redis setup failed: {e}")
        # Continue without Redis if it fails
        system_state.redis_client = None

# ============================================================================
# CSP SYSTEM INITIALIZATION
# ============================================================================

async def initialize_csp_system():
    """Initialize all CSP system components"""
    try:
        logger.info("Initializing Enhanced CSP System...")
        
        # 1. Initialize core CSP engine
        system_state.csp_engine = AdvancedCSPEngine()
        logger.info("✅ Core CSP engine initialized")
        
        # 2. Initialize AI-enhanced engine
        if config.ai.enable_llm_integration:
            system_state.ai_engine = AdvancedCSPEngineWithAI(
                base_engine=system_state.csp_engine,
                llm_config={
                    'model': config.ai.default_model,
                    'timeout': config.ai.reasoning_timeout
                }
            )
            logger.info("✅ AI-enhanced CSP engine initialized")
        
        # 3. Initialize runtime orchestrator
        runtime_config = RuntimeConfig(
            execution_model=getattr(ExecutionModel, config.runtime.execution_model),
            max_workers=config.runtime.max_workers,
            memory_limit_gb=config.runtime.memory_limit_gb,
            enable_optimization=config.runtime.enable_optimization,
            debug_mode=config.runtime.enable_debugging
        )
        system_state.runtime_orchestrator = CSPRuntimeOrchestrator(runtime_config)
        await system_state.runtime_orchestrator.start()
        logger.info("✅ Runtime orchestrator initialized")
        
        # 4. Initialize deployment orchestrator
        system_state.deployment_orchestrator = CSPDeploymentOrchestrator()
        logger.info("✅ Deployment orchestrator initialized")
        
        # 5. Initialize development tools
        system_state.dev_tools = CSPDevelopmentTools()
        await system_state.dev_tools.initialize()
        logger.info("✅ Development tools initialized")
        
        # 6. Initialize monitoring
        if config.monitoring.enable_prometheus:
            system_state.monitor = CSPMonitor(
                prometheus_enabled=True,
                metrics_port=config.monitoring.metrics_port
            )
            await system_state.monitor.start()
            logger.info("✅ Monitoring system initialized")
        
        logger.info("🚀 Enhanced CSP System fully initialized!")
        
    except Exception as e:
        logger.error(f"CSP system initialization failed: {e}")
        raise

# ============================================================================
# FASTAPI APPLICATION SETUP
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Enhanced CSP System...")
    
    try:
        await setup_database()
        await setup_redis()
        await initialize_csp_system()
        
        # Update system health
        if SYSTEM_HEALTH:
            SYSTEM_HEALTH.set(1.0)
        logger.info("System startup completed successfully")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        if SYSTEM_HEALTH:
            SYSTEM_HEALTH.set(0.0)
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Enhanced CSP System...")
    
    try:
        # Close WebSocket connections
        for ws in system_state.active_websockets:
            await ws.close()
        
        # Shutdown components
        if system_state.runtime_orchestrator:
            await system_state.runtime_orchestrator.stop()
        
        if system_state.monitor:
            await system_state.monitor.stop()
        
        if system_state.redis_client:
            await system_state.redis_client.close()
        
        if system_state.db_engine:
            await system_state.db_engine.dispose()
        
        logger.info("System shutdown completed")
        
    except Exception as e:
        logger.error(f"Shutdown error: {e}")

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
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_data = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system": {
            "cpu_percent": psutil.cpu_percent() if psutil else 0.0,
            "memory_percent": psutil.virtual_memory().percent if psutil else 0.0,
            "disk_percent": psutil.disk_usage('/') .percent if psutil else 0.0
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
                    "cpu": psutil.cpu_percent(),
                    "memory": psutil.virtual_memory().percent,
                    "disk": psutil.disk_usage('/').percent
                },
                "websocket_connections": len(system_state.active_websockets)
            }
            
            await websocket.send_json(status)
            await asyncio.sleep(5)  # Update every 5 seconds
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
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
        system_state.active_websockets.remove(websocket)

# ============================================================================
# STATIC FILES AND TEMPLATES
# ============================================================================

# Mount dashboard if available
try:
    dashboard_app = create_dashboard_app()
    app.mount("/dashboard", dashboard_app, name="dashboard")
    logger.info("Dashboard mounted at /dashboard")
except Exception as e:
    logger.warning(f"Dashboard not available: {e}")

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
    logger.info(f"Starting {config.app_name} v{config.version}")
    logger.info(f"Environment: {config.environment}")
    logger.info(f"Host: {config.host}:{config.port}")
    
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
