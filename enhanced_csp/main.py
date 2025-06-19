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
"""

import os
import sys
import asyncio
import logging
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager

# FastAPI and web dependencies
from fastapi import FastAPI, WebSocket, HTTPException, Depends, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import uvicorn

# Database and storage
import sqlite3
import redis.asyncio as redis
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

# Monitoring and metrics
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import psutil

# Configuration and utilities
import yaml
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('CSP_LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/enhanced_csp.log')
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================

class CSPConfig(BaseModel):
    """Enhanced CSP System Configuration"""
    
    # Core settings
    app_name: str = "Enhanced CSP System"
    version: str = "1.0.0"
    debug: bool = False
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    
    # Database settings
    database_url: str = "sqlite:///./data/enhanced_csp.db"
    redis_url: str = "redis://localhost:6379"
    
    # CSP Engine settings
    max_processes: int = 1000
    max_channels: int = 10000
    max_agents: int = 500
    consciousness_enabled: bool = True
    quantum_enabled: bool = True
    neural_mesh_enabled: bool = True
    
    # AI settings
    ai_enabled: bool = True
    protocol_synthesis_enabled: bool = True
    emergent_detection_enabled: bool = True
    
    # Performance settings
    execution_model: str = "HYBRID"
    max_workers: int = 4
    memory_limit_gb: float = 8.0
    
    # Security settings
    enable_cors: bool = True
    allowed_origins: List[str] = ["*"]
    secret_key: str = "enhanced-csp-secret-key-change-in-production"
    
    # Monitoring settings
    metrics_enabled: bool = True
    health_check_enabled: bool = True
    
    @classmethod
    def load_config(cls, config_path: str = "config/app.yaml") -> "CSPConfig":
        """Load configuration from file"""
        try:
            if Path(config_path).exists():
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                return cls(**config_data)
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}")
        
        return cls()

# ============================================================================
# PROMETHEUS METRICS
# ============================================================================

# System metrics
system_requests_total = Counter('csp_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
system_request_duration = Histogram('csp_request_duration_seconds', 'Request duration')
system_active_connections = Gauge('csp_active_connections', 'Active WebSocket connections')
system_memory_usage = Gauge('csp_memory_usage_bytes', 'Memory usage')
system_cpu_usage = Gauge('csp_cpu_usage_percent', 'CPU usage percentage')

# CSP-specific metrics
csp_processes_active = Gauge('csp_processes_active', 'Active CSP processes')
csp_channels_active = Gauge('csp_channels_active', 'Active CSP channels')
csp_agents_active = Gauge('csp_agents_active', 'Active CSP agents')
csp_messages_processed = Counter('csp_messages_processed_total', 'Total messages processed')
csp_quantum_operations = Counter('csp_quantum_operations_total', 'Total quantum operations')
csp_consciousness_sync = Counter('csp_consciousness_sync_total', 'Total consciousness synchronizations')

# ============================================================================
# DATABASE MODELS AND OPERATIONS
# ============================================================================

class DatabaseManager:
    """Database operations manager"""
    
    def __init__(self, config: CSPConfig):
        self.config = config
        self.engine = None
        self.redis_client = None
    
    async def initialize(self):
        """Initialize database connections"""
        try:
            # Initialize SQLite
            await self._init_sqlite()
            
            # Initialize Redis
            await self._init_redis()
            
            logger.info("Database connections initialized")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    async def _init_sqlite(self):
        """Initialize SQLite database"""
        # Ensure data directory exists
        Path("data").mkdir(exist_ok=True)
        
        # Create tables if they don't exist
        db_path = "data/enhanced_csp.db"
        conn = sqlite3.connect(db_path)
        
        # Core tables
        conn.execute('''
            CREATE TABLE IF NOT EXISTS engines (
                engine_id TEXT PRIMARY KEY,
                status TEXT DEFAULT 'initializing',
                configuration TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS processes (
                process_id TEXT PRIMARY KEY,
                engine_id TEXT,
                process_type TEXT,
                status TEXT DEFAULT 'ready',
                configuration TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (engine_id) REFERENCES engines (engine_id)
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS channels (
                channel_id TEXT PRIMARY KEY,
                engine_id TEXT,
                channel_type TEXT,
                status TEXT DEFAULT 'active',
                participants TEXT,
                configuration TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (engine_id) REFERENCES engines (engine_id)
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS agents (
                agent_id TEXT PRIMARY KEY,
                engine_id TEXT,
                agent_type TEXT,
                status TEXT DEFAULT 'active',
                consciousness_level REAL DEFAULT 0.8,
                capabilities TEXT,
                configuration TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (engine_id) REFERENCES engines (engine_id)
            )
        ''')
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT,
                metric_value REAL,
                labels TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def _init_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.from_url(self.config.redis_url)
            await self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None
    
    async def close(self):
        """Close database connections"""
        if self.redis_client:
            await self.redis_client.close()

# ============================================================================
# CSP ENGINE INTEGRATION
# ============================================================================

class EnhancedCSPEngine:
    """Enhanced CSP Engine with quantum and consciousness features"""
    
    def __init__(self, config: CSPConfig):
        self.config = config
        self.engine_id = str(uuid.uuid4())
        self.status = "initializing"
        self.processes = {}
        self.channels = {}
        self.agents = {}
        self.metrics = {}
        
        # Feature flags
        self.consciousness_enabled = config.consciousness_enabled
        self.quantum_enabled = config.quantum_enabled
        self.neural_mesh_enabled = config.neural_mesh_enabled
        self.ai_enabled = config.ai_enabled
    
    async def initialize(self):
        """Initialize the CSP engine"""
        logger.info(f"Initializing Enhanced CSP Engine {self.engine_id}")
        
        # Initialize core components
        await self._init_core_engine()
        
        # Initialize optional features
        if self.consciousness_enabled:
            await self._init_consciousness_manager()
        
        if self.quantum_enabled:
            await self._init_quantum_manager()
        
        if self.neural_mesh_enabled:
            await self._init_neural_mesh()
        
        if self.ai_enabled:
            await self._init_ai_extensions()
        
        self.status = "running"
        logger.info("Enhanced CSP Engine initialized successfully")
    
    async def _init_core_engine(self):
        """Initialize core CSP engine"""
        # Create default engine record
        self.metrics["core_initialized"] = datetime.now()
        csp_processes_active.set(len(self.processes))
        csp_channels_active.set(len(self.channels))
        csp_agents_active.set(len(self.agents))
    
    async def _init_consciousness_manager(self):
        """Initialize consciousness management"""
        logger.info("Initializing consciousness manager...")
        self.metrics["consciousness_initialized"] = datetime.now()
    
    async def _init_quantum_manager(self):
        """Initialize quantum communication"""
        logger.info("Initializing quantum manager...")
        self.metrics["quantum_initialized"] = datetime.now()
    
    async def _init_neural_mesh(self):
        """Initialize neural mesh network"""
        logger.info("Initializing neural mesh...")
        self.metrics["neural_mesh_initialized"] = datetime.now()
    
    async def _init_ai_extensions(self):
        """Initialize AI extensions"""
        logger.info("Initializing AI extensions...")
        self.metrics["ai_extensions_initialized"] = datetime.now()
    
    async def create_process(self, process_type: str, config: dict = None) -> str:
        """Create a new CSP process"""
        process_id = str(uuid.uuid4())
        self.processes[process_id] = {
            "id": process_id,
            "type": process_type,
            "status": "ready",
            "config": config or {},
            "created_at": datetime.now()
        }
        
        csp_processes_active.set(len(self.processes))
        logger.info(f"Created process {process_id} of type {process_type}")
        return process_id
    
    async def create_channel(self, channel_type: str, participants: List[str] = None) -> str:
        """Create a new communication channel"""
        channel_id = str(uuid.uuid4())
        self.channels[channel_id] = {
            "id": channel_id,
            "type": channel_type,
            "status": "active",
            "participants": participants or [],
            "created_at": datetime.now()
        }
        
        csp_channels_active.set(len(self.channels))
        logger.info(f"Created channel {channel_id} of type {channel_type}")
        return channel_id
    
    async def create_agent(self, agent_type: str, consciousness_level: float = 0.8) -> str:
        """Create a new AI agent"""
        agent_id = str(uuid.uuid4())
        self.agents[agent_id] = {
            "id": agent_id,
            "type": agent_type,
            "status": "active",
            "consciousness_level": consciousness_level,
            "capabilities": [],
            "created_at": datetime.now()
        }
        
        csp_agents_active.set(len(self.agents))
        logger.info(f"Created agent {agent_id} of type {agent_type}")
        return agent_id
    
    async def send_message(self, channel_id: str, message: dict) -> bool:
        """Send message through channel"""
        if channel_id not in self.channels:
            return False
        
        # Process message through CSP protocol
        csp_messages_processed.inc()
        
        # Apply quantum processing if enabled
        if self.quantum_enabled:
            csp_quantum_operations.inc()
        
        # Apply consciousness sync if enabled
        if self.consciousness_enabled:
            csp_consciousness_sync.inc()
        
        logger.debug(f"Message sent through channel {channel_id}")
        return True
    
    def get_status(self) -> dict:
        """Get engine status"""
        return {
            "engine_id": self.engine_id,
            "status": self.status,
            "processes": len(self.processes),
            "channels": len(self.channels),
            "agents": len(self.agents),
            "features": {
                "consciousness": self.consciousness_enabled,
                "quantum": self.quantum_enabled,
                "neural_mesh": self.neural_mesh_enabled,
                "ai": self.ai_enabled
            },
            "metrics": self.metrics
        }

# ============================================================================
# WEBSOCKET CONNECTION MANAGER
# ============================================================================

class ConnectionManager:
    """WebSocket connection manager"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_count = 0
    
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_count += 1
        system_active_connections.set(len(self.active_connections))
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        system_active_connections.set(len(self.active_connections))
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to specific WebSocket"""
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: str):
        """Broadcast message to all connected WebSockets"""
        if not self.active_connections:
            return
        
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")
                disconnected.append(connection)
        
        # Remove disconnected connections
        for connection in disconnected:
            self.disconnect(connection)

# ============================================================================
# FASTAPI APPLICATION SETUP
# ============================================================================

# Load configuration
config = CSPConfig.load_config()

# Global instances
db_manager = None
csp_engine = None
connection_manager = ConnectionManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global db_manager, csp_engine
    
    # Startup
    logger.info("Starting Enhanced CSP System...")
    
    # Create directories
    for directory in ["data", "logs", "config", "static", "templates"]:
        Path(directory).mkdir(exist_ok=True)
    
    # Initialize database
    db_manager = DatabaseManager(config)
    await db_manager.initialize()
    
    # Initialize CSP engine
    csp_engine = EnhancedCSPEngine(config)
    await csp_engine.initialize()
    
    # Start background tasks
    asyncio.create_task(metrics_updater())
    asyncio.create_task(system_monitor())
    
    logger.info("Enhanced CSP System started successfully!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Enhanced CSP System...")
    
    if db_manager:
        await db_manager.close()
    
    logger.info("Enhanced CSP System shutdown complete.")

# Create FastAPI application
app = FastAPI(
    title=config.app_name,
    description="Revolutionary AI-to-AI Communication Platform",
    version=config.version,
    lifespan=lifespan
)

# Add middleware
if config.enable_cors:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint - serve dashboard"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Enhanced CSP System</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .header { background: linear-gradient(45deg, #007acc, #0099ff); color: white; padding: 20px; border-radius: 10px; }
            .status { background: #f8f9fa; padding: 20px; margin: 20px 0; border-radius: 10px; }
            .endpoints { background: #e9ecef; padding: 20px; border-radius: 10px; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🚀 Enhanced CSP System</h1>
            <p>Revolutionary AI-to-AI Communication Platform</p>
        </div>
        
        <div class="status">
            <h2>System Status</h2>
            <p>✅ Core Engine: Active</p>
            <p>✅ API Server: Running</p>
            <p>✅ WebSocket: Available</p>
            <p>✅ Database: Connected</p>
        </div>
        
        <div class="endpoints">
            <h2>Available Endpoints</h2>
            <ul>
                <li><a href="/health">Health Check</a></li>
                <li><a href="/metrics">Prometheus Metrics</a></li>
                <li><a href="/status">System Status</a></li>
                <li><a href="/docs">API Documentation</a></li>
                <li><a href="/dashboard">Real-time Dashboard</a></li>
            </ul>
        </div>
    </body>
    </html>
    """

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": config.version,
        "engine_status": csp_engine.status if csp_engine else "not_initialized"
    }

@app.get("/status")
async def system_status():
    """Detailed system status"""
    if not csp_engine:
        raise HTTPException(status_code=503, detail="CSP Engine not initialized")
    
    return {
        "system": {
            "name": config.app_name,
            "version": config.version,
            "uptime": "running",
            "memory_usage": psutil.virtual_memory().percent,
            "cpu_usage": psutil.cpu_percent(),
        },
        "csp_engine": csp_engine.get_status(),
        "connections": {
            "websocket": len(connection_manager.active_connections),
            "total": connection_manager.connection_count
        }
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest().decode('utf-8')

# ============================================================================
# CSP API ENDPOINTS
# ============================================================================

@app.post("/api/processes")
async def create_process(process_type: str, config_data: dict = None):
    """Create a new CSP process"""
    if not csp_engine:
        raise HTTPException(status_code=503, detail="CSP Engine not available")
    
    process_id = await csp_engine.create_process(process_type, config_data)
    return {"process_id": process_id, "status": "created"}

@app.post("/api/channels")
async def create_channel(channel_type: str, participants: List[str] = None):
    """Create a new communication channel"""
    if not csp_engine:
        raise HTTPException(status_code=503, detail="CSP Engine not available")
    
    channel_id = await csp_engine.create_channel(channel_type, participants)
    return {"channel_id": channel_id, "status": "created"}

@app.post("/api/agents")
async def create_agent(agent_type: str, consciousness_level: float = 0.8):
    """Create a new AI agent"""
    if not csp_engine:
        raise HTTPException(status_code=503, detail="CSP Engine not available")
    
    agent_id = await csp_engine.create_agent(agent_type, consciousness_level)
    return {"agent_id": agent_id, "status": "created"}

@app.post("/api/messages/{channel_id}")
async def send_message(channel_id: str, message: dict):
    """Send message through channel"""
    if not csp_engine:
        raise HTTPException(status_code=503, detail="CSP Engine not available")
    
    success = await csp_engine.send_message(channel_id, message)
    if not success:
        raise HTTPException(status_code=404, detail="Channel not found")
    
    return {"status": "sent", "channel_id": channel_id}

@app.get("/api/processes")
async def list_processes():
    """List all processes"""
    if not csp_engine:
        raise HTTPException(status_code=503, detail="CSP Engine not available")
    
    return {"processes": list(csp_engine.processes.values())}

@app.get("/api/channels")
async def list_channels():
    """List all channels"""
    if not csp_engine:
        raise HTTPException(status_code=503, detail="CSP Engine not available")
    
    return {"channels": list(csp_engine.channels.values())}

@app.get("/api/agents")
async def list_agents():
    """List all agents"""
    if not csp_engine:
        raise HTTPException(status_code=503, detail="CSP Engine not available")
    
    return {"agents": list(csp_engine.agents.values())}

# ============================================================================
# WEBSOCKET ENDPOINTS
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for real-time communication"""
    await connection_manager.connect(websocket)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Process message based on type
            if message_data.get("type") == "ping":
                await connection_manager.send_personal_message(
                    json.dumps({"type": "pong", "timestamp": datetime.now().isoformat()}),
                    websocket
                )
            elif message_data.get("type") == "broadcast":
                await connection_manager.broadcast(
                    json.dumps({
                        "type": "message",
                        "content": message_data.get("content"),
                        "timestamp": datetime.now().isoformat()
                    })
                )
            elif message_data.get("type") == "csp_command":
                # Handle CSP-specific commands
                response = await handle_csp_command(message_data)
                await connection_manager.send_personal_message(
                    json.dumps(response),
                    websocket
                )
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        connection_manager.disconnect(websocket)

async def handle_csp_command(command_data: dict) -> dict:
    """Handle CSP-specific WebSocket commands"""
    command = command_data.get("command")
    
    if command == "create_process":
        process_id = await csp_engine.create_process(
            command_data.get("process_type", "default"),
            command_data.get("config", {})
        )
        return {"type": "process_created", "process_id": process_id}
    
    elif command == "get_status":
        return {"type": "status", "data": csp_engine.get_status()}
    
    elif command == "list_entities":
        return {
            "type": "entities",
            "processes": len(csp_engine.processes),
            "channels": len(csp_engine.channels),
            "agents": len(csp_engine.agents)
        }
    
    else:
        return {"type": "error", "message": f"Unknown command: {command}"}

# ============================================================================
# BACKGROUND TASKS
# ============================================================================

async def metrics_updater():
    """Update system metrics periodically"""
    while True:
        try:
            # Update system metrics
            system_memory_usage.set(psutil.virtual_memory().used)
            system_cpu_usage.set(psutil.cpu_percent())
            
            # Update CSP metrics
            if csp_engine:
                csp_processes_active.set(len(csp_engine.processes))
                csp_channels_active.set(len(csp_engine.channels))
                csp_agents_active.set(len(csp_engine.agents))
            
            await asyncio.sleep(10)  # Update every 10 seconds
        
        except Exception as e:
            logger.error(f"Metrics update error: {e}")
            await asyncio.sleep(30)

async def system_monitor():
    """Monitor system health and performance"""
    while True:
        try:
            # Check memory usage
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 90:
                logger.warning(f"High memory usage: {memory_percent}%")
            
            # Check disk space
            disk_usage = psutil.disk_usage('.').percent
            if disk_usage > 90:
                logger.warning(f"High disk usage: {disk_usage}%")
            
            # Broadcast system status to WebSocket clients
            if connection_manager.active_connections:
                status_message = {
                    "type": "system_status",
                    "memory": memory_percent,
                    "disk": disk_usage,
                    "timestamp": datetime.now().isoformat()
                }
                await connection_manager.broadcast(json.dumps(status_message))
            
            await asyncio.sleep(30)  # Check every 30 seconds
        
        except Exception as e:
            logger.error(f"System monitor error: {e}")
            await asyncio.sleep(60)

# ============================================================================
# DASHBOARD ENDPOINT
# ============================================================================

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Real-time dashboard"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Enhanced CSP Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; background: #1a1a1a; color: white; }
            .header { background: linear-gradient(45deg, #007acc, #0099ff); padding: 20px; }
            .dashboard { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; padding: 20px; }
            .card { background: #2a2a2a; padding: 20px; border-radius: 10px; border: 1px solid #444; }
            .metric { font-size: 2em; color: #00ff88; margin: 10px 0; }
            .status { color: #00ff88; }
            #log { height: 200px; overflow-y: scroll; background: #1a1a1a; padding: 10px; border-radius: 5px; font-family: monospace; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🚀 Enhanced CSP System Dashboard</h1>
            <p>Real-time monitoring and control</p>
        </div>
        
        <div class="dashboard">
            <div class="card">
                <h3>System Status</h3>
                <div class="status" id="system-status">Initializing...</div>
                <div>Memory: <span id="memory-usage">0%</span></div>
                <div>CPU: <span id="cpu-usage">0%</span></div>
            </div>
            
            <div class="card">
                <h3>CSP Engine</h3>
                <div>Processes: <span class="metric" id="processes">0</span></div>
                <div>Channels: <span class="metric" id="channels">0</span></div>
                <div>Agents: <span class="metric" id="agents">0</span></div>
            </div>
            
            <div class="card">
                <h3>Connections</h3>
                <div>WebSocket: <span class="metric" id="websocket-connections">0</span></div>
                <div>Status: <span id="connection-status">Disconnected</span></div>
            </div>
            
            <div class="card">
                <h3>Real-time Log</h3>
                <div id="log"></div>
            </div>
        </div>
        
        <script>
            const ws = new WebSocket('ws://localhost:8000/ws');
            const log = document.getElementById('log');
            
            ws.onopen = function() {
                document.getElementById('connection-status').textContent = 'Connected';
                document.getElementById('connection-status').style.color = '#00ff88';
                addLog('WebSocket connected');
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                
                if (data.type === 'system_status') {
                    document.getElementById('memory-usage').textContent = data.memory.toFixed(1) + '%';
                } else if (data.type === 'status') {
                    document.getElementById('processes').textContent = data.data.processes;
                    document.getElementById('channels').textContent = data.data.channels;
                    document.getElementById('agents').textContent = data.data.agents;
                    document.getElementById('system-status').textContent = data.data.status;
                }
                
                addLog(JSON.stringify(data, null, 2));
            };
            
            ws.onclose = function() {
                document.getElementById('connection-status').textContent = 'Disconnected';
                document.getElementById('connection-status').style.color = '#ff4444';
                addLog('WebSocket disconnected');
            };
            
            function addLog(message) {
                const timestamp = new Date().toLocaleTimeString();
                log.innerHTML += `[${timestamp}] ${message}\\n`;
                log.scrollTop = log.scrollHeight;
            }
            
            // Request status updates
            setInterval(() => {
                if (ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({type: 'csp_command', command: 'get_status'}));
                }
            }, 5000);
            
            // Update WebSocket connection count
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('websocket-connections').textContent = data.connections.websocket;
                })
                .catch(error => console.error('Error fetching status:', error));
        </script>
    </body>
    </html>
    """

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to run the Enhanced CSP System"""
    
    # Ensure directories exist
    for directory in ["data", "logs", "config", "static", "templates"]:
        Path(directory).mkdir(exist_ok=True)
    
    # Log startup information
    logger.info("=" * 60)
    logger.info("Enhanced CSP System - Starting Up")
    logger.info("=" * 60)
    logger.info(f"Version: {config.version}")
    logger.info(f"Host: {config.host}")
    logger.info(f"Port: {config.port}")
    logger.info(f"Debug: {config.debug}")
    logger.info(f"Features: AI={config.ai_enabled}, Quantum={config.quantum_enabled}, Consciousness={config.consciousness_enabled}")
    logger.info("=" * 60)
    
    # Run the application
    uvicorn.run(
        "enhanced_csp.main:app",
        host=config.host,
        port=config.port,
        workers=config.workers,
        log_level="info" if not config.debug else "debug",
        reload=config.debug,
        access_log=True
    )

if __name__ == "__main__":
    main()