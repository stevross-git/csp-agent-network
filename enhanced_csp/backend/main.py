# File: backend/main.py
"""
CSP Visual Designer Backend - Main FastAPI Application
=====================================================
Complete backend implementation with all endpoints and middleware
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
import redis.asyncio as redis

# Import all our backend modules
from backend.database.connection import (
    startup_database, shutdown_database, check_database_health,
    db_manager
)
from backend.auth.auth_system import (
    get_auth_service, AuthenticationService, LoginRequest, RegisterRequest,
    TokenResponse, UserInfo, create_initial_admin, get_current_user, get_current_user_optional,
    require_permission, Permission
)
from backend.components.registry import get_component_registry, get_available_components
from backend.execution.execution_engine import (
    get_execution_engine, execute_design_async, get_execution_metrics, ExecutionConfig
)
from backend.realtime.websocket_manager import (
    websocket_endpoint, connection_manager, init_websocket_manager, 
    shutdown_websocket_manager, broadcast_design_event, EventType
)

# Import API endpoints
from backend.api.endpoints.designs import router as designs_router
from backend.schemas.api_schemas import (
    BaseResponse, ErrorResponse, ExecutionConfig as ExecutionConfigSchema,
    ComponentTypeResponse, ComponentCategoryResponse
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# APPLICATION LIFECYCLE
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("🚀 Starting CSP Visual Designer Backend")
    
    try:
        # Initialize database connections
        await startup_database()
        
        # Initialize WebSocket manager with Redis
        redis_client = await db_manager.get_redis()
        await init_websocket_manager(redis_client)
        
        # Initialize component registry
        component_registry = await get_component_registry()
        
        # Initialize execution engine
        execution_engine = await get_execution_engine()
        
        # Create initial admin user
        await create_initial_admin()
        
        logger.info("✅ Backend startup completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down CSP Visual Designer Backend")
    
    try:
        await shutdown_websocket_manager()
        await shutdown_database()
        logger.info("✅ Backend shutdown completed")
    except Exception as e:
        logger.error(f"❌ Shutdown error: {e}")

# ============================================================================
# FASTAPI APPLICATION SETUP
# ============================================================================

app = FastAPI(
    title="CSP Visual Designer API",
    description="Advanced AI-Powered CSP Process Designer Backend",
    version="2.0.0",
    docs_url=None,  # Custom docs URL
    redoc_url="/redoc",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "authentication", "description": "User authentication and authorization"},
        {"name": "designs", "description": "Visual design management"},
        {"name": "components", "description": "Component registry and metadata"},
        {"name": "execution", "description": "Design execution and monitoring"},
        {"name": "websocket", "description": "Real-time collaboration"},
        {"name": "system", "description": "System health and metrics"}
    ]
)

# ============================================================================
# MIDDLEWARE CONFIGURATION
# ============================================================================

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gzip compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Timestamp"] = str(int(time.time()))
    return response

# Error handling middleware
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Global HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error_code": f"HTTP_{exc.status_code}",
            "message": exc.detail,
            "timestamp": datetime.now().isoformat(),
            "path": request.url.path
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error_code": "INTERNAL_ERROR",
            "message": "An internal server error occurred",
            "timestamp": datetime.now().isoformat(),
            "path": request.url.path
        }
    )

# ============================================================================
# INCLUDE ROUTERS
# ============================================================================

# Include design management router
app.include_router(designs_router)

# ============================================================================
# AUTHENTICATION ENDPOINTS
# ============================================================================

@app.post("/api/auth/register", response_model=UserInfo, tags=["authentication"])
async def register(
    registration: RegisterRequest,
    auth_service: AuthenticationService = Depends(get_auth_service),
    db_session = Depends(lambda: db_manager.get_session())
):
    """Register a new user"""
    return await auth_service.register_user(registration, db_session)

@app.post("/api/auth/login", response_model=TokenResponse, tags=["authentication"])
async def login(
    login_request: LoginRequest,
    auth_service: AuthenticationService = Depends(get_auth_service),
    db_session = Depends(lambda: db_manager.get_session())
):
    """Login and get access tokens"""
    return await auth_service.authenticate_user(login_request, db_session)

@app.post("/api/auth/refresh", response_model=TokenResponse, tags=["authentication"])
async def refresh_token(
    refresh_token: str,
    auth_service: AuthenticationService = Depends(get_auth_service),
    db_session = Depends(lambda: db_manager.get_session())
):
    """Refresh access token"""
    return await auth_service.refresh_token(refresh_token, db_session)

@app.post("/api/auth/logout", response_model=BaseResponse, tags=["authentication"])
async def logout(
    current_user: UserInfo = Depends(get_current_user),
    auth_service: AuthenticationService = Depends(get_auth_service)
):
    """Logout current user"""
    await auth_service.logout_user(current_user.id)
    return BaseResponse(message="Logged out successfully")

@app.get("/api/auth/me", response_model=UserInfo, tags=["authentication"])
async def get_current_user_info(current_user: UserInfo = Depends(get_current_user)):
    """Get current user information"""
    return current_user

# ============================================================================
# COMPONENT ENDPOINTS
# ============================================================================

@app.get("/api/components", response_model=Dict[str, List[ComponentTypeResponse]], tags=["components"])
async def list_components(
    category: Optional[str] = None,
    current_user: Optional[UserInfo] = Depends(get_current_user_optional)
):
    """List all available component types"""
    try:
        components = await get_available_components()
        
        if category:
            return {category: components.get(category, [])}
        
        return components
        
    except Exception as e:
        logger.error(f"Failed to list components: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve components")

@app.get("/api/components/categories", response_model=List[ComponentCategoryResponse], tags=["components"])
async def get_component_categories():
    """Get component categories with counts"""
    try:
        components = await get_available_components()
        
        categories = []
        for category_name, component_list in components.items():
            categories.append(ComponentCategoryResponse(
                category=category_name,
                components=component_list,
                count=len(component_list)
            ))
        
        return categories
        
    except Exception as e:
        logger.error(f"Failed to get component categories: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve component categories")

@app.get("/api/components/{component_type}", response_model=ComponentTypeResponse, tags=["components"])
async def get_component_info(
    component_type: str,
    current_user: Optional[UserInfo] = Depends(get_current_user_optional)
):
    """Get detailed information about a specific component type"""
    try:
        registry = await get_component_registry()
        metadata = registry.get_component_metadata(component_type)
        
        if not metadata:
            raise HTTPException(status_code=404, detail="Component type not found")
        
        return ComponentTypeResponse(
            id=component_type,  # Using type as ID for simplicity
            component_type=metadata.component_type,
            category=metadata.category.value,
            display_name=metadata.display_name,
            description=metadata.description,
            icon=metadata.icon,
            color=metadata.color,
            default_properties=metadata.default_properties,
            input_ports=[port.__dict__ for port in metadata.input_ports],
            output_ports=[port.__dict__ for port in metadata.output_ports],
            implementation_class=metadata.implementation_class,
            is_active=True,
            created_at=datetime.now()  # Would be actual creation time in production
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get component info: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve component information")

# ============================================================================
# EXECUTION ENDPOINTS
# ============================================================================

@app.post("/api/executions", tags=["execution"])
async def start_execution(
    design_id: str,
    config: ExecutionConfigSchema,
    current_user: UserInfo = Depends(require_permission(Permission.EXECUTE_DESIGN))
):
    """Start executing a design"""
    try:
        from uuid import UUID
        design_uuid = UUID(design_id)
        
        execution_session = await execute_design_async(design_uuid, config.dict())
        
        return {
            "success": True,
            "execution_id": execution_session.session_id,
            "design_id": design_id,
            "status": execution_session.status.value,
            "message": "Execution started successfully"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to start execution: {e}")
        raise HTTPException(status_code=500, detail="Failed to start execution")

@app.get("/api/executions/{execution_id}/status", tags=["execution"])
async def get_execution_status(
    execution_id: str,
    current_user: UserInfo = Depends(require_permission(Permission.VIEW_EXECUTION))
):
    """Get execution status"""
    try:
        engine = await get_execution_engine()
        status = engine.get_execution_status(execution_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="Execution not found")
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get execution status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve execution status")

@app.get("/api/executions/{execution_id}/metrics", tags=["execution"])
async def get_execution_metrics_endpoint(
    execution_id: str,
    current_user: UserInfo = Depends(require_permission(Permission.VIEW_EXECUTION))
):
    """Get detailed execution metrics"""
    try:
        metrics = await get_execution_metrics(execution_id)
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get execution metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve execution metrics")

@app.post("/api/executions/{execution_id}/pause", tags=["execution"])
async def pause_execution(
    execution_id: str,
    current_user: UserInfo = Depends(require_permission(Permission.CONTROL_EXECUTION))
):
    """Pause execution"""
    try:
        engine = await get_execution_engine()
        success = await engine.pause_execution(execution_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Execution not found or cannot be paused")
        
        return BaseResponse(message="Execution paused successfully")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to pause execution: {e}")
        raise HTTPException(status_code=500, detail="Failed to pause execution")

@app.post("/api/executions/{execution_id}/resume", tags=["execution"])
async def resume_execution(
    execution_id: str,
    current_user: UserInfo = Depends(require_permission(Permission.CONTROL_EXECUTION))
):
    """Resume execution"""
    try:
        engine = await get_execution_engine()
        success = await engine.resume_execution(execution_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Execution not found or cannot be resumed")
        
        return BaseResponse(message="Execution resumed successfully")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to resume execution: {e}")
        raise HTTPException(status_code=500, detail="Failed to resume execution")

@app.post("/api/executions/{execution_id}/cancel", tags=["execution"])
async def cancel_execution(
    execution_id: str,
    current_user: UserInfo = Depends(require_permission(Permission.CONTROL_EXECUTION))
):
    """Cancel execution"""
    try:
        engine = await get_execution_engine()
        success = await engine.cancel_execution(execution_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Execution not found")
        
        return BaseResponse(message="Execution cancelled successfully")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel execution: {e}")
        raise HTTPException(status_code=500, detail="Failed to cancel execution")

# ============================================================================
# WEBSOCKET ENDPOINTS
# ============================================================================

@app.websocket("/ws/{user_id}")
async def websocket_collaboration(websocket: WebSocket, user_id: str, design_id: Optional[str] = None):
    """WebSocket endpoint for real-time collaboration"""
    await websocket_endpoint(websocket, user_id, design_id)

@app.get("/api/websocket/stats", tags=["websocket"])
async def get_websocket_stats(
    current_user: UserInfo = Depends(require_permission(Permission.VIEW_METRICS))
):
    """Get WebSocket connection statistics"""
    try:
        stats = await connection_manager.get_connection_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get WebSocket stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve WebSocket statistics")

# ============================================================================
# SYSTEM ENDPOINTS
# ============================================================================

@app.get("/", response_model=Dict[str, Any], tags=["system"])
async def root():
    """Root endpoint with system information"""
    return {
        "name": "CSP Visual Designer API",
        "version": "2.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "features": {
            "authentication": True,
            "real_time_collaboration": True,
            "ai_integration": True,
            "component_registry": True,
            "execution_engine": True,
            "performance_monitoring": True
        },
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json",
            "websocket": "/ws/{user_id}",
            "health": "/health"
        }
    }

@app.get("/health", response_model=Dict[str, Any], tags=["system"])
async def health_check():
    """System health check"""
    try:
        # Check database health
        db_health = await check_database_health()
        
        # Check component registry
        registry = await get_component_registry()
        component_count = len(registry.get_all_components())
        
        # Check WebSocket connections
        ws_stats = await connection_manager.get_connection_stats()
        
        # Overall health status
        overall_status = "healthy"
        if db_health["database"]["status"] != "healthy" or db_health["redis"]["status"] != "healthy":
            overall_status = "degraded"
        
        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.0",
            "uptime": "calculated_by_process_manager",
            "components": {
                "database": db_health,
                "component_registry": {
                    "status": "healthy",
                    "component_count": component_count
                },
                "websocket_manager": {
                    "status": "healthy",
                    "active_connections": ws_stats["active_connections"]
                },
                "execution_engine": {
                    "status": "healthy"
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@app.get("/metrics", tags=["system"])
async def get_system_metrics(
    current_user: UserInfo = Depends(require_permission(Permission.VIEW_METRICS))
):
    """Get detailed system metrics"""
    try:
        import psutil
        
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Database metrics
        db_health = await check_database_health()
        
        # WebSocket metrics
        ws_stats = await connection_manager.get_connection_stats()
        
        # Component metrics
        registry = await get_component_registry()
        components_by_category = {}
        for comp_type, metadata in registry.get_all_components().items():
            category = metadata.category.value
            components_by_category[category] = components_by_category.get(category, 0) + 1
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": round(memory.used / (1024**3), 2),
                "memory_total_gb": round(memory.total / (1024**3), 2),
                "disk_percent": disk.percent,
                "disk_used_gb": round(disk.used / (1024**3), 2),
                "disk_total_gb": round(disk.total / (1024**3), 2)
            },
            "database": db_health,
            "websocket": ws_stats,
            "components": {
                "total_components": len(registry.get_all_components()),
                "by_category": components_by_category
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system metrics")

# ============================================================================
# CUSTOM DOCUMENTATION
# ============================================================================

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Custom Swagger UI"""
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="CSP Visual Designer API Documentation",
        swagger_favicon_url="/static/favicon.ico"
    )

def custom_openapi():
    """Custom OpenAPI schema"""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="CSP Visual Designer API",
        version="2.0.0",
        description="""
        ## Advanced AI-Powered CSP Process Designer Backend

        This API provides comprehensive functionality for building, executing, and monitoring
        CSP (Communicating Sequential Processes) designs through a visual interface.

        ### Key Features
        - **Visual Design Management**: Create, edit, and manage process designs
        - **Real-time Collaboration**: Multi-user editing with WebSocket support
        - **Component Registry**: Extensible library of AI, data, and security components
        - **Execution Engine**: Run designs with monitoring and metrics
        - **Authentication & Authorization**: JWT-based security with role-based access
        - **Performance Monitoring**: Comprehensive metrics and health checks

        ### Authentication
        Most endpoints require authentication. Use the `/api/auth/login` endpoint to obtain
        an access token, then include it in the `Authorization` header as `Bearer <token>`.

        ### WebSocket Collaboration
        Connect to `/ws/{user_id}?design_id={design_id}` for real-time collaboration features.
        """,
        routes=app.routes,
    )
    
    openapi_schema["info"]["x-logo"] = {
        "url": "/static/logo.png"
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# ============================================================================
# STATIC FILES (Optional)
# ============================================================================

# Mount static files for documentation assets
import os
static_dir = "backend/static"
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# ============================================================================
# DEVELOPMENT SERVER
# ============================================================================

if __name__ == "__main__":
    # Development server configuration
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True,
        reload_dirs=["backend"],
        reload_excludes=["*.pyc", "__pycache__", ".git"]
    )
