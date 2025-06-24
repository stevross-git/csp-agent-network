# File: backend/main.py
"""
CSP Visual Designer Backend - Main FastAPI Application with Azure AD
=====================================================
Complete backend implementation with all endpoints, middleware, and Azure AD authentication
"""

import asyncio
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.security import HTTPBearer
import redis.asyncio as redis
import httpx
import jwt
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# ENVIRONMENT CONFIGURATION
# ============================================================================

AZURE_TENANT_ID = os.getenv("AZURE_TENANT_ID", "622a5fe0-fac1-4213-9cf7-d5f6defdf4c4")
AZURE_CLIENT_ID = os.getenv("AZURE_CLIENT_ID", "53537e30-ae6b-48f7-9c7c-4db20fc27850")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:3001").split(",")

# ============================================================================
# AZURE AD AUTHENTICATION CLASSES
# ============================================================================

class AzureUserInfo(BaseModel):
    """Azure AD user information model"""
    user_id: str  # oid from Azure AD
    email: str
    name: str
    tenant_id: str
    roles: List[str] = []
    groups: List[str] = []
    app_roles: List[str] = []

class MinimalAzureValidator:
    """Azure AD token validator with caching"""
    
    def __init__(self, tenant_id: str, client_id: str):
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.jwks_uri = f"https://login.microsoftonline.com/{tenant_id}/discovery/v2.0/keys"
        self.issuer = f"https://login.microsoftonline.com/{tenant_id}/v2.0"
        self._keys_cache = None

    async def get_signing_keys(self):
        """Get Azure AD signing keys"""
        if self._keys_cache:
            return self._keys_cache
            
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.get(self.jwks_uri)
                response.raise_for_status()
                self._keys_cache = response.json()
                return self._keys_cache
            except Exception as e:
                logger.error(f"Failed to fetch Azure AD keys: {e}")
                raise HTTPException(503, "Authentication service unavailable")

    async def validate_token(self, token: str) -> AzureUserInfo:
        """Validate Azure AD token"""
        try:
            # Get header to find key
            header = jwt.get_unverified_header(token)
            kid = header.get('kid')
            
            # Get signing keys
            keys_data = await self.get_signing_keys()
            keys = keys_data.get('keys', [])
            
            # Find matching key
            key = next((k for k in keys if k.get('kid') == kid), None)
            if not key:
                raise HTTPException(401, "Invalid token - key not found")
            
            # Convert JWK to PEM
            public_key = jwt.algorithms.RSAAlgorithm.from_jwk(key)
            
            # Decode token
            decoded = jwt.decode(
                token,
                public_key,
                algorithms=['RS256'],
                audience=self.client_id,
                issuer=self.issuer
            )
            
            # Extract user info and map roles
            user_roles = decoded.get('roles', [])
            email = decoded.get('preferred_username', '') or decoded.get('upn', '')
            
            # Map email domains to roles if no Azure AD roles
            if not user_roles:
                user_roles = self._map_email_to_roles(email)
            
            return AzureUserInfo(
                user_id=decoded.get('oid', ''),
                email=email,
                name=decoded.get('name', ''),
                tenant_id=decoded.get('tid', ''),
                roles=user_roles,
                app_roles=user_roles
            )
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(401, "Token expired")
        except jwt.InvalidTokenError as e:
            raise HTTPException(401, f"Invalid token: {str(e)}")
        except Exception as e:
            logger.error(f"Token validation error: {e}")
            raise HTTPException(500, "Authentication error")
    
    def _map_email_to_roles(self, email: str) -> List[str]:
        """Map email to default roles"""
        if 'admin@' in email:
            return ['super_admin', 'admin', 'developer', 'user']
        elif 'dev@' in email or 'developer@' in email:
            return ['developer', 'user']
        elif 'analyst@' in email:
            return ['analyst', 'user']
        else:
            return ['user']

# Global Azure AD validator
azure_validator = MinimalAzureValidator(AZURE_TENANT_ID, AZURE_CLIENT_ID)
security = HTTPBearer()

# ============================================================================
# AUTHENTICATION DEPENDENCIES
# ============================================================================

async def get_current_user_azure(credentials: HTTPBearer = Depends(security)) -> AzureUserInfo:
    """Get current authenticated user via Azure AD"""
    return await azure_validator.validate_token(credentials.credentials)

async def get_current_user_optional_azure(request: Request) -> Optional[AzureUserInfo]:
    """Get current user if authenticated (optional)"""
    auth_header = request.headers.get('Authorization')
    
    if not auth_header or not auth_header.startswith('Bearer '):
        return None
    
    try:
        token = auth_header.split(' ')[1]
        return await azure_validator.validate_token(token)
    except Exception:
        return None

def require_role_azure(required_roles: List[str]):
    """Decorator to require specific roles"""
    def role_checker(user: AzureUserInfo = Depends(get_current_user_azure)) -> AzureUserInfo:
        user_roles = user.app_roles + user.roles
        
        if not any(role in user_roles for role in required_roles):
            raise HTTPException(
                403, 
                f"Insufficient permissions. Required roles: {required_roles}"
            )
        
        return user
    
    return role_checker

def require_permission_azure(permission: str):
    """Decorator to require specific permission"""
    permission_to_roles = {
        'read': ['user', 'analyst', 'developer', 'admin', 'super_admin'],
        'write': ['developer', 'admin', 'super_admin'],
        'execute': ['developer', 'admin', 'super_admin'],
        'admin': ['admin', 'super_admin'],
        'super_admin': ['super_admin']
    }
    
    required_roles = permission_to_roles.get(permission, [])
    return require_role_azure(required_roles)

# ============================================================================
# FALLBACK IMPORTS AND CLASSES
# ============================================================================

# Try to import existing modules, fallback to placeholders if not available
try:
    from backend.database.connection import (
        startup_database, shutdown_database, check_database_health,
        db_manager
    )
    DATABASE_AVAILABLE = True
except ImportError:
    logger.warning("Database modules not available, using fallbacks")
    DATABASE_AVAILABLE = False
    
    async def startup_database():
        logger.info("Database startup (placeholder)")
    
    async def shutdown_database():
        logger.info("Database shutdown (placeholder)")
    
    async def check_database_health():
        return {
            "database": {"status": "not_configured"},
            "redis": {"status": "not_configured"}
        }
    
    class MockDBManager:
        async def get_redis(self):
            return None
        
        async def get_session(self):
            return None
    
    db_manager = MockDBManager()

try:
    from backend.auth.auth_system import (
        get_auth_service, AuthenticationService, LoginRequest, RegisterRequest,
        TokenResponse, UserInfo, create_initial_admin, get_current_user, get_current_user_optional,
        require_permission, Permission
    )
    AUTH_SYSTEM_AVAILABLE = True
except ImportError:
    logger.warning("Auth system modules not available, using Azure AD only")
    AUTH_SYSTEM_AVAILABLE = False
    
    # Placeholder classes
    class UserInfo(BaseModel):
        id: str
        username: str
        email: str
        roles: List[str] = []
    
    class LoginRequest(BaseModel):
        username: str
        password: str
    
    class RegisterRequest(BaseModel):
        username: str
        email: str
        password: str
    
    class TokenResponse(BaseModel):
        access_token: str
        token_type: str = "bearer"
    
    class Permission:
        VIEW_DESIGN = "view_design"
        EDIT_DESIGN = "edit_design"
        DELETE_DESIGN = "delete_design"
        EXECUTE_DESIGN = "execute_design"
        VIEW_EXECUTION = "view_execution"
        CONTROL_EXECUTION = "control_execution"
        VIEW_METRICS = "view_metrics"
    
    async def create_initial_admin():
        pass
    
    def get_current_user():
        return get_current_user_azure
    
    def get_current_user_optional():
        return get_current_user_optional_azure
    
    def require_permission(permission: str):
        return require_permission_azure(permission)

try:
    from backend.components.registry import get_component_registry, get_available_components
    COMPONENTS_AVAILABLE = True
except ImportError:
    logger.warning("Component registry not available, using placeholders")
    COMPONENTS_AVAILABLE = False
    
    async def get_component_registry():
        class MockRegistry:
            def get_component_metadata(self, component_type):
                return None
            def get_all_components(self):
                return {}
        return MockRegistry()
    
    async def get_available_components():
        return {
            "AI": [
                {
                    "type": "ai_agent",
                    "name": "AI Agent",
                    "description": "Intelligent processing agent",
                    "icon": "🤖"
                }
            ],
            "Data": [
                {
                    "type": "data_processor", 
                    "name": "Data Processor",
                    "description": "Process and transform data",
                    "icon": "⚡"
                }
            ]
        }

try:
    from backend.execution.execution_engine import (
        get_execution_engine, execute_design_async, get_execution_metrics, ExecutionConfig
    )
    EXECUTION_AVAILABLE = True
except ImportError:
    logger.warning("Execution engine not available, using placeholders")
    EXECUTION_AVAILABLE = False
    
    class ExecutionConfig(BaseModel):
        timeout: int = 300
        max_workers: int = 4
    
    async def get_execution_engine():
        class MockEngine:
            def get_execution_status(self, execution_id):
                return {"status": "running", "progress": 50}
            async def pause_execution(self, execution_id):
                return True
            async def resume_execution(self, execution_id):
                return True
            async def cancel_execution(self, execution_id):
                return True
        return MockEngine()
    
    async def execute_design_async(design_id, config):
        class MockSession:
            session_id = f"exec_{datetime.now().timestamp()}"
            status = type('Status', (), {'value': 'running'})()
        return MockSession()
    
    async def get_execution_metrics(execution_id):
        return {"execution_id": execution_id, "metrics": {}}

try:
    from backend.realtime.websocket_manager import (
        websocket_endpoint, connection_manager, init_websocket_manager, 
        shutdown_websocket_manager, broadcast_design_event, EventType
    )
    WEBSOCKET_AVAILABLE = True
except ImportError:
    logger.warning("WebSocket manager not available, using placeholders")
    WEBSOCKET_AVAILABLE = False
    
    async def websocket_endpoint(websocket, user_id, design_id=None):
        await websocket.accept()
        try:
            while True:
                data = await websocket.receive_text()
                await websocket.send_text(f"Echo: {data}")
        except WebSocketDisconnect:
            pass
    
    class MockConnectionManager:
        async def get_connection_stats(self):
            return {"active_connections": 0}
    
    connection_manager = MockConnectionManager()
    
    async def init_websocket_manager(redis_client):
        pass
    
    async def shutdown_websocket_manager():
        pass

try:
    from backend.api.endpoints.designs import router as designs_router
    DESIGNS_ROUTER_AVAILABLE = True
except ImportError:
    logger.warning("Designs router not available, using placeholder")
    DESIGNS_ROUTER_AVAILABLE = False
    from fastapi import APIRouter
    designs_router = APIRouter()

try:
    from backend.schemas.api_schemas import (
        BaseResponse, ErrorResponse, ExecutionConfig as ExecutionConfigSchema,
        ComponentTypeResponse, ComponentCategoryResponse
    )
    SCHEMAS_AVAILABLE = True
except ImportError:
    logger.warning("API schemas not available, using placeholders")
    SCHEMAS_AVAILABLE = False
    
    class BaseResponse(BaseModel):
        message: str
        success: bool = True
    
    class ErrorResponse(BaseModel):
        error: str
        success: bool = False
    
    class ExecutionConfigSchema(BaseModel):
        timeout: int = 300
        max_workers: int = 4
    
    class ComponentTypeResponse(BaseModel):
        id: str
        component_type: str
        category: str
        display_name: str
        description: str
        icon: str
        color: str = "#6B73FF"
        default_properties: Dict[str, Any] = {}
        input_ports: List[Dict[str, Any]] = []
        output_ports: List[Dict[str, Any]] = []
        implementation_class: str = ""
        is_active: bool = True
        created_at: datetime
    
    class ComponentCategoryResponse(BaseModel):
        category: str
        components: List[Dict[str, Any]]
        count: int

# ============================================================================
# APPLICATION LIFECYCLE
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("🚀 Starting CSP Visual Designer Backend with Azure AD")
    
    try:
        # Initialize database connections if available
        if DATABASE_AVAILABLE:
            await startup_database()
            
            # Initialize WebSocket manager with Redis
            redis_client = await db_manager.get_redis()
            if WEBSOCKET_AVAILABLE:
                await init_websocket_manager(redis_client)
        
        # Initialize component registry if available
        if COMPONENTS_AVAILABLE:
            component_registry = await get_component_registry()
        
        # Initialize execution engine if available
        if EXECUTION_AVAILABLE:
            execution_engine = await get_execution_engine()
        
        # Create initial admin user if auth system available
        if AUTH_SYSTEM_AVAILABLE:
            await create_initial_admin()
        
        # Startup logging
        logger.info("🚀 Enhanced CSP System Backend Starting...")
        logger.info(f"📋 Azure AD Tenant: {AZURE_TENANT_ID}")
        logger.info(f"🔑 Azure AD Client: {AZURE_CLIENT_ID}")
        logger.info(f"🌐 Allowed Origins: {ALLOWED_ORIGINS}")
        logger.info("✅ Azure AD authentication initialized")
        logger.info("🌐 API documentation: http://localhost:8000/docs")
        logger.info("✅ Backend startup completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        # Don't raise in minimal mode, just log the error
        logger.warning("Continuing with minimal functionality...")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down CSP Visual Designer Backend")
    
    try:
        if WEBSOCKET_AVAILABLE:
            await shutdown_websocket_manager()
        if DATABASE_AVAILABLE:
            await shutdown_database()
        logger.info("✅ Backend shutdown completed")
    except Exception as e:
        logger.error(f"❌ Shutdown error: {e}")

# ============================================================================
# FASTAPI APPLICATION SETUP
# ============================================================================

app = FastAPI(
    title="CSP Visual Designer API",
    description="Advanced AI-Powered CSP Process Designer Backend with Azure AD",
    version="2.1.0",
    docs_url=None,  # Custom docs URL
    redoc_url="/redoc",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "authentication", "description": "Azure AD authentication and authorization"},
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
    allow_origins=ALLOWED_ORIGINS,
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

# Include design management router if available
if DESIGNS_ROUTER_AVAILABLE:
    app.include_router(designs_router)

# ============================================================================
# AZURE AD AUTHENTICATION ENDPOINTS
# ============================================================================

@app.get("/api/auth/info", tags=["authentication"])
async def get_auth_info():
    """Get Azure AD authentication configuration (public)"""
    return {
        "auth_type": "azure_ad",
        "tenant_id": AZURE_TENANT_ID,
        "client_id": AZURE_CLIENT_ID,
        "authority": f"https://login.microsoftonline.com/{AZURE_TENANT_ID}",
        "scopes": ["User.Read", "User.ReadBasic.All", "Group.Read.All"]
    }

@app.get("/api/auth/me", tags=["authentication"])
async def get_current_user_info_azure(
    current_user: AzureUserInfo = Depends(get_current_user_azure)
) -> Dict[str, Any]:
    """Get current authenticated user information"""
    return {
        "user_id": current_user.user_id,
        "email": current_user.email,
        "name": current_user.name,
        "roles": current_user.app_roles,
        "groups": current_user.groups,
        "tenant_id": current_user.tenant_id,
        "auth_method": "azure_ad"
    }

@app.get("/api/auth/permissions", tags=["authentication"])
async def get_user_permissions_azure(
    current_user: AzureUserInfo = Depends(get_current_user_azure)
) -> Dict[str, Any]:
    """Get current user's permissions"""
    # Map roles to permissions
    role_permissions = {
        'super_admin': ['read', 'write', 'execute', 'admin', 'super_admin'],
        'admin': ['read', 'write', 'execute', 'admin'],
        'developer': ['read', 'write', 'execute'],
        'analyst': ['read'],
        'user': ['read']
    }
    
    permissions = set()
    for role in current_user.app_roles + current_user.roles:
        permissions.update(role_permissions.get(role, []))
    
    return {
        "user_id": current_user.user_id,
        "permissions": list(permissions),
        "roles": current_user.app_roles
    }

@app.post("/api/auth/logout", tags=["authentication"])
async def logout_azure(current_user: AzureUserInfo = Depends(get_current_user_azure)):
    """Logout endpoint (client-side logout mainly for Azure AD)"""
    return {"message": "Logout successful", "user_id": current_user.user_id}

# ============================================================================
# LEGACY AUTHENTICATION ENDPOINTS (if auth system available)
# ============================================================================

if AUTH_SYSTEM_AVAILABLE:
    @app.post("/api/auth/register", response_model=UserInfo, tags=["authentication"])
    async def register(
        registration: RegisterRequest,
        auth_service: AuthenticationService = Depends(get_auth_service),
        db_session = Depends(lambda: db_manager.get_session())
    ):
        """Register a new user (legacy system)"""
        return await auth_service.register_user(registration, db_session)

    @app.post("/api/auth/login", response_model=TokenResponse, tags=["authentication"])
    async def login(
        login_request: LoginRequest,
        auth_service: AuthenticationService = Depends(get_auth_service),
        db_session = Depends(lambda: db_manager.get_session())
    ):
        """Login and get access tokens (legacy system)"""
        return await auth_service.authenticate_user(login_request, db_session)

    @app.post("/api/auth/refresh", response_model=TokenResponse, tags=["authentication"])
    async def refresh_token(
        refresh_token: str,
        auth_service: AuthenticationService = Depends(get_auth_service),
        db_session = Depends(lambda: db_manager.get_session())
    ):
        """Refresh access token (legacy system)"""
        return await auth_service.refresh_token(refresh_token, db_session)

# ============================================================================
# COMPONENT ENDPOINTS
# ============================================================================

@app.get("/api/components", response_model=Dict[str, List], tags=["components"])
async def list_components(
    category: Optional[str] = None,
    current_user: Optional[AzureUserInfo] = Depends(get_current_user_optional_azure)
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
    current_user: Optional[AzureUserInfo] = Depends(get_current_user_optional_azure)
):
    """Get detailed information about a specific component type"""
    try:
        registry = await get_component_registry()
        metadata = registry.get_component_metadata(component_type)
        
        if not metadata and COMPONENTS_AVAILABLE:
            raise HTTPException(status_code=404, detail="Component type not found")
        
        # Fallback for when component registry isn't available
        if not metadata:
            return ComponentTypeResponse(
                id=component_type,
                component_type=component_type,
                category="AI",
                display_name=component_type.replace("_", " ").title(),
                description=f"Component of type {component_type}",
                icon="🔧",
                color="#6B73FF",
                default_properties={},
                input_ports=[],
                output_ports=[],
                implementation_class="",
                is_active=True,
                created_at=datetime.now()
            )
        
        return ComponentTypeResponse(
            id=component_type,
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
            created_at=datetime.now()
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
    current_user: AzureUserInfo = Depends(require_permission_azure("execute"))
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
            "status": execution_session.status.value if hasattr(execution_session.status, 'value') else str(execution_session.status),
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
    current_user: AzureUserInfo = Depends(require_permission_azure("read"))
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
    current_user: AzureUserInfo = Depends(require_permission_azure("read"))
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
    current_user: AzureUserInfo = Depends(require_permission_azure("execute"))
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
    current_user: AzureUserInfo = Depends(require_permission_azure("execute"))
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
    current_user: AzureUserInfo = Depends(require_permission_azure("execute"))
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
# DESIGN ENDPOINTS (if not available via router)
# ============================================================================

if not DESIGNS_ROUTER_AVAILABLE:
    @app.get("/api/designs", tags=["designs"])
    async def get_designs(
        current_user: AzureUserInfo = Depends(get_current_user_azure)
    ):
        """Get user's designs (placeholder implementation)"""
        return [
            {
                "id": "sample_design_1",
                "name": "Sample CSP Design",
                "description": "A sample design for testing",
                "owner": current_user.email,
                "created_at": datetime.utcnow().isoformat(),
                "status": "draft"
            }
        ]

    @app.post("/api/designs", tags=["designs"])
    async def create_design(
        design_data: Dict[str, Any],
        current_user: AzureUserInfo = Depends(require_permission_azure("write"))
    ):
        """Create new design (placeholder implementation)"""
        return {
            "id": f"design_{datetime.utcnow().timestamp()}",
            "name": design_data.get("name", "Untitled Design"),
            "description": design_data.get("description", ""),
            "owner": current_user.email,
            "created_at": datetime.utcnow().isoformat(),
            "status": "draft",
            "message": "Design created successfully"
        }

# ============================================================================
# WEBSOCKET ENDPOINTS
# ============================================================================

@app.websocket("/ws/{user_id}")
async def websocket_collaboration(websocket: WebSocket, user_id: str, design_id: Optional[str] = None):
    """WebSocket endpoint for real-time collaboration"""
    await websocket_endpoint(websocket, user_id, design_id)

@app.get("/api/websocket/stats", tags=["websocket"])
async def get_websocket_stats(
    current_user: AzureUserInfo = Depends(require_permission_azure("read"))
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
        "version": "2.1.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "auth_method": "azure_ad",
        "features": {
            "azure_ad_authentication": True,
            "real_time_collaboration": WEBSOCKET_AVAILABLE,
            "ai_integration": True,
            "component_registry": COMPONENTS_AVAILABLE,
            "execution_engine": EXECUTION_AVAILABLE,
            "performance_monitoring": True,
            "database": DATABASE_AVAILABLE
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
        # Check database health if available
        if DATABASE_AVAILABLE:
            db_health = await check_database_health()
        else:
            db_health = {
                "database": {"status": "not_configured"},
                "redis": {"status": "not_configured"}
            }
        
        # Check component registry if available
        if COMPONENTS_AVAILABLE:
            registry = await get_component_registry()
            component_count = len(registry.get_all_components())
        else:
            component_count = 0
        
        # Check WebSocket connections
        ws_stats = await connection_manager.get_connection_stats()
        
        # Check Azure AD connectivity
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(azure_validator.jwks_uri)
                azure_ad_status = "healthy" if response.status_code == 200 else "degraded"
        except Exception:
            azure_ad_status = "unhealthy"
        
        # Overall health status
        overall_status = "healthy"
        if DATABASE_AVAILABLE and (db_health["database"]["status"] != "healthy" or db_health["redis"]["status"] != "healthy"):
            overall_status = "degraded"
        if azure_ad_status == "unhealthy":
            overall_status = "degraded"
        
        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "version": "2.1.0",
            "uptime": "calculated_by_process_manager",
            "components": {
                "database": db_health,
                "azure_ad": {
                    "status": azure_ad_status,
                    "tenant_id": AZURE_TENANT_ID
                },
                "component_registry": {
                    "status": "healthy" if COMPONENTS_AVAILABLE else "not_configured",
                    "component_count": component_count
                },
                "websocket_manager": {
                    "status": "healthy" if WEBSOCKET_AVAILABLE else "not_configured",
                    "active_connections": ws_stats.get("active_connections", 0)
                },
                "execution_engine": {
                    "status": "healthy" if EXECUTION_AVAILABLE else "not_configured"
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
    current_user: AzureUserInfo = Depends(require_permission_azure("read"))
):
    """Get detailed system metrics"""
    try:
        # System metrics
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            system_metrics = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": round(memory.used / (1024**3), 2),
                "memory_total_gb": round(memory.total / (1024**3), 2),
                "disk_percent": disk.percent,
                "disk_used_gb": round(disk.used / (1024**3), 2),
                "disk_total_gb": round(disk.total / (1024**3), 2)
            }
        except ImportError:
            system_metrics = {"status": "psutil not available"}
        
        # Database metrics
        if DATABASE_AVAILABLE:
            db_health = await check_database_health()
        else:
            db_health = {"status": "not_configured"}
        
        # WebSocket metrics
        ws_stats = await connection_manager.get_connection_stats()
        
        # Component metrics
        if COMPONENTS_AVAILABLE:
            registry = await get_component_registry()
            components_by_category = {}
            for comp_type, metadata in registry.get_all_components().items():
                category = metadata.category.value
                components_by_category[category] = components_by_category.get(category, 0) + 1
            total_components = len(registry.get_all_components())
        else:
            components_by_category = {}
            total_components = 0
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system": system_metrics,
            "database": db_health,
            "websocket": ws_stats,
            "azure_ad": {
                "tenant_id": AZURE_TENANT_ID,
                "client_id": AZURE_CLIENT_ID
            },
            "components": {
                "total_components": total_components,
                "by_category": components_by_category
            },
            "features": {
                "database_available": DATABASE_AVAILABLE,
                "components_available": COMPONENTS_AVAILABLE,
                "execution_available": EXECUTION_AVAILABLE,
                "websocket_available": WEBSOCKET_AVAILABLE,
                "auth_system_available": AUTH_SYSTEM_AVAILABLE
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
        swagger_favicon_url="/static/favicon.ico" if os.path.exists("backend/static/favicon.ico") else None
    )

def custom_openapi():
    """Custom OpenAPI schema"""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="CSP Visual Designer API",
        version="2.1.0",
        description="""
        ## Advanced AI-Powered CSP Process Designer Backend with Azure AD

        This API provides comprehensive functionality for building, executing, and monitoring
        CSP (Communicating Sequential Processes) designs through a visual interface.

        ### Key Features
        - **Azure AD Authentication**: Secure authentication using Microsoft Azure AD
        - **Visual Design Management**: Create, edit, and manage process designs
        - **Real-time Collaboration**: Multi-user editing with WebSocket support
        - **Component Registry**: Extensible library of AI, data, and security components
        - **Execution Engine**: Run designs with monitoring and metrics
        - **Performance Monitoring**: Comprehensive metrics and health checks

        ### Authentication
        This API uses Azure AD authentication. Include your Azure AD access token in the 
        `Authorization` header as `Bearer <token>`.

        ### WebSocket Collaboration
        Connect to `/ws/{user_id}?design_id={design_id}` for real-time collaboration features.
        """,
        routes=app.routes,
    )
    
    # Add Azure AD security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "AzureADBearer": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "Azure AD access token"
        }
    }
    
    # Apply security to all endpoints except public ones
    for path, path_item in openapi_schema["paths"].items():
        if not any(path.startswith(public) for public in ["/health", "/api/auth/info", "/docs", "/redoc", "/openapi.json"]):
            for method, operation in path_item.items():
                if method in ["get", "post", "put", "delete", "patch"]:
                    operation["security"] = [{"AzureADBearer": []}]
    
    if not os.path.exists("backend/static/logo.png"):
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
static_dir = "backend/static"
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# ============================================================================
# STARTUP MESSAGE (moved to lifespan)
# ============================================================================
# Startup logging is now handled in the lifespan context manager above

# ============================================================================
# DEVELOPMENT SERVER
# ============================================================================

if __name__ == "__main__":
    print("🚀 Starting Enhanced CSP System Backend with Azure AD...")
    print("=" * 60)
    print(f"🔗 API: http://localhost:8000")
    print(f"📚 Docs: http://localhost:8000/docs")
    print(f"🔍 Health: http://localhost:8000/health")
    print(f"🔐 Auth Info: http://localhost:8000/api/auth/info")
    print("=" * 60)
    
    # Development server configuration
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True,
        reload_dirs=["backend"] if os.path.exists("backend") else ["."],
        reload_excludes=["*.pyc", "__pycache__", ".git"]
    )