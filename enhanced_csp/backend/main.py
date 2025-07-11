# backend/main.py

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
from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, Request, Response, BackgroundTasks
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
from backend.monitoring.monitoring_integration import integrate_enhanced_monitoring
from pydantic import BaseModel, EmailStr

# Configure logging first
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
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:3001,http://localhost:8000").split(",")

# Local authentication configuration
LOCAL_AUTH_SECRET_KEY = os.getenv("LOCAL_AUTH_SECRET_KEY", "your-secret-key-here-change-in-production")
LOCAL_JWT_ALGORITHM = "HS256"
LOCAL_ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours
LOCAL_REFRESH_TOKEN_EXPIRE_DAYS = 30

# ============================================================================
# AI COORDINATION SYSTEM INTEGRATION
# ============================================================================
from backend.api.endpoints.memory import router as memory_router


try:
    from backend.api.endpoints.ai_coordination import router as ai_coordination_router
    from backend.ai.ai_coordination_engine import coordination_engine
    AI_COORDINATION_AVAILABLE = True
    logger.info("✅ AI Coordination system available")
except ImportError as e:
    logger.warning(f"AI Coordination system not available: {e}")
    AI_COORDINATION_AVAILABLE = False
    from fastapi import APIRouter
    ai_coordination_router = APIRouter()

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

# ============================================================================
# LOCAL AUTHENTICATION CLASSES
# ============================================================================

class LocalUserInfo(BaseModel):
    """Local user information model"""
    user_id: str
    email: str
    name: str
    roles: List[str] = []
    is_active: bool = True
    is_email_verified: bool = False
    auth_method: str = "local"

class UnifiedUserInfo(BaseModel):
    """Unified user info from both Azure AD and local auth"""
    user_id: str
    email: str
    name: str
    roles: List[str]
    auth_method: str  # 'azure' or 'local'
    tenant_id: Optional[str] = None
    groups: List[str] = []
    is_active: bool = True
    is_email_verified: Optional[bool] = None

class LocalAuthSchemas:
    """Local authentication schemas"""
    
    class UserRegistration(BaseModel):
        email: EmailStr
        password: str
        confirm_password: str
        full_name: str
        
        def validate_password_strength(self):
            """Validate password strength"""
            if len(self.password) < 8:
                raise ValueError('Password must be at least 8 characters long')
            if not any(c.isupper() for c in self.password):
                raise ValueError('Password must contain at least one uppercase letter')
            if not any(c.islower() for c in self.password):
                raise ValueError('Password must contain at least one lowercase letter')
            if not any(c.isdigit() for c in self.password):
                raise ValueError('Password must contain at least one digit')
            if not any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in self.password):
                raise ValueError('Password must contain at least one special character')
            if self.password != self.confirm_password:
                raise ValueError('Passwords do not match')
            return True

    class UserLogin(BaseModel):
        email: EmailStr
        password: str
        remember_me: bool = False

    class TokenRefresh(BaseModel):
        refresh_token: str

    class TokenResponse(BaseModel):
        access_token: str
        refresh_token: str
        token_type: str = "bearer"
        expires_in: int
        user: Dict[str, Any]

    class UserUpdate(BaseModel):
        full_name: Optional[str] = None
        password: Optional[str] = None
        roles: Optional[List[str]] = None
        is_active: Optional[bool] = None

class LocalAuthService:
    """Minimal local authentication service"""
    
    def __init__(self):
        # In-memory user storage (replace with database in production)
        self.users = {}
        self.refresh_tokens = {}
        
        # Create default admin user
        self._create_default_admin()
    
    def _create_default_admin(self):
        """Create default admin user"""
        admin_user = {
            "id": "local_admin_001",
            "email": "admin@csp-system.com",
            "password_hash": self._hash_password("AdminPass123!"),
            "full_name": "System Administrator",
            "roles": ["super_admin", "admin", "developer", "user"],
            "is_active": True,
            "is_email_verified": True,
            "created_at": datetime.now()
        }
        self.users["admin@csp-system.com"] = admin_user
        logger.info("Created default admin user: admin@csp-system.com / AdminPass123!")
    
    def _hash_password(self, password: str) -> str:
        """Hash password (use proper hashing in production)"""
        import hashlib
        return hashlib.pbkdf2_hmac('sha256', password.encode(), b'salt', 100000).hex()
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password"""
        return self._hash_password(password) == password_hash
    
    def _create_token(self, user_data: Dict[str, Any], token_type: str = "access") -> str:
        """Create JWT token"""
        from datetime import timedelta
        
        payload = {
            "sub": user_data["id"],
            "email": user_data["email"],
            "roles": user_data["roles"],
            "auth_method": "local",
            "type": token_type
        }
        
        if token_type == "access":
            payload["exp"] = datetime.utcnow() + timedelta(minutes=LOCAL_ACCESS_TOKEN_EXPIRE_MINUTES)
        else:  # refresh
            payload["exp"] = datetime.utcnow() + timedelta(days=LOCAL_REFRESH_TOKEN_EXPIRE_DAYS)
        
        return jwt.encode(payload, LOCAL_AUTH_SECRET_KEY, algorithm=LOCAL_JWT_ALGORITHM)
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, LOCAL_AUTH_SECRET_KEY, algorithms=[LOCAL_JWT_ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(401, "Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(401, "Invalid token")
    
    async def register_user(self, registration: LocalAuthSchemas.UserRegistration) -> LocalUserInfo:
        """Register new user"""
        # Validate password
        registration.validate_password_strength()
        
        # Check if user exists
        if registration.email in self.users:
            raise HTTPException(400, "Email already registered")
        
        # Create user
        user_id = f"local_{len(self.users) + 1}"
        user_data = {
            "id": user_id,
            "email": registration.email,
            "password_hash": self._hash_password(registration.password),
            "full_name": registration.full_name,
            "roles": ["user"],  # Default role
            "is_active": True,
            "is_email_verified": False,
            "created_at": datetime.now()
        }
        
        self.users[registration.email] = user_data
        
        return LocalUserInfo(
            user_id=user_id,
            email=registration.email,
            name=registration.full_name,
            roles=["user"],
            is_active=True,
            is_email_verified=False
        )
    
    async def authenticate_user(self, login: LocalAuthSchemas.UserLogin) -> LocalAuthSchemas.TokenResponse:
        """Authenticate user"""
        user = self.users.get(login.email)
        if not user:
            raise HTTPException(401, "Invalid credentials")
        
        if not self._verify_password(login.password, user["password_hash"]):
            raise HTTPException(401, "Invalid credentials")
        
        if not user["is_active"]:
            raise HTTPException(401, "Account deactivated")
        
        # Create tokens
        access_token = self._create_token(user, "access")
        refresh_token = self._create_token(user, "refresh")
        
        # Store refresh token
        self.refresh_tokens[refresh_token] = user["id"]
        
        user_info = LocalUserInfo(
            user_id=user["id"],
            email=user["email"],
            name=user["full_name"],
            roles=user["roles"],
            is_active=user["is_active"],
            is_email_verified=user["is_email_verified"]
        )
        
        return LocalAuthSchemas.TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=LOCAL_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user=user_info.dict()
        )
    
    async def refresh_token(self, refresh_token: str) -> LocalAuthSchemas.TokenResponse:
        """Refresh access token"""
        if refresh_token not in self.refresh_tokens:
            raise HTTPException(401, "Invalid refresh token")
        
        try:
            payload = self.verify_token(refresh_token)
            if payload.get("type") != "refresh":
                raise HTTPException(401, "Invalid refresh token")
            
            user_email = payload["email"]
            user = self.users.get(user_email)
            if not user:
                raise HTTPException(401, "User not found")
            
            # Create new tokens
            new_access_token = self._create_token(user, "access")
            new_refresh_token = self._create_token(user, "refresh")
            
            # Replace old refresh token
            del self.refresh_tokens[refresh_token]
            self.refresh_tokens[new_refresh_token] = user["id"]
            
            user_info = LocalUserInfo(
                user_id=user["id"],
                email=user["email"],
                name=user["full_name"],
                roles=user["roles"],
                is_active=user["is_active"],
                is_email_verified=user["is_email_verified"]
            )
            
            return LocalAuthSchemas.TokenResponse(
                access_token=new_access_token,
                refresh_token=new_refresh_token,
                expires_in=LOCAL_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
                user=user_info.dict()
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            raise HTTPException(401, "Token refresh failed")
    
    async def get_user_by_token(self, token: str) -> LocalUserInfo:
        """Get user info from token"""
        payload = self.verify_token(token)
        user_email = payload["email"]
        user = self.users.get(user_email)
        
        if not user:
            raise HTTPException(401, "User not found")
        
        return LocalUserInfo(
            user_id=user["id"],
            email=user["email"],
            name=user["full_name"],
            roles=user["roles"],
            is_active=user["is_active"],
            is_email_verified=user["is_email_verified"]
        )

    def _get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        for user in self.users.values():
            if user["id"] == user_id:
                return user
        return None

    async def update_user(self, user_id: str, updates: LocalAuthSchemas.UserUpdate) -> LocalUserInfo:
        user = self._get_user_by_id(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        if updates.full_name is not None:
            user["full_name"] = updates.full_name
        if updates.password:
            user["password_hash"] = self._hash_password(updates.password)
        if updates.roles is not None:
            user["roles"] = updates.roles
        if updates.is_active is not None:
            user["is_active"] = updates.is_active

        return LocalUserInfo(
            user_id=user["id"],
            email=user["email"],
            name=user["full_name"],
            roles=user["roles"],
            is_active=user["is_active"],
            is_email_verified=user["is_email_verified"]
        )

    async def delete_user(self, user_id: str) -> None:
        for email, data in list(self.users.items()):
            if data["id"] == user_id:
                del self.users[email]
                return
        raise HTTPException(status_code=404, detail="User not found")

# Global instances
azure_validator = MinimalAzureValidator(AZURE_TENANT_ID, AZURE_CLIENT_ID)
local_auth_service = LocalAuthService()
security = HTTPBearer()

# ============================================================================
# ENHANCED AUTHENTICATION DEPENDENCIES
# ============================================================================

async def get_current_user_azure(credentials: HTTPBearer = Depends(security)) -> AzureUserInfo:
    """Get current authenticated user via Azure AD"""
    return await azure_validator.validate_token(credentials.credentials)

async def get_current_user_local(credentials: HTTPBearer = Depends(security)) -> LocalUserInfo:
    """Get current authenticated user via local auth"""
    return await local_auth_service.get_user_by_token(credentials.credentials)

async def get_current_user_unified(credentials: HTTPBearer = Depends(security)) -> UnifiedUserInfo:
    """Get current user from either Azure AD or local auth"""
    token = credentials.credentials
    
    # Try Azure AD first
    try:
        azure_user = await azure_validator.validate_token(token)
        return UnifiedUserInfo(
            user_id=azure_user.user_id,
            email=azure_user.email,
            name=azure_user.name,
            roles=azure_user.app_roles,
            auth_method="azure",
            tenant_id=azure_user.tenant_id,
            groups=azure_user.groups,
            is_active=True
        )
    except HTTPException:
        pass  # Try local auth
    
    # Try local auth
    try:
        local_user = await local_auth_service.get_user_by_token(token)
        return UnifiedUserInfo(
            user_id=local_user.user_id,
            email=local_user.email,
            name=local_user.name,
            roles=local_user.roles,
            auth_method="local",
            is_active=local_user.is_active,
            is_email_verified=local_user.is_email_verified
        )
    except HTTPException:
        pass
    
    raise HTTPException(401, "Invalid authentication token")

async def get_current_user_optional_unified(request: Request) -> Optional[UnifiedUserInfo]:
    """Get current user if authenticated (optional, supports both auth methods)"""
    auth_header = request.headers.get('Authorization')
    
    if not auth_header or not auth_header.startswith('Bearer '):
        return None
    
    try:
        token = auth_header.split(' ')[1]
        credentials = HTTPBearer().__call__(Request({"type": "http", "headers": [("authorization", auth_header)]}))
        return await get_current_user_unified(credentials)
    except Exception:
        return None

def require_role_unified(required_roles: List[str]):
    """Decorator to require specific roles (supports both auth methods)"""
    def role_checker(user: UnifiedUserInfo = Depends(get_current_user_unified)) -> UnifiedUserInfo:
        if not any(role in user.roles for role in required_roles):
            raise HTTPException(
                403, 
                f"Insufficient permissions. Required roles: {required_roles}"
            )
        return user
    return role_checker

def require_permission_unified(permission: str):
    """Decorator to require specific permission (supports both auth methods)"""
    permission_to_roles = {
        'read': ['user', 'analyst', 'developer', 'admin', 'super_admin'],
        'write': ['developer', 'admin', 'super_admin'],
        'execute': ['developer', 'admin', 'super_admin'],
        'admin': ['admin', 'super_admin'],
        'super_admin': ['super_admin']
    }
    
    required_roles = permission_to_roles.get(permission, [])
    return require_role_unified(required_roles)

# ============================================================================
# AUDIT LOGGING
# ============================================================================

async def log_security_event(
    action: str,
    user_id: Optional[str] = None,
    success: bool = True,
    details: Dict[str, Any] = None,
    request: Optional[Request] = None
):
    """Log security events for audit trail"""
    try:
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "user_id": user_id,
            "success": success,
            "details": details or {},
            "ip_address": request.client.host if request else None,
            "user_agent": request.headers.get("user-agent") if request else None
        }
        
        # In production, this would write to a database
        logger.info(f"AUDIT: {audit_entry}")
        
    except Exception as e:
        logger.error(f"Failed to log audit event: {e}")

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
    logger.info("🚀 Starting Enhanced CSP Visual Designer Backend with Dual Authentication")
    
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
        
        # Initialize network service
        if NETWORK_AVAILABLE:
            await initialize_network_service(app)
            
            # Integrate WebSocket with network
            if WEBSOCKET_AVAILABLE and network_service and network_service.is_initialized:
                integrate_websocket_with_network(connection_manager, network_service)
        
        # Initialize execution engine if available
        if EXECUTION_AVAILABLE:
            execution_engine = await get_execution_engine()
        
        # Initialize AI coordination if available
        if AI_COORDINATION_AVAILABLE:
            await initialize_ai_coordination()
        
        # Startup logging
        logger.info("🚀 Enhanced CSP System Backend Starting...")
        logger.info(f"📋 Azure AD Tenant: {AZURE_TENANT_ID}")
        logger.info(f"🔑 Azure AD Client: {AZURE_CLIENT_ID}")
        logger.info(f"🌐 Allowed Origins: {ALLOWED_ORIGINS}")
        logger.info("✅ Azure AD authentication initialized")
        logger.info("✅ Local email/password authentication initialized")
        logger.info("✅ Dual authentication system ready")
        logger.info("🌐 API documentation: http://localhost:8000/docs")
        logger.info("✅ Backend startup completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        # Don't raise in minimal mode, just log the error
        logger.warning("Continuing with minimal functionality...")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Enhanced CSP Visual Designer Backend")
        
        # Shutdown network service
    if NETWORK_AVAILABLE and network_service:
            await shutdown_network_service()
    
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
    title="Enhanced CSP Visual Designer API",
    description="Advanced AI-Powered CSP Process Designer Backend with Dual Authentication",
    version="2.2.0",
    docs_url=None,  # Custom docs URL
    redoc_url="/redoc",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "authentication", "description": "Dual authentication (Azure AD + Email/Password)"},
        {"name": "designs", "description": "Visual design management"},
        {"name": "components", "description": "Component registry and metadata"},
        {"name": "execution", "description": "Design execution and monitoring"},
        {"name": "websocket", "description": "Real-time collaboration"},
        {"name": "system", "description": "System health and metrics"},
        {"name": "admin", "description": "Administrative functions"}
    ]
)
app.include_router(memory_router)
app = integrate_enhanced_monitoring(app)



# ============================================================================
# CORS MIDDLEWARE CONFIGURATION - DEVELOPMENT MODE (ALLOW ALL)
# ============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"],
)

# ============================================================================
# OTHER MIDDLEWARE CONFIGURATION
# ============================================================================

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
    response.headers["X-Auth-Methods"] = "azure,local"
    return response

# Enhanced error handling middleware
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Global HTTP exception handler with audit logging"""
    # Log security-related errors
    if exc.status_code in [401, 403]:
        await log_security_event(
            action="security_error",
            success=False,
            details={"status_code": exc.status_code, "detail": exc.detail},
            request=request
        )
    
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

# Include AI coordination router if available
if AI_COORDINATION_AVAILABLE:
    app.include_router(
        ai_coordination_router,
        tags=["AI Coordination"],
        dependencies=[]
    )
    logger.info("✅ AI Coordination router registered")

# Include AI coordination monitoring router if available
try:
    from backend.api.endpoints.ai_coordination_monitoring import monitoring_router
    app.include_router(
        monitoring_router,
        tags=["AI Coordination Monitoring"]
    )
    logger.info("✅ AI Coordination monitoring router registered")
    AI_MONITORING_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ AI Coordination monitoring router not available")
    AI_MONITORING_AVAILABLE = False

# Include infrastructure management router
try:
    from backend.api.endpoints.infrastructure import router as infrastructure_router
    app.include_router(infrastructure_router)
    logger.info("✅ Infrastructure router registered")
    INFRASTRUCTURE_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ Infrastructure router not available")
    INFRASTRUCTURE_AVAILABLE = False

# Include system settings router
try:
    from backend.api.endpoints import settings as settings_router
    app.include_router(settings_router.router)
    logger.info("✅ Settings router registered")
except ImportError:
    logger.warning("⚠️ Settings router not available")

# Include licenses router
try:
    from backend.api.endpoints.licenses import router as licenses_router
    app.include_router(licenses_router)
    logger.info("✅ Licenses router registered")
except ImportError:
    logger.warning("⚠️ Licenses router not available")

# ============================================================================
# AUTHENTICATION ENDPOINTS
# ============================================================================

@app.get("/api/auth/info", tags=["authentication"])
async def get_auth_info():
    """Get authentication configuration (public)"""
    return {
        "auth_type": "dual",
        "methods": ["azure_ad", "local"],
        "azure_ad": {
            "tenant_id": AZURE_TENANT_ID,
            "client_id": AZURE_CLIENT_ID,
            "authority": f"https://login.microsoftonline.com/{AZURE_TENANT_ID}",
            "scopes": ["User.Read", "User.ReadBasic.All", "Group.Read.All"]
        },
        "local_auth": {
            "enabled": True,
            "features": ["registration", "password_reset", "email_verification"]
        }
    }

@app.get("/api/auth/me", tags=["authentication"])
async def get_current_user_info_unified_endpoint(
    current_user: UnifiedUserInfo = Depends(get_current_user_unified)
) -> Dict[str, Any]:
    """Get current authenticated user information (supports both auth methods)"""
    return {
        "user_id": current_user.user_id,
        "email": current_user.email,
        "name": current_user.name,
        "roles": current_user.roles,
        "auth_method": current_user.auth_method,
        "tenant_id": current_user.tenant_id,
        "groups": current_user.groups,
        "is_active": current_user.is_active,
        "is_email_verified": current_user.is_email_verified
    }


@app.get("/api/auth/validate", tags=["authentication"])
async def validate_current_token(
    current_user: UnifiedUserInfo = Depends(get_current_user_unified),
) -> Dict[str, Any]:
    """Validate the provided authentication token and return the user"""
    return {"valid": True, "user": current_user}

@app.get("/api/auth/permissions", tags=["authentication"])
async def get_user_permissions_unified(
    current_user: UnifiedUserInfo = Depends(get_current_user_unified)
) -> Dict[str, Any]:
    """Get current user's permissions (supports both auth methods)"""
    # Map roles to permissions
    role_permissions = {
        'super_admin': ['read', 'write', 'execute', 'admin', 'super_admin'],
        'admin': ['read', 'write', 'execute', 'admin'],
        'developer': ['read', 'write', 'execute'],
        'analyst': ['read'],
        'user': ['read']
    }
    
    permissions = set()
    for role in current_user.roles:
        permissions.update(role_permissions.get(role, []))
    
    return {
        "user_id": current_user.user_id,
        "permissions": list(permissions),
        "roles": current_user.roles,
        "auth_method": current_user.auth_method
    }

@app.post("/api/auth/logout", tags=["authentication"])
async def logout_unified(
    current_user: UnifiedUserInfo = Depends(get_current_user_unified),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Logout endpoint (supports both auth methods)"""
    # Log logout event
    background_tasks.add_task(
        log_security_event,
        action="logout",
        user_id=current_user.user_id,
        success=True,
        details={"auth_method": current_user.auth_method}
    )
    
    return {
        "message": "Logout successful", 
        "user_id": current_user.user_id,
        "auth_method": current_user.auth_method
    }

# ============================================================================
# LOCAL AUTHENTICATION ENDPOINTS
# ============================================================================

@app.post("/api/auth/local/register", 
        response_model=LocalUserInfo, 
        status_code=201,
        tags=["authentication"])
async def register_local_user(
    registration: LocalAuthSchemas.UserRegistration,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Register new local user with email/password"""
    try:
        user_info = await local_auth_service.register_user(registration)
        
        # Log registration event
        background_tasks.add_task(
            log_security_event,
            action="user_registration",
            user_id=user_info.user_id,
            success=True,
            details={"email": user_info.email, "auth_method": "local"}
        )
        
        return user_info
        
    except HTTPException:
        # Log failed registration
        background_tasks.add_task(
            log_security_event,
            action="user_registration",
            success=False,
            details={"email": registration.email, "auth_method": "local"}
        )
        raise
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise HTTPException(500, "Registration failed")

@app.post("/api/auth/local/login", 
        response_model=LocalAuthSchemas.TokenResponse, 
        tags=["authentication"])
async def login_local_user(
    login: LocalAuthSchemas.UserLogin,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Login with email/password"""
    try:
        token_response = await local_auth_service.authenticate_user(login)
        
        # Log successful login
        background_tasks.add_task(
            log_security_event,
            action="login",
            user_id=token_response.user["user_id"],
            success=True,
            details={"email": login.email, "auth_method": "local", "remember_me": login.remember_me}
        )
        
        return token_response
        
    except HTTPException:
        # Log failed login
        background_tasks.add_task(
            log_security_event,
            action="login",
            success=False,
            details={"email": login.email, "auth_method": "local"}
        )
        raise
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise HTTPException(500, "Authentication failed")

@app.post("/api/auth/local/refresh", 
        response_model=LocalAuthSchemas.TokenResponse, 
        tags=["authentication"])
async def refresh_local_token(
    token_refresh: LocalAuthSchemas.TokenRefresh,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Refresh access token using refresh token"""
    try:
        token_response = await local_auth_service.refresh_token(token_refresh.refresh_token)
        
        # Log token refresh
        background_tasks.add_task(
            log_security_event,
            action="token_refresh",
            user_id=token_response.user["user_id"],
            success=True,
            details={"auth_method": "local"}
        )
        
        return token_response
        
    except HTTPException:
        # Log failed refresh
        background_tasks.add_task(
            log_security_event,
            action="token_refresh",
            success=False,
            details={"auth_method": "local"}
        )
        raise
    except Exception as e:
        logger.error(f"Token refresh failed: {e}")
        raise HTTPException(500, "Token refresh failed")

@app.post("/api/auth/local/forgot-password", 
        response_model=BaseResponse, 
        tags=["authentication"])
async def forgot_password(
    request: Dict[str, str],  # {"email": "user@example.com"}
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Request password reset (placeholder implementation)"""
    email = request.get("email")
    if not email:
        raise HTTPException(400, "Email is required")
    
    # Log password reset request
    background_tasks.add_task(
        log_security_event,
        action="password_reset_request",
        success=True,
        details={"email": email, "auth_method": "local"}
    )
    
    # In production, this would send a real email
    logger.info(f"Password reset requested for: {email}")
    
    return BaseResponse(message="Password reset email sent if account exists")

# ============================================================================
# AZURE AD SPECIFIC ENDPOINTS (for backward compatibility)
# ============================================================================

class AzureLoginRequest(BaseModel):
    azure_token: str

@app.post("/api/auth/azure-login", tags=["authentication"])
async def azure_login(request: AzureLoginRequest) -> Dict[str, Any]:
    """Validate an Azure AD token and return user info"""
    try:
        azure_user = await azure_validator.validate_token(request.azure_token)
        return {
            "user_id": azure_user.user_id,
            "email": azure_user.email,
            "name": azure_user.name,
            "roles": azure_user.app_roles,
            "tenant_id": azure_user.tenant_id,
            "auth_method": "azure",
        }
    except HTTPException:
        raise HTTPException(status_code=401, detail="Invalid Azure token")

@app.get("/api/auth/azure/me", tags=["authentication"])
async def get_current_user_info_azure(
    current_user: AzureUserInfo = Depends(get_current_user_azure)
) -> Dict[str, Any]:
    """Get current authenticated Azure AD user information"""
    return {
        "user_id": current_user.user_id,
        "email": current_user.email,
        "name": current_user.name,
        "roles": current_user.app_roles,
        "groups": current_user.groups,
        "tenant_id": current_user.tenant_id,
        "auth_method": "azure_ad"
    }

@app.get("/api/auth/azure/permissions", tags=["authentication"])
async def get_user_permissions_azure(
    current_user: AzureUserInfo = Depends(get_current_user_azure)
) -> Dict[str, Any]:
    """Get current Azure AD user's permissions"""
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

@app.post("/api/auth/azure/logout", tags=["authentication"])
async def logout_azure(current_user: AzureUserInfo = Depends(get_current_user_azure)):
    """Logout endpoint (client-side logout mainly for Azure AD)"""
    return {"message": "Logout successful", "user_id": current_user.user_id}

# ============================================================================
# LEGACY AUTHENTICATION ENDPOINTS (if auth system available)
# ============================================================================

try:
    from backend.auth.auth_system import (
        get_auth_service, AuthenticationService, LoginRequest, RegisterRequest,
        TokenResponse, UserInfo, create_initial_admin, get_current_user, get_current_user_optional,
        require_permission, Permission
    )
    LEGACY_AUTH_SYSTEM_AVAILABLE = True
    
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

except ImportError:
    LEGACY_AUTH_SYSTEM_AVAILABLE = False
    logger.info("Legacy auth system not available")

# ============================================================================
# ADMIN ENDPOINTS WITH RBAC
# ============================================================================

@app.get("/api/admin/users", tags=["admin"])
async def list_users(
    current_user: UnifiedUserInfo = Depends(require_role_unified(["admin", "super_admin"]))
):
    """List all users (admin only)"""
    # Return both Azure AD and local users
    local_users = []
    for email, user_data in local_auth_service.users.items():
        local_users.append({
            "id": user_data["id"],
            "email": user_data["email"],
            "name": user_data["full_name"],
            "roles": user_data["roles"],
            "auth_method": "local",
            "is_active": user_data["is_active"],
            "created_at": user_data["created_at"].isoformat()
        })
    
    return {
        "local_users": local_users,
        "azure_users": [],  # Would query Azure AD in production
        "total": len(local_users)
    }


@app.post("/api/admin/users", response_model=LocalUserInfo, status_code=201, tags=["admin"])
async def create_user(
    user: LocalAuthSchemas.UserRegistration,
    current_user: UnifiedUserInfo = Depends(require_role_unified(["admin", "super_admin"]))
):
    """Create a new local user (admin only)"""
    try:
        new_user = await local_auth_service.register_user(user)
        return new_user
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create user: {e}")
        raise HTTPException(status_code=500, detail="User creation failed")


@app.put("/api/admin/users/{user_id}", response_model=LocalUserInfo, tags=["admin"])
async def update_user(
    user_id: str,
    updates: LocalAuthSchemas.UserUpdate,
    current_user: UnifiedUserInfo = Depends(require_role_unified(["admin", "super_admin"]))
):
    """Update a local user (admin only)"""
    try:
        updated_user = await local_auth_service.update_user(user_id, updates)
        return updated_user
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="User update failed")


@app.delete("/api/admin/users/{user_id}", status_code=204, tags=["admin"])
async def delete_user(
    user_id: str,
    current_user: UnifiedUserInfo = Depends(require_role_unified(["admin", "super_admin"]))
):
    """Delete a local user (admin only)"""
    try:
        await local_auth_service.delete_user(user_id)
        return Response(status_code=204)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="User deletion failed")

@app.get("/api/admin/audit-logs", tags=["admin"])
async def get_audit_logs(
    limit: int = 100,
    current_user: UnifiedUserInfo = Depends(require_role_unified(["admin", "super_admin"]))
):
    """Get audit logs (admin only)"""
    # In production, this would query a database
    return {
        "logs": [],
        "message": "Audit logs would be retrieved from database in production",
        "limit": limit
    }

# ============================================================================
# COMPONENT ENDPOINTS
# ============================================================================

@app.get("/api/components", response_model=Dict[str, List], tags=["components"])
async def list_components(
    category: Optional[str] = None,
    current_user: Optional[UnifiedUserInfo] = Depends(get_current_user_optional_unified)
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
    current_user: Optional[UnifiedUserInfo] = Depends(get_current_user_optional_unified)
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
    current_user: UnifiedUserInfo = Depends(require_permission_unified("execute"))
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
    current_user: UnifiedUserInfo = Depends(require_permission_unified("read"))
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
    current_user: UnifiedUserInfo = Depends(require_permission_unified("read"))
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
    current_user: UnifiedUserInfo = Depends(require_permission_unified("execute"))
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
    current_user: UnifiedUserInfo = Depends(require_permission_unified("execute"))
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
    current_user: UnifiedUserInfo = Depends(require_permission_unified("execute"))
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
        current_user: UnifiedUserInfo = Depends(get_current_user_unified)
    ):
        """Get user's designs (placeholder implementation)"""
        return [
            {
                "id": "sample_design_1",
                "name": "Sample CSP Design",
                "description": "A sample design for testing",
                "owner": current_user.email,
                "created_at": datetime.utcnow().isoformat(),
                "status": "draft",
                "auth_method": current_user.auth_method
            }
        ]

    @app.post("/api/designs", tags=["designs"])
    async def create_design(
        design_data: Dict[str, Any],
        current_user: UnifiedUserInfo = Depends(require_permission_unified("write"))
    ):
        """Create new design (placeholder implementation)"""
        return {
            "id": f"design_{datetime.utcnow().timestamp()}",
            "name": design_data.get("name", "Untitled Design"),
            "description": design_data.get("description", ""),
            "owner": current_user.email,
            "created_at": datetime.utcnow().isoformat(),
            "status": "draft",
            "message": "Design created successfully",
            "auth_method": current_user.auth_method
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
    current_user: UnifiedUserInfo = Depends(require_permission_unified("read"))
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
        "name": "Enhanced CSP Visual Designer API",
        "version": "2.2.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "auth_methods": ["azure_ad", "local"],
        "features": {
            "azure_ad_authentication": True,
            "local_email_authentication": True,
            "dual_authentication": True,
            "role_based_access_control": True,
            "audit_logging": True,
            "token_refresh_rotation": True,
            "real_time_collaboration": WEBSOCKET_AVAILABLE,
            "ai_integration": True,
            "component_registry": COMPONENTS_AVAILABLE,
            "network": {
                "enabled": NETWORK_AVAILABLE and network_service and network_service.is_initialized,
                "node_id": network_service.network.node_id.to_base58() if (network_service and network_service.is_initialized) else None,
                "peers": len(network_service.node_registry) - 1 if (network_service and network_service.is_initialized) else 0
            },
            "execution_engine": EXECUTION_AVAILABLE,
            "performance_monitoring": True,
            "database": DATABASE_AVAILABLE
        },
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json",
            "websocket": "/ws/{user_id}",
            "health": "/health",
            "local_auth": "/api/auth/local",
            "azure_auth": "/api/auth/azure"
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
        
        # Check local auth status
        local_auth_status = "healthy" if len(local_auth_service.users) > 0 else "no_users"
        
        # Overall health status
        overall_status = "healthy"
        if DATABASE_AVAILABLE and (db_health["database"]["status"] != "healthy" or db_health["redis"]["status"] != "healthy"):
            overall_status = "degraded"
        if azure_ad_status == "unhealthy":
            overall_status = "degraded"
        
        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "version": "2.2.0",
            "uptime": "calculated_by_process_manager",
            "components": {
                "database": db_health,
                "azure_ad": {
                    "status": azure_ad_status,
                    "tenant_id": AZURE_TENANT_ID
                },
                "local_auth": {
                    "status": local_auth_status,
                    "user_count": len(local_auth_service.users),
                    "active_sessions": len(local_auth_service.refresh_tokens)
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
    current_user: UnifiedUserInfo = Depends(require_permission_unified("read"))
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
        
        # Authentication metrics
        auth_metrics = {
            "local_users": len(local_auth_service.users),
            "active_local_sessions": len(local_auth_service.refresh_tokens),
            "azure_ad_available": True
        }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system": system_metrics,
            "database": db_health,
            "websocket": ws_stats,
            "authentication": auth_metrics,
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
                "dual_authentication": True
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system metrics")

# ============================================================================
# AI COORDINATION INITIALIZATION AND HEALTH ENDPOINTS
# ============================================================================

async def initialize_ai_coordination():
    """Initialize AI coordination systems on startup"""
    if not AI_COORDINATION_AVAILABLE:
        logger.info("🔄 AI Coordination not available, skipping initialization")
        return
    
    try:
        logger.info("🤖 Initializing AI Coordination System...")
        
        # Initialize system parameters for optimal performance
        await coordination_engine.optimize_system_parameters()
        
        # Run a quick validation test
        system_status = await coordination_engine.get_system_status()
        performance = system_status.get('recent_performance', 0.0)
        
        if performance >= 95.0:
            logger.info(f"✅ AI Coordination System initialized successfully - Performance: {performance:.1f}%")
        elif performance >= 85.0:
            logger.info(f"⚠️ AI Coordination System initialized with reduced performance: {performance:.1f}%")
        else:
            logger.warning(f"❌ AI Coordination System performance below target: {performance:.1f}%")
        
        # Log available features
        features = [
            "Multi-Dimensional Consciousness Synchronization",
            "Quantum Knowledge Osmosis", 
            "Meta-Wisdom Convergence",
            "Temporal Entanglement",
            "Emergent Behavior Detection"
        ]
        logger.info(f"🔧 Available AI Features: {', '.join(features)}")
        
    except Exception as e:
        logger.error(f"❌ AI Coordination System initialization failed: {e}")

@app.get("/health/ai-coordination")
async def ai_coordination_health():
    """Health check specifically for AI coordination systems"""
    if not AI_COORDINATION_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unavailable",
                "message": "AI Coordination system not installed",
                "timestamp": datetime.now().isoformat()
            }
        )
    
    try:
        # Get system status
        status = await coordination_engine.get_system_status()
        performance = status.get('recent_performance', 0.0)
        
        # Determine health status
        if performance >= 95.0:
            health_status = "excellent"
        elif performance >= 85.0:
            health_status = "good" 
        elif performance >= 70.0:
            health_status = "degraded"
        else:
            health_status = "poor"
        
        return JSONResponse(
            status_code=200 if health_status in ["excellent", "good"] else 503,
            content={
                "status": health_status,
                "performance": f"{performance:.1f}%",
                "ai_coordination": {
                    "system_status": status.get('system_status', 'unknown'),
                    "coordination_sessions": status.get('coordination_sessions', 0),
                    "registered_agents": status.get('registered_agents', 0),
                    "target_achievement": performance >= 95.0
                },
                "available_endpoints": [
                    "/api/ai-coordination/synchronize",
                    "/api/ai-coordination/performance/metrics", 
                    "/api/ai-coordination/system/status",
                    "/api/ai-coordination/consciousness/sync",
                    "/api/ai-coordination/quantum/entangle",
                    "/api/ai-coordination/wisdom/converge",
                    "/api/ai-coordination/temporal/entangle",
                    "/api/ai-coordination/emergence/detect"
                ] if AI_COORDINATION_AVAILABLE else [],
                "timestamp": datetime.now().isoformat()
            }
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@app.get("/api/ai-coordination/features")
async def list_ai_coordination_features():
    """List available AI coordination features and capabilities"""
    if not AI_COORDINATION_AVAILABLE:
        raise HTTPException(status_code=503, detail="AI Coordination system not available")
    
    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "message": "AI Coordination Features",
            "features": {
                "consciousness_synchronization": {
                    "description": "Multi-dimensional consciousness alignment across AI agents",
                    "target_performance": ">95% consciousness coherence",
                    "dimensions": {
                        "attention": 128,
                        "emotion": 64, 
                        "metacognition": 5
                    },
                    "endpoint": "/api/ai-coordination/consciousness/sync"
                },
                "quantum_knowledge_osmosis": {
                    "description": "Quantum entanglement-based knowledge sharing",
                    "target_performance": ">95% Bell state fidelity",
                    "capabilities": ["knowledge_transfer", "belief_synchronization", "memory_synthesis"],
                    "endpoint": "/api/ai-coordination/quantum/entangle"
                },
                "meta_wisdom_convergence": {
                    "description": "Advanced reasoning and wisdom synthesis",
                    "target_performance": ">85% convergence rate",
                    "algorithms": ["dialectical_synthesis", "consensus_building", "wisdom_distillation"],
                    "endpoint": "/api/ai-coordination/wisdom/converge"
                },
                "temporal_entanglement": {
                    "description": "Cross-temporal state synchronization",
                    "target_performance": ">95% temporal coherence",
                    "features": ["state_history", "causal_consistency", "temporal_alignment"],
                    "endpoint": "/api/ai-coordination/temporal/entangle"
                },
                "emergent_behavior_detection": {
                    "description": "Detection and amplification of emergent collective intelligence",
                    "target_performance": ">95% emergence detection",
                    "capabilities": ["collective_reasoning", "metacognitive_resonance", "consciousness_amplification"],
                    "endpoint": "/api/ai-coordination/emergence/detect"
                }
            },
            "system_capabilities": {
                "max_agents_per_coordination": 50,
                "supported_ai_models": ["gpt-4", "claude-3", "custom"],
                "performance_monitoring": True,
                "real_time_optimization": True,
                "audit_logging": True
            },
            "timestamp": datetime.now().isoformat()
        }
    )

# ============================================================================
# CUSTOM DOCUMENTATION
# ============================================================================

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Custom Swagger UI"""
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Enhanced CSP Visual Designer API Documentation",
        swagger_favicon_url="/static/favicon.ico" if os.path.exists("backend/static/favicon.ico") else None
    )

def custom_openapi():
    """Custom OpenAPI schema"""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Enhanced CSP Visual Designer API",
        version="2.2.0",
        description="""
        ## Advanced AI-Powered CSP Process Designer Backend with Dual Authentication

        This API provides comprehensive functionality for building, executing, and monitoring
        CSP (Communicating Sequential Processes) designs through a visual interface.

        ### 🔐 **Dual Authentication System**
        
        This API supports two authentication methods:
        
        #### Azure AD Authentication
        - Enterprise single sign-on using Microsoft Azure AD
        - Supports role assignment through Azure AD groups
        - Automatic user provisioning
        
        #### Local Email/Password Authentication  
        - Traditional registration and login with email/password
        - Password reset functionality
        - Email verification support
        - JWT token management with refresh rotation
        
        ### 🛡 **Security Features**
        - Role-based access control (RBAC) with granular permissions
        - JWT token blacklisting and refresh rotation for security
        - Comprehensive audit logging for compliance
        - Rate limiting and input validation
        - Secure password hashing and validation

        ### 🚀 **Key Features**
        - **Visual Design Management**: Create, edit, and manage process designs
        - **Real-time Collaboration**: Multi-user editing with WebSocket support
        - **Component Registry**: Extensible library of AI, data, and security components
        - **Execution Engine**: Run designs with monitoring and metrics
        - **Performance Monitoring**: Comprehensive metrics and health checks
        - **Toast Notifications**: Global notification system for consistent UX

        ### 📋 **Permission System**
        - **Super Admin**: Full system access and user management
        - **Admin**: Application administration and user oversight
        - **Developer**: Create, edit, execute designs and components
        - **Analyst**: View and analyze data with limited edit access
        - **User**: Basic access to view and create simple designs
        - **Viewer**: Read-only access to public designs

        ### Authentication
        Include your access token in the `Authorization` header as `Bearer <token>`.
        
        For Azure AD: Use your Azure AD access token
        For Local Auth: Use the JWT token from `/api/auth/local/login`

        ### WebSocket Collaboration
        Connect to `/ws/{user_id}?design_id={design_id}` for real-time collaboration features.
        """,
        routes=app.routes,
    )
    
    # Enhanced security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "AzureADBearer": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "Azure AD access token"
        },
        "LocalAuthBearer": {
            "type": "http",
            "scheme": "bearer", 
            "bearerFormat": "JWT",
            "description": "Local authentication JWT token"
        }
    }
    
    # Apply security to endpoints except public ones
    public_paths = ["/health", "/api/auth/info", "/docs", "/redoc", "/openapi.json", 
                "/api/auth/local/register", "/api/auth/local/login"]
    
    for path, path_item in openapi_schema["paths"].items():
        if not any(path.startswith(public) for public in public_paths):
            for method, operation in path_item.items():
                if method in ["get", "post", "put", "delete", "patch"]:
                    operation["security"] = [
                        {"AzureADBearer": []}, 
                        {"LocalAuthBearer": []}
                    ]
    
    if not os.path.exists("backend/static/logo.png"):
        openapi_schema["info"]["x-logo"] = {
            "url": "/static/logo.png"
        }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# ============================================================================
# INTEGRATION INSTRUCTIONS SECTION  
# ============================================================================

"""
INTEGRATION INSTRUCTIONS:
========================

1. **Add the imports** to your existing backend/main.py file after your current imports

2. **Register the router** by adding the router registration code where you register other routers 

3. **Update your startup function** by adding the initialize_ai_coordination() call to your existing startup handler:

@app.on_event("startup") 
async def startup():
    # Your existing startup code...
    
    # Add AI coordination initialization
    await initialize_ai_coordination()

4. **Add the health endpoints** to provide AI system monitoring

5. **Test the integration** by starting your server and checking:
- GET /health/ai-coordination
- GET /api/ai-coordination/features  
- GET /api/ai-coordination/system/status

6. **Verify all endpoints work**:
- POST /api/ai-coordination/synchronize (requires authentication)
- GET /api/ai-coordination/performance/metrics (requires authentication)
- All individual system endpoints (consciousness, quantum, wisdom, temporal, emergence)

The AI coordination system provides 5 advanced algorithms targeting >95% performance:
✅ Multi-Dimensional Consciousness Synchronization 
✅ Quantum Knowledge Osmosis
✅ Meta-Wisdom Convergence  
✅ Temporal Entanglement
✅ Emergent Behavior Detection

All features are already implemented in your project - this just ensures proper integration!
"""

# ============================================================================
# STATIC FILES
# ============================================================================

# Mount static files for documentation assets
static_dir = "backend/static"
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# ============================================================================
# DEVELOPMENT SERVER
# ============================================================================

if __name__ == "__main__":
    print("🚀 Starting Enhanced CSP System Backend with Dual Authentication...")
    print("=" * 70)
    print(f"🔗 API: http://localhost:8000")
    print(f"📚 Docs: http://localhost:8000/docs")
    print(f"🔍 Health: http://localhost:8000/health")
    print(f"🔐 Auth Info: http://localhost:8000/api/auth/info")
    print(f"🌐 Azure AD: http://localhost:8000/api/auth/azure/me")
    print(f"✉️ Local Auth: http://localhost:8000/api/auth/local/login")
    print(f"👤 Local Register: http://localhost:8000/api/auth/local/register")
    print(f"📊 Metrics: http://localhost:8000/metrics")
    print("=" * 70)
    print("🔑 Default Admin User:")
    print("   Email: admin@csp-system.com")
    print("   Password: AdminPass123!")
    print("=" * 70)
    
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