# File: backend/auth/auth_system.py
"""
Authentication & Authorization System
====================================
JWT-based authentication with role-based access control
"""

import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import secrets
import hashlib
from enum import Enum

import jwt
from passlib.context import CryptContext
from passlib.hash import bcrypt
from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from pydantic import BaseModel, EmailStr, validator
import redis.asyncio as redis

from backend.models.database_models import User
from backend.database.connection import get_db_session, get_redis_client

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

# JWT Configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
JWT_ALGORITHM = "HS256"
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours
JWT_REFRESH_TOKEN_EXPIRE_DAYS = 30

# Security Configuration
PASSWORD_MIN_LENGTH = 8
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_DURATION_MINUTES = 15

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# HTTP Bearer token handler
security = HTTPBearer(auto_error=False)

# ============================================================================
# MODELS AND SCHEMAS
# ============================================================================

class UserRole(str, Enum):
    """User roles for authorization"""
    ADMIN = "admin"
    DESIGNER = "designer"
    VIEWER = "viewer"
    API_USER = "api_user"

class Permission(str, Enum):
    """Granular permissions"""
    # Design permissions
    CREATE_DESIGN = "create_design"
    READ_DESIGN = "read_design"
    UPDATE_DESIGN = "update_design"
    DELETE_DESIGN = "delete_design"
    EXECUTE_DESIGN = "execute_design"
    
    # Component permissions
    CREATE_COMPONENT = "create_component"
    UPDATE_COMPONENT = "update_component"
    DELETE_COMPONENT = "delete_component"
    
    # Execution permissions
    VIEW_EXECUTION = "view_execution"
    CONTROL_EXECUTION = "control_execution"
    
    # System permissions
    MANAGE_USERS = "manage_users"
    VIEW_METRICS = "view_metrics"
    SYSTEM_CONFIG = "system_config"

class LoginRequest(BaseModel):
    """Login request schema"""
    username: str
    password: str
    remember_me: bool = False

class RegisterRequest(BaseModel):
    """User registration schema"""
    username: str
    email: EmailStr
    password: str
    full_name: Optional[str] = None
    
    @validator('username')
    def validate_username(cls, v):
        if len(v) < 3:
            raise ValueError('Username must be at least 3 characters long')
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Username can only contain letters, numbers, hyphens, and underscores')
        return v
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < PASSWORD_MIN_LENGTH:
            raise ValueError(f'Password must be at least {PASSWORD_MIN_LENGTH} characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v

class TokenResponse(BaseModel):
    """Token response schema"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: Dict[str, Any]

class UserInfo(BaseModel):
    """User information schema"""
    id: str
    username: str
    email: str
    full_name: Optional[str]
    roles: List[UserRole]
    permissions: List[Permission]
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime]

# ============================================================================
# ROLE-BASED ACCESS CONTROL
# ============================================================================

# Role to permissions mapping
ROLE_PERMISSIONS = {
    UserRole.ADMIN: [
        # All permissions
        Permission.CREATE_DESIGN, Permission.READ_DESIGN, Permission.UPDATE_DESIGN, 
        Permission.DELETE_DESIGN, Permission.EXECUTE_DESIGN,
        Permission.CREATE_COMPONENT, Permission.UPDATE_COMPONENT, Permission.DELETE_COMPONENT,
        Permission.VIEW_EXECUTION, Permission.CONTROL_EXECUTION,
        Permission.MANAGE_USERS, Permission.VIEW_METRICS, Permission.SYSTEM_CONFIG
    ],
    UserRole.DESIGNER: [
        # Design and execution permissions
        Permission.CREATE_DESIGN, Permission.READ_DESIGN, Permission.UPDATE_DESIGN, 
        Permission.DELETE_DESIGN, Permission.EXECUTE_DESIGN,
        Permission.VIEW_EXECUTION, Permission.CONTROL_EXECUTION,
        Permission.VIEW_METRICS
    ],
    UserRole.VIEWER: [
        # Read-only permissions
        Permission.READ_DESIGN, Permission.VIEW_EXECUTION, Permission.VIEW_METRICS
    ],
    UserRole.API_USER: [
        # API access permissions
        Permission.CREATE_DESIGN, Permission.READ_DESIGN, Permission.UPDATE_DESIGN,
        Permission.EXECUTE_DESIGN, Permission.VIEW_EXECUTION
    ]
}

def get_permissions_for_roles(roles: List[UserRole]) -> List[Permission]:
    """Get all permissions for a list of roles"""
    permissions = set()
    for role in roles:
        permissions.update(ROLE_PERMISSIONS.get(role, []))
    return list(permissions)

# ============================================================================
# AUTHENTICATION UTILITIES
# ============================================================================

class PasswordManager:
    """Password hashing and verification utilities"""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password"""
        return pwd_context.hash(password)
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    @staticmethod
    def generate_reset_token() -> str:
        """Generate a password reset token"""
        return secrets.token_urlsafe(32)

class TokenManager:
    """JWT token management"""
    
    @staticmethod
    def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create an access token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire, "type": "access"})
        
        return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    
    @staticmethod
    def create_refresh_token(data: Dict[str, Any]) -> str:
        """Create a refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=JWT_REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode.update({"exp": expire, "type": "refresh"})
        
        return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    
    @staticmethod
    def verify_token(token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode a token"""
        try:
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    @staticmethod
    def refresh_access_token(refresh_token: str) -> str:
        """Create new access token from refresh token"""
        payload = TokenManager.verify_token(refresh_token)
        
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid refresh token")
        
        # Create new access token
        access_data = {
            "sub": payload["sub"],
            "username": payload["username"],
            "roles": payload["roles"]
        }
        
        return TokenManager.create_access_token(access_data)

# ============================================================================
# AUTHENTICATION SERVICE
# ============================================================================

class AuthenticationService:
    """Main authentication service"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.password_manager = PasswordManager()
        self.token_manager = TokenManager()
    
    async def register_user(self, registration: RegisterRequest, 
                          db_session: AsyncSession) -> UserInfo:
        """Register a new user"""
        # Check if username already exists
        existing_user = await self._get_user_by_username(registration.username, db_session)
        if existing_user:
            raise HTTPException(status_code=400, detail="Username already exists")
        
        # Check if email already exists
        existing_email = await self._get_user_by_email(registration.email, db_session)
        if existing_email:
            raise HTTPException(status_code=400, detail="Email already exists")
        
        # Create user
        hashed_password = self.password_manager.hash_password(registration.password)
        
        user = User(
            username=registration.username,
            email=registration.email,
            hashed_password=hashed_password,
            full_name=registration.full_name,
            is_active=True,
            preferences={"default_role": UserRole.DESIGNER.value}
        )
        
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)
        
        logger.info(f"User registered: {registration.username}")
        
        return await self._user_to_info(user)
    
    async def authenticate_user(self, login: LoginRequest, 
                              db_session: AsyncSession) -> TokenResponse:
        """Authenticate user and return tokens"""
        # Check rate limiting
        if await self._is_rate_limited(login.username):
            raise HTTPException(
                status_code=429, 
                detail=f"Too many login attempts. Please try again in {LOCKOUT_DURATION_MINUTES} minutes."
            )
        
        # Get user
        user = await self._get_user_by_username(login.username, db_session)
        if not user:
            await self._record_failed_attempt(login.username)
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Verify password
        if not self.password_manager.verify_password(login.password, user.hashed_password):
            await self._record_failed_attempt(login.username)
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Check if user is active
        if not user.is_active:
            raise HTTPException(status_code=401, detail="Account is deactivated")
        
        # Clear failed attempts
        await self._clear_failed_attempts(login.username)
        
        # Update last login
        user.last_login = datetime.now()
        await db_session.commit()
        
        # Get user roles and permissions
        user_info = await self._user_to_info(user)
        
        # Create tokens
        token_data = {
            "sub": str(user.id),
            "username": user.username,
            "roles": user_info.roles
        }
        
        expires_delta = timedelta(days=30) if login.remember_me else None
        access_token = self.token_manager.create_access_token(token_data, expires_delta)
        refresh_token = self.token_manager.create_refresh_token(token_data)
        
        # Store refresh token in Redis
        if self.redis_client:
            await self.redis_client.setex(
                f"refresh_token:{user.id}",
                JWT_REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,
                refresh_token
            )
        
        logger.info(f"User authenticated: {user.username}")
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user=user_info.dict()
        )
    
    async def refresh_token(self, refresh_token: str, db_session: AsyncSession) -> TokenResponse:
        """Refresh access token"""
        payload = self.token_manager.verify_token(refresh_token)
        
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid refresh token")
        
        user_id = payload["sub"]
        
        # Verify refresh token exists in Redis
        if self.redis_client:
            stored_token = await self.redis_client.get(f"refresh_token:{user_id}")
            if not stored_token or stored_token != refresh_token:
                raise HTTPException(status_code=401, detail="Invalid refresh token")
        
        # Get user
        user = await self._get_user_by_id(user_id, db_session)
        if not user or not user.is_active:
            raise HTTPException(status_code=401, detail="User not found or inactive")
        
        # Create new tokens
        user_info = await self._user_to_info(user)
        token_data = {
            "sub": str(user.id),
            "username": user.username,
            "roles": user_info.roles
        }
        
        new_access_token = self.token_manager.create_access_token(token_data)
        new_refresh_token = self.token_manager.create_refresh_token(token_data)
        
        # Update refresh token in Redis
        if self.redis_client:
            await self.redis_client.setex(
                f"refresh_token:{user.id}",
                JWT_REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,
                new_refresh_token
            )
        
        return TokenResponse(
            access_token=new_access_token,
            refresh_token=new_refresh_token,
            expires_in=JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user=user_info.dict()
        )
    
    async def logout_user(self, user_id: str):
        """Logout user and invalidate refresh token"""
        if self.redis_client:
            await self.redis_client.delete(f"refresh_token:{user_id}")
        
        logger.info(f"User logged out: {user_id}")
    
    async def get_current_user(self, token: str, db_session: AsyncSession) -> UserInfo:
        """Get current user from token"""
        payload = self.token_manager.verify_token(token)
        
        if payload.get("type") != "access":
            raise HTTPException(status_code=401, detail="Invalid access token")
        
        user_id = payload["sub"]
        user = await self._get_user_by_id(user_id, db_session)
        
        if not user or not user.is_active:
            raise HTTPException(status_code=401, detail="User not found or inactive")
        
        return await self._user_to_info(user)
    
    async def _get_user_by_username(self, username: str, db_session: AsyncSession) -> Optional[User]:
        """Get user by username"""
        query = select(User).where(User.username == username)
        result = await db_session.execute(query)
        return result.scalar_one_or_none()
    
    async def _get_user_by_email(self, email: str, db_session: AsyncSession) -> Optional[User]:
        """Get user by email"""
        query = select(User).where(User.email == email)
        result = await db_session.execute(query)
        return result.scalar_one_or_none()
    
    async def _get_user_by_id(self, user_id: str, db_session: AsyncSession) -> Optional[User]:
        """Get user by ID"""
        query = select(User).where(User.id == user_id)
        result = await db_session.execute(query)
        return result.scalar_one_or_none()
    
    async def _user_to_info(self, user: User) -> UserInfo:
        """Convert User model to UserInfo"""
        # Get user roles (simplified - in production this might come from a separate table)
        roles = [UserRole.DESIGNER]  # Default role
        if user.is_admin:
            roles.append(UserRole.ADMIN)
        
        # Get permissions
        permissions = get_permissions_for_roles(roles)
        
        return UserInfo(
            id=str(user.id),
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            roles=roles,
            permissions=permissions,
            is_active=user.is_active,
            created_at=user.created_at,
            last_login=user.last_login
        )
    
    async def _is_rate_limited(self, username: str) -> bool:
        """Check if user is rate limited"""
        if not self.redis_client:
            return False
        
        key = f"login_attempts:{username}"
        attempts = await self.redis_client.get(key)
        
        return attempts and int(attempts) >= MAX_LOGIN_ATTEMPTS
    
    async def _record_failed_attempt(self, username: str):
        """Record a failed login attempt"""
        if not self.redis_client:
            return
        
        key = f"login_attempts:{username}"
        current = await self.redis_client.get(key)
        
        if current:
            attempts = int(current) + 1
        else:
            attempts = 1
        
        await self.redis_client.setex(
            key,
            LOCKOUT_DURATION_MINUTES * 60,
            attempts
        )
    
    async def _clear_failed_attempts(self, username: str):
        """Clear failed login attempts"""
        if not self.redis_client:
            return
        
        await self.redis_client.delete(f"login_attempts:{username}")

# ============================================================================
# AUTHORIZATION DECORATORS AND DEPENDENCIES
# ============================================================================

def require_permissions(*required_perms: Permission):
    """Decorator to require specific permissions"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Get current user from kwargs
            current_user = kwargs.get('current_user')
            if not current_user:
                raise HTTPException(status_code=401, detail="Authentication required")
            
            # Check permissions
            user_perms = set(current_user.permissions)
            required_perms_set = set(required_perms)
            
            if not required_perms_set.issubset(user_perms):
                missing_perms = required_perms_set - user_perms
                raise HTTPException(
                    status_code=403, 
                    detail=f"Missing required permissions: {[p.value for p in missing_perms]}"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def require_roles(*required_roles: UserRole):
    """Decorator to require specific roles"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            current_user = kwargs.get('current_user')
            if not current_user:
                raise HTTPException(status_code=401, detail="Authentication required")
            
            user_roles = set(current_user.roles)
            required_roles_set = set(required_roles)
            
            if not required_roles_set.intersection(user_roles):
                raise HTTPException(
                    status_code=403,
                    detail=f"Required roles: {[r.value for r in required_roles]}"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# ============================================================================
# FASTAPI DEPENDENCIES
# ============================================================================

# Global authentication service
auth_service = None

async def get_auth_service() -> AuthenticationService:
    """Get authentication service instance"""
    global auth_service
    
    if auth_service is None:
        try:
            redis_client = await get_redis_client()
            auth_service = AuthenticationService(redis_client)
        except:
            # Fallback without Redis
            auth_service = AuthenticationService()
        
        logger.info("✅ Authentication service initialized")
    
    return auth_service

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db_session: AsyncSession = Depends(get_db_session),
    auth_svc: AuthenticationService = Depends(get_auth_service)
) -> UserInfo:
    """FastAPI dependency to get current authenticated user"""
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    return await auth_svc.get_current_user(credentials.credentials, db_session)

async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db_session: AsyncSession = Depends(get_db_session),
    auth_svc: AuthenticationService = Depends(get_auth_service)
) -> Optional[UserInfo]:
    """FastAPI dependency to get current user (optional)"""
    if not credentials:
        return None
    
    try:
        return await auth_svc.get_current_user(credentials.credentials, db_session)
    except HTTPException:
        return None

def require_permission(permission: Permission):
    """FastAPI dependency factory for permission checking"""
    def permission_checker(current_user: UserInfo = Depends(get_current_user)) -> UserInfo:
        if permission not in current_user.permissions:
            raise HTTPException(
                status_code=403,
                detail=f"Missing required permission: {permission.value}"
            )
        return current_user
    
    return permission_checker

def require_role(role: UserRole):
    """FastAPI dependency factory for role checking"""
    def role_checker(current_user: UserInfo = Depends(get_current_user)) -> UserInfo:
        if role not in current_user.roles:
            raise HTTPException(
                status_code=403,
                detail=f"Required role: {role.value}"
            )
        return current_user
    
    return role_checker

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

async def create_initial_admin(username: str = "admin", password: str = "admin123", 
                             email: str = "admin@example.com"):
    """Create initial admin user"""
    try:
        auth_svc = await get_auth_service()
        
        async for db_session in get_db_session():
            # Check if admin already exists
            existing_admin = await auth_svc._get_user_by_username(username, db_session)
            if existing_admin:
                logger.info("Admin user already exists")
                return
            
            # Create admin user
            hashed_password = auth_svc.password_manager.hash_password(password)
            
            admin_user = User(
                username=username,
                email=email,
                hashed_password=hashed_password,
                full_name="System Administrator",
                is_active=True,
                is_admin=True,
                preferences={"created_by": "system"}
            )
            
            db_session.add(admin_user)
            await db_session.commit()
            
            logger.info(f"✅ Initial admin user created: {username}")
            break
            
    except Exception as e:
        logger.error(f"Failed to create initial admin user: {e}")

async def hash_existing_passwords():
    """Utility to hash existing plain text passwords in database"""
    try:
        async for db_session in get_db_session():
            # This would be used for migrating existing users
            # Implementation depends on your migration needs
            pass
    except Exception as e:
        logger.error(f"Failed to hash existing passwords: {e}")
