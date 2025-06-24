# backend/auth/rbac.py
"""
Role-Based Access Control (RBAC) System
=======================================
Consistent role enforcement across all endpoints
"""

import logging
from enum import Enum
from functools import wraps
from typing import List, Dict, Any, Optional, Set

from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from backend.auth.auth_system import get_current_user_azure, AzureUserInfo
from backend.auth.local_auth import LocalAuthService, get_local_auth_service
from backend.database.connection import get_db_session

logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)

# NEW - Role definitions
class UserRole(str, Enum):
    """System roles with hierarchy"""
    SUPER_ADMIN = "super_admin"      # Highest level - system administration
    ADMIN = "admin"                  # Application administration
    DESIGNER = "designer"            # Can create and edit designs
    ANALYST = "analyst"              # Can view and analyze data
    USER = "user"                    # Basic user access
    VIEWER = "viewer"                # Read-only access

# NEW - Permission definitions
class Permission(str, Enum):
    """Granular permissions"""
    # Design permissions
    CREATE_DESIGN = "create_design"
    READ_DESIGN = "read_design"
    UPDATE_DESIGN = "update_design"
    DELETE_DESIGN = "delete_design"
    EXECUTE_DESIGN = "execute_design"
    SHARE_DESIGN = "share_design"
    
    # Component permissions
    CREATE_COMPONENT = "create_component"
    UPDATE_COMPONENT = "update_component"
    DELETE_COMPONENT = "delete_component"
    
    # User management
    CREATE_USER = "create_user"
    UPDATE_USER = "update_user"
    DELETE_USER = "delete_user"
    VIEW_USERS = "view_users"
    
    # System administration
    SYSTEM_CONFIG = "system_config"
    VIEW_METRICS = "view_metrics"
    MANAGE_INFRASTRUCTURE = "manage_infrastructure"
    
    # Analytics
    VIEW_ANALYTICS = "view_analytics"
    EXPORT_DATA = "export_data"

# NEW - Role-Permission mapping
ROLE_PERMISSIONS: Dict[UserRole, Set[Permission]] = {
    UserRole.SUPER_ADMIN: {
        # All permissions
        Permission.CREATE_DESIGN, Permission.READ_DESIGN, Permission.UPDATE_DESIGN,
        Permission.DELETE_DESIGN, Permission.EXECUTE_DESIGN, Permission.SHARE_DESIGN,
        Permission.CREATE_COMPONENT, Permission.UPDATE_COMPONENT, Permission.DELETE_COMPONENT,
        Permission.CREATE_USER, Permission.UPDATE_USER, Permission.DELETE_USER, Permission.VIEW_USERS,
        Permission.SYSTEM_CONFIG, Permission.VIEW_METRICS, Permission.MANAGE_INFRASTRUCTURE,
        Permission.VIEW_ANALYTICS, Permission.EXPORT_DATA
    },
    UserRole.ADMIN: {
        Permission.CREATE_DESIGN, Permission.READ_DESIGN, Permission.UPDATE_DESIGN,
        Permission.DELETE_DESIGN, Permission.EXECUTE_DESIGN, Permission.SHARE_DESIGN,
        Permission.CREATE_COMPONENT, Permission.UPDATE_COMPONENT, Permission.DELETE_COMPONENT,
        Permission.VIEW_USERS, Permission.VIEW_METRICS,
        Permission.VIEW_ANALYTICS, Permission.EXPORT_DATA
    },
    UserRole.DESIGNER: {
        Permission.CREATE_DESIGN, Permission.READ_DESIGN, Permission.UPDATE_DESIGN,
        Permission.DELETE_DESIGN, Permission.EXECUTE_DESIGN, Permission.SHARE_DESIGN,
        Permission.VIEW_ANALYTICS
    },
    UserRole.ANALYST: {
        Permission.READ_DESIGN, Permission.EXECUTE_DESIGN,
        Permission.VIEW_ANALYTICS, Permission.EXPORT_DATA
    },
    UserRole.USER: {
        Permission.READ_DESIGN, Permission.CREATE_DESIGN, Permission.UPDATE_DESIGN,
        Permission.EXECUTE_DESIGN
    },
    UserRole.VIEWER: {
        Permission.READ_DESIGN
    }
}

# NEW - User info unified model
class UnifiedUserInfo:
    """Unified user info from both Azure AD and local auth"""
    
    def __init__(self, user_id: str, email: str, name: str, roles: List[str], 
                 auth_method: str, **kwargs):
        self.user_id = user_id
        self.email = email
        self.name = name
        self.roles = [UserRole(role) for role in roles if role in [r.value for r in UserRole]]
        self.auth_method = auth_method  # 'azure' or 'local'
        self.extra_data = kwargs
    
    def has_role(self, role: UserRole) -> bool:
        """Check if user has specific role"""
        return role in self.roles
    
    def has_any_role(self, roles: List[UserRole]) -> bool:
        """Check if user has any of the specified roles"""
        return any(role in self.roles for role in roles)
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if user has specific permission"""
        for role in self.roles:
            if permission in ROLE_PERMISSIONS.get(role, set()):
                return True
        return False
    
    def has_any_permission(self, permissions: List[Permission]) -> bool:
        """Check if user has any of the specified permissions"""
        return any(self.has_permission(perm) for perm in permissions)
    
    def get_all_permissions(self) -> Set[Permission]:
        """Get all permissions for user"""
        all_permissions = set()
        for role in self.roles:
            all_permissions.update(ROLE_PERMISSIONS.get(role, set()))
        return all_permissions

# NEW - RBAC Service
class RBACService:
    """Role-based access control service"""
    
    @staticmethod
    def check_permission(user: UnifiedUserInfo, permission: Permission) -> bool:
        """Check if user has permission"""
        return user.has_permission(permission)
    
    @staticmethod
    def check_role(user: UnifiedUserInfo, role: UserRole) -> bool:
        """Check if user has role"""
        return user.has_role(role)
    
    @staticmethod
    def enforce_permission(user: UnifiedUserInfo, permission: Permission):
        """Enforce permission or raise HTTPException"""
        if not user.has_permission(permission):
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions. Required: {permission.value}"
            )
    
    @staticmethod
    def enforce_role(user: UnifiedUserInfo, role: UserRole):
        """Enforce role or raise HTTPException"""
        if not user.has_role(role):
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient privileges. Required role: {role.value}"
            )
    
    @staticmethod
    def enforce_any_role(user: UnifiedUserInfo, roles: List[UserRole]):
        """Enforce any of the specified roles"""
        if not user.has_any_role(roles):
            role_names = [role.value for role in roles]
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient privileges. Required one of: {', '.join(role_names)}"
            )

# NEW - Unified authentication dependency
async def get_current_user_unified(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    local_auth: LocalAuthService = Depends(get_local_auth_service),
    db_session = Depends(get_db_session)
) -> UnifiedUserInfo:
    """Get current user from either Azure AD or local auth"""
    
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    token = credentials.credentials
    
    try:
        # Try Azure AD first
        try:
            azure_user = await get_current_user_azure(credentials)
            return UnifiedUserInfo(
                user_id=azure_user.user_id,
                email=azure_user.email,
                name=azure_user.name,
                roles=azure_user.app_roles + azure_user.roles,
                auth_method="azure",
                tenant_id=azure_user.tenant_id,
                groups=azure_user.groups
            )
        except HTTPException as azure_error:
            # If Azure AD fails, try local auth
            if azure_error.status_code == 401:
                try:
                    payload = local_auth.verify_token(token)
                    if payload.get("auth_method") == "local":
                        user = await local_auth._get_user_by_email(payload["email"], db_session)
                        if user and user.is_active:
                            return UnifiedUserInfo(
                                user_id=str(user.id),
                                email=user.email,
                                name=user.full_name,
                                roles=user.roles,
                                auth_method="local",
                                is_email_verified=user.is_email_verified
                            )
                        else:
                            raise HTTPException(status_code=401, detail="User not found or inactive")
                    else:
                        raise HTTPException(status_code=401, detail="Invalid token format")
                except Exception as local_error:
                    logger.error(f"Local auth failed: {local_error}")
                    raise HTTPException(status_code=401, detail="Authentication failed")
            else:
                raise azure_error
                
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed")

# NEW - Role-based decorators and dependencies
def require_roles(required_roles: List[UserRole]):
    """Decorator to require specific roles"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract current_user from kwargs
            current_user = kwargs.get('current_user')
            if not current_user:
                raise HTTPException(status_code=401, detail="Authentication required")
            
            rbac = RBACService()
            rbac.enforce_any_role(current_user, required_roles)
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def require_permissions(required_permissions: List[Permission]):
    """Decorator to require specific permissions"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_user = kwargs.get('current_user')
            if not current_user:
                raise HTTPException(status_code=401, detail="Authentication required")
            
            rbac = RBACService()
            for permission in required_permissions:
                rbac.enforce_permission(current_user, permission)
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# NEW - FastAPI dependencies for common role checks
async def require_admin(current_user: UnifiedUserInfo = Depends(get_current_user_unified)) -> UnifiedUserInfo:
    """Require admin role"""
    RBACService.enforce_any_role(current_user, [UserRole.ADMIN, UserRole.SUPER_ADMIN])
    return current_user

async def require_designer(current_user: UnifiedUserInfo = Depends(get_current_user_unified)) -> UnifiedUserInfo:
    """Require designer role or higher"""
    RBACService.enforce_any_role(current_user, [UserRole.DESIGNER, UserRole.ADMIN, UserRole.SUPER_ADMIN])
    return current_user

async def require_analyst(current_user: UnifiedUserInfo = Depends(get_current_user_unified)) -> UnifiedUserInfo:
    """Require analyst role or higher"""
    RBACService.enforce_any_role(current_user, [UserRole.ANALYST, UserRole.DESIGNER, UserRole.ADMIN, UserRole.SUPER_ADMIN])
    return current_user

async def require_authenticated(current_user: UnifiedUserInfo = Depends(get_current_user_unified)) -> UnifiedUserInfo:
    """Require any authenticated user"""
    return current_user

# NEW - Permission-based dependencies
async def require_design_create(current_user: UnifiedUserInfo = Depends(get_current_user_unified)) -> UnifiedUserInfo:
    """Require design creation permission"""
    RBACService.enforce_permission(current_user, Permission.CREATE_DESIGN)
    return current_user

async def require_design_delete(current_user: UnifiedUserInfo = Depends(get_current_user_unified)) -> UnifiedUserInfo:
    """Require design deletion permission"""
    RBACService.enforce_permission(current_user, Permission.DELETE_DESIGN)
    return current_user

async def require_user_management(current_user: UnifiedUserInfo = Depends(get_current_user_unified)) -> UnifiedUserInfo:
    """Require user management permissions"""
    RBACService.enforce_permission(current_user, Permission.VIEW_USERS)
    return current_user