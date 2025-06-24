# backend/auth/azure_auth.py
"""
Azure AD Authentication Module for Enhanced CSP System
=====================================================
Provides server-side Azure AD token validation and user management
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Dict, List, Optional, Any

import httpx
import jwt
from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import redis.asyncio as redis

logger = logging.getLogger(__name__)

# ============================================================================
# MODELS
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

class TokenValidationResult(BaseModel):
    """Token validation result"""
    valid: bool
    user_info: Optional[AzureUserInfo] = None
    error: Optional[str] = None

# ============================================================================
# AZURE AD TOKEN VALIDATOR
# ============================================================================

class AzureADValidator:
    """Azure AD token validator with caching and error handling"""
    
    def __init__(self, tenant_id: str, client_id: str, redis_client: Optional[redis.Redis] = None):
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.redis_client = redis_client
        
        # Azure AD endpoints
        self.jwks_uri = f"https://login.microsoftonline.com/{tenant_id}/discovery/v2.0/keys"
        self.issuer = f"https://login.microsoftonline.com/{tenant_id}/v2.0"
        self.graph_endpoint = "https://graph.microsoft.com/v1.0"
        
        # Cache settings
        self.keys_cache_ttl = 3600  # 1 hour
        self.user_cache_ttl = 300   # 5 minutes
        
    async def _get_signing_keys(self) -> Dict[str, Any]:
        """Get Azure AD signing keys with caching"""
        cache_key = f"azure_keys:{self.tenant_id}"
        
        # Try cache first
        if self.redis_client:
            try:
                cached_keys = await self.redis_client.get(cache_key)
                if cached_keys:
                    return json.loads(cached_keys)
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")
        
        # Fetch from Azure AD
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(self.jwks_uri)
                response.raise_for_status()
                keys_data = response.json()
                
                # Cache the keys
                if self.redis_client:
                    try:
                        await self.redis_client.setex(
                            cache_key, 
                            self.keys_cache_ttl, 
                            json.dumps(keys_data)
                        )
                    except Exception as e:
                        logger.warning(f"Failed to cache keys: {e}")
                
                return keys_data
                
        except httpx.RequestError as e:
            logger.error(f"Failed to fetch Azure AD keys: {e}")
            raise HTTPException(503, "Unable to validate token - service unavailable")
        except Exception as e:
            logger.error(f"Unexpected error fetching keys: {e}")
            raise HTTPException(500, "Token validation error")
    
    async def _get_key_for_token(self, token: str) -> Dict[str, Any]:
        """Get the signing key for a specific token"""
        try:
            # Decode header without verification to get kid
            header = jwt.get_unverified_header(token)
            kid = header.get('kid')
            
            if not kid:
                raise HTTPException(401, "Token missing key ID")
            
            # Get signing keys
            keys_data = await self._get_signing_keys()
            keys = keys_data.get('keys', [])
            
            # Find matching key
            key = next((k for k in keys if k.get('kid') == kid), None)
            if not key:
                raise HTTPException(401, "Invalid token - key not found")
            
            return key
            
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token header: {e}")
            raise HTTPException(401, "Invalid token format")
    
    async def validate_token(self, token: str) -> TokenValidationResult:
        """Validate Azure AD access token"""
        try:
            # Get signing key
            key = await self._get_key_for_token(token)
            
            # Convert JWK to PEM format for PyJWT
            public_key = jwt.algorithms.RSAAlgorithm.from_jwk(key)
            
            # Decode and validate token
            decoded_token = jwt.decode(
                token,
                public_key,
                algorithms=['RS256'],
                audience=self.client_id,
                issuer=self.issuer,
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_aud": True,
                    "verify_iss": True
                }
            )
            
            # Extract user information
            user_info = await self._extract_user_info(decoded_token, token)
            
            return TokenValidationResult(
                valid=True,
                user_info=user_info
            )
            
        except jwt.ExpiredSignatureError:
            logger.info("Token expired")
            return TokenValidationResult(
                valid=False,
                error="Token expired"
            )
        except jwt.InvalidAudienceError:
            logger.warning("Invalid token audience")
            return TokenValidationResult(
                valid=False,
                error="Invalid token audience"
            )
        except jwt.InvalidIssuerError:
            logger.warning("Invalid token issuer")
            return TokenValidationResult(
                valid=False,
                error="Invalid token issuer"
            )
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return TokenValidationResult(
                valid=False,
                error=f"Invalid token: {str(e)}"
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Token validation error: {e}")
            return TokenValidationResult(
                valid=False,
                error="Token validation failed"
            )
    
    async def _extract_user_info(self, decoded_token: Dict[str, Any], access_token: str) -> AzureUserInfo:
        """Extract user information from token and optionally from Microsoft Graph"""
        
        # Basic info from token
        user_info = AzureUserInfo(
            user_id=decoded_token.get('oid', ''),
            email=decoded_token.get('preferred_username', '') or decoded_token.get('upn', ''),
            name=decoded_token.get('name', ''),
            tenant_id=decoded_token.get('tid', ''),
            roles=decoded_token.get('roles', []),
            app_roles=decoded_token.get('roles', [])
        )
        
        # Try to get additional info from Microsoft Graph
        try:
            groups = await self._get_user_groups(access_token, user_info.user_id)
            user_info.groups = groups
            
            # Map groups to application roles
            app_roles = self._map_groups_to_roles(groups)
            user_info.app_roles.extend(app_roles)
            
        except Exception as e:
            logger.warning(f"Failed to get user groups: {e}")
        
        return user_info
    
    async def _get_user_groups(self, access_token: str, user_id: str) -> List[str]:
        """Get user's group memberships from Microsoft Graph"""
        cache_key = f"user_groups:{user_id}"
        
        # Try cache first
        if self.redis_client:
            try:
                cached_groups = await self.redis_client.get(cache_key)
                if cached_groups:
                    return json.loads(cached_groups)
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")
        
        try:
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json'
            }
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.graph_endpoint}/me/memberOf",
                    headers=headers
                )
                
                if response.status_code == 200:
                    data = response.json()
                    groups = [group.get('displayName', '') for group in data.get('value', [])]
                    
                    # Cache the groups
                    if self.redis_client:
                        try:
                            await self.redis_client.setex(
                                cache_key,
                                self.user_cache_ttl,
                                json.dumps(groups)
                            )
                        except Exception as e:
                            logger.warning(f"Failed to cache groups: {e}")
                    
                    return groups
                else:
                    logger.warning(f"Graph API error: {response.status_code}")
                    return []
                    
        except Exception as e:
            logger.warning(f"Failed to fetch user groups: {e}")
            return []
    
    def _map_groups_to_roles(self, groups: List[str]) -> List[str]:
        """Map Azure AD groups to application roles"""
        role_mapping = {
            'CSP-Super-Admins': 'super_admin',
            'CSP-Administrators': 'admin',
            'CSP-Developers': 'developer',
            'CSP-Analysts': 'analyst',
            'CSP-Users': 'user'
        }
        
        mapped_roles = []
        for group in groups:
            if group in role_mapping:
                mapped_roles.append(role_mapping[group])
        
        # Default role if no groups match
        if not mapped_roles:
            mapped_roles.append('user')
        
        return mapped_roles

# ============================================================================
# FASTAPI DEPENDENCIES
# ============================================================================

# Security scheme
security = HTTPBearer()

# Global validator instance (will be initialized in main.py)
azure_validator: Optional[AzureADValidator] = None

def init_azure_validator(tenant_id: str, client_id: str, redis_client: Optional[redis.Redis] = None):
    """Initialize the global Azure AD validator"""
    global azure_validator
    azure_validator = AzureADValidator(tenant_id, client_id, redis_client)
    logger.info("Azure AD validator initialized")

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> AzureUserInfo:
    """FastAPI dependency to get current authenticated user"""
    if not azure_validator:
        raise HTTPException(500, "Azure AD validator not initialized")
    
    try:
        result = await azure_validator.validate_token(credentials.credentials)
        
        if not result.valid:
            raise HTTPException(401, result.error or "Authentication failed")
        
        return result.user_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(500, "Authentication service error")

async def get_current_user_optional(
    request: Request
) -> Optional[AzureUserInfo]:
    """FastAPI dependency to get current user if authenticated (optional)"""
    auth_header = request.headers.get('Authorization')
    
    if not auth_header or not auth_header.startswith('Bearer '):
        return None
    
    if not azure_validator:
        return None
    
    try:
        token = auth_header.split(' ')[1]
        result = await azure_validator.validate_token(token)
        return result.user_info if result.valid else None
    except Exception:
        return None

def require_role(required_roles: List[str]):
    """Decorator to require specific roles"""
    def role_checker(user: AzureUserInfo = Depends(get_current_user)) -> AzureUserInfo:
        user_roles = user.app_roles + user.roles
        
        if not any(role in user_roles for role in required_roles):
            raise HTTPException(
                403, 
                f"Insufficient permissions. Required roles: {required_roles}"
            )
        
        return user
    
    return role_checker

def require_permission(permission: str):
    """Decorator to require specific permission (simplified role checking)"""
    permission_to_roles = {
        'read': ['user', 'analyst', 'developer', 'admin', 'super_admin'],
        'write': ['developer', 'admin', 'super_admin'],
        'admin': ['admin', 'super_admin'],
        'super_admin': ['super_admin']
    }
    
    required_roles = permission_to_roles.get(permission, [])
    return require_role(required_roles)

# ============================================================================
# MIDDLEWARE
# ============================================================================

async def azure_auth_middleware(request: Request, call_next):
    """Middleware to validate Azure AD tokens on protected routes"""
    
    # Skip auth for public routes
    public_routes = [
        '/docs', '/redoc', '/openapi.json', '/health', 
        '/api/auth/info', '/api/system/health'
    ]
    
    if any(request.url.path.startswith(route) for route in public_routes):
        return await call_next(request)
    
    # Extract and validate token
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return JSONResponse(
            status_code=401,
            content={"error": "Missing or invalid authorization header"}
        )
    
    try:
        token = auth_header.split(' ')[1]
        
        if not azure_validator:
            raise HTTPException(500, "Authentication service not available")
        
        result = await azure_validator.validate_token(token)
        
        if not result.valid:
            return JSONResponse(
                status_code=401,
                content={"error": result.error or "Authentication failed"}
            )
        
        # Add user context to request
        request.state.user = result.user_info
        request.state.user_id = result.user_info.user_id
        request.state.user_roles = result.user_info.app_roles
        
        response = await call_next(request)
        return response
        
    except HTTPException as e:
        logger.warning(f"Authentication failed: {e.detail}")
        return JSONResponse(
            status_code=e.status_code,
            content={"error": e.detail}
        )
    except Exception as e:
        logger.error(f"Auth middleware error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Authentication service error"}
        )

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

async def get_user_permissions(user: AzureUserInfo) -> List[str]:
    """Get user permissions based on roles"""
    role_permissions = {
        'super_admin': ['read', 'write', 'admin', 'super_admin'],
        'admin': ['read', 'write', 'admin'],
        'developer': ['read', 'write'],
        'analyst': ['read'],
        'user': ['read']
    }
    
    permissions = set()
    for role in user.app_roles + user.roles:
        permissions.update(role_permissions.get(role, []))
    
    return list(permissions)

async def check_user_permission(user: AzureUserInfo, required_permission: str) -> bool:
    """Check if user has specific permission"""
    permissions = await get_user_permissions(user)
    return required_permission in permissions

# ============================================================================
# HEALTH CHECK
# ============================================================================

async def azure_auth_health_check() -> Dict[str, Any]:
    """Health check for Azure AD authentication"""
    try:
        if not azure_validator:
            return {"status": "unhealthy", "error": "Validator not initialized"}
        
        # Test connectivity to Azure AD
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(azure_validator.jwks_uri)
            response.raise_for_status()
        
        return {
            "status": "healthy",
            "tenant_id": azure_validator.tenant_id,
            "client_id": azure_validator.client_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }