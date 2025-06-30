from functools import wraps
from typing import Optional, Callable
from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

class AuthMiddleware:
    """Authentication middleware for FastAPI"""
    
    def __init__(self, jwt_service: JWTService):
        self.jwt_service = jwt_service
        
    async def verify_token(
        self, 
        credentials: HTTPAuthorizationCredentials = security
    ) -> Dict:
        """Verify JWT token from request"""
        token = credentials.credentials
        
        payload = self.jwt_service.validate_token(token)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        return payload
    
    def require_auth(self, required_roles: Optional[list] = None):
        """Decorator to require authentication"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Extract request from args
                request = None
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break
                
                if not request:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Request object not found"
                    )
                
                # Get token from header
                auth_header = request.headers.get("Authorization")
                if not auth_header or not auth_header.startswith("Bearer "):
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Missing authentication token",
                        headers={"WWW-Authenticate": "Bearer"},
                    )
                
                token = auth_header.split(" ")[1]
                payload = self.jwt_service.validate_token(token)
                
                if not payload:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid or expired token",
                        headers={"WWW-Authenticate": "Bearer"},
                    )
                
                # Check roles if required
                if required_roles:
                    user_roles = payload.get('roles', [])
                    if not any(role in user_roles for role in required_roles):
                        raise HTTPException(
                            status_code=status.HTTP_403_FORBIDDEN,
                            detail="Insufficient permissions"
                        )
                
                # Add user info to request state
                request.state.user_id = payload['user_id']
                request.state.user_roles = payload.get('roles', [])
                
                return await func(*args, **kwargs)
            
            return wrapper
        return decorator