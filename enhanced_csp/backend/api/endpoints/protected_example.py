from fastapi import APIRouter, Depends, Request
from backend.middleware.auth_middleware import AuthMiddleware
from backend.services.auth.jwt_service import JWTService
from backend.config.security import SecurityConfig

router = APIRouter()

# Initialize services
security_config = SecurityConfig()
jwt_service = JWTService(security_config)
auth_middleware = AuthMiddleware(jwt_service)

@router.post("/api/login")
async def login(username: str, password: str):
    """Login endpoint - returns JWT tokens"""
    # TODO: Validate credentials against database
    # This is a simplified example
    
    if username == "admin" and password == "secure_password":
        user_id = "123"
        roles = ["admin", "user"]
        
        access_token = jwt_service.create_access_token(user_id, roles)
        refresh_token = jwt_service.create_refresh_token(user_id)
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer"
        }
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials"
    )

@router.get("/api/protected")
@auth_middleware.require_auth()
async def protected_route(request: Request):
    """Protected endpoint - requires valid JWT"""
    return {
        "message": "This is a protected resource",
        "user_id": request.state.user_id,
        "roles": request.state.user_roles
    }

@router.get("/api/admin-only")
@auth_middleware.require_auth(required_roles=["admin"])
async def admin_only_route(request: Request):
    """Admin-only endpoint - requires admin role"""
    return {
        "message": "This is an admin-only resource",
        "user_id": request.state.user_id
    }