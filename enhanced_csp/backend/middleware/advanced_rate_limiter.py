# File: backend/middleware/advanced_rate_limiter.py
"""
Advanced rate limiter with per-route configuration
"""

from typing import Dict, Optional, Callable
from fastapi import Request, HTTPException
from functools import wraps
import inspect

class ConfigurableRateLimiter(RateLimiter):
    """Rate limiter with per-route configuration"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.route_limits: Dict[str, tuple] = {}
    
    def limit(self, rate_string: str):
        """Decorator for per-route rate limits"""
        def decorator(func: Callable) -> Callable:
            # Parse rate string (e.g., "5/minute", "100/hour")
            limit, period = self._parse_rate_string(rate_string)
            
            # Store route configuration
            endpoint_name = f"{func.__module__}.{func.__name__}"
            self.route_limits[endpoint_name] = (limit, period)
            
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Rate limit will be checked by middleware
                return await func(*args, **kwargs)
            
            # Mark function with rate limit info
            wrapper._rate_limit = rate_string
            return wrapper
        
        return decorator
    
    def _parse_rate_string(self, rate_string: str) -> tuple:
        """Parse rate limit string like '5/minute' """
        parts = rate_string.split("/")
        limit = int(parts[0])
        
        period_map = {
            "second": 1,
            "minute": 60,
            "hour": 3600,
            "day": 86400
        }
        
        period = parts[1].rstrip("s")  # Remove plural
        seconds = period_map.get(period, 3600)
        
        return limit, seconds
    
    async def check_route_limit(self, request: Request) -> tuple:
        """Check rate limit for specific route"""
        # Get route info from request
        route = request.scope.get("route")
        if route and hasattr(route.endpoint, "_rate_limit"):
            rate_string = route.endpoint._rate_limit
            limit, period = self._parse_rate_string(rate_string)
            
            # Use route-specific limits
            return await self.check_rate_limit(
                request,
                endpoint=str(route.path),
                limit=limit,
                window=period
            )
        
        # Fall back to default
        return await self.check_rate_limit(request)

# Usage example:
"""
from backend.middleware.advanced_rate_limiter import ConfigurableRateLimiter

rate_limiter = ConfigurableRateLimiter(redis_client)

@router.post("/api/auth/login")
@rate_limiter.limit("5/minute")
async def login(credentials: LoginRequest):
    # Strict rate limit for auth endpoints
    pass

@router.get("/api/public/status")
@rate_limiter.limit("1000/hour")
async def status():
    # More relaxed limit for public endpoints
    pass

@router.post("/api/quantum/entangle")
@rate_limiter.limit("10/hour")
async def quantum_entangle():
    # Expensive operations get strict limits
    pass
"""