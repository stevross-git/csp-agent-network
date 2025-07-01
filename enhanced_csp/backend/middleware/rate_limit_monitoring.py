"""
Rate limiting middleware with monitoring
"""
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable
import time

try:
    from monitoring import get_default
    monitor = get_default()
    MONITORING_ENABLED = True
except ImportError:
    monitor = None
    MONITORING_ENABLED = False

class RateLimitMonitoringMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware with metrics collection"""
    
    def __init__(self, app, rate_limit: int = 100):
        super().__init__(app)
        self.rate_limit = rate_limit
        self.request_counts = {}
        self.window_start = time.time()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get client identifier
        client_id = request.client.host if request.client else "unknown"
        endpoint = request.url.path
        
        # Simple rate limiting (per minute)
        current_time = time.time()
        if current_time - self.window_start > 60:
            self.request_counts.clear()
            self.window_start = current_time
        
        # Check rate limit
        request_key = f"{client_id}:{endpoint}"
        self.request_counts[request_key] = self.request_counts.get(request_key, 0) + 1
        
        if self.request_counts[request_key] > self.rate_limit:
            # Rate limit exceeded
            if MONITORING_ENABLED:
                monitor.record_rate_limit_hit(endpoint, "per_minute")
                monitor.update_rate_limit_remaining(endpoint, client_id, 0)
            
            return Response(
                content="Rate limit exceeded",
                status_code=429,
                headers={"Retry-After": "60"}
            )
        
        # Update remaining capacity
        if MONITORING_ENABLED:
            remaining = self.rate_limit - self.request_counts[request_key]
            monitor.update_rate_limit_remaining(endpoint, client_id, remaining)
        
        # Process request
        response = await call_next(request)
        return response
