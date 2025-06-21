# DDoS Protection for Enhanced CSP
# Implement rate limiting and request filtering

import time
import asyncio
from collections import defaultdict, deque
from typing import Dict, Optional
from fastapi import Request, HTTPException
from functools import wraps
import logging

logger = logging.getLogger(__name__)

class DDoSProtector:
    """Advanced DDoS protection with rate limiting and pattern detection"""
    
    def __init__(self):
        self.request_counts = defaultdict(deque)  # IP -> request timestamps
        self.blocked_ips = set()
        self.suspicious_patterns = defaultdict(int)
        
        # Configuration
        self.rate_limits = {
            'requests_per_minute': 60,
            'requests_per_second': 10,
            'burst_threshold': 20,  # Rapid requests trigger immediate block
            'block_duration': 300,  # 5 minutes
            'warning_threshold': 40,
            'critical_threshold': 100
        }
        
        # Whitelist for trusted IPs
        self.whitelist = {
            '127.0.0.1',
            '::1',
            'localhost'
        }
    
    def get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request"""
        # Check for forwarded headers first
        forwarded_for = request.headers.get('X-Forwarded-For')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        
        real_ip = request.headers.get('X-Real-IP')
        if real_ip:
            return real_ip
        
        # Fall back to direct connection
        return request.client.host if request.client else 'unknown'
    
    def is_rate_limited(self, client_ip: str) -> bool:
        """Check if client IP is rate limited"""
        if client_ip in self.whitelist:
            return False
        
        if client_ip in self.blocked_ips:
            return True
        
        now = time.time()
        requests = self.request_counts[client_ip]
        
        # Remove old requests (older than 1 minute)
        while requests and requests[0] < now - 60:
            requests.popleft()
        
        # Add current request
        requests.append(now)
        
        # Check rate limits
        recent_requests = len([req for req in requests if req > now - 60])  # Last minute
        burst_requests = len([req for req in requests if req > now - 10])   # Last 10 seconds
        
        # Immediate blocking for burst attacks
        if burst_requests >= self.rate_limits['burst_threshold']:
            self.blocked_ips.add(client_ip)
            logger.warning(f"DDOS: Blocked {client_ip} for burst attack ({burst_requests} requests in 10s)")
            return True
        
        # Rate limiting
        if recent_requests > self.rate_limits['requests_per_minute']:
            logger.warning(f"DDOS: Rate limited {client_ip} ({recent_requests} requests/min)")
            return True
        
        # Pattern detection
        if recent_requests > self.rate_limits['warning_threshold']:
            self.suspicious_patterns[client_ip] += 1
            logger.info(f"DDOS: Suspicious pattern from {client_ip} ({recent_requests} requests)")
        
        return False
    
    def analyze_request_pattern(self, request: Request) -> Dict[str, float]:
        """Analyze request for suspicious patterns"""
        suspicious_score = 0.0
        reasons = []
        
        # Check user agent
        user_agent = request.headers.get('User-Agent', '').lower()
        if not user_agent or 'bot' in user_agent or 'crawler' in user_agent:
            suspicious_score += 0.3
            reasons.append("suspicious_user_agent")
        
        # Check for missing common headers
        common_headers = ['Accept', 'Accept-Language', 'Accept-Encoding']
        missing_headers = sum(1 for h in common_headers if h not in request.headers)
        if missing_headers >= 2:
            suspicious_score += 0.2
            reasons.append("missing_headers")
        
        # Check request method patterns
        if request.method in ['POST', 'PUT', 'DELETE'] and not request.headers.get('Content-Type'):
            suspicious_score += 0.2
            reasons.append("missing_content_type")
        
        return {
            'score': suspicious_score,
            'reasons': reasons,
            'threshold_exceeded': suspicious_score > 0.5
        }
    
    async def cleanup_old_data(self):
        """Cleanup old request data and unblock IPs after timeout"""
        now = time.time()
        
        # Clean up old request counts
        for ip in list(self.request_counts.keys()):
            requests = self.request_counts[ip]
            while requests and requests[0] < now - 300:  # 5 minutes
                requests.popleft()
            
            if not requests:
                del self.request_counts[ip]
        
        # Unblock IPs after timeout (implement with timestamp tracking)
        # This is simplified - in production, store block timestamps
        if len(self.blocked_ips) > 100:  # Basic cleanup
            self.blocked_ips.clear()
            logger.info("DDOS: Cleared blocked IPs list")

# Middleware for FastAPI
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

ddos_protector = DDoSProtector()

async def ddos_protection_middleware(request: Request, call_next):
    """DDoS protection middleware"""
    client_ip = ddos_protector.get_client_ip(request)
    
    # Check rate limiting
    if ddos_protector.is_rate_limited(client_ip):
        logger.warning(f"DDOS: Blocked request from {client_ip}")
        return JSONResponse(
            status_code=429,
            content={
                "error": "Too Many Requests",
                "message": "Rate limit exceeded. Please try again later.",
                "retry_after": 60
            },
            headers={"Retry-After": "60"}
        )
    
    # Analyze request pattern
    pattern_analysis = ddos_protector.analyze_request_pattern(request)
    if pattern_analysis['threshold_exceeded']:
        logger.warning(f"DDOS: Suspicious pattern from {client_ip}: {pattern_analysis['reasons']}")
        # Could implement additional verification here (CAPTCHA, etc.)
    
    # Process request
    response = await call_next(request)
    
    return response

# Decorator for individual endpoints
def ddos_protection(max_requests_per_minute: int = 60):
    """Decorator for endpoint-specific rate limiting"""
    def decorator(func):
        endpoint_limits = defaultdict(deque)
        
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            client_ip = ddos_protector.get_client_ip(request)
            
            if client_ip in ddos_protector.whitelist:
                return await func(request, *args, **kwargs)
            
            now = time.time()
            requests = endpoint_limits[client_ip]
            
            # Remove old requests
            while requests and requests[0] < now - 60:
                requests.popleft()
            
            # Check limit
            if len(requests) >= max_requests_per_minute:
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded for this endpoint: {max_requests_per_minute} requests/minute"
                )
            
            requests.append(now)
            return await func(request, *args, **kwargs)
        
        return wrapper
    return decorator

# Integration with Enhanced CSP
app = FastAPI()

# Add middleware
app.middleware("http")(ddos_protection_middleware)

# Example protected endpoints
@app.post("/api/login")
@ddos_protection(max_requests_per_minute=10)  # Stricter limit for login
async def login_endpoint(request: Request, credentials: dict):
    """Login endpoint with DDoS protection"""
    # Your login logic here
    return {"status": "authenticated"}

@app.get("/api/processes")
@ddos_protection(max_requests_per_minute=30)
async def get_processes(request: Request):
    """Process listing with rate limiting"""
    # Your process logic here
    return {"processes": []}

# Background task for cleanup
@app.on_event("startup")
async def start_cleanup_task():
    """Start background cleanup task"""
    async def cleanup_loop():
        while True:
            await asyncio.sleep(300)  # Every 5 minutes
            await ddos_protector.cleanup_old_data()
    
    asyncio.create_task(cleanup_loop())

# Configuration endpoint for runtime adjustments
@app.post("/admin/ddos-config")
async def update_ddos_config(request: Request, config: dict):
    """Update DDoS protection configuration (admin only)"""
    # Implement admin authentication here
    ddos_protector.rate_limits.update(config)
    logger.info(f"DDOS: Configuration updated: {config}")
    return {"status": "updated", "config": ddos_protector.rate_limits}

# Monitoring endpoint
@app.get("/admin/ddos-status")
async def get_ddos_status(request: Request):
    """Get current DDoS protection status"""
    return {
        "blocked_ips": len(ddos_protector.blocked_ips),
        "active_sessions": len(ddos_protector.request_counts),
        "suspicious_patterns": len(ddos_protector.suspicious_patterns),
        "rate_limits": ddos_protector.rate_limits
    }
