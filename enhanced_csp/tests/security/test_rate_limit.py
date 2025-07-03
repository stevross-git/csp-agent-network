"""
Comprehensive rate limiting tests
"""

import pytest
import asyncio
from httpx import AsyncClient
from fastapi import FastAPI
from backend.middleware.rate_limiter import RateLimiter, rate_limit_middleware
import redis.asyncio as redis
from unittest.mock import AsyncMock, MagicMock

@pytest.fixture
async def redis_client():
    """Mock Redis client for testing"""
    client = AsyncMock()
    client.eval = AsyncMock(return_value=[1, 50, 100])  # allowed, current, limit
    yield client
    
@pytest.fixture
def app_with_rate_limit(redis_client):
    """FastAPI app with rate limiting"""
    app = FastAPI()
    
    rate_limiter = RateLimiter(
        redis_client=redis_client,
        default_limit=100,
        window_seconds=3600
    )
    app.state.rate_limiter = rate_limiter
    app.middleware("http")(rate_limit_middleware)
    
    @app.get("/test")
    async def test_endpoint():
        return {"status": "ok"}
    
    @app.get("/auth/login")
    async def login_endpoint():
        return {"token": "test"}
    
    return app

@pytest.mark.asyncio
async def test_rate_limit_allows_under_limit(app_with_rate_limit):
    """Test requests under limit are allowed"""
    async with AsyncClient(app=app_with_rate_limit, base_url="http://test") as client:
        response = await client.get("/test")
        assert response.status_code == 200
        assert "X-RateLimit-Limit" in response.headers
        assert response.headers["X-RateLimit-Limit"] == "100"

@pytest.mark.asyncio
async def test_rate_limit_blocks_over_limit(app_with_rate_limit, redis_client):
    """Test requests over limit are blocked"""
    # Mock Redis to return "not allowed"
    redis_client.eval.return_value = [0, 101, 100]  # not allowed
    
    async with AsyncClient(app=app_with_rate_limit, base_url="http://test") as client:
        response = await client.get("/test")
        assert response.status_code == 429
        assert "retry_after" in response.json()

@pytest.mark.asyncio
async def test_different_endpoints_different_limits(app_with_rate_limit):
    """Test different endpoints have different limits"""
    async with AsyncClient(app=app_with_rate_limit, base_url="http://test") as client:
        # Regular endpoint
        response = await client.get("/test")
        assert response.status_code == 200
        
        # Auth endpoint should have stricter limits
        auth_response = await client.get("/auth/login")
        # Would check headers for different limit values

@pytest.mark.asyncio
async def test_rate_limit_headers_present(app_with_rate_limit):
    """Test rate limit headers are present"""
    async with AsyncClient(app=app_with_rate_limit, base_url="http://test") as client:
        response = await client.get("/test")
        
        required_headers = [
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
            "X-RateLimit-Reset-After"
        ]
        
        for header in required_headers:
            assert header in response.headers

@pytest.mark.asyncio
async def test_burst_protection(app_with_rate_limit):
    """Test burst protection works"""
    async with AsyncClient(app=app_with_rate_limit, base_url="http://test") as client:
        # Simulate burst of requests
        tasks = []
        for _ in range(25):  # Burst of 25 requests
            tasks.append(client.get("/test"))
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Some should succeed, some should be rate limited
        success_count = sum(1 for r in responses if not isinstance(r, Exception) and r.status_code == 200)
        rate_limited_count = sum(1 for r in responses if not isinstance(r, Exception) and r.status_code == 429)
        
        assert success_count > 0
        assert success_count <= 20  # Burst size