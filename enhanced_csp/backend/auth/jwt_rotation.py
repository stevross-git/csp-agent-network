# ==============================================================================
# JWT DUAL-KEY ROTATION SYSTEM - Zero Downtime Implementation
# ==============================================================================

# File: backend/auth/jwt_rotation.py
"""
Production-ready JWT rotation with zero-downtime support.
Validates against both current and previous keys during rotation period.
"""

import os
import jwt
import json
from typing import Optional, Dict, List, Tuple, Union
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from functools import lru_cache
import asyncio
from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import redis.asyncio as redis
import logging
from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)

@dataclass
class JWTKey:
    """Represents a JWT signing key with metadata"""
    key_id: str
    secret: str
    algorithm: str
    created_at: datetime
    expires_at: datetime
    is_active: bool = True
    
    def is_expired(self) -> bool:
        """Check if key has expired"""
        return datetime.now(timezone.utc) > self.expires_at

@dataclass
class JWTKeySet:
    """Manages current and previous JWT keys for rotation"""
    context: str  # 'rest', 'websocket', 'refresh'
    current: JWTKey
    previous: Optional[JWTKey] = None
    rotation_overlap_hours: int = 24  # How long to accept old tokens
    
    def get_signing_key(self) -> JWTKey:
        """Get the current key for signing new tokens"""
        if self.current.is_expired():
            raise ValueError(f"Current {self.context} key has expired")
        return self.current
    
    def get_verification_keys(self) -> List[JWTKey]:
        """Get all keys that should be tried for verification"""
        keys = [self.current]
        
        # Include previous key if within rotation window
        if self.previous and not self._is_rotation_complete():
            keys.append(self.previous)
        
        return keys
    
    def _is_rotation_complete(self) -> bool:
        """Check if rotation grace period has ended"""
        if not self.previous:
            return True
        
        rotation_end = self.previous.expires_at + timedelta(hours=self.rotation_overlap_hours)
        return datetime.now(timezone.utc) > rotation_end

class JWTRotationManager:
    """
    Manages JWT key rotation with zero downtime.
    Supports multiple contexts (REST, WebSocket, Refresh tokens).
    """
    
    def __init__(
        self, 
        redis_client: redis.Redis,
        key_lifetime_days: int = 30,
        rotation_overlap_hours: int = 24
    ):
        self.redis = redis_client
        self.key_lifetime_days = key_lifetime_days
        self.rotation_overlap_hours = rotation_overlap_hours
        self._key_sets: Dict[str, JWTKeySet] = {}
        self._lock = asyncio.Lock()
        
    async def initialize(self):
        """Initialize key sets from environment or Redis"""
        contexts = ['rest', 'websocket', 'refresh']
        
        for context in contexts:
            key_set = await self._load_or_create_keyset(context)
            self._key_sets[context] = key_set
            
        # Start background rotation checker
        asyncio.create_task(self._rotation_checker())
    
    async def _load_or_create_keyset(self, context: str) -> JWTKeySet:
        """Load existing keyset from Redis or create new one"""
        # Try to load from Redis
        current_data = await self.redis.get(f"jwt:keys:{context}:current")
        previous_data = await self.redis.get(f"jwt:keys:{context}:previous")
        
        if current_data:
            current = JWTKey(**json.loads(current_data))
            previous = JWTKey(**json.loads(previous_data)) if previous_data else None
            
            # Check if current key needs rotation
            if current.is_expired() or self._should_rotate(current):
                return await self._rotate_keys(context, current)
            
            return JWTKeySet(
                context=context,
                current=current,
                previous=previous,
                rotation_overlap_hours=self.rotation_overlap_hours
            )
        else:
            # Create new keyset
            return await self._create_new_keyset(context)
    
    async def _create_new_keyset(self, context: str) -> JWTKeySet:
        """Create a new keyset for a context"""
        current = self._generate_key(context)
        
        # Store in Redis
        await self.redis.set(
            f"jwt:keys:{context}:current",
            json.dumps({
                'key_id': current.key_id,
                'secret': current.secret,
                'algorithm': current.algorithm,
                'created_at': current.created_at.isoformat(),
                'expires_at': current.expires_at.isoformat(),
                'is_active': current.is_active
            }),
            ex=self.key_lifetime_days * 24 * 3600 * 2  # Keep for 2x lifetime
        )
        
        logger.info(f"Created new JWT keyset for context: {context}")
        
        return JWTKeySet(
            context=context,
            current=current,
            rotation_overlap_hours=self.rotation_overlap_hours
        )
    
    def _generate_key(self, context: str) -> JWTKey:
        """Generate a new JWT key"""
        import secrets
        
        now = datetime.now(timezone.utc)
        key_id = f"{context}_{int(now.timestamp())}"
        
        # Use URL-safe base64 encoding
        secret = secrets.token_urlsafe(64)
        
        return JWTKey(
            key_id=key_id,
            secret=secret,
            algorithm="HS512",
            created_at=now,
            expires_at=now + timedelta(days=self.key_lifetime_days),
            is_active=True
        )
    
    def _should_rotate(self, key: JWTKey) -> bool:
        """Check if a key should be rotated (e.g., 7 days before expiry)"""
        days_until_expiry = (key.expires_at - datetime.now(timezone.utc)).days
        return days_until_expiry <= 7
    
    async def _rotate_keys(self, context: str, current: JWTKey) -> JWTKeySet:
        """Rotate keys for a context"""
        async with self._lock:
            # Generate new key
            new_key = self._generate_key(context)
            
            # Move current to previous
            await self.redis.set(
                f"jwt:keys:{context}:previous",
                await self.redis.get(f"jwt:keys:{context}:current"),
                ex=self.rotation_overlap_hours * 3600 * 2
            )
            
            # Set new current
            await self.redis.set(
                f"jwt:keys:{context}:current",
                json.dumps({
                    'key_id': new_key.key_id,
                    'secret': new_key.secret,
                    'algorithm': new_key.algorithm,
                    'created_at': new_key.created_at.isoformat(),
                    'expires_at': new_key.expires_at.isoformat(),
                    'is_active': new_key.is_active
                }),
                ex=self.key_lifetime_days * 24 * 3600 * 2
            )
            
            logger.warning(f"Rotated JWT keys for context: {context}")
            
            # Notify monitoring
            await self._notify_rotation(context, new_key.key_id)
            
            return JWTKeySet(
                context=context,
                current=new_key,
                previous=current,
                rotation_overlap_hours=self.rotation_overlap_hours
            )
    
    async def _rotation_checker(self):
        """Background task to check for needed rotations"""
        while True:
            try:
                for context, key_set in self._key_sets.items():
                    if self._should_rotate(key_set.current):
                        new_keyset = await self._rotate_keys(context, key_set.current)
                        self._key_sets[context] = new_keyset
                
                # Check every hour
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"Error in rotation checker: {e}")
                await asyncio.sleep(300)  # Retry in 5 minutes
    
    async def _notify_rotation(self, context: str, new_key_id: str):
        """Notify monitoring system of key rotation"""
        # Publish to Redis pub/sub for other instances
        await self.redis.publish(
            "jwt:rotation",
            json.dumps({
                "context": context,
                "new_key_id": new_key_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        )
        
        # Log metric
        from backend.monitoring.enhanced_metrics import MetricsCollector
        MetricsCollector.record_security_event(
            event_type="jwt_rotation",
            endpoint=f"jwt_{context}",
            severity="info"
        )
    
    def create_token(
        self, 
        payload: Dict, 
        context: str = "rest",
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create a new JWT token with current key"""
        key_set = self._key_sets.get(context)
        if not key_set:
            raise ValueError(f"No keyset found for context: {context}")
        
        signing_key = key_set.get_signing_key()
        
        # Add standard claims
        now = datetime.now(timezone.utc)
        expires = now + (expires_delta or timedelta(hours=1))
        
        full_payload = {
            **payload,
            "iat": int(now.timestamp()),
            "exp": int(expires.timestamp()),
            "kid": signing_key.key_id,  # Key ID for rotation support
            "ctx": context  # Context identifier
        }
        
        return jwt.encode(
            full_payload,
            signing_key.secret,
            algorithm=signing_key.algorithm
        )
    
    def verify_token(self, token: str, context: str = "rest") -> Dict:
        """
        Verify a JWT token, trying multiple keys during rotation.
        Returns decoded payload if valid.
        """
        key_set = self._key_sets.get(context)
        if not key_set:
            raise ValueError(f"No keyset found for context: {context}")
        
        # Try to decode with each valid key
        verification_keys = key_set.get_verification_keys()
        last_error = None
        
        for key in verification_keys:
            try:
                # Decode and verify
                payload = jwt.decode(
                    token,
                    key.secret,
                    algorithms=[key.algorithm],
                    options={"verify_exp": True}
                )
                
                # Verify context matches
                if payload.get("ctx") != context:
                    continue
                
                # Log if using old key (for monitoring)
                if key.key_id != key_set.current.key_id:
                    logger.info(f"Token verified with previous key: {key.key_id}")
                
                return payload
                
            except jwt.ExpiredSignatureError:
                last_error = "Token has expired"
            except jwt.InvalidTokenError as e:
                last_error = str(e)
                continue
        
        # All keys failed
        raise HTTPException(
            status_code=401,
            detail=last_error or "Invalid token"
        )

# ==============================================================================
# FASTAPI INTEGRATION
# ==============================================================================

# Global instance (initialized at startup)
jwt_manager: Optional[JWTRotationManager] = None

async def get_jwt_manager() -> JWTRotationManager:
    """Get the global JWT manager instance"""
    if not jwt_manager:
        raise RuntimeError("JWT manager not initialized")
    return jwt_manager

class JWTBearer(HTTPBearer):
    """Custom JWT bearer authentication with rotation support"""
    
    def __init__(self, context: str = "rest", auto_error: bool = True):
        super().__init__(auto_error=auto_error)
        self.context = context
    
    async def __call__(self, request: Request, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
        if not credentials:
            raise HTTPException(status_code=403, detail="Invalid authorization code.")
        
        if credentials.scheme != "Bearer":
            raise HTTPException(status_code=403, detail="Invalid authentication scheme.")
        
        # Get JWT manager
        manager = await get_jwt_manager()
        
        try:
            payload = manager.verify_token(credentials.credentials, self.context)
            return payload
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=403, detail=f"Invalid token: {str(e)}")

# Dependency for different contexts
jwt_rest = JWTBearer(context="rest")
jwt_websocket = JWTBearer(context="websocket")
jwt_refresh = JWTBearer(context="refresh")

# ==============================================================================
# USAGE IN ENDPOINTS
# ==============================================================================

"""
# In your FastAPI app startup:

@app.on_event("startup")
async def startup_event():
    global jwt_manager
    
    redis_client = redis.asyncio.from_url(
        os.getenv("REDIS_URL", "redis://localhost:6379")
    )
    
    jwt_manager = JWTRotationManager(
        redis_client=redis_client,
        key_lifetime_days=30,
        rotation_overlap_hours=24
    )
    
    await jwt_manager.initialize()

# In your endpoints:

@router.post("/api/auth/login")
async def login(credentials: LoginRequest):
    # Validate credentials...
    
    # Create token with rotation support
    manager = await get_jwt_manager()
    token = manager.create_token(
        payload={"sub": user.id, "email": user.email},
        context="rest",
        expires_delta=timedelta(hours=1)
    )
    
    return {"access_token": token, "token_type": "bearer"}

@router.get("/api/protected")
async def protected_route(token_data: Dict = Depends(jwt_rest)):
    # token_data contains the decoded JWT payload
    user_id = token_data["sub"]
    return {"user_id": user_id}

# For WebSocket:
@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    token_data: Dict = Depends(jwt_websocket)
):
    # WebSocket connection with separate JWT context
    user_id = token_data["sub"]
    await websocket.accept()
"""

# ==============================================================================
# TESTS FOR JWT ROTATION
# ==============================================================================

# File: tests/security/test_jwt_rotation.py
"""
import pytest
import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock
import json
import jwt as pyjwt

from backend.auth.jwt_rotation import JWTRotationManager, JWTKey, JWTKeySet

@pytest.fixture
async def redis_mock():
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock(return_value=True)
    redis.publish = AsyncMock(return_value=1)
    return redis

@pytest.fixture
async def jwt_manager(redis_mock):
    manager = JWTRotationManager(
        redis_client=redis_mock,
        key_lifetime_days=30,
        rotation_overlap_hours=24
    )
    await manager.initialize()
    return manager

@pytest.mark.asyncio
async def test_token_creation_and_verification(jwt_manager):
    # Create token
    payload = {"sub": "user123", "email": "test@example.com"}
    token = jwt_manager.create_token(payload, context="rest")
    
    # Verify token
    decoded = jwt_manager.verify_token(token, context="rest")
    
    assert decoded["sub"] == "user123"
    assert decoded["email"] == "test@example.com"
    assert "exp" in decoded
    assert "iat" in decoded
    assert "kid" in decoded

@pytest.mark.asyncio
async def test_token_verification_with_wrong_context(jwt_manager):
    # Create REST token
    token = jwt_manager.create_token({"sub": "user123"}, context="rest")
    
    # Try to verify as WebSocket token
    with pytest.raises(Exception):
        jwt_manager.verify_token(token, context="websocket")

@pytest.mark.asyncio
async def test_dual_key_verification_during_rotation(jwt_manager, redis_mock):
    # Simulate key rotation scenario
    old_key = JWTKey(
        key_id="rest_old",
        secret="old_secret_key",
        algorithm="HS512",
        created_at=datetime.now(timezone.utc) - timedelta(days=25),
        expires_at=datetime.now(timezone.utc) + timedelta(days=5),
        is_active=True
    )
    
    new_key = JWTKey(
        key_id="rest_new",
        secret="new_secret_key",
        algorithm="HS512",
        created_at=datetime.now(timezone.utc),
        expires_at=datetime.now(timezone.utc) + timedelta(days=30),
        is_active=True
    )
    
    # Set up keyset with both keys
    jwt_manager._key_sets["rest"] = JWTKeySet(
        context="rest",
        current=new_key,
        previous=old_key,
        rotation_overlap_hours=24
    )
    
    # Create token with old key
    old_token = pyjwt.encode(
        {
            "sub": "user123",
            "exp": datetime.now(timezone.utc) + timedelta(hours=1),
            "iat": datetime.now(timezone.utc),
            "kid": "rest_old",
            "ctx": "rest"
        },
        old_key.secret,
        algorithm="HS512"
    )
    
    # Should still verify during overlap period
    decoded = jwt_manager.verify_token(old_token, context="rest")
    assert decoded["sub"] == "user123"

@pytest.mark.asyncio
async def test_expired_token_rejection(jwt_manager):
    # Create expired token
    expired_payload = {
        "sub": "user123",
        "exp": int((datetime.now(timezone.utc) - timedelta(hours=1)).timestamp())
    }
    
    key_set = jwt_manager._key_sets["rest"]
    token = pyjwt.encode(
        expired_payload,
        key_set.current.secret,
        algorithm=key_set.current.algorithm
    )
    
    with pytest.raises(Exception) as exc_info:
        jwt_manager.verify_token(token, context="rest")
    
    assert "expired" in str(exc_info.value).lower()

@pytest.mark.asyncio
async def test_automatic_rotation_trigger(jwt_manager, redis_mock):
    # Create key that needs rotation (expires in 5 days)
    old_key = JWTKey(
        key_id="rest_old",
        secret="old_secret",
        algorithm="HS512",
        created_at=datetime.now(timezone.utc) - timedelta(days=25),
        expires_at=datetime.now(timezone.utc) + timedelta(days=5),
        is_active=True
    )
    
    jwt_manager._key_sets["rest"].current = old_key
    
    # Trigger rotation check
    new_keyset = await jwt_manager._rotate_keys("rest", old_key)
    
    # Verify rotation occurred
    assert new_keyset.current.key_id != old_key.key_id
    assert new_keyset.previous.key_id == old_key.key_id
    
    # Verify Redis was updated
    assert redis_mock.set.call_count >= 2  # current and previous
"""

# ==============================================================================
# MONITORING AND ALERTING
# ==============================================================================

# File: backend/auth/jwt_monitoring.py
"""
JWT rotation monitoring and alerting
"""

from prometheus_client import Counter, Gauge, Histogram
import logging

logger = logging.getLogger(__name__)

# Metrics
jwt_rotations_total = Counter(
    'jwt_rotations_total',
    'Total number of JWT key rotations',
    ['context', 'status']
)

jwt_keys_age_days = Gauge(
    'jwt_keys_age_days',
    'Age of current JWT keys in days',
    ['context']
)

jwt_verification_duration = Histogram(
    'jwt_verification_duration_seconds',
    'Time taken to verify JWT tokens',
    ['context', 'key_type']  # key_type: current/previous
)

jwt_verification_failures = Counter(
    'jwt_verification_failures_total',
    'JWT verification failures',
    ['context', 'reason']
)

class JWTMonitor:
    """Monitor JWT rotation health"""
    
    @staticmethod
    async def check_rotation_health(manager: JWTRotationManager) -> Dict[str, Any]:
        """Check health of JWT rotation system"""
        health = {
            "status": "healthy",
            "contexts": {},
            "alerts": []
        }
        
        for context, key_set in manager._key_sets.items():
            # Calculate key age
            key_age_days = (
                datetime.now(timezone.utc) - key_set.current.created_at
            ).days
            
            jwt_keys_age_days.labels(context=context).set(key_age_days)
            
            # Check if rotation is needed soon
            days_until_expiry = (
                key_set.current.expires_at - datetime.now(timezone.utc)
            ).days
            
            context_health = {
                "current_key_id": key_set.current.key_id,
                "key_age_days": key_age_days,
                "days_until_expiry": days_until_expiry,
                "has_previous_key": key_set.previous is not None
            }
            
            # Generate alerts
            if days_until_expiry <= 3:
                health["alerts"].append({
                    "severity": "critical",
                    "context": context,
                    "message": f"JWT key expires in {days_until_expiry} days"
                })
                health["status"] = "critical"
            elif days_until_expiry <= 7:
                health["alerts"].append({
                    "severity": "warning",
                    "context": context,
                    "message": f"JWT key rotation needed soon ({days_until_expiry} days)"
                })
                if health["status"] == "healthy":
                    health["status"] = "warning"
            
            health["contexts"][context] = context_health
        
        return health

# Prometheus alerting rules
"""
groups:
  - name: jwt_rotation
    interval: 5m
    rules:
      - alert: JWTKeyExpiringSoon
        expr: jwt_keys_age_days > 23
        for: 1h
        labels:
          severity: warning
        annotations:
          summary: "JWT key rotation needed for {{ $labels.context }}"
          description: "JWT key for {{ $labels.context }} is {{ $value }} days old"
      
      - alert: JWTKeyExpiryCritical
        expr: jwt_keys_age_days > 27
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "JWT key expiring soon for {{ $labels.context }}"
          description: "JWT key for {{ $labels.context }} expires in less than 3 days"
      
      - alert: JWTVerificationFailureRate
        expr: rate(jwt_verification_failures_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High JWT verification failure rate"
          description: "{{ $value }} failures per second for {{ $labels.context }}"
"""