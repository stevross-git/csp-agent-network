# backend/auth/token_blacklist.py
"""
JWT Token Blacklist System
==========================
Secure token revocation and refresh rotation
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Set, Dict, Any
from uuid import uuid4

import redis.asyncio as redis
from fastapi import HTTPException

from backend.database.connection import get_redis_client

logger = logging.getLogger(__name__)

# NEW - Token blacklist configuration
BLACKLIST_KEY_PREFIX = "token_blacklist:"
REFRESH_TOKEN_PREFIX = "refresh_token:"
TOKEN_FAMILY_PREFIX = "token_family:"
BLACKLIST_CLEANUP_INTERVAL = 3600  # 1 hour

class TokenBlacklistService:
    """Token blacklist and refresh rotation service"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
    
    async def blacklist_token(self, token_jti: str, expires_at: datetime, reason: str = "logout"):
        """Add token to blacklist"""
        if not self.redis_client:
            return
        
        try:
            expiry_seconds = int((expires_at - datetime.utcnow()).total_seconds())
            if expiry_seconds > 0:
                blacklist_data = {
                    "blacklisted_at": datetime.utcnow().isoformat(),
                    "reason": reason,
                    "expires_at": expires_at.isoformat()
                }
                
                await self.redis_client.setex(
                    f"{BLACKLIST_KEY_PREFIX}{token_jti}",
                    expiry_seconds,
                    json.dumps(blacklist_data)
                )
                
                logger.info(f"Token {token_jti} blacklisted: {reason}")
                
        except Exception as e:
            logger.error(f"Failed to blacklist token {token_jti}: {e}")
    
    async def is_token_blacklisted(self, token_jti: str) -> bool:
        """Check if token is blacklisted"""
        if not self.redis_client:
            return False
        
        try:
            result = await self.redis_client.get(f"{BLACKLIST_KEY_PREFIX}{token_jti}")
            return result is not None
        except Exception as e:
            logger.error(f"Failed to check blacklist for token {token_jti}: {e}")
            return False
    
    async def create_token_family(self, user_id: str, refresh_token: str) -> str:
        """Create a new token family for refresh rotation"""
        if not self.redis_client:
            return str(uuid4())
        
        try:
            family_id = str(uuid4())
            family_data = {
                "user_id": user_id,
                "created_at": datetime.utcnow().isoformat(),
                "tokens": [refresh_token],
                "active": True
            }
            
            # Store family for 30 days
            await self.redis_client.setex(
                f"{TOKEN_FAMILY_PREFIX}{family_id}",
                30 * 24 * 60 * 60,
                json.dumps(family_data)
            )
            
            return family_id
            
        except Exception as e:
            logger.error(f"Failed to create token family: {e}")
            return str(uuid4())
    
    async def rotate_refresh_token(self, old_refresh_token: str, new_refresh_token: str, 
                                  family_id: str) -> bool:
        """Rotate refresh token within family"""
        if not self.redis_client:
            return True
        
        try:
            family_key = f"{TOKEN_FAMILY_PREFIX}{family_id}"
            family_data_str = await self.redis_client.get(family_key)
            
            if not family_data_str:
                raise HTTPException(status_code=401, detail="Invalid token family")
            
            family_data = json.loads(family_data_str)
            
            # Check if family is still active
            if not family_data.get("active", False):
                raise HTTPException(status_code=401, detail="Token family revoked")
            
            # Check if old token is in family
            if old_refresh_token not in family_data.get("tokens", []):
                # Potential token reuse attack - blacklist entire family
                await self.revoke_token_family(family_id, "token_reuse_detected")
                raise HTTPException(status_code=401, detail="Token reuse detected")
            
            # Add new token to family
            family_data["tokens"].append(new_refresh_token)
            family_data["last_rotation"] = datetime.utcnow().isoformat()
            
            # Keep only last 5 tokens in family
            if len(family_data["tokens"]) > 5:
                family_data["tokens"] = family_data["tokens"][-5:]
            
            # Update family
            await self.redis_client.setex(
                family_key,
                30 * 24 * 60 * 60,
                json.dumps(family_data)
            )
            
            # Store new refresh token
            await self.redis_client.setex(
                f"{REFRESH_TOKEN_PREFIX}{new_refresh_token}",
                30 * 24 * 60 * 60,
                json.dumps({
                    "family_id": family_id,
                    "created_at": datetime.utcnow().isoformat()
                })
            )
            
            return True
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to rotate refresh token: {e}")
            return False
    
    async def revoke_token_family(self, family_id: str, reason: str = "manual_revocation"):
        """Revoke entire token family"""
        if not self.redis_client:
            return
        
        try:
            family_key = f"{TOKEN_FAMILY_PREFIX}{family_id}"
            family_data_str = await self.redis_client.get(family_key)
            
            if family_data_str:
                family_data = json.loads(family_data_str)
                family_data["active"] = False
                family_data["revoked_at"] = datetime.utcnow().isoformat()
                family_data["revocation_reason"] = reason
                
                # Update family as revoked
                await self.redis_client.setex(
                    family_key,
                    7 * 24 * 60 * 60,  # Keep for 7 days for audit
                    json.dumps(family_data)
                )
                
                logger.info(f"Token family {family_id} revoked: {reason}")
                
        except Exception as e:
            logger.error(f"Failed to revoke token family {family_id}: {e}")
    
    async def validate_refresh_token(self, refresh_token: str) -> Dict[str, Any]:
        """Validate refresh token and get family info"""
        if not self.redis_client:
            return {"valid": True, "family_id": None}
        
        try:
            token_data_str = await self.redis_client.get(f"{REFRESH_TOKEN_PREFIX}{refresh_token}")
            
            if not token_data_str:
                raise HTTPException(status_code=401, detail="Invalid refresh token")
            
            token_data = json.loads(token_data_str)
            family_id = token_data.get("family_id")
            
            if family_id:
                family_data_str = await self.redis_client.get(f"{TOKEN_FAMILY_PREFIX}{family_id}")
                if family_data_str:
                    family_data = json.loads(family_data_str)
                    if not family_data.get("active", False):
                        raise HTTPException(status_code=401, detail="Token family revoked")
            
            return {
                "valid": True,
                "family_id": family_id,
                "token_data": token_data
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to validate refresh token: {e}")
            raise HTTPException(status_code=401, detail="Token validation failed")
    
    async def cleanup_expired_tokens(self):
        """Clean up expired blacklisted tokens (background task)"""
        if not self.redis_client:
            return
        
        try:
            # Redis automatically expires keys, but we can do additional cleanup
            # This is mainly for logging and monitoring
            cursor = 0
            cleaned_count = 0
            
            while True:
                cursor, keys = await self.redis_client.scan(
                    cursor=cursor,
                    match=f"{BLACKLIST_KEY_PREFIX}*",
                    count=100
                )
                
                for key in keys:
                    # Check if key still exists (not expired)
                    if not await self.redis_client.exists(key):
                        cleaned_count += 1
                
                if cursor == 0:
                    break
            
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} expired blacklisted tokens")
                
        except Exception as e:
            logger.error(f"Token cleanup failed: {e}")

# NEW - Dependency to get blacklist service
async def get_token_blacklist_service(redis_client=Depends(get_redis_client)) -> TokenBlacklistService:
    """Get token blacklist service"""
    return TokenBlacklistService(redis_client)