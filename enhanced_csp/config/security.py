# File: backend/config/security.py
"""
Enhanced security configuration with context-specific secrets
"""

import os
from typing import Dict, Optional
from functools import lru_cache
import secrets
from cryptography.fernet import Fernet
from datetime import timedelta

class SecurityConfig:
    """Centralized security configuration with context isolation"""
    
    def __init__(self):
        # Separate secrets for different contexts
        self.jwt_rest_secret = os.getenv("JWT_REST_SECRET", self._generate_secret())
        self.jwt_ws_secret = os.getenv("JWT_WS_SECRET", self._generate_secret())
        self.jwt_refresh_secret = os.getenv("JWT_REFRESH_SECRET", self._generate_secret())
        self.cookie_sign_key = os.getenv("COOKIE_SIGN_KEY", self._generate_secret())
        self.csrf_secret = os.getenv("CSRF_SECRET", self._generate_secret())
        
        # Encryption keys for sensitive data
        self.field_encryption_key = os.getenv(
            "FIELD_ENCRYPTION_KEY", 
            Fernet.generate_key().decode()
        )
        
        # Token lifetimes
        self.access_token_expire = timedelta(minutes=15)  # Short-lived
        self.refresh_token_expire = timedelta(days=7)
        self.ws_token_expire = timedelta(hours=1)  # WebSocket-specific
        
        # Algorithms
        self.jwt_algorithm = "HS512"  # Upgrade from HS256
        self.refresh_algorithm = "HS512"
        
        # Rate limiting per context
        self.rate_limits = {
            "api_default": "100/hour",
            "api_auth": "5/minute",
            "ws_connect": "10/minute",
            "quantum_ops": "20/hour",
        }
    
    def _generate_secret(self, length: int = 64) -> str:
        """Generate cryptographically secure secret"""
        return secrets.token_urlsafe(length)
    
    def get_context_secret(self, context: str) -> str:
        """Get secret for specific context"""
        context_map = {
            "rest": self.jwt_rest_secret,
            "websocket": self.jwt_ws_secret,
            "refresh": self.jwt_refresh_secret,
            "cookie": self.cookie_sign_key,
            "csrf": self.csrf_secret,
        }
        return context_map.get(context, self.jwt_rest_secret)
    
    def rotate_secret(self, context: str) -> str:
        """Rotate a specific secret (requires restart)"""
        new_secret = self._generate_secret()
        # In production, this would update Azure Key Vault
        return new_secret

@lru_cache()
def get_security_config() -> SecurityConfig:
    """Get cached security configuration"""
    return SecurityConfig()