import os
import secrets
from typing import Optional
from datetime import timedelta

class SecurityConfig:
    """Security configuration with validation"""
    
    def __init__(self):
        self.jwt_secret_key = self._get_jwt_secret()
        self.jwt_algorithm = "HS256"
        self.jwt_expiration = timedelta(hours=24)
        self.refresh_token_expiration = timedelta(days=7)
        
    def _get_jwt_secret(self) -> str:
        """Get JWT secret with validation"""
        jwt_secret = os.environ.get('JWT_SECRET_KEY')
        
        if not jwt_secret:
            # In development, generate a warning
            if os.environ.get('ENVIRONMENT') == 'development':
                print("WARNING: JWT_SECRET_KEY not set, generating temporary key")
                jwt_secret = secrets.token_urlsafe(32)
            else:
                raise ValueError(
                    "JWT_SECRET_KEY must be set in production environment. "
                    "Generate with: openssl rand -hex 32"
                )
        
        # Validate key strength
        if len(jwt_secret) < 32:
            raise ValueError("JWT_SECRET_KEY must be at least 32 characters")
            
        return jwt_secret