# backend/auth/local_auth.py
"""
Local Email/Password Authentication System
========================================
Traditional authentication alongside Azure AD
"""

import os
import secrets
import smtplib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import jwt
from passlib.context import CryptContext
from passlib.hash import bcrypt
from fastapi import HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from pydantic import BaseModel, EmailStr, validator
import redis.asyncio as redis

from backend.models.database_models import LocalUser  # NEW
from backend.database.connection import get_db_session, get_redis_client

# NEW - Local authentication configuration
LOCAL_AUTH_SECRET_KEY = os.getenv("LOCAL_AUTH_SECRET_KEY", secrets.token_urlsafe(32))
LOCAL_JWT_ALGORITHM = "HS256"
LOCAL_ACCESS_TOKEN_EXPIRE_MINUTES = 60
LOCAL_REFRESH_TOKEN_EXPIRE_DAYS = 30

# NEW - Password requirements
PASSWORD_MIN_LENGTH = 8
PASSWORD_REQUIRE_UPPERCASE = True
PASSWORD_REQUIRE_LOWERCASE = True
PASSWORD_REQUIRE_DIGITS = True
PASSWORD_REQUIRE_SPECIAL = True

# NEW - Email configuration
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
FROM_EMAIL = os.getenv("FROM_EMAIL", "noreply@csp-system.com")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# NEW - Local authentication schemas
class LocalUserRegistration(BaseModel):
    """User registration request"""
    email: EmailStr
    password: str
    full_name: str
    confirm_password: str
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < PASSWORD_MIN_LENGTH:
            raise ValueError(f'Password must be at least {PASSWORD_MIN_LENGTH} characters')
        
        if PASSWORD_REQUIRE_UPPERCASE and not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        
        if PASSWORD_REQUIRE_LOWERCASE and not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        
        if PASSWORD_REQUIRE_DIGITS and not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        
        if PASSWORD_REQUIRE_SPECIAL and not any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in v):
            raise ValueError('Password must contain at least one special character')
        
        return v
    
    @validator('confirm_password')
    def validate_password_match(cls, v, values):
        if 'password' in values and v != values['password']:
            raise ValueError('Passwords do not match')
        return v

class LocalUserLogin(BaseModel):
    """User login request"""
    email: EmailStr
    password: str
    remember_me: bool = False

class LocalUserInfo(BaseModel):
    """Local user information"""
    id: str
    email: str
    full_name: str
    roles: List[str]
    is_active: bool
    is_email_verified: bool
    created_at: datetime
    last_login: Optional[datetime]

class LocalTokenResponse(BaseModel):
    """Local authentication token response"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: Dict[str, Any]

class PasswordResetRequest(BaseModel):
    """Password reset request"""
    email: EmailStr

class PasswordReset(BaseModel):
    """Password reset submission"""
    token: str
    new_password: str
    confirm_password: str
    
    @validator('confirm_password')
    def validate_password_match(cls, v, values):
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('Passwords do not match')
        return v

# NEW - Local authentication service
class LocalAuthService:
    """Local email/password authentication service"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
    
    def hash_password(self, password: str) -> str:
        """Hash a password"""
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create access token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=LOCAL_ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire, "type": "access", "auth_method": "local"})
        return jwt.encode(to_encode, LOCAL_AUTH_SECRET_KEY, algorithm=LOCAL_JWT_ALGORITHM)
    
    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Create refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=LOCAL_REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode.update({"exp": expire, "type": "refresh", "auth_method": "local"})
        return jwt.encode(to_encode, LOCAL_AUTH_SECRET_KEY, algorithm=LOCAL_JWT_ALGORITHM)
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode token"""
        try:
            payload = jwt.decode(token, LOCAL_AUTH_SECRET_KEY, algorithms=[LOCAL_JWT_ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    async def send_verification_email(self, email: str, token: str):
        """Send email verification"""
        if not SMTP_USERNAME or not SMTP_PASSWORD:
            print(f"Email verification token for {email}: {token}")  # Development fallback
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = FROM_EMAIL
            msg['To'] = email
            msg['Subject'] = "Verify Your CSP System Account"
            
            verification_url = f"{os.getenv('FRONTEND_URL', 'http://localhost:8000')}/verify-email?token={token}"
            
            body = f"""
            Welcome to CSP System!
            
            Please click the link below to verify your email address:
            {verification_url}
            
            This link will expire in 24 hours.
            
            If you didn't create this account, please ignore this email.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(SMTP_USERNAME, SMTP_PASSWORD)
                server.send_message(msg)
                
        except Exception as e:
            print(f"Failed to send verification email: {e}")
    
    async def register_user(self, registration: LocalUserRegistration, db_session: AsyncSession) -> LocalUserInfo:
        """Register new local user"""
        # Check if email already exists
        existing_user = await self._get_user_by_email(registration.email, db_session)
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")
        
        # Create user
        hashed_password = self.hash_password(registration.password)
        
        user = LocalUser(
            email=registration.email,
            hashed_password=hashed_password,
            full_name=registration.full_name,
            is_active=True,
            is_email_verified=False,
            roles=["user"]  # Default role
        )
        
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)
        
        # Send verification email
        verification_token = self.create_verification_token(user.id)
        await self.send_verification_email(user.email, verification_token)
        
        return await self._user_to_info(user)
    
    async def authenticate_user(self, login: LocalUserLogin, db_session: AsyncSession) -> LocalTokenResponse:
        """Authenticate user and return tokens"""
        # Get user
        user = await self._get_user_by_email(login.email, db_session)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Verify password
        if not self.verify_password(login.password, user.hashed_password):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Check if user is active
        if not user.is_active:
            raise HTTPException(status_code=401, detail="Account is deactivated")
        
        # Update last login
        user.last_login = datetime.now()
        await db_session.commit()
        
        # Create tokens
        user_info = await self._user_to_info(user)
        token_data = {
            "sub": str(user.id),
            "email": user.email,
            "roles": user.roles,
            "auth_method": "local"
        }
        
        expires_delta = timedelta(days=30) if login.remember_me else None
        access_token = self.create_access_token(token_data, expires_delta)
        refresh_token = self.create_refresh_token(token_data)
        
        # Store refresh token
        if self.redis_client:
            await self.redis_client.setex(
                f"local_refresh_token:{user.id}",
                LOCAL_REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,
                refresh_token
            )
        
        return LocalTokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=LOCAL_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user=user_info.dict()
        )
    
    def create_verification_token(self, user_id: str) -> str:
        """Create email verification token"""
        data = {
            "sub": str(user_id),
            "type": "email_verification",
            "exp": datetime.utcnow() + timedelta(hours=24)
        }
        return jwt.encode(data, LOCAL_AUTH_SECRET_KEY, algorithm=LOCAL_JWT_ALGORITHM)
    
    async def _get_user_by_email(self, email: str, db_session: AsyncSession) -> Optional[LocalUser]:
        """Get user by email"""
        query = select(LocalUser).where(LocalUser.email == email)
        result = await db_session.execute(query)
        return result.scalar_one_or_none()
    
    async def _user_to_info(self, user: LocalUser) -> LocalUserInfo:
        """Convert user model to info"""
        return LocalUserInfo(
            id=str(user.id),
            email=user.email,
            full_name=user.full_name,
            roles=user.roles,
            is_active=user.is_active,
            is_email_verified=user.is_email_verified,
            created_at=user.created_at,
            last_login=user.last_login
        )

# NEW - Dependency to get local auth service
async def get_local_auth_service(redis_client=Depends(get_redis_client)) -> LocalAuthService:
    """Get local authentication service"""
    return LocalAuthService(redis_client)