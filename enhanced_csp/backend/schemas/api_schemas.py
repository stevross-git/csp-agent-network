# backend/schemas/api_schemas.py
"""
API Schemas for CSP Visual Designer
==================================
Pydantic schemas for request/response validation with examples
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from uuid import UUID
import uuid
from pydantic import BaseModel, EmailStr, validator, Field
from enum import Enum

# ============================================================================
# EXISTING SCHEMAS (preserved)
# ============================================================================

class BaseResponse(BaseModel):
    """Base response model"""
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)
    success: bool = True

class ErrorResponse(BaseModel):
    """Error response model"""
    success: bool = False
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)

# Design related schemas (existing, preserved)
class DesignCreate(BaseModel):
    """Design creation request"""
    name: str = Field(..., min_length=1, max_length=255, description="Design name")
    description: Optional[str] = Field(None, max_length=2000, description="Design description")
    version: str = Field("1.0.0", description="Design version")
    canvas_settings: Dict[str, Any] = Field(default_factory=dict, description="Canvas configuration")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    is_public: bool = Field(False, description="Whether design is publicly visible")
    is_template: bool = Field(False, description="Whether design is a template")
    tags: List[str] = Field(default_factory=list, description="Design tags")

    class Config:
        schema_extra = {
            "example": {
                "name": "Customer Data Pipeline",
                "description": "A pipeline for processing customer data",
                "version": "1.0.0",
                "canvas_settings": {"zoom": 1.0, "pan_x": 0, "pan_y": 0},
                "metadata": {"category": "data_processing"},
                "is_public": False,
                "is_template": False,
                "tags": ["data", "pipeline", "customer"]
            }
        }

class DesignUpdate(BaseModel):
    """Design update request"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=2000)
    version: Optional[str] = None
    canvas_settings: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    is_public: Optional[bool] = None
    is_template: Optional[bool] = None
    tags: Optional[List[str]] = None

class DesignResponse(BaseModel):
    """Design response"""
    id: str
    name: str
    description: Optional[str]
    version: str
    canvas_settings: Dict[str, Any]
    metadata: Dict[str, Any]
    is_public: bool
    is_template: bool
    tags: List[str]
    created_by: Optional[str]
    created_by_local_user_id: Optional[str]  # NEW
    created_at: datetime
    updated_at: datetime
    node_count: int
    connection_count: int

class DesignListResponse(BaseModel):
    """Design list response"""
    designs: List[DesignResponse]
    total: int
    page: int = 1
    page_size: int = 20

# Node related schemas (existing, preserved)
class NodeCreate(BaseModel):
    """Node creation request"""
    node_id: str = Field(..., description="Unique node identifier within design")
    component_type: str = Field(..., description="Component type")
    component_config: Dict[str, Any] = Field(default_factory=dict, description="Component configuration")
    position: Dict[str, float] = Field(..., description="Node position")
    size: Dict[str, float] = Field(default_factory=lambda: {"width": 200, "height": 100}, description="Node size")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @validator('position')
    def validate_position(cls, v):
        if 'x' not in v or 'y' not in v:
            raise ValueError('Position must contain x and y coordinates')
        return v

    @validator('size')
    def validate_size(cls, v):
        if 'width' not in v or 'height' not in v:
            raise ValueError('Size must contain width and height')
        if v['width'] <= 0 or v['height'] <= 0:
            raise ValueError('Width and height must be positive')
        return v

class NodeUpdate(BaseModel):
    """Node update request"""
    component_config: Optional[Dict[str, Any]] = None
    position: Optional[Dict[str, float]] = None
    size: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None
    is_locked: Optional[bool] = None

class NodeResponse(BaseModel):
    """Node response"""
    id: str
    design_id: str
    node_id: str
    component_type: str
    component_config: Dict[str, Any]
    position: Dict[str, float]
    size: Dict[str, float]
    z_index: int
    is_locked: bool
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

# Connection related schemas (existing, preserved)
class ConnectionCreate(BaseModel):
    """Connection creation request"""
    connection_id: str = Field(..., description="Unique connection identifier within design")
    from_node_id: str = Field(..., description="Source node ID")
    from_port: str = Field(..., description="Source port")
    to_node_id: str = Field(..., description="Target node ID")
    to_port: str = Field(..., description="Target port")
    connection_type: str = Field("data", description="Connection type")
    style: Dict[str, Any] = Field(default_factory=dict, description="Connection style")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class ConnectionUpdate(BaseModel):
    """Connection update request"""
    connection_type: Optional[str] = None
    style: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class ConnectionResponse(BaseModel):
    """Connection response"""
    id: str
    design_id: str
    connection_id: str
    from_node_id: str
    from_port: str
    to_node_id: str
    to_port: str
    connection_type: str
    style: Dict[str, Any]
    metadata: Dict[str, Any]
    created_at: datetime

# Execution related schemas (existing, preserved)
class ExecutionConfig(BaseModel):
    """Execution configuration"""
    timeout_seconds: int = Field(300, ge=1, le=3600, description="Execution timeout")
    parallel_execution: bool = Field(True, description="Enable parallel execution")
    debug_mode: bool = Field(False, description="Enable debug mode")
    environment: Dict[str, Any] = Field(default_factory=dict, description="Environment variables")
    resource_limits: Dict[str, Any] = Field(default_factory=dict, description="Resource limits")

class ExecutionResponse(BaseModel):
    """Execution response"""
    id: str
    design_id: str
    status: str
    started_at: Optional[datetime]
    ended_at: Optional[datetime]
    configuration: Dict[str, Any]
    results: Dict[str, Any]
    error_message: Optional[str]
    performance_metrics: Dict[str, Any]
    created_at: datetime
    duration_seconds: Optional[float]

# ============================================================================
# NEW AUTHENTICATION SCHEMAS
# ============================================================================

class UserRegistration(BaseModel):
    """User registration request"""
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=8, description="User password")
    confirm_password: str = Field(..., description="Password confirmation")
    full_name: str = Field(..., min_length=2, max_length=100, description="Full name")
    
    @validator('password')
    def validate_password_strength(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        if not any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in v):
            raise ValueError('Password must contain at least one special character')
        return v
    
    @validator('confirm_password')
    def validate_passwords_match(cls, v, values):
        if 'password' in values and v != values['password']:
            raise ValueError('Passwords do not match')
        return v

    class Config:
        schema_extra = {
            "example": {
                "email": "user@example.com",
                "password": "SecurePass123!",
                "confirm_password": "SecurePass123!",
                "full_name": "John Doe"
            }
        }

class UserLogin(BaseModel):
    """User login request"""
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., description="User password")
    remember_me: bool = Field(False, description="Remember login for extended period")

    class Config:
        schema_extra = {
            "example": {
                "email": "user@example.com",
                "password": "SecurePass123!",
                "remember_me": False
            }
        }

class TokenRefresh(BaseModel):
    """Token refresh request"""
    refresh_token: str = Field(..., description="Refresh token")

class PasswordResetRequest(BaseModel):
    """Password reset request"""
    email: EmailStr = Field(..., description="User email address")

class PasswordReset(BaseModel):
    """Password reset"""
    token: str = Field(..., description="Password reset token")
    new_password: str = Field(..., min_length=8, description="New password")
    confirm_password: str = Field(..., description="Password confirmation")
    
    @validator('confirm_password')
    def validate_passwords_match(cls, v, values):
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('Passwords do not match')
        return v

class UserInfo(BaseModel):
    """User information response"""
    id: str
    email: str
    full_name: str
    roles: List[str]
    auth_method: str  # 'azure' or 'local'
    is_active: bool
    is_email_verified: Optional[bool] = None
    last_login: Optional[datetime]
    created_at: datetime

class TokenResponse(BaseModel):
    """Authentication token response"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: Dict[str, Any]

# ============================================================================
# NEW RBAC SCHEMAS
# ============================================================================

class RoleUpdate(BaseModel):
    """Role update request"""
    user_id: str = Field(..., description="User ID")
    roles: List[str] = Field(..., description="New roles")
    
    @validator('roles')
    def validate_roles(cls, v):
        valid_roles = ['super_admin', 'admin', 'designer', 'analyst', 'user', 'viewer']
        for role in v:
            if role not in valid_roles:
                raise ValueError(f'Invalid role: {role}. Valid roles: {valid_roles}')
        return v

class PermissionCheck(BaseModel):
    """Permission check request"""
    permission: str = Field(..., description="Permission to check")
    resource_id: Optional[str] = Field(None, description="Resource ID (optional)")

class PermissionResponse(BaseModel):
    """Permission check response"""
    has_permission: bool
    permission: str
    user_roles: List[str]
    all_permissions: List[str]

# ============================================================================
# NEW SYSTEM SCHEMAS
# ============================================================================

class SystemHealth(BaseModel):
    """System health response"""
    status: str = Field(..., description="Overall system status")
    timestamp: datetime = Field(default_factory=datetime.now)
    services: Dict[str, Any] = Field(..., description="Service status details")
    version: str = Field(..., description="System version")
    uptime_seconds: float = Field(..., description="System uptime in seconds")

class SystemMetrics(BaseModel):
    """System metrics response"""
    cpu_usage_percent: float
    memory_usage_percent: float
    disk_usage_percent: float
    active_users: int
    active_designs: int
    total_executions: int
    average_response_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.now)

class AuditLogEntry(BaseModel):
    """Audit log entry"""
    id: str
    user_id: Optional[str]
    user_email: Optional[str]
    action: str
    resource_type: Optional[str]
    resource_id: Optional[str]
    details: Dict[str, Any]
    ip_address: Optional[str]
    success: bool
    error_message: Optional[str]
    timestamp: datetime

class AuditLogQuery(BaseModel):
    """Audit log query parameters"""
    user_id: Optional[str] = None
    action: Optional[str] = None
    resource_type: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    success: Optional[bool] = None
    page: int = Field(1, ge=1)
    page_size: int = Field(20, ge=1, le=100)

class AuditLogResponse(BaseModel):
    """Audit log response"""
    logs: List[AuditLogEntry]
    total: int
    page: int
    page_size: int

# ============================================================================
# NEW NOTIFICATION SCHEMAS
# ============================================================================

class NotificationCreate(BaseModel):
    """Notification creation request"""
    title: str = Field(..., max_length=200, description="Notification title")
    message: str = Field(..., max_length=1000, description="Notification message")
    type: str = Field("info", description="Notification type: info, success, warning, error")
    user_id: Optional[str] = Field(None, description="Target user ID (if personal)")
    persistent: bool = Field(False, description="Whether notification persists across sessions")
    actions: List[Dict[str, Any]] = Field(default_factory=list, description="Notification actions")

    @validator('type')
    def validate_notification_type(cls, v):
        valid_types = ['info', 'success', 'warning', 'error']
        if v not in valid_types:
            raise ValueError(f'Invalid notification type: {v}. Valid types: {valid_types}')
        return v

class NotificationResponse(BaseModel):
    """Notification response"""
    id: str
    title: str
    message: str
    type: str
    user_id: Optional[str]
    persistent: bool
    read: bool
    actions: List[Dict[str, Any]]
    created_at: datetime
    read_at: Optional[datetime]

# ============================================================================
# NEW LICENSE SCHEMAS
# ============================================================================

class LicenseCreate(BaseModel):
    """License creation request"""
    product: str
    key: str
    expires_at: Optional[datetime] = None
    active: bool = True

class LicenseResponse(LicenseCreate):
    id: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

# ============================================================================
# NEW COMPONENT SCHEMAS
# ============================================================================

class ComponentInfo(BaseModel):
    """Component information"""
    component_type: str
    name: str
    description: str
    category: str
    icon: str
    input_ports: List[Dict[str, Any]]
    output_ports: List[Dict[str, Any]]
    configuration_schema: Dict[str, Any]
    documentation_url: Optional[str]
    version: str

class ComponentValidation(BaseModel):
    """Component validation result"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]

# ============================================================================
# NEW ANALYTICS SCHEMAS
# ============================================================================

class DesignAnalytics(BaseModel):
    """Design analytics"""
    design_id: str
    total_executions: int
    successful_executions: int
    failed_executions: int
    average_execution_time_seconds: float
    last_execution: Optional[datetime]
    most_used_components: List[Dict[str, Any]]
    performance_trend: List[Dict[str, Any]]

class SystemAnalytics(BaseModel):
    """System-wide analytics"""
    total_designs: int
    active_users: int
    total_executions: int
    popular_components: List[Dict[str, Any]]
    usage_by_day: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)