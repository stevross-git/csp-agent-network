# backend/models/database_models.py
"""
Database Models for Enhanced CSP Visual Designer
===============================================
SQLAlchemy models with UUID primary keys and comprehensive relationships
"""

import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from sqlalchemy import Column, String, Text, DateTime, Boolean, Integer, Float, JSON, ForeignKey, Table
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref
from sqlalchemy.sql import func

Base = declarative_base()

# ============================================================================
# USER MODELS
# ============================================================================

class User(Base):
    """Azure AD User model"""
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    azure_user_id = Column(String(100), unique=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True)
    full_name = Column(String(255))
    hashed_password = Column(String(255))  # For legacy compatibility
    is_active = Column(Boolean, default=True)
    last_login = Column(DateTime)
    preferences = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    designs = relationship("Design", back_populates="created_by_user")

class LocalUser(Base):
    """Local email/password user model"""
    __tablename__ = "local_users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_email_verified = Column(Boolean, default=False)
    email_verification_token = Column(String(255))
    password_reset_token = Column(String(255))
    password_reset_expires = Column(DateTime)
    roles = Column(ARRAY(String), default=["user"])
    last_login = Column(DateTime)
    login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime)
    preferences = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    local_designs = relationship("Design", foreign_keys="Design.created_by_local_user_id", back_populates="created_by_local_user")
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            "id": str(self.id),
            "email": self.email,
            "full_name": self.full_name,
            "is_active": self.is_active,
            "is_email_verified": self.is_email_verified,
            "roles": self.roles,
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

# ============================================================================
# DESIGN MODELS
# ============================================================================

class Design(Base):
    """Visual design model"""
    __tablename__ = "designs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    version = Column(String(50), default="1.0.0")
    canvas_settings = Column(JSON, default=dict)
    design_metadata = Column(JSON, default=dict)  # FIXED: Renamed from 'metadata' to avoid SQLAlchemy conflict
    is_public = Column(Boolean, default=False)
    is_template = Column(Boolean, default=False)
    tags = Column(ARRAY(String), default=list)
    
    # Enhanced creator tracking (supports both auth methods)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)  # Azure AD user
    created_by_local_user_id = Column(UUID(as_uuid=True), ForeignKey("local_users.id"), nullable=True)  # Local user
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    created_by_user = relationship("User", back_populates="designs")
    created_by_local_user = relationship("LocalUser", back_populates="local_designs")
    nodes = relationship("DesignNode", back_populates="design", cascade="all, delete-orphan")
    connections = relationship("DesignConnection", back_populates="design", cascade="all, delete-orphan")
    executions = relationship("ExecutionSession", back_populates="design")
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "canvas_settings": self.canvas_settings,
            "metadata": self.design_metadata,  # Map back to 'metadata' for API compatibility
            "is_public": self.is_public,
            "is_template": self.is_template,
            "tags": self.tags,
            "created_by": str(self.created_by) if self.created_by else None,
            "created_by_local_user_id": str(self.created_by_local_user_id) if self.created_by_local_user_id else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "node_count": len(self.nodes) if self.nodes else 0,
            "connection_count": len(self.connections) if self.connections else 0
        }

class DesignNode(Base):
    """Design node model"""
    __tablename__ = "design_nodes"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    design_id = Column(UUID(as_uuid=True), ForeignKey("designs.id"), nullable=False)
    node_id = Column(String(100), nullable=False)  # Unique within design
    component_type = Column(String(100), nullable=False)
    component_config = Column(JSON, default=dict)
    position_x = Column(Float, default=0)
    position_y = Column(Float, default=0)
    width = Column(Float, default=200)
    height = Column(Float, default=100)
    z_index = Column(Integer, default=0)
    is_locked = Column(Boolean, default=False)
    node_metadata = Column(JSON, default=dict)  # FIXED: Renamed from 'metadata'
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    design = relationship("Design", back_populates="nodes")
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            "id": str(self.id),
            "design_id": str(self.design_id),
            "node_id": self.node_id,
            "component_type": self.component_type,
            "component_config": self.component_config,
            "position": {"x": self.position_x, "y": self.position_y},
            "size": {"width": self.width, "height": self.height},
            "z_index": self.z_index,
            "is_locked": self.is_locked,
            "metadata": self.node_metadata,  # Map back to 'metadata' for API compatibility
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }

class DesignConnection(Base):
    """Design connection model"""
    __tablename__ = "design_connections"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    design_id = Column(UUID(as_uuid=True), ForeignKey("designs.id"), nullable=False)
    connection_id = Column(String(100), nullable=False)  # Unique within design
    from_node_id = Column(String(100), nullable=False)
    from_port = Column(String(50), nullable=False)
    to_node_id = Column(String(100), nullable=False)
    to_port = Column(String(50), nullable=False)
    connection_type = Column(String(50), default="data")
    style = Column(JSON, default=dict)
    connection_metadata = Column(JSON, default=dict)  # FIXED: Renamed from 'metadata'
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    design = relationship("Design", back_populates="connections")
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            "id": str(self.id),
            "design_id": str(self.design_id),
            "connection_id": self.connection_id,
            "from_node_id": self.from_node_id,
            "from_port": self.from_port,
            "to_node_id": self.to_node_id,
            "to_port": self.to_port,
            "connection_type": self.connection_type,
            "style": self.style,
            "metadata": self.connection_metadata,  # Map back to 'metadata' for API compatibility
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

# ============================================================================
# EXECUTION MODELS
# ============================================================================

class ExecutionSession(Base):
    """Execution session model"""
    __tablename__ = "execution_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    design_id = Column(UUID(as_uuid=True), ForeignKey("designs.id"), nullable=False)
    status = Column(String(50), default="pending")  # pending, running, completed, failed, cancelled
    started_at = Column(DateTime)
    ended_at = Column(DateTime)
    configuration = Column(JSON, default=dict)
    results = Column(JSON, default=dict)
    error_message = Column(Text)
    performance_metrics = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    design = relationship("Design", back_populates="executions")
    metrics = relationship("ExecutionMetric", back_populates="session", cascade="all, delete-orphan")
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            "id": str(self.id),
            "design_id": str(self.design_id),
            "status": self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "configuration": self.configuration,
            "results": self.results,
            "error_message": self.error_message,
            "performance_metrics": self.performance_metrics,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "duration_seconds": (self.ended_at - self.started_at).total_seconds() if self.started_at and self.ended_at else None
        }

class ExecutionMetric(Base):
    """Execution metrics model"""
    __tablename__ = "execution_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("execution_sessions.id"), nullable=False)
    node_id = Column(String(100))  # Optional - node-specific metrics
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(50))
    tags = Column(JSON, default=dict)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    session = relationship("ExecutionSession", back_populates="metrics")

# ============================================================================
# COMPONENT MODELS
# ============================================================================

class ComponentType(Base):
    """Component type definition"""
    __tablename__ = "component_types"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    component_type = Column(String(100), unique=True, nullable=False)
    category = Column(String(50), nullable=False)
    display_name = Column(String(255), nullable=False)
    description = Column(Text)
    icon = Column(String(50))
    color = Column(String(7), default="#6B73FF")  # Hex color
    default_properties = Column(JSON, default=dict)
    input_ports = Column(JSON, default=list)
    output_ports = Column(JSON, default=list)
    implementation_class = Column(String(255))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            "id": str(self.id),
            "component_type": self.component_type,
            "category": self.category,
            "display_name": self.display_name,
            "description": self.description,
            "icon": self.icon,
            "color": self.color,
            "default_properties": self.default_properties,
            "input_ports": self.input_ports,
            "output_ports": self.output_ports,
            "implementation_class": self.implementation_class,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }

# ============================================================================
# AUDIT AND SYSTEM MODELS
# ============================================================================

class AuditLog(Base):
    """Audit log for security events"""
    __tablename__ = "audit_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(100))  # Can be Azure AD or local user ID
    user_email = Column(String(255))
    action = Column(String(100), nullable=False)  # login, logout, create_design, etc.
    resource_type = Column(String(50))  # design, user, etc.
    resource_id = Column(String(100))
    details = Column(JSON, default=dict)
    ip_address = Column(String(45))  # Support IPv6
    user_agent = Column(Text)
    success = Column(Boolean, nullable=False)
    error_message = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            "id": str(self.id),
            "user_id": self.user_id,
            "user_email": self.user_email,
            "action": self.action,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "details": self.details,
            "ip_address": self.ip_address,
            "success": self.success,
            "error_message": self.error_message,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }

class SystemConfig(Base):
    """System configuration settings"""
    __tablename__ = "system_config"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key = Column(String(100), unique=True, nullable=False)
    value = Column(JSON, nullable=False)
    description = Column(Text)
    is_public = Column(Boolean, default=False)  # Can be accessed by non-admin users
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_by = Column(String(100))  # User ID who made the change

# ============================================================================
# NOTIFICATION MODELS
# ============================================================================

class Notification(Base):
    """User notifications"""
    __tablename__ = "notifications"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(100), nullable=False)  # Azure AD or local user ID
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    notification_type = Column(String(50), default="info")  # info, success, warning, error
    is_read = Column(Boolean, default=False)
    is_persistent = Column(Boolean, default=False)
    actions = Column(JSON, default=list)  # Action buttons
    created_at = Column(DateTime, default=datetime.utcnow)
    read_at = Column(DateTime)
    expires_at = Column(DateTime)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            "id": str(self.id),
            "user_id": self.user_id,
            "title": self.title,
            "message": self.message,
            "type": self.notification_type,
            "is_read": self.is_read,
            "is_persistent": self.is_persistent,
            "actions": self.actions,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "read_at": self.read_at.isoformat() if self.read_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None
        }