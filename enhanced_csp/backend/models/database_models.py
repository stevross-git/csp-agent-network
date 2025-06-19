# File: backend/models/database_models.py
"""
Database Models for CSP Visual Designer Backend
==============================================
SQLAlchemy models for the visual designer system
"""

from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, JSON, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

Base = declarative_base()

class Design(Base):
    """Visual design model"""
    __tablename__ = "designs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    version = Column(String(50), default="1.0.0")
    created_by = Column(UUID(as_uuid=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    canvas_settings = Column(JSON, default=dict)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    nodes = relationship("DesignNode", back_populates="design", cascade="all, delete-orphan")
    connections = relationship("DesignConnection", back_populates="design", cascade="all, delete-orphan")
    execution_sessions = relationship("ExecutionSession", back_populates="design")
    
    def to_dict(self):
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "created_by": str(self.created_by) if self.created_by else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "canvas_settings": self.canvas_settings,
            "is_active": self.is_active
        }

class DesignNode(Base):
    """Node in a visual design"""
    __tablename__ = "design_nodes"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    design_id = Column(UUID(as_uuid=True), ForeignKey("designs.id"), nullable=False)
    node_id = Column(String(100), nullable=False)  # Visual node identifier
    component_type = Column(String(100), nullable=False)
    position_x = Column(Float, default=0.0)
    position_y = Column(Float, default=0.0)
    width = Column(Float, default=120.0)
    height = Column(Float, default=80.0)
    properties = Column(JSON, default=dict)
    visual_style = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    design = relationship("Design", back_populates="nodes")
    
    def to_dict(self):
        return {
            "id": str(self.id),
            "design_id": str(self.design_id),
            "node_id": self.node_id,
            "component_type": self.component_type,
            "position": {"x": self.position_x, "y": self.position_y},
            "size": {"width": self.width, "height": self.height},
            "properties": self.properties,
            "visual_style": self.visual_style,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class DesignConnection(Base):
    """Connection between nodes in a design"""
    __tablename__ = "design_connections"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    design_id = Column(UUID(as_uuid=True), ForeignKey("designs.id"), nullable=False)
    connection_id = Column(String(100), nullable=False)  # Visual connection identifier
    from_node_id = Column(String(100), nullable=False)
    to_node_id = Column(String(100), nullable=False)
    from_port = Column(String(50), default="output")
    to_port = Column(String(50), default="input")
    connection_type = Column(String(50), default="data_flow")
    properties = Column(JSON, default=dict)
    visual_style = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    design = relationship("Design", back_populates="connections")
    
    def to_dict(self):
        return {
            "id": str(self.id),
            "design_id": str(self.design_id),
            "connection_id": self.connection_id,
            "from_node_id": self.from_node_id,
            "to_node_id": self.to_node_id,
            "from_port": self.from_port,
            "to_port": self.to_port,
            "connection_type": self.connection_type,
            "properties": self.properties,
            "visual_style": self.visual_style,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class ComponentType(Base):
    """Available component types for the visual designer"""
    __tablename__ = "component_types"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    component_type = Column(String(100), nullable=False, unique=True)
    category = Column(String(50), nullable=False)
    display_name = Column(String(255), nullable=False)
    description = Column(Text)
    icon = Column(String(255))
    color = Column(String(7))  # Hex color code
    default_properties = Column(JSON, default=dict)
    input_ports = Column(JSON, default=list)
    output_ports = Column(JSON, default=list)
    implementation_class = Column(String(255))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def to_dict(self):
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
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class ExecutionSession(Base):
    """Execution session for a design"""
    __tablename__ = "execution_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    design_id = Column(UUID(as_uuid=True), ForeignKey("designs.id"), nullable=False)
    session_name = Column(String(255))
    status = Column(String(50), default="pending")  # pending, running, completed, failed, paused
    started_at = Column(DateTime(timezone=True))
    ended_at = Column(DateTime(timezone=True))
    configuration = Column(JSON, default=dict)
    performance_metrics = Column(JSON, default=dict)
    error_logs = Column(JSON, default=list)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    design = relationship("Design", back_populates="execution_sessions")
    metrics = relationship("ComponentMetric", back_populates="session")
    
    def to_dict(self):
        return {
            "id": str(self.id),
            "design_id": str(self.design_id),
            "session_name": self.session_name,
            "status": self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "configuration": self.configuration,
            "performance_metrics": self.performance_metrics,
            "error_logs": self.error_logs,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class ComponentMetric(Base):
    """Performance metrics for individual components during execution"""
    __tablename__ = "component_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("execution_sessions.id"), nullable=False)
    node_id = Column(String(100), nullable=False)
    metric_name = Column(String(100), nullable=False)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(50))
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    session = relationship("ExecutionSession", back_populates="metrics")
    
    def to_dict(self):
        return {
            "id": str(self.id),
            "session_id": str(self.session_id),
            "node_id": self.node_id,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "metric_unit": self.metric_unit,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }

class DesignTemplate(Base):
    """Reusable design templates"""
    __tablename__ = "design_templates"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    category = Column(String(100))
    tags = Column(JSON, default=list)
    template_data = Column(JSON, nullable=False)  # Complete design structure
    preview_image = Column(String(255))
    is_public = Column(Boolean, default=False)
    created_by = Column(UUID(as_uuid=True))
    usage_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def to_dict(self):
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "tags": self.tags,
            "template_data": self.template_data,
            "preview_image": self.preview_image,
            "is_public": self.is_public,
            "created_by": str(self.created_by) if self.created_by else None,
            "usage_count": self.usage_count,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class User(Base):
    """User model for authentication"""
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(100), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    preferences = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_login = Column(DateTime(timezone=True))
    
    def to_dict(self):
        return {
            "id": str(self.id),
            "username": self.username,
            "email": self.email,
            "full_name": self.full_name,
            "is_active": self.is_active,
            "is_admin": self.is_admin,
            "preferences": self.preferences,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None
        }
