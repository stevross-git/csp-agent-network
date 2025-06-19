# File: backend/schemas/api_schemas.py
"""
Pydantic Schemas for CSP Visual Designer API
===========================================
Request/Response models for API validation
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from uuid import UUID
import uuid

# ============================================================================
# BASE SCHEMAS
# ============================================================================

class BaseResponse(BaseModel):
    """Base response model"""
    success: bool = True
    message: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)

class ErrorResponse(BaseResponse):
    """Error response model"""
    success: bool = False
    error_code: str
    details: Optional[Dict[str, Any]] = None

# ============================================================================
# DESIGN SCHEMAS
# ============================================================================

class DesignCreate(BaseModel):
    """Schema for creating a new design"""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    version: str = Field(default="1.0.0", max_length=50)
    canvas_settings: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "My AI Processing Pipeline",
                "description": "A pipeline for processing AI requests",
                "version": "1.0.0",
                "canvas_settings": {
                    "width": 1200,
                    "height": 800,
                    "zoom": 1.0,
                    "grid_enabled": True
                }
            }
        }

class DesignUpdate(BaseModel):
    """Schema for updating a design"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    version: Optional[str] = Field(None, max_length=50)
    canvas_settings: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None

class DesignResponse(BaseModel):
    """Schema for design response"""
    id: UUID
    name: str
    description: Optional[str]
    version: str
    created_by: Optional[UUID]
    created_at: datetime
    updated_at: datetime
    canvas_settings: Dict[str, Any]
    is_active: bool
    node_count: int = 0
    connection_count: int = 0
    
    class Config:
        from_attributes = True

class DesignListResponse(BaseResponse):
    """Schema for design list response"""
    designs: List[DesignResponse]
    total_count: int
    page: int = 1
    page_size: int = 10

# ============================================================================
# NODE SCHEMAS
# ============================================================================

class Position(BaseModel):
    """Position model"""
    x: float
    y: float

class Size(BaseModel):
    """Size model"""
    width: float = 120.0
    height: float = 80.0

class NodeCreate(BaseModel):
    """Schema for creating a node"""
    node_id: str = Field(..., min_length=1, max_length=100)
    component_type: str = Field(..., min_length=1, max_length=100)
    position: Position
    size: Size = Field(default_factory=Size)
    properties: Dict[str, Any] = Field(default_factory=dict)
    visual_style: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_schema_extra = {
            "example": {
                "node_id": "ai_agent_1",
                "component_type": "ai_agent",
                "position": {"x": 100, "y": 200},
                "size": {"width": 150, "height": 100},
                "properties": {
                    "model": "gpt-4",
                    "temperature": 0.7,
                    "max_tokens": 1000
                },
                "visual_style": {
                    "color": "#4CAF50",
                    "border_color": "#2E7D32"
                }
            }
        }

class NodeUpdate(BaseModel):
    """Schema for updating a node"""
    position: Optional[Position] = None
    size: Optional[Size] = None
    properties: Optional[Dict[str, Any]] = None
    visual_style: Optional[Dict[str, Any]] = None

class NodeResponse(BaseModel):
    """Schema for node response"""
    id: UUID
    design_id: UUID
    node_id: str
    component_type: str
    position: Position
    size: Size
    properties: Dict[str, Any]
    visual_style: Dict[str, Any]
    created_at: datetime
    
    class Config:
        from_attributes = True

# ============================================================================
# CONNECTION SCHEMAS
# ============================================================================

class ConnectionCreate(BaseModel):
    """Schema for creating a connection"""
    connection_id: str = Field(..., min_length=1, max_length=100)
    from_node_id: str = Field(..., min_length=1, max_length=100)
    to_node_id: str = Field(..., min_length=1, max_length=100)
    from_port: str = Field(default="output", max_length=50)
    to_port: str = Field(default="input", max_length=50)
    connection_type: str = Field(default="data_flow", max_length=50)
    properties: Dict[str, Any] = Field(default_factory=dict)
    visual_style: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('to_node_id')
    def validate_no_self_connection(cls, v, values):
        if 'from_node_id' in values and v == values['from_node_id']:
            raise ValueError('Cannot connect node to itself')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "connection_id": "conn_1",
                "from_node_id": "ai_agent_1",
                "to_node_id": "data_processor_1",
                "from_port": "output",
                "to_port": "input",
                "connection_type": "data_flow",
                "properties": {
                    "data_type": "text",
                    "buffer_size": 1000
                },
                "visual_style": {
                    "color": "#2196F3",
                    "width": 2
                }
            }
        }

class ConnectionUpdate(BaseModel):
    """Schema for updating a connection"""
    from_port: Optional[str] = Field(None, max_length=50)
    to_port: Optional[str] = Field(None, max_length=50)
    connection_type: Optional[str] = Field(None, max_length=50)
    properties: Optional[Dict[str, Any]] = None
    visual_style: Optional[Dict[str, Any]] = None

class ConnectionResponse(BaseModel):
    """Schema for connection response"""
    id: UUID
    design_id: UUID
    connection_id: str
    from_node_id: str
    to_node_id: str
    from_port: str
    to_port: str
    connection_type: str
    properties: Dict[str, Any]
    visual_style: Dict[str, Any]
    created_at: datetime
    
    class Config:
        from_attributes = True

# ============================================================================
# COMPONENT TYPE SCHEMAS
# ============================================================================

class ComponentTypeResponse(BaseModel):
    """Schema for component type response"""
    id: UUID
    component_type: str
    category: str
    display_name: str
    description: Optional[str]
    icon: Optional[str]
    color: Optional[str]
    default_properties: Dict[str, Any]
    input_ports: List[Dict[str, Any]]
    output_ports: List[Dict[str, Any]]
    implementation_class: Optional[str]
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

class ComponentCategoryResponse(BaseModel):
    """Schema for component categories"""
    category: str
    components: List[ComponentTypeResponse]
    count: int

# ============================================================================
# EXECUTION SCHEMAS
# ============================================================================

class ExecutionConfig(BaseModel):
    """Schema for execution configuration"""
    session_name: Optional[str] = None
    max_execution_time: Optional[int] = Field(default=3600, ge=1, le=86400)  # 1 hour default, max 24 hours
    enable_monitoring: bool = True
    enable_profiling: bool = False
    custom_settings: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_name": "Test Run 1",
                "max_execution_time": 1800,
                "enable_monitoring": True,
                "enable_profiling": True,
                "custom_settings": {
                    "log_level": "INFO",
                    "performance_sampling_rate": 0.1
                }
            }
        }

class ExecutionResponse(BaseModel):
    """Schema for execution session response"""
    id: UUID
    design_id: UUID
    session_name: Optional[str]
    status: str
    started_at: Optional[datetime]
    ended_at: Optional[datetime]
    configuration: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    error_logs: List[Dict[str, Any]]
    created_at: datetime
    
    class Config:
        from_attributes = True

class ExecutionStatusResponse(BaseResponse):
    """Schema for execution status response"""
    execution: ExecutionResponse
    current_step: Optional[str] = None
    progress_percentage: float = 0.0
    active_nodes: List[str] = Field(default_factory=list)
    
# ============================================================================
# TEMPLATE SCHEMAS
# ============================================================================

class TemplateCreate(BaseModel):
    """Schema for creating a template"""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    category: Optional[str] = Field(None, max_length=100)
    tags: List[str] = Field(default_factory=list)
    template_data: Dict[str, Any] = Field(..., description="Complete design structure")
    preview_image: Optional[str] = None
    is_public: bool = False

class TemplateResponse(BaseModel):
    """Schema for template response"""
    id: UUID
    name: str
    description: Optional[str]
    category: Optional[str]
    tags: List[str]
    template_data: Dict[str, Any]
    preview_image: Optional[str]
    is_public: bool
    created_by: Optional[UUID]
    usage_count: int
    created_at: datetime
    
    class Config:
        from_attributes = True

# ============================================================================
# METRICS SCHEMAS
# ============================================================================

class MetricData(BaseModel):
    """Schema for individual metric data"""
    node_id: str
    metric_name: str
    metric_value: float
    metric_unit: Optional[str] = None
    timestamp: datetime

class MetricsResponse(BaseResponse):
    """Schema for metrics response"""
    session_id: UUID
    metrics: List[MetricData]
    summary: Dict[str, Any] = Field(default_factory=dict)

# ============================================================================
# USER SCHEMAS
# ============================================================================

class UserCreate(BaseModel):
    """Schema for user creation"""
    username: str = Field(..., min_length=3, max_length=100)
    email: str = Field(..., regex=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = Field(None, max_length=255)

class UserLogin(BaseModel):
    """Schema for user login"""
    username: str
    password: str

class UserResponse(BaseModel):
    """Schema for user response"""
    id: UUID
    username: str
    email: str
    full_name: Optional[str]
    is_active: bool
    is_admin: bool
    preferences: Dict[str, Any]
    created_at: datetime
    last_login: Optional[datetime]
    
    class Config:
        from_attributes = True

class TokenResponse(BaseModel):
    """Schema for authentication token response"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse

# ============================================================================
# WEBSOCKET SCHEMAS
# ============================================================================

class WebSocketMessage(BaseModel):
    """Schema for WebSocket messages"""
    type: str  # 'node_update', 'connection_update', 'execution_status', etc.
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)
    sender_id: Optional[str] = None

class DesignCollaborationEvent(BaseModel):
    """Schema for design collaboration events"""
    event_type: str  # 'cursor_move', 'node_select', 'node_drag', etc.
    user_id: UUID
    design_id: UUID
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)

# ============================================================================
# BATCH OPERATIONS
# ============================================================================

class BatchNodeCreate(BaseModel):
    """Schema for batch node creation"""
    nodes: List[NodeCreate]

class BatchConnectionCreate(BaseModel):
    """Schema for batch connection creation"""
    connections: List[ConnectionCreate]

class BatchOperationResponse(BaseResponse):
    """Schema for batch operation response"""
    successful_operations: int
    failed_operations: int
    results: List[Dict[str, Any]]
    errors: List[Dict[str, Any]] = Field(default_factory=list)
