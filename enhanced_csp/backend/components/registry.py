# File: backend/components/registry.py
"""
Component Registry & Factory System
==================================
Dynamic component loading and instantiation for the visual designer
"""

import asyncio
import inspect
import importlib
import logging
from typing import Dict, List, Any, Optional, Type, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import json
from pathlib import Path

from pydantic import BaseModel, Field, validator
from sqlalchemy.ext.asyncio import AsyncSession

# Import base CSP components
from core.advanced_csp_core import Process, ProcessContext, Channel
from backend.models.database_models import ComponentType
from backend.database.connection import get_db_session

logger = logging.getLogger(__name__)

# ============================================================================
# COMPONENT INTERFACES AND BASE CLASSES
# ============================================================================

class ComponentCategory(str, Enum):
    """Component categories for organization"""
    AI = "AI"
    DATA = "Data"
    SECURITY = "Security" 
    MONITORING = "Monitoring"
    COMMUNICATION = "Communication"
    PROCESSING = "Processing"
    INTEGRATION = "Integration"
    CONTROL_FLOW = "Control Flow"
    UTILITY = "Utility"

class PortType(str, Enum):
    """Port data types"""
    TEXT = "text"
    DATA = "data"
    JSON = "json"
    BINARY = "binary"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    METRICS = "metrics"
    ERROR = "error"
    SIGNAL = "signal"
    ANY = "any"

@dataclass
class ComponentPort:
    """Component input/output port definition"""
    name: str
    port_type: PortType
    required: bool = True
    description: str = ""
    default_value: Any = None
    validation_schema: Optional[Dict[str, Any]] = None

@dataclass
class ComponentMetadata:
    """Metadata for a component type"""
    component_type: str
    category: ComponentCategory
    display_name: str
    description: str
    version: str = "1.0.0"
    author: str = "CSP System"
    icon: str = "component"
    color: str = "#6B73FF"
    tags: List[str] = field(default_factory=list)
    
    # Port definitions
    input_ports: List[ComponentPort] = field(default_factory=list)
    output_ports: List[ComponentPort] = field(default_factory=list)
    
    # Configuration
    default_properties: Dict[str, Any] = field(default_factory=dict)
    property_schema: Dict[str, Any] = field(default_factory=dict)
    
    # Technical details
    implementation_class: str = ""
    dependencies: List[str] = field(default_factory=list)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    
    # UI hints
    ui_hints: Dict[str, Any] = field(default_factory=dict)

class ComponentBase(ABC):
    """Base class for all visual designer components"""
    
    def __init__(self, node_id: str, properties: Dict[str, Any] = None):
        self.node_id = node_id
        self.properties = properties or {}
        self.input_channels: Dict[str, Channel] = {}
        self.output_channels: Dict[str, Channel] = {}
        self.is_running = False
        self.context: Optional[ProcessContext] = None
        
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the component"""
        pass
    
    @abstractmethod
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the component logic"""
        pass
    
    @abstractmethod
    async def cleanup(self):
        """Cleanup component resources"""
        pass
    
    async def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """Validate input data"""
        return True
    
    async def get_status(self) -> Dict[str, Any]:
        """Get component status"""
        return {
            "node_id": self.node_id,
            "is_running": self.is_running,
            "properties": self.properties
        }

# ============================================================================
# COMPONENT IMPLEMENTATIONS
# ============================================================================

class AIAgentComponent(ComponentBase):
    """AI Agent component implementation"""
    
    metadata = ComponentMetadata(
        component_type="ai_agent",
        category=ComponentCategory.AI,
        display_name="AI Agent",
        description="Intelligent AI agent for processing and decision making",
        icon="robot",
        color="#4CAF50",
        input_ports=[
            ComponentPort("input", PortType.TEXT, required=True, description="Input text to process"),
            ComponentPort("system_prompt", PortType.TEXT, required=False, description="System prompt override"),
        ],
        output_ports=[
            ComponentPort("output", PortType.TEXT, description="AI response"),
            ComponentPort("tokens_used", PortType.DATA, description="Token usage statistics"),
        ],
        default_properties={
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 1000,
            "system_prompt": "You are a helpful AI assistant."
        },
        property_schema={
            "type": "object",
            "properties": {
                "model": {"type": "string", "enum": ["gpt-4", "gpt-3.5-turbo", "claude-3"]},
                "temperature": {"type": "number", "minimum": 0, "maximum": 2},
                "max_tokens": {"type": "integer", "minimum": 1, "maximum": 4000},
                "system_prompt": {"type": "string"}
            }
        }
    )
    
    async def initialize(self) -> bool:
        """Initialize AI agent"""
        try:
            # Initialize AI client based on model
            model = self.properties.get("model", "gpt-4")
            
            if model.startswith("gpt"):
                from openai import AsyncOpenAI
                self.client = AsyncOpenAI()
            elif model.startswith("claude"):
                # Initialize Anthropic client
                self.client = None  # Would initialize Anthropic client
            
            logger.info(f"AI Agent {self.node_id} initialized with model {model}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize AI Agent {self.node_id}: {e}")
            return False
    
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute AI processing"""
        try:
            input_text = inputs.get("input", "")
            system_prompt = inputs.get("system_prompt", self.properties.get("system_prompt"))
            
            if not input_text:
                return {"error": "No input text provided"}
            
            # Call AI model
            if self.properties.get("model", "").startswith("gpt"):
                response = await self._call_openai(input_text, system_prompt)
            else:
                response = {"output": "AI processing simulation", "tokens_used": {"total": 100}}
            
            return response
            
        except Exception as e:
            logger.error(f"AI Agent {self.node_id} execution error: {e}")
            return {"error": str(e)}
    
    async def _call_openai(self, input_text: str, system_prompt: str) -> Dict[str, Any]:
        """Call OpenAI API"""
        try:
            response = await self.client.chat.completions.create(
                model=self.properties.get("model", "gpt-4"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_text}
                ],
                temperature=self.properties.get("temperature", 0.7),
                max_tokens=self.properties.get("max_tokens", 1000)
            )
            
            return {
                "output": response.choices[0].message.content,
                "tokens_used": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
        except Exception as e:
            return {"error": f"OpenAI API error: {str(e)}"}
    
    async def cleanup(self):
        """Cleanup AI agent resources"""
        if hasattr(self, "client"):
            del self.client

class DataProcessorComponent(ComponentBase):
    """Data processing component implementation"""
    
    metadata = ComponentMetadata(
        component_type="data_processor",
        category=ComponentCategory.DATA,
        display_name="Data Processor",
        description="Process and transform data",
        icon="database",
        color="#2196F3",
        input_ports=[
            ComponentPort("input", PortType.DATA, required=True, description="Input data to process"),
            ComponentPort("config", PortType.JSON, required=False, description="Processing configuration"),
        ],
        output_ports=[
            ComponentPort("output", PortType.DATA, description="Processed data"),
            ComponentPort("stats", PortType.JSON, description="Processing statistics"),
        ],
        default_properties={
            "operation": "transform",
            "format": "json",
            "filters": [],
            "transformations": []
        }
    )
    
    async def initialize(self) -> bool:
        """Initialize data processor"""
        self.operation = self.properties.get("operation", "transform")
        logger.info(f"Data Processor {self.node_id} initialized with operation {self.operation}")
        return True
    
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data processing"""
        try:
            input_data = inputs.get("input")
            config = inputs.get("config", {})
            
            if input_data is None:
                return {"error": "No input data provided"}
            
            # Apply transformations
            processed_data = await self._process_data(input_data, config)
            
            return {
                "output": processed_data,
                "stats": {
                    "input_size": len(str(input_data)),
                    "output_size": len(str(processed_data)),
                    "operation": self.operation
                }
            }
            
        except Exception as e:
            logger.error(f"Data Processor {self.node_id} execution error: {e}")
            return {"error": str(e)}
    
    async def _process_data(self, data: Any, config: Dict[str, Any]) -> Any:
        """Process data based on configuration"""
        # This is a simplified implementation
        # In production, this would handle various data transformations
        
        if self.operation == "transform":
            # Apply transformations
            transformations = self.properties.get("transformations", [])
            result = data
            
            for transform in transformations:
                # Apply each transformation
                if transform.get("type") == "filter":
                    # Apply filtering logic
                    pass
                elif transform.get("type") == "map":
                    # Apply mapping logic
                    pass
            
            return result
        
        elif self.operation == "aggregate":
            # Aggregation logic
            return {"aggregated": True, "original": data}
        
        else:
            return data
    
    async def cleanup(self):
        """Cleanup data processor resources"""
        pass

class InputValidatorComponent(ComponentBase):
    """Input validation component implementation"""
    
    metadata = ComponentMetadata(
        component_type="input_validator",
        category=ComponentCategory.SECURITY,
        display_name="Input Validator",
        description="Validate and sanitize input data",
        icon="shield",
        color="#FF9800",
        input_ports=[
            ComponentPort("input", PortType.ANY, required=True, description="Data to validate"),
            ComponentPort("schema", PortType.JSON, required=False, description="Validation schema"),
        ],
        output_ports=[
            ComponentPort("valid", PortType.ANY, description="Valid data output"),
            ComponentPort("invalid", PortType.ERROR, description="Invalid data and errors"),
        ],
        default_properties={
            "strict_mode": True,
            "max_length": 1000,
            "allowed_types": ["string", "number", "boolean"],
            "sanitize": True
        }
    )
    
    async def initialize(self) -> bool:
        """Initialize input validator"""
        self.strict_mode = self.properties.get("strict_mode", True)
        self.max_length = self.properties.get("max_length", 1000)
        logger.info(f"Input Validator {self.node_id} initialized")
        return True
    
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute input validation"""
        try:
            input_data = inputs.get("input")
            schema = inputs.get("schema", {})
            
            validation_result = await self._validate_input(input_data, schema)
            
            if validation_result["is_valid"]:
                return {
                    "valid": validation_result["sanitized_data"],
                    "validation_info": validation_result["info"]
                }
            else:
                return {
                    "invalid": {
                        "original_data": input_data,
                        "errors": validation_result["errors"]
                    }
                }
                
        except Exception as e:
            logger.error(f"Input Validator {self.node_id} execution error: {e}")
            return {"invalid": {"errors": [str(e)]}}
    
    async def _validate_input(self, data: Any, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input data"""
        errors = []
        
        # Length validation
        if isinstance(data, str) and len(data) > self.max_length:
            errors.append(f"Input length {len(data)} exceeds maximum {self.max_length}")
        
        # Type validation
        allowed_types = self.properties.get("allowed_types", [])
        if allowed_types:
            data_type = type(data).__name__
            if data_type not in allowed_types:
                errors.append(f"Type {data_type} not in allowed types {allowed_types}")
        
        # Schema validation (simplified)
        if schema:
            # Would implement JSON schema validation here
            pass
        
        # Sanitization
        sanitized_data = data
        if self.properties.get("sanitize", True) and isinstance(data, str):
            # Basic HTML/script tag removal
            import re
            sanitized_data = re.sub(r'<[^>]*>', '', data)
        
        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "sanitized_data": sanitized_data,
            "info": {"original_length": len(str(data)), "sanitized_length": len(str(sanitized_data))}
        }
    
    async def cleanup(self):
        """Cleanup validator resources"""
        pass

class MetricsCollectorComponent(ComponentBase):
    """Metrics collection component implementation"""
    
    metadata = ComponentMetadata(
        component_type="metrics_collector",
        category=ComponentCategory.MONITORING,
        display_name="Metrics Collector",
        description="Collect and report performance metrics",
        icon="chart",
        color="#9C27B0",
        input_ports=[
            ComponentPort("data", PortType.ANY, required=False, description="Data to analyze"),
            ComponentPort("trigger", PortType.SIGNAL, required=False, description="Collection trigger"),
        ],
        output_ports=[
            ComponentPort("metrics", PortType.METRICS, description="Collected metrics"),
        ],
        default_properties={
            "sampling_rate": 1.0,
            "metrics": ["cpu", "memory", "latency"],
            "aggregation_window": 60
        }
    )
    
    async def initialize(self) -> bool:
        """Initialize metrics collector"""
        self.sampling_rate = self.properties.get("sampling_rate", 1.0)
        self.metrics_buffer = []
        logger.info(f"Metrics Collector {self.node_id} initialized")
        return True
    
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute metrics collection"""
        try:
            import psutil
            import time
            
            # Collect system metrics
            metrics = {
                "timestamp": time.time(),
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent if hasattr(psutil, 'disk_usage') else 0,
            }
            
            # Add custom metrics if data provided
            if "data" in inputs:
                data_size = len(str(inputs["data"]))
                metrics["data_size"] = data_size
                metrics["processing_latency"] = 0.1  # Simulated
            
            # Store in buffer
            self.metrics_buffer.append(metrics)
            
            # Keep only recent metrics
            window_size = self.properties.get("aggregation_window", 60)
            if len(self.metrics_buffer) > window_size:
                self.metrics_buffer = self.metrics_buffer[-window_size:]
            
            return {"metrics": metrics}
            
        except Exception as e:
            logger.error(f"Metrics Collector {self.node_id} execution error: {e}")
            return {"error": str(e)}
    
    async def cleanup(self):
        """Cleanup metrics collector resources"""
        self.metrics_buffer.clear()

# ============================================================================
# COMPONENT REGISTRY
# ============================================================================

class ComponentRegistry:
    """Registry for managing component types and factories"""
    
    def __init__(self):
        self.components: Dict[str, ComponentMetadata] = {}
        self.factories: Dict[str, Type[ComponentBase]] = {}
        self.component_paths: Dict[str, str] = {}
        self._initialized = False
    
    async def initialize(self, db_session: AsyncSession = None):
        """Initialize the component registry"""
        if self._initialized:
            return
        
        # Register built-in components
        await self._register_builtin_components()
        
        # Load components from database
        if db_session:
            await self._load_components_from_db(db_session)
        
        # Load components from filesystem
        await self._discover_components()
        
        self._initialized = True
        logger.info(f"✅ Component registry initialized with {len(self.components)} components")
    
    async def _register_builtin_components(self):
        """Register built-in component types"""
        builtin_components = [
            (AIAgentComponent.metadata, AIAgentComponent),
            (DataProcessorComponent.metadata, DataProcessorComponent),
            (InputValidatorComponent.metadata, InputValidatorComponent),
            (MetricsCollectorComponent.metadata, MetricsCollectorComponent),
        ]
        
        for metadata, component_class in builtin_components:
            self.components[metadata.component_type] = metadata
            self.factories[metadata.component_type] = component_class
            
            logger.info(f"Registered built-in component: {metadata.component_type}")
    
    async def _load_components_from_db(self, db_session: AsyncSession):
        """Load component definitions from database"""
        try:
            from sqlalchemy import select
            
            result = await db_session.execute(select(ComponentType))
            db_components = result.scalars().all()
            
            for db_component in db_components:
                if db_component.component_type not in self.components:
                    # Create metadata from database record
                    metadata = ComponentMetadata(
                        component_type=db_component.component_type,
                        category=ComponentCategory(db_component.category),
                        display_name=db_component.display_name,
                        description=db_component.description or "",
                        icon=db_component.icon or "component",
                        color=db_component.color or "#6B73FF",
                        input_ports=[ComponentPort(**port) for port in db_component.input_ports],
                        output_ports=[ComponentPort(**port) for port in db_component.output_ports],
                        default_properties=db_component.default_properties,
                        implementation_class=db_component.implementation_class or ""
                    )
                    
                    self.components[db_component.component_type] = metadata
                    
                    # Try to load implementation class
                    if db_component.implementation_class:
                        await self._load_component_class(
                            db_component.component_type,
                            db_component.implementation_class
                        )
            
            logger.info(f"Loaded {len(db_components)} components from database")
            
        except Exception as e:
            logger.error(f"Failed to load components from database: {e}")
    
    async def _discover_components(self):
        """Discover components from filesystem"""
        component_dirs = [
            "backend/components/implementations",
            "plugins/components",
            "custom_components"
        ]
        
        for component_dir in component_dirs:
            path = Path(component_dir)
            if path.exists():
                await self._scan_component_directory(path)
    
    async def _scan_component_directory(self, directory: Path):
        """Scan directory for component implementations"""
        try:
            for py_file in directory.glob("*.py"):
                if py_file.name.startswith("__"):
                    continue
                
                module_name = f"{directory.as_posix().replace('/', '.')}.{py_file.stem}"
                await self._load_component_module(module_name)
                
        except Exception as e:
            logger.error(f"Error scanning component directory {directory}: {e}")
    
    async def _load_component_module(self, module_name: str):
        """Load a component module and register components"""
        try:
            module = importlib.import_module(module_name)
            
            # Look for component classes
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, ComponentBase) and 
                    obj != ComponentBase and
                    hasattr(obj, 'metadata')):
                    
                    metadata = obj.metadata
                    if metadata.component_type not in self.components:
                        self.components[metadata.component_type] = metadata
                        self.factories[metadata.component_type] = obj
                        self.component_paths[metadata.component_type] = module_name
                        
                        logger.info(f"Discovered component: {metadata.component_type} from {module_name}")
            
        except Exception as e:
            logger.error(f"Failed to load component module {module_name}: {e}")
    
    async def _load_component_class(self, component_type: str, class_path: str):
        """Load a specific component class by path"""
        try:
            module_path, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            component_class = getattr(module, class_name)
            
            if issubclass(component_class, ComponentBase):
                self.factories[component_type] = component_class
                self.component_paths[component_type] = class_path
                logger.info(f"Loaded component class: {component_type} from {class_path}")
            
        except Exception as e:
            logger.error(f"Failed to load component class {class_path}: {e}")
    
    def get_component_metadata(self, component_type: str) -> Optional[ComponentMetadata]:
        """Get metadata for a component type"""
        return self.components.get(component_type)
    
    def get_all_components(self) -> Dict[str, ComponentMetadata]:
        """Get all registered component metadata"""
        return self.components.copy()
    
    def get_components_by_category(self, category: ComponentCategory) -> Dict[str, ComponentMetadata]:
        """Get components filtered by category"""
        return {
            comp_type: metadata 
            for comp_type, metadata in self.components.items()
            if metadata.category == category
        }
    
    async def create_component(self, component_type: str, node_id: str, 
                             properties: Dict[str, Any] = None) -> Optional[ComponentBase]:
        """Create a component instance"""
        if component_type not in self.factories:
            logger.error(f"No factory found for component type: {component_type}")
            return None
        
        try:
            factory = self.factories[component_type]
            component = factory(node_id, properties)
            
            # Initialize the component
            if await component.initialize():
                logger.info(f"Created component: {component_type} with ID {node_id}")
                return component
            else:
                logger.error(f"Failed to initialize component: {component_type}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to create component {component_type}: {e}")
            return None
    
    def validate_component_properties(self, component_type: str, 
                                    properties: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component properties against schema"""
        metadata = self.get_component_metadata(component_type)
        if not metadata:
            return {"valid": False, "errors": ["Unknown component type"]}
        
        errors = []
        
        # Basic validation against property schema
        if metadata.property_schema:
            # Would implement JSON schema validation here
            pass
        
        # Validate required properties
        for key, default_value in metadata.default_properties.items():
            if key not in properties and default_value is None:
                errors.append(f"Required property '{key}' is missing")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "normalized_properties": {**metadata.default_properties, **properties}
        }
    
    async def register_component(self, metadata: ComponentMetadata, 
                               component_class: Type[ComponentBase]):
        """Register a new component type"""
        self.components[metadata.component_type] = metadata
        self.factories[metadata.component_type] = component_class
        
        logger.info(f"Registered new component: {metadata.component_type}")
    
    def unregister_component(self, component_type: str):
        """Unregister a component type"""
        self.components.pop(component_type, None)
        self.factories.pop(component_type, None)
        self.component_paths.pop(component_type, None)
        
        logger.info(f"Unregistered component: {component_type}")

# Global component registry instance
component_registry = ComponentRegistry()

# Dependency injection for FastAPI
async def get_component_registry() -> ComponentRegistry:
    """FastAPI dependency for component registry"""
    if not component_registry._initialized:
        db_session = None
        try:
            # Try to get a database session for loading components
            async for session in get_db_session():
                db_session = session
                break
        except:
            pass
        
        await component_registry.initialize(db_session)
    
    return component_registry

# Utility functions
async def get_available_components() -> Dict[str, Any]:
    """Get all available components organized by category"""
    registry = await get_component_registry()
    
    categories = {}
    for component_type, metadata in registry.get_all_components().items():
        category = metadata.category.value
        if category not in categories:
            categories[category] = []
        
        categories[category].append({
            "component_type": component_type,
            "display_name": metadata.display_name,
            "description": metadata.description,
            "icon": metadata.icon,
            "color": metadata.color,
            "input_ports": len(metadata.input_ports),
            "output_ports": len(metadata.output_ports)
        })
    
    return categories

async def validate_design_components(design_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate all components in a design"""
    registry = await get_component_registry()
    
    validation_results = []
    all_valid = True
    
    for node in design_data.get("nodes", []):
        component_type = node.get("component_type")
        properties = node.get("properties", {})
        
        if not component_type:
            validation_results.append({
                "node_id": node.get("node_id"),
                "valid": False,
                "errors": ["Missing component_type"]
            })
            all_valid = False
            continue
        
        # Validate component exists
        metadata = registry.get_component_metadata(component_type)
        if not metadata:
            validation_results.append({
                "node_id": node.get("node_id"),
                "valid": False,
                "errors": [f"Unknown component type: {component_type}"]
            })
            all_valid = False
            continue
        
        # Validate properties
        prop_validation = registry.validate_component_properties(component_type, properties)
        validation_results.append({
            "node_id": node.get("node_id"),
            "component_type": component_type,
            "valid": prop_validation["valid"],
            "errors": prop_validation["errors"],
            "normalized_properties": prop_validation.get("normalized_properties")
        })
        
        if not prop_validation["valid"]:
            all_valid = False
    
    return {
        "design_valid": all_valid,
        "component_validations": validation_results
    }
