# enhanced_csp/__init__.py
"""
Enhanced CSP - Communicating Sequential Processes with AI Integration
"""

# Version
__version__ = "1.0.0"

# Core imports
from .core.types import (
    Message,
    Channel,
    Process,
    ProcessState,
    ChannelType,
    MessagePriority,
    ProcessContext
)

from .core.engine import (
    CSPEngine,
    EngineConfig,
    ExecutionMode
)

# Agents
from .agents.base import BaseAgent
from .agents.cleaner import DataCleanerAgent
from .agents.planner import PlannerAgent

# AI Communication
from .ai_comm.channel import (
    AdvancedAICommChannel,
    AdvancedCommPattern,
    ChannelConfig
)

from .ai_comm.patterns import (
    NeuralMeshPattern,
    BroadcastPattern,
    PipelinePattern,
    QuantumEntangledPattern
)

# API and Storage
from .api.log_store import CSPLogStore
from .api.metrics import MetricsCollector, MetricsExporter

# Memory
from .memory.vector_store import ChromaVectorStore
from .memory.document_store import DocumentStore
from .memory.cache import MemoryCache

# Protocols
from .protocols.csp_protocol import (
    create_csp_message,
    parse_csp_message,
    validate_csp_message,
    CSPProtocol,
    ProtocolVersion
)

# Network Optimization
from .network import (
    MessageCompressor,
    MessageBatcher,
    ConnectionPool,
    BinaryProtocol,
    AdaptiveNetworkOptimizer,
    OptimizedNetworkChannel,
    NetworkMetricsCollector,
    CompressionAlgorithm,
    BatchConfig
)

# Configuration
from .config import (
    settings,
    CSPSettings,
    NetworkOptimizationConfig,
    AgentConfig,
    ChannelConfig as ConfigChannelConfig,
    load_config,
    load_defaults
)

# Utilities
from .utils import (
    generate_id,
    get_timestamp,
    format_bytes,
    AsyncContextManager,
    retry_async,
    timeout_async
)

# Monitoring
from .monitoring import (
    HealthChecker,
    SystemMonitor,
    PerformanceTracker
)

# Exceptions
from .exceptions import (
    CSPException,
    ChannelClosedException,
    ProcessTerminatedException,
    ConfigurationException,
    NetworkException,
    TimeoutException
)

# Main entry points
from .main import (
    create_app,
    run_server,
    bootstrap_system
)

__all__ = [
    # Version
    '__version__',
    
    # Core types
    'Message',
    'Channel',
    'Process',
    'ProcessState',
    'ChannelType',
    'MessagePriority',
    'ProcessContext',
    
    # Engine
    'CSPEngine',
    'EngineConfig',
    'ExecutionMode',
    
    # Agents
    'BaseAgent',
    'DataCleanerAgent',
    'PlannerAgent',
    
    # AI Communication
    'AdvancedAICommChannel',
    'AdvancedCommPattern',
    'ChannelConfig',
    'NeuralMeshPattern',
    'BroadcastPattern',
    'PipelinePattern',
    'QuantumEntangledPattern',
    
    # API and Storage
    'CSPLogStore',
    'MetricsCollector',
    'MetricsExporter',
    
    # Memory
    'ChromaVectorStore',
    'DocumentStore',
    'MemoryCache',
    
    # Protocols
    'create_csp_message',
    'parse_csp_message',
    'validate_csp_message',
    'CSPProtocol',
    'ProtocolVersion',
    
    # Network Optimization
    'MessageCompressor',
    'MessageBatcher',
    'ConnectionPool',
    'BinaryProtocol',
    'AdaptiveNetworkOptimizer',
    'OptimizedNetworkChannel',
    'NetworkMetricsCollector',
    'CompressionAlgorithm',
    'BatchConfig',
    
    # Configuration
    'settings',
    'CSPSettings',
    'NetworkOptimizationConfig',
    'AgentConfig',
    'ConfigChannelConfig',
    'load_config',
    'load_defaults',
    
    # Utilities
    'generate_id',
    'get_timestamp',
    'format_bytes',
    'AsyncContextManager',
    'retry_async',
    'timeout_async',
    
    # Monitoring
    'HealthChecker',
    'SystemMonitor',
    'PerformanceTracker',
    
    # Exceptions
    'CSPException',
    'ChannelClosedException',
    'ProcessTerminatedException',
    'ConfigurationException',
    'NetworkException',
    'TimeoutException',
    
    # Main entry points
    'create_app',
    'run_server',
    'bootstrap_system'
]

# Initialize logging
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Check for required dependencies
def check_dependencies():
    """Check if all required dependencies are installed"""
    required = {
        'aiohttp': 'aiohttp>=3.8.0',
        'msgpack': 'msgpack>=1.0.0',
        'pydantic': 'pydantic>=2.0.0',
        'chromadb': 'chromadb>=0.4.0'
    }
    
    missing = []
    for module, requirement in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(requirement)
    
    if missing:
        import warnings
        warnings.warn(
            f"Missing required dependencies: {', '.join(missing)}. "
            f"Please install with: pip install {' '.join(missing)}",
            ImportWarning,
            stacklevel=2
        )

# Check dependencies on import
check_dependencies()

# Package metadata
__author__ = "Enhanced CSP Team"
__email__ = "team@enhanced-csp.io"
__license__ = "MIT"
__url__ = "https://github.com/enhanced-csp/enhanced-csp"
__description__ = "Advanced Communicating Sequential Processes with AI Integration and Network Optimization"