# enhanced_csp/network/__init__.py
"""
Enhanced CSP Network Module
Peer-to-peer networking, mesh topology, and adaptive routing
"""

__version__ = "1.0.0"

# Core network types and classes
from .core.types import (
    NodeID,
    NodeCapabilities,
    PeerInfo,
    NetworkMessage,
    MessageType,
)

from .core.config import (
    NetworkConfig,
    SecurityConfig,
    P2PConfig,
    MeshConfig,
    DNSConfig,
    RoutingConfig,
    PQCConfig,
)

from .errors import (
    NetworkError,
    ConnectionError,
    TimeoutError,
    ProtocolError,
    SecurityError,
    ValidationError,
    ErrorMetrics,
    CircuitBreakerOpen,
)

from .utils import (
    setup_logging,
    get_logger,
    NetworkLogger,
    SecurityLogger,
    PerformanceLogger,
    AuditLogger,
)

# Avoid importing heavy classes during module import to reduce optional
# dependencies for consumers that only need basic types.

def _lazy_network_node():
    from .core.node import NetworkNode
    return NetworkNode


def _lazy_enhanced_network():
    from .core.node import EnhancedCSPNetwork
    return EnhancedCSPNetwork

# Convenience functions
def create_network(config: NetworkConfig | None = None):
    """Create a new Enhanced CSP Network instance."""
    return _lazy_enhanced_network()(config)


def create_node(config: NetworkConfig | None = None):
    """Create a new network node."""
    return _lazy_network_node()(config)

# Export main classes and functions
__all__ = [
    # Version
    "__version__",

    # Core types
    "NodeID",
    "NodeCapabilities",
    "PeerInfo",
    "NetworkMessage",
    "MessageType",

    # Configuration
    "NetworkConfig",
    "SecurityConfig",
    "P2PConfig",
    "MeshConfig",
    "DNSConfig",
    "RoutingConfig",
    "PQCConfig",

    # Errors
    "NetworkError",
    "ConnectionError",
    "TimeoutError",
    "ProtocolError",
    "SecurityError",
    "ValidationError",
    "ErrorMetrics",
    "CircuitBreakerOpen",

    # Logging utilities
    "setup_logging",
    "get_logger",
    "NetworkLogger",
    "SecurityLogger",
    "PerformanceLogger",
    "AuditLogger",

    # Convenience functions
    "create_network",
    "create_node",
]
