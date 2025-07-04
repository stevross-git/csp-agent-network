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
    MessageType
)

from .core.config import (
    NetworkConfig,
    SecurityConfig,
    P2PConfig,
    MeshConfig,
    DNSConfig,
    RoutingConfig
)

from .core.node import (
    NetworkNode,
    EnhancedCSPNetwork
)

# Convenience functions
def create_network(config: NetworkConfig = None) -> EnhancedCSPNetwork:
    """Create a new Enhanced CSP Network instance."""
    return EnhancedCSPNetwork(config)

def create_node(config: NetworkConfig = None) -> NetworkNode:
    """Create a new network node."""
    return NetworkNode(config)

# Export main classes and functions
__all__ = [
    # Version
    '__version__',
    
    # Core types
    'NodeID',
    'NodeCapabilities', 
    'PeerInfo',
    'NetworkMessage',
    'MessageType',
    
    # Configuration
    'NetworkConfig',
    'SecurityConfig',
    'P2PConfig',
    'MeshConfig',
    'DNSConfig',
    'RoutingConfig',
    
    # Core classes
    'NetworkNode',
    'EnhancedCSPNetwork',
    
    # Convenience functions
    'create_network',
    'create_node',
]