# network/core/__init__.py
"""Core network components."""
from .config import (
    NetworkConfig,
    SecurityConfig,
    P2PConfig,
    MeshConfig,
    DNSConfig,
    RoutingConfig,
    PQCConfig,
)
# Avoid heavy imports on module import; import classes lazily in convenience
# functions when needed.
from .types import (
    NodeID, NodeCapabilities, MessageType, PeerInfo, NetworkMessage
)

__all__ = [
    # Configuration
    'NetworkConfig',
    'SecurityConfig',
    'P2PConfig',
    'MeshConfig',
    'DNSConfig',
    'RoutingConfig',
    'PQCConfig',
    
    # Types
    'NodeID',
    'NodeCapabilities',
    'MessageType',
    'PeerInfo',
    'NetworkMessage',
]