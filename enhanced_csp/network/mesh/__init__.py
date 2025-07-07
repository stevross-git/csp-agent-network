# enhanced_csp/network/mesh/__init__.py
"""
Enhanced CSP Mesh Network Module

Advanced mesh networking capabilities with topology management,
BATMAN routing protocol, and adaptive network optimization.
"""

__version__ = "1.0.0"

# Import topology management components
from .topology import (
    MeshTopologyManager,
    TopologyType,
    NodeRole,
    LinkState,
    NetworkMetrics,
    MeshLink,
    TopologyOptimization
)

# Import routing protocol components
from .routing import (
    BatmanRouting,
    OriginatorMessage,
    RoutingEntry,
    RoutingTableEntry
)

# Convenience functions
def create_topology_manager(node_id, config, send_message_fn):
    """Create a mesh topology manager instance."""
    return MeshTopologyManager(node_id, config, send_message_fn)

def create_batman_routing(node, topology_manager):
    """Create a BATMAN routing protocol instance."""
    return BatmanRouting(node, topology_manager)

def create_mesh_link(local_node, remote_node, **kwargs):
    """Create a mesh link between two nodes."""
    return MeshLink(local_node=local_node, remote_node=remote_node, **kwargs)

# Export all public classes and functions
__all__ = [
    # Version
    '__version__',
    
    # Topology Management
    'MeshTopologyManager',
    'TopologyType',
    'NodeRole', 
    'LinkState',
    'NetworkMetrics',
    'MeshLink',
    'TopologyOptimization',
    
    # Routing Protocol
    'BatmanRouting',
    'OriginatorMessage',
    'RoutingEntry',
    'RoutingTableEntry',
    
    # Convenience functions
    'create_topology_manager',
    'create_batman_routing',
    'create_mesh_link',
]