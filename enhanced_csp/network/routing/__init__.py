# enhanced_csp/network/routing/__init__.py
"""
Enhanced CSP Network Routing Module

Provides adaptive routing, metrics collection, and multipath management
for optimal network performance and reliability.
"""

__version__ = "1.0.0"

# Import core routing components
from .adaptive import (
    AdaptiveRoutingEngine,
    RouteMetrics,
    FlowState,
    RoutePredictor
)

from .metrics import (
    MetricsCollector,
    ProbePacket,
    MeasurementSession
)

from .multipath import (
    MultipathManager,
    PathDiversity
)

# Convenience functions
def create_adaptive_engine(node, config, batman_routing):
    """Create an adaptive routing engine instance."""
    return AdaptiveRoutingEngine(node, config, batman_routing)

def create_metrics_collector(node):
    """Create a metrics collector instance."""
    return MetricsCollector(node)

def create_multipath_manager(max_paths=3):
    """Create a multipath manager instance."""
    return MultipathManager(max_paths)

# Export all public classes and functions
__all__ = [
    # Version
    '__version__',
    
    # Adaptive routing
    'AdaptiveRoutingEngine',
    'RouteMetrics', 
    'FlowState',
    'RoutePredictor',
    
    # Metrics collection
    'MetricsCollector',
    'ProbePacket',
    'MeasurementSession',
    
    # Multipath routing
    'MultipathManager',
    'PathDiversity',
    
    # Convenience functions
    'create_adaptive_engine',
    'create_metrics_collector',
    'create_multipath_manager',
]