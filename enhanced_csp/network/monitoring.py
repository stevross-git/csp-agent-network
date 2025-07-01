"""
Network node monitoring instrumentation
"""
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
from typing import Dict, Any

# Create registry for network metrics
NETWORK_REGISTRY = CollectorRegistry()

# Enhanced network metrics
network_peer_connections = Gauge(
    'enhanced_csp_peers',
    'Number of connected peers',
    ['node_id'],
    registry=NETWORK_REGISTRY
)

network_bandwidth_in = Counter(
    'enhanced_csp_bandwidth_in_bytes',
    'Incoming bandwidth in bytes',
    ['node_id'],
    registry=NETWORK_REGISTRY
)

network_bandwidth_out = Counter(
    'enhanced_csp_bandwidth_out_bytes',
    'Outgoing bandwidth in bytes',
    ['node_id'],
    registry=NETWORK_REGISTRY
)

network_message_latency = Histogram(
    'enhanced_csp_message_latency_seconds',
    'Message latency in seconds',
    ['message_type'],
    registry=NETWORK_REGISTRY
)

network_routing_table_size = Gauge(
    'enhanced_csp_routing_table_size',
    'Number of entries in routing table',
    ['node_id'],
    registry=NETWORK_REGISTRY
)

network_peer_reputation = Gauge(
    'enhanced_csp_peer_reputation',
    'Peer reputation score',
    ['peer_id'],
    registry=NETWORK_REGISTRY
)

class NetworkMonitor:
    """Monitor network node operations"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.message_count = 0
        self.bytes_in = 0
        self.bytes_out = 0
    
    def record_peer_connected(self, peer_count: int):
        """Record peer connection"""
        network_peer_connections.labels(node_id=self.node_id).set(peer_count)
    
    def record_bandwidth(self, bytes_in: int, bytes_out: int):
        """Record bandwidth usage"""
        network_bandwidth_in.labels(node_id=self.node_id).inc(bytes_in)
        network_bandwidth_out.labels(node_id=self.node_id).inc(bytes_out)
        self.bytes_in += bytes_in
        self.bytes_out += bytes_out
    
    def record_message_latency(self, message_type: str, latency: float):
        """Record message latency"""
        network_message_latency.labels(message_type=message_type).observe(latency)
    
    def update_routing_table(self, size: int):
        """Update routing table size"""
        network_routing_table_size.labels(node_id=self.node_id).set(size)
    
    def update_peer_reputation(self, peer_id: str, reputation: float):
        """Update peer reputation"""
        network_peer_reputation.labels(peer_id=peer_id).set(reputation)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return {
            "node_id": self.node_id,
            "message_count": self.message_count,
            "bytes_in": self.bytes_in,
            "bytes_out": self.bytes_out
        }
