# network/core/config.py
"""
Network configuration classes for Enhanced CSP.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class SecurityConfig:
    """Security configuration for network communication."""
    enable_tls: bool = True
    tls_version: str = "1.3"
    enable_pqc: bool = True  # Post-quantum cryptography
    pqc_algorithm: str = "kyber768"
    enable_encryption: bool = True
    enable_authentication: bool = True
    trusted_nodes: List[str] = field(default_factory=list)
    blocked_nodes: List[str] = field(default_factory=list)


@dataclass
class P2PConfig:
    """P2P network configuration."""
    listen_address: str = "0.0.0.0"
    listen_port: int = 9000
    enable_quic: bool = True
    enable_tcp: bool = True
    enable_websocket: bool = False
    enable_mdns: bool = True
    bootstrap_nodes: List[str] = field(default_factory=list)
    bootstrap_api_url: Optional[str] = None
    dns_seed_domain: Optional[str] = None
    stun_servers: List[str] = field(default_factory=lambda: [
        "stun:stun.l.google.com:19302",
        "stun:global.stun.twilio.com:3478"
    ])
    turn_servers: List[Dict[str, Any]] = field(default_factory=list)
    connection_timeout: int = 30
    max_connections: int = 100
    min_peers: int = 3
    max_peers: int = 50


@dataclass
class MeshConfig:
    """Mesh network topology configuration."""
    topology_type: str = "dynamic_partial"  # full_mesh, partial_mesh, dynamic_partial, hierarchical
    enable_super_peers: bool = True
    super_peer_capacity_threshold: float = 100.0  # Mbps
    max_peers: int = 20
    routing_update_interval: int = 10  # seconds
    link_quality_threshold: float = 0.5
    enable_multi_hop: bool = True
    max_hop_count: int = 10


@dataclass
class DNSConfig:
    """DNS overlay configuration."""
    root_domain: str = ".web4ai"
    enable_dnssec: bool = True
    default_ttl: int = 3600  # 1 hour
    cache_size: int = 10000
    enable_recursive: bool = True
    upstream_dns: List[str] = field(default_factory=lambda: [
        "8.8.8.8",
        "1.1.1.1"
    ])


@dataclass
class RoutingConfig:
    """Adaptive routing configuration."""
    enable_multipath: bool = True
    enable_ml_predictor: bool = True
    max_paths_per_destination: int = 3
    failover_threshold_ms: int = 500
    path_quality_update_interval: int = 30  # seconds
    enable_congestion_control: bool = True
    enable_qos: bool = True
    priority_levels: int = 4


@dataclass
class NetworkConfig:
    """Complete network stack configuration."""
    # Sub-configurations
    security: SecurityConfig = field(default_factory=SecurityConfig)
    p2p: P2PConfig = field(default_factory=P2PConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    dns: DNSConfig = field(default_factory=DNSConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    
    # Feature flags
    enable_discovery: bool = True
    enable_dht: bool = True
    enable_nat_traversal: bool = True
    enable_mesh: bool = True
    enable_dns: bool = True
    enable_adaptive_routing: bool = True
    enable_metrics: bool = True
    enable_compression: bool = True
    
    # Performance settings
    message_buffer_size: int = 1000
    max_message_size: int = 16 * 1024 * 1024  # 16MB
    worker_threads: int = 4
    
    # Storage settings
    enable_storage: bool = True
    storage_path: str = "./network_data"
    max_storage_size: int = 1024 * 1024 * 1024  # 1GB
    
    # Advanced features
    enable_quantum: bool = False
    enable_blockchain: bool = False
    enable_compute: bool = False
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'NetworkConfig':
        """Create NetworkConfig from dictionary."""
        config = cls()
        
        # Update security config
        if 'security' in config_dict:
            for key, value in config_dict['security'].items():
                setattr(config.security, key, value)
        
        # Update p2p config
        if 'p2p' in config_dict:
            for key, value in config_dict['p2p'].items():
                setattr(config.p2p, key, value)
        
        # Update mesh config
        if 'mesh' in config_dict:
            for key, value in config_dict['mesh'].items():
                setattr(config.mesh, key, value)
        
        # Update dns config
        if 'dns' in config_dict:
            for key, value in config_dict['dns'].items():
                setattr(config.dns, key, value)
        
        # Update routing config
        if 'routing' in config_dict:
            for key, value in config_dict['routing'].items():
                setattr(config.routing, key, value)
        
        # Update top-level settings
        for key, value in config_dict.items():
            if key not in ['security', 'p2p', 'mesh', 'dns', 'routing']:
                if hasattr(config, key):
                    setattr(config, key, value)
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert NetworkConfig to dictionary."""
        return {
            'security': self.security.__dict__,
            'p2p': self.p2p.__dict__,
            'mesh': self.mesh.__dict__,
            'dns': self.dns.__dict__,
            'routing': self.routing.__dict__,
            'enable_discovery': self.enable_discovery,
            'enable_dht': self.enable_dht,
            'enable_nat_traversal': self.enable_nat_traversal,
            'enable_mesh': self.enable_mesh,
            'enable_dns': self.enable_dns,
            'enable_adaptive_routing': self.enable_adaptive_routing,
            'enable_metrics': self.enable_metrics,
            'enable_compression': self.enable_compression,
            'message_buffer_size': self.message_buffer_size,
            'max_message_size': self.max_message_size,
            'worker_threads': self.worker_threads,
            'enable_storage': self.enable_storage,
            'storage_path': self.storage_path,
            'max_storage_size': self.max_storage_size,
            'enable_quantum': self.enable_quantum,
            'enable_blockchain': self.enable_blockchain,
            'enable_compute': self.enable_compute
        }