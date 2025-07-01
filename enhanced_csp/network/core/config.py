"""Network configuration classes."""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path


@dataclass
class SecurityConfig:
    """Security configuration for the network."""
    enable_tls: bool = True
    enable_mtls: bool = False
    enable_pq_crypto: bool = True
    enable_zero_trust: bool = False
    tls_cert_path: Optional[str] = None
    tls_key_path: Optional[str] = None
    ca_cert_path: Optional[str] = None
    allowed_cipher_suites: List[str] = field(default_factory=lambda: ["TLS_AES_256_GCM_SHA384"])
    min_tls_version: str = "1.3"
    audit_log_path: Optional[Path] = None
    rate_limit_requests: int = 100
    rate_limit_window: int = 60
    enable_threat_detection: bool = True
    threat_detection_threshold: float = 0.8
    enable_intrusion_prevention: bool = True
    max_connection_rate: int = 50
    enable_compliance_mode: bool = False
    compliance_standards: List[str] = field(default_factory=list)
    data_retention_days: int = 90
    enable_data_encryption: bool = True
    encryption_algorithm: str = "AES-256-GCM"
    key_rotation_interval: int = 86400
    tls_rotation_interval: int = 2592000
    enable_ca_mode: bool = False
    trust_anchors: List[str] = field(default_factory=list)


@dataclass
class NetworkConfig:
    """Main network configuration."""
    bootstrap_nodes: List[str] = field(default_factory=list)
    listen_address: str = "0.0.0.0"
    listen_port: int = 30300
    stun_servers: List[str] = field(default_factory=list)
    turn_servers: List[str] = field(default_factory=list)
    is_super_peer: bool = False
    enable_relay: bool = True
    enable_nat_traversal: bool = True
    enable_upnp: bool = True
    max_peers: int = 100
    peer_discovery_interval: int = 30
    peer_cleanup_interval: int = 300
    message_ttl: int = 64
    max_message_size: int = 1048576
    enable_compression: bool = True
    enable_encryption: bool = True
    node_capabilities: Dict[str, Any] = field(default_factory=dict)
    network_id: str = "enhanced-csp"
    protocol_version: str = "1.0.0"
    enable_metrics: bool = True
    metrics_interval: int = 60
    enable_dht: bool = True
    dht_bootstrap_nodes: List[str] = field(default_factory=list)
    routing_algorithm: str = "batman-adv"
    enable_qos: bool = True
    bandwidth_limit: int = 0
    enable_ipv6: bool = True
    dns_seeds: List[str] = field(default_factory=list)
    gossip_interval: int = 5
    gossip_fanout: int = 6
    enable_mdns: bool = True
    security: SecurityConfig = field(default_factory=SecurityConfig)