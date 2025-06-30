# enhanced_csp/network/core/types.py
"""
Core types and protocols for the Enhanced CSP Network Stack
"""

from dataclasses import dataclass, field
from typing import Protocol, Optional, Dict, List, Any, Tuple, Set, runtime_checkable
from enum import Enum, auto
import asyncio
from datetime import datetime
import ipaddress
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization


class NetworkProtocol(Enum):
    """Supported network protocols"""
    QUIC = "quic"
    TCP = "tcp"
    UDP = "udp"


class NodeStatus(Enum):
    """Node operational status"""
    STARTING = auto()
    DISCOVERING = auto()
    CONNECTED = auto()
    DISCONNECTED = auto()
    ERROR = auto()


class PeerType(Enum):
    """Peer node types in the mesh"""
    REGULAR = "regular"
    SUPER_PEER = "super_peer"
    BOOTSTRAP = "bootstrap"
    RELAY = "relay"


@dataclass(frozen=True)
class NodeID:
    """Self-certifying node identifier (multihash of Ed25519 public key)"""
    raw_id: bytes
    public_key: ed25519.Ed25519PublicKey
    
    @classmethod
    def from_public_key(cls, public_key: ed25519.Ed25519PublicKey) -> 'NodeID':
        """Create NodeID from public key using multihash format"""
        # Multihash: <hash-func><digest-size><digest>
        # Using identity hash (0x00) for now, can upgrade to SHA3-256 (0x16)
        key_bytes = public_key.public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
        multihash = b'\x00' + len(key_bytes).to_bytes(1, 'big') + key_bytes
        return cls(raw_id=multihash, public_key=public_key)
    
    def to_base58(self) -> str:
        """Convert to base58 string for human-readable format"""
        # Simple base58 implementation (production should use proper library)
        alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
        num = int.from_bytes(self.raw_id, 'big')
        encoded = ""
        while num > 0:
            num, remainder = divmod(num, 58)
            encoded = alphabet[remainder] + encoded

        n_pad = len(self.raw_id) - len(self.raw_id.lstrip(b"\x00"))
        return "1" * n_pad + encoded


@dataclass
class PeerInfo:
    """Information about a peer node"""
    node_id: NodeID
    addresses: List[str]  # multiaddr format: /ip4/1.2.3.4/tcp/4001/p2p/QmNodeID
    peer_type: PeerType = PeerType.REGULAR
    latency_ms: Optional[float] = None
    bandwidth_mbps: Optional[float] = None
    packet_loss: float = 0.0
    security_score: float = 1.0
    last_seen: datetime = field(default_factory=datetime.now)
    capabilities: Set[str] = field(default_factory=set)
    pow_nonce: Optional[str] = None
    
    @property
    def routing_cost(self) -> float:
        """Calculate routing cost metric"""
        if self.latency_ms is None or self.bandwidth_mbps is None:
            return float('inf')
        
        # Cost = latency × (1/bandwidth) × loss_factor × security_weight
        loss_factor = 1.0 + self.packet_loss
        security_weight = 2.0 - self.security_score  # Lower score = higher cost
        
        bandwidth = self.bandwidth_mbps if self.bandwidth_mbps > 0 else 1.0
        return (
            self.latency_ms * (1000.0 / bandwidth) *
            loss_factor * security_weight
        )


@dataclass
class NetworkStats:
    """Network performance statistics"""
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    active_connections: int = 0
    total_peers: int = 0
    super_peers: int = 0
    average_latency_ms: float = 0.0
    packet_loss_rate: float = 0.0
    uptime_seconds: float = 0.0


@runtime_checkable
class Transport(Protocol):
    """Transport layer protocol interface"""
    
    async def connect(self, address: str) -> 'Connection':
        """Establish connection to peer"""
        ...
    
    async def listen(self, address: str) -> None:
        """Listen for incoming connections"""
        ...
    
    async def close(self) -> None:
        """Close transport"""
        ...


@runtime_checkable
class Connection(Protocol):
    """Network connection protocol"""
    
    async def send(self, data: bytes) -> None:
        """Send data over connection"""
        ...
    
    async def receive(self) -> bytes:
        """Receive data from connection"""
        ...
    
    async def close(self) -> None:
        """Close connection"""
        ...
    
    @property
    def remote_peer(self) -> Optional[PeerInfo]:
        """Get remote peer information"""
        ...


@runtime_checkable
class DHT(Protocol):
    """Distributed Hash Table protocol"""
    
    async def get(self, key: bytes) -> Optional[bytes]:
        """Get value from DHT"""
        ...
    
    async def put(self, key: bytes, value: bytes) -> None:
        """Store value in DHT"""
        ...
    
    async def find_peer(self, node_id: NodeID) -> Optional[PeerInfo]:
        """Find peer by node ID"""
        ...
    
    async def announce(self, key: bytes, port: int) -> None:
        """Announce availability of a resource"""
        ...

    async def find_providers(self, key: bytes, count: int = 10) -> List[Dict[str, Any]]:
        """Find nodes providing a resource"""
        ...

    async def find_closest_peers(self, target: NodeID, k: int = 20) -> List[Dict[str, Any]]:
        """Find k closest peers to target"""
        ...


@dataclass
class RoutingEntry:
    """Entry in the routing table"""
    destination: NodeID
    next_hop: NodeID
    metric: float
    path: List[NodeID]
    last_updated: datetime = field(default_factory=datetime.now)
    is_active: bool = True


@dataclass
class DNSRecord:
    """DNS record in the overlay network"""
    name: str  # e.g., "alice.web4ai"
    record_type: str  # A, AAAA, TXT, SRV
    value: str
    ttl: int = 3600
    signature: Optional[bytes] = None  # Ed25519 signature
    created: datetime = field(default_factory=datetime.now)
    
    def is_expired(self) -> bool:
        """Check if record has expired"""
        age = (datetime.now() - self.created).total_seconds()
        return age > self.ttl


# Configuration types
@dataclass
class P2PConfig:
    """P2P network configuration"""
    listen_address: str = "0.0.0.0"
    listen_port: int = 4001
    enable_quic: bool = True
    bootstrap_nodes: List[str] = field(default_factory=list)
    bootstrap_api_url: str = ""
    dns_seed_domain: str = ""
    enable_mdns: bool = True
    enable_dht: bool = True
    dht_protocol: str = "kademlia"
    nat_traversal: bool = True
    stun_servers: List[str] = field(default_factory=lambda: [
        "stun:stun.l.google.com:19302",
        "stun:global.stun.twilio.com:3478"
    ])
    turn_servers: List[Dict[str, str]] = field(default_factory=list)
    stun_secret: str = ""
    max_peers: int = 50
    connection_timeout: int = 30


@dataclass
class MeshConfig:
    """Mesh network configuration"""
    topology_type: str = "dynamic_partial"
    enable_super_peers: bool = True
    super_peer_capacity_threshold: float = 100.0  # Mbps
    peer_selection_weights: Dict[str, float] = field(default_factory=lambda: {
        "latency": 0.7,
        "capacity": 0.3
    })
    routing_protocol: str = "batman_inspired"
    routing_update_interval: int = 10  # seconds
    max_hops: int = 10
    max_peers: int = 50


@dataclass
class DNSConfig:
    """DNS overlay configuration"""
    root_domain: str = ".web4ai"
    enable_dnssec: bool = True
    default_ttl: int = 3600
    cache_size: int = 10000
    resolver_timeout: int = 5
    supported_records: List[str] = field(default_factory=lambda: [
        "A", "AAAA", "TXT", "SRV"
    ])


@dataclass
class RoutingConfig:
    """Adaptive routing configuration"""
    enable_multipath: bool = True
    enable_ml_predictor: bool = True
    metric_update_interval: int = 1  # seconds
    ml_update_interval: int = 60  # seconds
    max_paths_per_destination: int = 3
    failover_threshold_ms: int = 500
    load_balance_algorithm: str = "weighted_round_robin"


@dataclass
class NetworkConfig:
    """Complete network stack configuration"""
    node_key_path: Optional[str] = None  # Path to Ed25519 private key
    data_dir: str = "./network_data"
    log_level: str = "INFO"
    enable_metrics: bool = True
    metrics_port: int = 9090
    
    # Sub-configurations
    p2p: P2PConfig = field(default_factory=P2PConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    dns: DNSConfig = field(default_factory=DNSConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    
    # Performance limits
    max_message_size: int = 1024 * 1024  # 1MB
    max_concurrent_connections: int = 2000
    connection_pool_size: int = 100
    
    # Security
    enable_tls: bool = True
    tls_version: str = "1.3"
    enable_pqc: bool = True  # Post-quantum crypto
    pqc_algorithm: str = "kyber768"
