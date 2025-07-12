# network/core/config.py
"""Network configuration dataclasses for Enhanced CSP."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, TYPE_CHECKING

from pathlib import Path
import json
import time

try:  # optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional
    yaml = None

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from .types import NodeCapabilities


NIST_KEM_ALGORITHMS = {"ML-KEM-768"}
NIST_SIGNATURE_ALGORITHMS = {"ML-DSA-65"}
NIST_BACKUP_SIGNATURES = {"SLH-DSA-SHAKE-128s"}


def _normalize_algorithm(value: str, mapping: Dict[str, str]) -> str:
    value_lower = value.lower().strip()
    return mapping.get(value_lower, value)


@dataclass
class PQCConfig:
    """Post-quantum cryptography settings."""

    kem_algorithm: str = "ML-KEM-768"
    """Key encapsulation mechanism algorithm."""

    signature_algorithm: str = "ML-DSA-65"
    """Digital signature algorithm."""

    backup_signature: str = "SLH-DSA-SHAKE-128s"
    """Backup signature scheme used as fallback."""

    key_rotation_interval: int = 86400 * 7
    """Interval in seconds to rotate PQC keys."""

    enable_hybrid_mode: bool = True
    """Use classical algorithms in parallel with PQC."""

    def __post_init__(self) -> None:
        old_kem = {"kyber768": "ML-KEM-768"}
        old_sig = {"dilithium3": "ML-DSA-65"}
        old_backup = {"sphincs+": "SLH-DSA-SHAKE-128s"}

        self.kem_algorithm = _normalize_algorithm(self.kem_algorithm, old_kem)
        self.signature_algorithm = _normalize_algorithm(
            self.signature_algorithm, old_sig
        )
        self.backup_signature = _normalize_algorithm(self.backup_signature, old_backup)

        if self.kem_algorithm not in NIST_KEM_ALGORITHMS:
            raise ValueError(f"Unsupported KEM algorithm: {self.kem_algorithm}")
        if self.signature_algorithm not in NIST_SIGNATURE_ALGORITHMS:
            raise ValueError(
                f"Unsupported signature algorithm: {self.signature_algorithm}"
            )
        if self.backup_signature not in NIST_BACKUP_SIGNATURES:
            raise ValueError(
                f"Unsupported backup signature: {self.backup_signature}"
            )
        if self.key_rotation_interval <= 0:
            raise ValueError("key_rotation_interval must be positive")



@dataclass
class SecurityConfig:
    """Security related settings."""

    enable_tls: bool = True
    """Enable TLS for all transports."""

    enable_mtls: bool = False
    """Require mutual TLS authentication."""

    tls_version: str = "1.3"
    """TLS protocol version."""

    tls_cert_path: Optional[Path] = None
    """Path to the TLS certificate file."""

    tls_key_path: Optional[Path] = None
    """Path to the TLS private key file."""

    ca_cert_path: Optional[Path] = None
    """Path to the CA certificate file."""

    enable_pq_crypto: bool = True
    """Enable post-quantum cryptography features."""

    pqc: PQCConfig = field(default_factory=PQCConfig)
    """Post-quantum cryptography configuration."""

    def __post_init__(self) -> None:
        for attr in ("tls_cert_path", "tls_key_path", "ca_cert_path"):
            value = getattr(self, attr)
            if isinstance(value, str):
                value = Path(value)
                object.__setattr__(self, attr, value)
            if value is not None and not value.exists():
                raise ValueError(f"{attr} does not exist: {value}")
        if isinstance(self.pqc, dict):
            object.__setattr__(self, "pqc", PQCConfig(**self.pqc))
        self.pqc.__post_init__()


@dataclass
class P2PConfig:
    """Peer-to-peer connectivity settings."""

    listen_address: str = "0.0.0.0"
    """IP address to bind to."""

    listen_port: int = 9000
    """Port for incoming connections."""

    enable_quic: bool = True
    """Enable QUIC transport."""

    enable_tcp: bool = True
    enable_websocket: bool = False
    enable_mdns: bool = True

    bootstrap_nodes: List[str] = field(default_factory=list)
    bootstrap_api_url: Optional[str] = None
    dns_seed_domain: Optional[str] = None
    stun_servers: List[str] = field(
        default_factory=lambda: [
            "stun:stun.l.google.com:19302",
            "stun:global.stun.twilio.com:3478",
        ]
    )
    turn_servers: List[Dict[str, Any]] = field(default_factory=list)

    connection_timeout: int = 30
    max_connections: int = 100
    min_peers: int = 3
    max_peers: int = 50
    max_message_size: int = 1024 * 1024
    """Maximum size of a message in bytes."""

    def __post_init__(self) -> None:
        from ..utils import validate_ip_address, validate_port_number

        validate_ip_address(self.listen_address)
        self.listen_port = validate_port_number(self.listen_port)
        if self.max_message_size <= 0:
            raise ValueError("max_message_size must be positive")


@dataclass
class MeshConfig:
    """Mesh topology parameters."""

    topology_type: str = "dynamic_partial"
    """Type of mesh topology to use."""

    enable_super_peers: bool = True
    super_peer_capacity_threshold: float = 100.0
    max_peers: int = 20
    routing_update_interval: int = 10
    link_quality_threshold: float = 0.5
    enable_multi_hop: bool = True
    max_hop_count: int = 10

    def __post_init__(self) -> None:
        if self.max_peers <= 0:
            raise ValueError("max_peers must be positive")


@dataclass
class DNSConfig:
    """Distributed DNS overlay settings."""

    root_domain: str = ".web4ai"
    """Root domain used for DNS entries."""

    enable_dnssec: bool = True
    default_ttl: int = 3600
    cache_size: int = 10000
    enable_recursive: bool = True
    upstream_dns: List[str] = field(default_factory=lambda: ["8.8.8.8", "1.1.1.1"])

    def __post_init__(self) -> None:
        if not self.root_domain.startswith('.'):
            raise ValueError("root_domain must start with '.'")


@dataclass
class RoutingConfig:
    """Settings for adaptive routing."""

    enable_multipath: bool = True
    enable_ml_predictor: bool = True
    max_paths_per_destination: int = 3
    failover_threshold_ms: int = 500
    path_quality_update_interval: int = 30
    metric_update_interval: int = 30
    route_optimization_interval: int = 60
    enable_congestion_control: bool = True
    enable_qos: bool = True
    priority_levels: int = 4

    def __post_init__(self) -> None:
        if self.max_paths_per_destination <= 0:
            raise ValueError("max_paths_per_destination must be positive")


class NetworkConfig:
    """Top-level configuration for a network node."""

    security: SecurityConfig = field(default_factory=SecurityConfig)
    p2p: P2PConfig = field(default_factory=P2PConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    dns: DNSConfig = field(default_factory=DNSConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)

    node_name: str = "csp-node"
    node_type: str = "standard"
    capabilities: 'NodeCapabilities' = field(default_factory=lambda: __import__(
        'enhanced_csp.network.core.types', fromlist=['NodeCapabilities']).NodeCapabilities())
    data_dir: Path = Path("./data")

    def __post_init__(self) -> None:
        self.p2p.__post_init__()
        self.mesh.__post_init__()
        self.dns.__post_init__()
        self.routing.__post_init__()
        self.data_dir = Path(self.data_dir).expanduser()
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Convenience factory methods
    # ------------------------------------------------------------------
    @classmethod
    def development(cls) -> 'NetworkConfig':
        """Configuration suitable for development."""
        return cls(
            node_name="dev-node",
            data_dir=Path("./dev_data"),
            p2p=P2PConfig(listen_port=9000),
        )

    @classmethod
    def production(cls) -> 'NetworkConfig':
        """Configuration with production defaults."""
        return cls()

    @classmethod
    def test(cls) -> 'NetworkConfig':
        """Configuration for testing environments."""
        return cls(
            node_name="test-node",
            data_dir=Path("./test_data"),
            p2p=P2PConfig(listen_port=0),
        )

    # ------------------------------------------------------------------
    # (De)serialisation helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NetworkConfig':
        return cls(
            security=SecurityConfig(**data.get('security', {})),
            p2p=P2PConfig(**data.get('p2p', {})),
            mesh=MeshConfig(**data.get('mesh', {})),
            dns=DNSConfig(**data.get('dns', {})),
            routing=RoutingConfig(**data.get('routing', {})),
            node_name=data.get('node_name', 'csp-node'),
            node_type=data.get('node_type', 'standard'),
            capabilities=__import__('enhanced_csp.network.core.types', fromlist=['NodeCapabilities']).NodeCapabilities(**data.get('capabilities', {})),
            data_dir=Path(data.get('data_dir', './data')),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'security': asdict(self.security),
            'p2p': asdict(self.p2p),
            'mesh': asdict(self.mesh),
            'dns': asdict(self.dns),
            'routing': asdict(self.routing),
            'node_name': self.node_name,
            'node_type': self.node_type,
            'capabilities': asdict(self.capabilities),
            'data_dir': str(self.data_dir),
        }

    @classmethod
    def load(cls, path: Path) -> 'NetworkConfig':
        path = Path(path)
        with path.open('r', encoding='utf-8') as fh:
            if path.suffix.lower() in {'.json'} or yaml is None:
                data = json.load(fh)
            else:
                data = yaml.safe_load(fh)
        return cls.from_dict(data)

    def save(self, path: Path) -> None:
        path = Path(path)
        with path.open('w', encoding='utf-8') as fh:
            if path.suffix.lower() in {'.json'} or yaml is None:
                json.dump(self.to_dict(), fh, indent=2)
            else:
                yaml.safe_dump(self.to_dict(), fh)
