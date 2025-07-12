# In enhanced_csp/network/core/types.py
# Update the NodeID dataclass to be hashable

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List, TYPE_CHECKING
import uuid
import hashlib
from cryptography.hazmat.primitives import serialization
from ..utils.secure_random import secure_bytes
from cryptography.hazmat.primitives.asymmetric import ed25519

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from .config import (
        SecurityConfig,
        P2PConfig,
        MeshConfig,
        DNSConfig,
        RoutingConfig,
        NetworkConfig,
    )
else:
    from .config import NetworkConfig


class MessageType(Enum):
    """Types of network messages."""
    PING = "ping"
    PONG = "pong"
    DATA = "data"
    BATCH = "batch"
    DHT_QUERY = "dht_query"
    DHT_RESPONSE = "dht_response"
    DNS_QUERY = "dns_query"
    DNS_RESPONSE = "dns_response"
    BOOTSTRAP = "bootstrap"
    PEER_ANNOUNCE = "peer_announce"
    ROUTE_UPDATE = "route_update"


BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def _b58encode(data: bytes) -> str:
    """Encode bytes to a base58 string."""
    num = int.from_bytes(data, "big")
    if num == 0:
        encoded = "0"
    else:
        encoded = ""
        while num > 0:
            num, rem = divmod(num, 58)
            encoded = BASE58_ALPHABET[rem] + encoded
    # preserve leading zeros
    pad = 0
    for b in data:
        if b == 0:
            pad += 1
        else:
            break
    return BASE58_ALPHABET[0] * pad + encoded


def _b58decode(s: str) -> bytes:
    """Decode a base58 string to bytes."""
    num = 0
    for ch in s:
        num = num * 58 + BASE58_ALPHABET.index(ch)
    byte_length = (num.bit_length() + 7) // 8
    data = num.to_bytes(byte_length, "big") if num else b""
    pad = 0
    for ch in s:
        if ch == BASE58_ALPHABET[0]:
            pad += 1
        else:
            break
    return b"\x00" * pad + data


@dataclass(frozen=True)
class NodeID:
    """Node identifier."""

    value: str = field(default="")
    raw_id: bytes = field(default=b"", repr=False)
    public_key: Optional[ed25519.Ed25519PublicKey] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.value and not self.raw_id:
            raise ValueError("Either value or raw_id must be provided")

        if self.value and not self.raw_id:
            object.__setattr__(self, "raw_id", _b58decode(self.value))
        elif self.raw_id and not self.value:
            object.__setattr__(self, "value", _b58encode(self.raw_id))

    def __str__(self) -> str:  # pragma: no cover - simple accessor
        return self.value

    def __hash__(self) -> int:  # pragma: no cover - trivial
        return hash(self.value)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, NodeID):
            return self.value == other.value
        return False

    @classmethod
    def generate(cls) -> "NodeID":
        """Generate a new node ID."""
        random_bytes = secure_bytes(32)
        raw = b"\x12\x20" + hashlib.sha256(random_bytes).digest()
        return cls(raw_id=raw)

    @classmethod
    def from_string(cls, value: str) -> "NodeID":
        """Create NodeID from string."""
        return cls(value=value)

    @classmethod
    def from_public_key(cls, public_key: ed25519.Ed25519PublicKey) -> "NodeID":
        """Create NodeID from an Ed25519 public key."""
        pk_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        raw = b"\x12\x20" + hashlib.sha256(pk_bytes).digest()
        return cls(raw_id=raw, public_key=public_key)

    def to_base58(self) -> str:
        """Return the base58 representation."""
        return self.value


@dataclass
class NodeCapabilities:
    """Node capability flags."""
    relay: bool = False
    storage: bool = False
    compute: bool = False
    quantum: bool = False
    blockchain: bool = False
    dns: bool = False
    bootstrap: bool = False
    

@dataclass
class PeerInfo:
    """Information about a peer node."""
    id: NodeID
    address: str
    port: int
    capabilities: NodeCapabilities
    last_seen: datetime = field(default_factory=datetime.utcnow)
    latency: Optional[float] = None
    reputation: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NetworkMessage:
    """Network message structure."""
    type: MessageType
    sender: NodeID
    recipient: Optional[NodeID] = None
    payload: Any = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    ttl: int = 64
    signature: Optional[bytes] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    @classmethod
    def create(cls, msg_type: MessageType, sender: NodeID, payload: Any,
               recipient: Optional[NodeID] = None, ttl: int = 64) -> 'NetworkMessage':
        """Create a new network message."""
        return cls(
            id=str(uuid.uuid4()),
            type=msg_type,
            sender=sender,
            recipient=recipient,
            payload=payload,
            timestamp=datetime.utcnow(),
            ttl=ttl,
        )


__all__ = [
    "NodeID",
    "NodeCapabilities",
    "PeerInfo",
    "NetworkMessage",
    "MessageType",
]
