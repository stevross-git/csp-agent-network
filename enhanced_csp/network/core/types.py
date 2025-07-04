"""Core network types and data structures."""
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
import uuid
import hashlib
from .config import NetworkConfig




class MessageType(Enum):
    """Types of network messages."""
    PING = "ping"
    PONG = "pong"
    DATA = "data"
    DHT_QUERY = "dht_query"
    DHT_RESPONSE = "dht_response"
    DNS_QUERY = "dns_query"
    DNS_RESPONSE = "dns_response"
    BOOTSTRAP = "bootstrap"
    PEER_ANNOUNCE = "peer_announce"
    ROUTE_UPDATE = "route_update"


@dataclass
class NodeID:
    """Node identifier."""
    value: str
    
    def __str__(self):
        return self.value
        
    @classmethod
    def generate(cls) -> 'NodeID':
        """Generate a new node ID."""
        return cls(f"Qm{hashlib.sha256(uuid.uuid4().bytes).hexdigest()[:44]}")
        
    @classmethod
    def from_string(cls, value: str) -> 'NodeID':
        """Create NodeID from string."""
        return cls(value)


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
    last_seen: datetime
    latency: Optional[float] = None
    reputation: float = 1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class NetworkMessage:
    """Network message structure."""
    id: str
    type: MessageType
    sender: NodeID
    recipient: Optional[NodeID]
    payload: Any
    timestamp: datetime
    ttl: int = 64
    signature: Optional[bytes] = None
    
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
            ttl=ttl
        )
