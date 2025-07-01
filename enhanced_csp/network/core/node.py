import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import sys

from .config import NetworkConfig, SecurityConfig
from .types import NodeID, PeerInfo

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import network components with fallbacks
try:
    from enhanced_csp.network.dns.overlay import DNSOverlay
    from enhanced_csp.network.p2p.transport import P2PTransport
    from enhanced_csp.network.mesh.topology import MeshTopology
    DNS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Failed to import network components: {e}")
    DNS_AVAILABLE = False
    
    # Create minimal stubs
    class DNSOverlay:
        def __init__(self, node):
            self.node = node
            self.records = {}
            
        async def start(self): pass
        async def stop(self): pass
        
        async def register(self, name: str, value: str, ttl: int = 3600):
            print(f"Stub DNS register: {name} -> {value}")
            self.records[name] = value
            
        async def resolve(self, name: str):
            return self.records.get(name)
            
        async def list_records(self):
            return self.records
        
    class P2PTransport:
        def __init__(self, config): pass
        async def start(self): pass
        async def stop(self): pass
        async def connect(self, addr): pass
        async def send(self, peer_id, msg): pass
        
    class MeshTopology:
        def __init__(self, node): pass
        async def start(self): pass
        async def stop(self): pass

# Optional subsystems - imported when available
try:
    from enhanced_csp.security_hardening import SecurityOrchestrator
except ImportError:
    class SecurityOrchestrator:
        def __init__(self, *args, **kwargs): pass
        async def initialize(self): pass
        async def shutdown(self): pass
        async def monitor_threats(self): pass
        async def rotate_tls_certificates(self): pass

try:
    from enhanced_csp.quantum_csp_engine import QuantumCSPEngine
except ImportError:
    class QuantumCSPEngine:
        def __init__(self, *args, **kwargs): pass
        async def initialize(self): pass
        async def shutdown(self): pass

try:
    from enhanced_csp.blockchain_csp_network import BlockchainCSPNetwork
except ImportError:
    class BlockchainCSPNetwork:
        def __init__(self, *args, **kwargs): pass
        async def initialize(self): pass
        async def shutdown(self): pass


@dataclass
class NodeStats:
    start_time: datetime
    peers: int = 0
    messages_sent: int = 0
    messages_recv: int = 0
    bandwidth_in: int = 0
    bandwidth_out: int = 0
    bootstrap_reqs: int = 0


class NetworkNode:
    """Enhanced CSP Network node implementation."""
    
    def __init__(self, config: NetworkConfig, 
                 enable_quantum: bool = False,
                 enable_blockchain: bool = False,
                 logger: Optional[logging.Logger] = None):
        self.config = config
        self.node_id = NodeID.generate()
        self.is_running = False
        self.logger = logger or logging.getLogger(f"enhanced_csp.NetworkNode.{self.node_id}")
        
        # Stats tracking
        self.stats = NodeStats(start_time=datetime.utcnow())
        
        # Peer management
        self.peers: List[PeerInfo] = []
        
        # Initialize network components
        self.transport = P2PTransport(config)
        self.topology = MeshTopology(self)
        self.dns_overlay = DNSOverlay(self)
        
        # Log initialization status
        if DNS_AVAILABLE:
            self.logger.info("DNS overlay initialized successfully")
        else:
            self.logger.warning("Using stub DNS overlay implementation")
        
        # Security orchestrator
        self.security = SecurityOrchestrator(config.security)
        
        # Optional subsystems
        self.quantum_engine = QuantumCSPEngine(self) if enable_quantum else None
        self.blockchain = BlockchainCSPNetwork(self) if enable_blockchain else None
        
        # Background tasks
        self._tasks: List[asyncio.Task] = []
        
    async def start(self):
        """Start the network node."""
        self.logger.info(f"Starting Enhanced CSP Node {self.node_id}")
        
        # Initialize security
        await self.security.initialize()
        
        # Initialize optional subsystems
        if self.quantum_engine:
            await self.quantum_engine.initialize()
        if self.blockchain:
            await self.blockchain.initialize()
        
        # Start network components
        await self.transport.start()
        await self.dns_overlay.start()
        await self.topology.start()
        
        self.is_running = True
        self.stats.start_time = datetime.utcnow()
        
        # Start background tasks
        self._tasks.append(asyncio.create_task(self._background_security()))
        self._tasks.append(asyncio.create_task(self._background_metrics()))
        
        # Bootstrap if not genesis
        if self.config.bootstrap_nodes:
            await self._bootstrap()
            
    async def stop(self):
        """Stop the network node."""
        self.logger.info("Stopping Enhanced CSP Node")
        
        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        
        # Stop network components
        await self.topology.stop()
        await self.dns_overlay.stop()
        await self.transport.stop()
        
        # Shutdown subsystems
        if self.blockchain:
            await self.blockchain.shutdown()
        if self.quantum_engine:
            await self.quantum_engine.shutdown()
        await self.security.shutdown()
        
        self.is_running = False
        
    async def _bootstrap(self):
        """Bootstrap the node by connecting to bootstrap nodes."""
        for bootstrap in self.config.bootstrap_nodes:
            try:
                self.logger.info(f"Bootstrapping from {bootstrap}")
                # Resolve DNS if needed
                if bootstrap.endswith('.web4ai'):
                    resolved = await self.dns_overlay.resolve(bootstrap)
                    if resolved:
                        bootstrap = resolved
                        
                # Connect to bootstrap node
                await self.transport.connect(bootstrap)
                self.stats.bootstrap_reqs += 1
            except Exception as e:
                self.logger.warning(f"Failed to connect to bootstrap {bootstrap}: {e}")
                
    async def _background_security(self):
        """Background security monitoring."""
        while self.is_running:
            try:
                await self.security.monitor_threats()
                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Security monitoring error: {e}")
                
    async def _background_metrics(self):
        """Background metrics collection."""
        while self.is_running:
            try:
                self.stats.peers = len(self.peers)
                self.logger.debug(f"Metrics: {self.stats}")
                await asyncio.sleep(self.config.metrics_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Metrics error: {e}")
    
    def get_peers(self) -> List[PeerInfo]:
        """Get list of connected peers."""
        return self.peers
        
    async def send_message(self, peer_id: str, message: Any):
        """Send a message to a peer."""
        await self.transport.send(peer_id, message)
        self.stats.messages_sent += 1
        
    async def connect(self, multiaddr: str):
        """Connect to a peer."""
        await self.transport.connect(multiaddr)
        
    async def broadcast(self, payload: Any):
        """Broadcast to all peers."""
        for peer in self.peers:
            await self.send_message(str(peer.id), payload)
            
    async def metrics(self) -> Dict[str, Any]:
        """Get node metrics."""
        return {
            "node_id": str(self.node_id),
            "uptime": (datetime.utcnow() - self.stats.start_time).total_seconds(),
            "peers": self.stats.peers,
            "messages_sent": self.stats.messages_sent,
            "messages_received": self.stats.messages_recv,
            "bandwidth_in": self.stats.bandwidth_in,
            "bandwidth_out": self.stats.bandwidth_out,
        }


# Alias for backward compatibility
EnhancedCSPNetwork = NetworkNode