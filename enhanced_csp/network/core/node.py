"""Enhanced CSP Network node implementation."""
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import os

from .config import NetworkConfig, SecurityConfig
from .types import NodeID, PeerInfo

# Import DNS and other components with proper paths
try:
    from ..dns.overlay import DNSOverlay
    from ..p2p.transport import P2PTransport
    from ..mesh.topology import MeshTopology
except ImportError:
    # Fallback imports for when running from different locations
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from dns.overlay import DNSOverlay
    from p2p.transport import P2PTransport
    from mesh.topology import MeshTopology


class EnhancedCSPNetwork:
    """Main network node implementation."""
    
    def __init__(self, config: NetworkConfig):
        self.config = config
        self.node_id = NodeID.generate()
        self.is_running = False
        self.start_time = datetime.utcnow()
        self.stats: Dict[str, Any] = {
            "messages_sent": 0,
            "messages_received": 0,
            "bandwidth_in": 0,
            "bandwidth_out": 0,
            "bootstrap_requests": 0,
        }
        
        # Initialize components
        self.transport = P2PTransport(config)
        self.topology = MeshTopology(self)
        self.dns_overlay = DNSOverlay(self)
        self.peers: List[PeerInfo] = []
        
        self.logger = logging.getLogger(f"enhanced_csp.node.{self.node_id}")
        
    async def start(self):
        """Start the network node."""
        self.logger.info(f"Starting Enhanced CSP Node {self.node_id}")
        
        # Start transport layer
        await self.transport.start()
        
        # Start DNS overlay
        await self.dns_overlay.start()
        
        # Start topology management
        await self.topology.start()
        
        self.is_running = True
        self.start_time = datetime.utcnow()
        
        # Bootstrap if not genesis
        if self.config.bootstrap_nodes:
            await self._bootstrap()
            
    async def stop(self):
        """Stop the network node."""
        self.logger.info("Stopping Enhanced CSP Node")
        self.is_running = False
        
        await self.topology.stop()
        await self.dns_overlay.stop()
        await self.transport.stop()
        
    async def _bootstrap(self):
        """Bootstrap the node by connecting to bootstrap nodes."""
        for bootstrap in self.config.bootstrap_nodes:
            try:
                # Resolve DNS if needed
                if bootstrap.endswith('.web4ai'):
                    resolved = await self.dns_overlay.resolve(bootstrap)
                    if resolved:
                        bootstrap = resolved
                        
                # Connect to bootstrap node
                await self.transport.connect(bootstrap)
                self.stats["bootstrap_requests"] += 1
            except Exception as e:
                self.logger.warning(f"Failed to connect to bootstrap {bootstrap}: {e}")
                
    def get_peers(self) -> List[PeerInfo]:
        """Get list of connected peers."""
        return self.peers
        
    async def send_message(self, peer_id: str, message: Any):
        """Send a message to a peer."""
        await self.transport.send(peer_id, message)
        self.stats["messages_sent"] += 1


# Re-export config classes
__all__ = ['NetworkConfig', 'SecurityConfig', 'EnhancedCSPNetwork']
