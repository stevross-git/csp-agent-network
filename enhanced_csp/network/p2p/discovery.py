# network/p2p/discovery.py
"""
Hybrid peer discovery implementation for Enhanced CSP.
Combines mDNS, bootstrap nodes, and DHT for robust peer discovery.
"""

import asyncio
import logging
import socket
import json
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from datetime import datetime

try:
    from zeroconf import ServiceBrowser, ServiceInfo, Zeroconf
    HAS_ZEROCONF = True
except ImportError:
    HAS_ZEROCONF = False
    
try:
    import aiodns
    HAS_AIODNS = True
except ImportError:
    HAS_AIODNS = False

from ..core.types import NodeID
from enhanced_csp.network.core.config import P2PConfig

logger = logging.getLogger(__name__)


class PeerType(Enum):
    """Types of peers in the network."""
    REGULAR = "regular"
    BOOTSTRAP = "bootstrap"
    SUPER_PEER = "super_peer"
    RELAY = "relay"


class HybridDiscovery:
    """
    Hybrid peer discovery using multiple mechanisms:
    - mDNS for local network discovery
    - Bootstrap nodes for initial connections
    - DHT for decentralized discovery
    - DNS seeds for fallback
    """
    
    def __init__(self, config: P2PConfig, node_id: NodeID):
        """Initialize hybrid discovery."""
        self.config = config
        self.node_id = node_id
        self.is_running = False
        
        # Discovery mechanisms
        self.zeroconf: Optional[Zeroconf] = None
        self.mdns_browser: Optional[ServiceBrowser] = None
        self.bootstrap_nodes = config.bootstrap_nodes or []
        
        # Callbacks
        self.on_peer_discovered: Optional[Callable[[Dict[str, Any]], Any]] = None
        
        # Discovered peers cache
        self.discovered_peers: Dict[str, Dict[str, Any]] = {}
        
        # Service info for mDNS
        self.service_type = "_enhanced-csp._tcp.local."
        self.service_name = f"csp-{node_id.value[:16]}.{self.service_type}"
        
    async def start(self):
        """Start all discovery mechanisms."""
        if self.is_running:
            return
            
        logger.info(f"Starting hybrid discovery for node {self.node_id}")
        self.is_running = True
        
        # Start mDNS discovery if available
        if HAS_ZEROCONF and self.config.enable_mdns:
            await self._start_mdns()
        else:
            logger.warning("mDNS discovery disabled or zeroconf not available")
        
        # Connect to bootstrap nodes
        if self.bootstrap_nodes:
            asyncio.create_task(self._connect_bootstrap_nodes())
        
        # Start DNS seed discovery if configured
        if self.config.dns_seed_domain and HAS_AIODNS:
            asyncio.create_task(self._dns_seed_discovery())
            
    async def stop(self):
        """Stop all discovery mechanisms."""
        if not self.is_running:
            return
            
        logger.info("Stopping hybrid discovery")
        self.is_running = False
        
        # Stop mDNS
        if self.mdns_browser:
            self.mdns_browser.cancel()
            self.mdns_browser = None
            
        if self.zeroconf:
            self.zeroconf.close()
            self.zeroconf = None
    
    async def find_peers(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Actively search for peers using all available mechanisms.
        
        Args:
            count: Maximum number of peers to find
            
        Returns:
            List of discovered peer information
        """
        peers = []
        
        # Return cached peers first
        cached = list(self.discovered_peers.values())[:count]
        peers.extend(cached)
        
        if len(peers) >= count:
            return peers[:count]
        
        # Try bootstrap nodes
        for bootstrap in self.bootstrap_nodes[:count - len(peers)]:
            peer_info = self._parse_multiaddr(bootstrap)
            if peer_info:
                peers.append(peer_info)
        
        return peers
    
    async def announce_peer(self, peer_info: Dict[str, Any]):
        """Announce a peer to the network."""
        # Store in cache
        peer_id = peer_info.get("node_id", "")
        if peer_id and peer_id != self.node_id.value:
            self.discovered_peers[peer_id] = peer_info
            
            # Notify callback
            if self.on_peer_discovered:
                await self.on_peer_discovered(peer_info)
    
    # mDNS Discovery
    
    async def _start_mdns(self):
        """Start mDNS discovery and advertisement."""
        try:
            self.zeroconf = Zeroconf()
            
            # Register our service
            addresses = [socket.inet_aton(self._get_local_ip())]
            
            service_info = ServiceInfo(
                self.service_type,
                self.service_name,
                addresses=addresses,
                port=self.config.listen_port,
                properties={
                    b"node_id": self.node_id.value.encode(),
                    b"version": b"1.0.0",
                    b"capabilities": b"relay,storage,compute",
                    b"protocols": b"tcp,quic"
                }
            )
            
            await asyncio.get_event_loop().run_in_executor(
                None, self.zeroconf.register_service, service_info
            )
            
            logger.info(f"Registered mDNS service: {self.service_name}")
            
            # Browse for other services
            self.mdns_browser = ServiceBrowser(
                self.zeroconf,
                self.service_type,
                self
            )
            
        except Exception as e:
            logger.error(f"Failed to start mDNS: {e}")
    
    def add_service(self, zeroconf: Zeroconf, service_type: str, name: str):
        """Called when a new mDNS service is discovered."""
        asyncio.create_task(self._handle_mdns_service(zeroconf, service_type, name))
    
    def remove_service(self, zeroconf: Zeroconf, service_type: str, name: str):
        """Called when an mDNS service is removed."""
        logger.debug(f"mDNS service removed: {name}")
    
    def update_service(self, zeroconf: Zeroconf, service_type: str, name: str):
        """Called when an mDNS service is updated."""
        asyncio.create_task(self._handle_mdns_service(zeroconf, service_type, name))
    
    async def _handle_mdns_service(self, zeroconf: Zeroconf, service_type: str, name: str):
        """Handle discovered mDNS service."""
        try:
            info = zeroconf.get_service_info(service_type, name)
            if not info:
                return
                
            properties = info.properties
            node_id_bytes = properties.get(b"node_id")
            if not node_id_bytes:
                return
                
            node_id_str = node_id_bytes.decode("utf-8")
            
            # Skip self
            if node_id_str == self.node_id.value:
                return
            
            # Parse addresses
            addresses = []
            for addr in info.addresses:
                ip = socket.inet_ntoa(addr)
                port = info.port
                
                # Create multiaddr format
                addresses.append(f"/ip4/{ip}/tcp/{port}/p2p/{node_id_str}")
                if b'quic' in properties.get(b'protocols', b''):
                    addresses.append(f"/ip4/{ip}/udp/{port}/quic/p2p/{node_id_str}")
            
            # Create peer info
            peer_info = {
                'node_id': node_id_str,
                'addresses': addresses,
                'source': 'mdns',
                'peer_type': PeerType.REGULAR.value,
                'capabilities': properties.get(b'capabilities', b'').decode('utf-8').split(','),
                'discovered_at': datetime.utcnow().isoformat()
            }
            
            logger.info(f"Discovered peer via mDNS: {node_id_str[:16]}...")
            
            # Announce the peer
            await self.announce_peer(peer_info)
            
        except Exception as e:
            logger.error(f"Error handling mDNS service {name}: {e}")
    
    # Bootstrap Discovery
    
    async def _connect_bootstrap_nodes(self):
        """Connect to configured bootstrap nodes."""
        logger.info(f"Connecting to {len(self.bootstrap_nodes)} bootstrap nodes...")
        
        tasks = []
        for bootstrap_addr in self.bootstrap_nodes:
            tasks.append(self._connect_bootstrap(bootstrap_addr))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        connected = sum(1 for r in results if r and not isinstance(r, Exception))
        logger.info(f"Successfully bootstrapped from {connected}/{len(self.bootstrap_nodes)} nodes")
    
    async def _connect_bootstrap(self, address: str) -> bool:
        """Connect to a single bootstrap node."""
        try:
            peer_info = self._parse_multiaddr(address)
            if peer_info:
                peer_info['peer_type'] = PeerType.BOOTSTRAP.value
                await self.announce_peer(peer_info)
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to connect to bootstrap node {address}: {e}")
            return False
    
    # DNS Seed Discovery
    
    async def _dns_seed_discovery(self):
        """Discover peers via DNS seeds."""
        if not HAS_AIODNS:
            logger.warning("aiodns not available, skipping DNS discovery")
            return
            
        try:
            resolver = aiodns.DNSResolver()
            
            # Query TXT records for peer addresses
            result = await resolver.query(self.config.dns_seed_domain, 'TXT')
            
            for record in result:
                try:
                    # Parse TXT record as multiaddr
                    addresses = record.text.split(',')
                    for addr in addresses:
                        peer_info = self._parse_multiaddr(addr.strip())
                        if peer_info:
                            peer_info['source'] = 'dns'
                            await self.announce_peer(peer_info)
                except Exception as e:
                    logger.error(f"Error parsing DNS record: {e}")
                    
        except Exception as e:
            logger.error(f"DNS seed discovery failed: {e}")
    
    # Utility methods
    
    def _parse_multiaddr(self, address: str) -> Optional[Dict[str, Any]]:
        """Parse a multiaddr string into peer info."""
        try:
            # Format: /ip4/1.2.3.4/tcp/4001/p2p/QmNodeID
            parts = address.strip('/').split('/')
            
            if len(parts) < 6:
                logger.error(f"Invalid multiaddr: {address}")
                return None
            
            peer_info = {
                'node_id': parts[5],  # p2p ID
                'addresses': [address],
                'source': 'bootstrap',
                'discovered_at': datetime.utcnow().isoformat()
            }
            
            return peer_info
            
        except Exception as e:
            logger.error(f"Failed to parse multiaddr {address}: {e}")
            return None
    
    def _get_local_ip(self) -> str:
        """Get local IP address."""
        try:
            # Create a socket to an external address to get local IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except:
            return "127.0.0.1"


class PeerExchange:
    """
    Peer exchange protocol for sharing peer lists.
    Helps nodes discover peers through existing connections.
    """
    
    def __init__(self, node: 'NetworkNode'):
        """Initialize peer exchange."""
        self.node = node
        self.exchange_interval = 60  # seconds
        self._exchange_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start peer exchange protocol."""
        self._exchange_task = asyncio.create_task(self._exchange_loop())
        
    async def stop(self):
        """Stop peer exchange protocol."""
        if self._exchange_task:
            self._exchange_task.cancel()
            await asyncio.gather(self._exchange_task, return_exceptions=True)
    
    async def _exchange_loop(self):
        """Periodically exchange peers with connected nodes."""
        while True:
            try:
                await asyncio.sleep(self.exchange_interval)
                
                # Select random peers to exchange with
                import random
                peers = list(self.node.peers.keys())
                if len(peers) > 3:
                    selected = random.sample(peers, 3)
                else:
                    selected = peers
                
                for peer_id in selected:
                    await self._exchange_with_peer(peer_id)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in peer exchange loop: {e}")
    
    async def _exchange_with_peer(self, peer_id: NodeID):
        """Exchange peer lists with a specific peer."""
        try:
            # Prepare our peer list (excluding the peer we're sending to)
            our_peers = []
            for pid, pinfo in list(self.node.peers.items())[:20]:
                if pid != peer_id:
                    our_peers.append({
                        'node_id': pid.value,
                        'addresses': [pinfo.address],
                        'capabilities': pinfo.capabilities.__dict__,
                        'latency_ms': pinfo.latency
                    })
            
            # Send peer exchange message
            from ..core.types import MessageType
            await self.node.send_message(
                peer_id,
                {
                    'type': 'peer_exchange',
                    'peers': our_peers
                },
                MessageType.CONTROL
            )
            
            logger.debug(f"Sent peer exchange to {peer_id}")
            
        except Exception as e:
            logger.error(f"Peer exchange failed with {peer_id}: {e}")