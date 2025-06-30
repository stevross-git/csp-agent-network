# enhanced_csp/network/p2p/discovery.py
"""
Hybrid peer discovery implementation:
- mDNS/Bonjour for local network
- Bootstrap nodes for initial contact
- Kademlia DHT for global discovery
"""

import asyncio
import logging
import socket
import struct
import json
from typing import Dict, List, Optional, Set, Callable
from datetime import datetime
import ipaddress

logger = logging.getLogger(__name__)

# For mDNS we'll use zeroconf
try:
    from zeroconf import ServiceBrowser, ServiceInfo, Zeroconf
    MDNS_AVAILABLE = True
except ImportError:
    MDNS_AVAILABLE = False
    Zeroconf = object  # type: ignore
    ServiceBrowser = ServiceInfo = object  # type: ignore
    logger.warning("zeroconf not available, mDNS discovery disabled")

from enhanced_csp.network.core.node import NetworkNode
from enhanced_csp.network.core.types import NodeID, PeerInfo, PeerType, P2PConfig


logger = logging.getLogger(__name__)


class HybridDiscovery:
    """Hybrid peer discovery using mDNS, bootstrap nodes, and DHT"""
    
    SERVICE_TYPE = "_enhanced-csp._tcp.local."
    
    def __init__(self, node: NetworkNode, config: P2PConfig):
        self.node = node
        self.config = config
        self.discovered_peers: Set[str] = set()
        self._lock = asyncio.Lock()
        
        # mDNS components
        self.zeroconf: Optional[Zeroconf] = None
        self.service_info: Optional[ServiceInfo] = None
        self.browser: Optional[ServiceBrowser] = None
        
        # Callbacks
        self.on_peer_discovered: Optional[Callable] = None
    
    async def start(self):
        """Start all discovery mechanisms"""
        logger.info("Starting hybrid peer discovery...")

        # Fetch additional bootstrap nodes from external sources
        if self.config.bootstrap_api_url:
            nodes = await self._fetch_bootstrap_api()
            for n in nodes:
                if n not in self.config.bootstrap_nodes:
                    self.config.bootstrap_nodes.append(n)

        if self.config.dns_seed_domain:
            nodes = await self._fetch_dns_seed()
            for n in nodes:
                if n not in self.config.bootstrap_nodes:
                    self.config.bootstrap_nodes.append(n)

        # Start mDNS discovery for local network
        if self.config.enable_mdns and MDNS_AVAILABLE:
            await self._start_mdns()
        
        # Connect to bootstrap nodes
        await self._connect_bootstrap_nodes()
        
        # DHT discovery will be started after initial peers are connected
        logger.info("Hybrid discovery started")
    
    async def stop(self):
        """Stop all discovery mechanisms"""
        if self.zeroconf:
            await self._stop_mdns()
    
    async def _start_mdns(self):
        """Start mDNS/Bonjour discovery"""
        try:
            logger.info("Starting mDNS discovery...")
            
            # Initialize Zeroconf
            self.zeroconf = Zeroconf()
            
            # Get local addresses
            local_ips = self._get_local_ips()
            if not local_ips:
                logger.warning("No local IP addresses found for mDNS")
                return
            
            # Create service info
            node_id_str = self.node.node_id.to_base58()
            service_name = f"enhanced-csp-{node_id_str[:8]}.{self.SERVICE_TYPE}"
            
            # Service properties
            properties = {
                'node_id': node_id_str,
                'version': '1.0',
                'protocols': 'quic,tcp',
                'port': str(self.node.config.p2p.listen_port),
                'capabilities': ','.join(['mesh', 'dns', 'routing'])
            }
            
            self.service_info = ServiceInfo(
                self.SERVICE_TYPE,
                service_name,
                addresses=[socket.inet_aton(ip) for ip in local_ips],
                port=self.node.config.p2p.listen_port,
                properties=properties,
            )
            
            # Register service
            await asyncio.get_event_loop().run_in_executor(
                None, self.zeroconf.register_service, self.service_info
            )
            
            logger.info(f"Registered mDNS service: {service_name}")
            
            # Start browsing for other services
            self.browser = ServiceBrowser(
                self.zeroconf, 
                self.SERVICE_TYPE, 
                self
            )
            
        except Exception as e:
            logger.error(f"Failed to start mDNS discovery: {e}")
    
    async def _stop_mdns(self):
        """Stop mDNS discovery"""
        try:
            if self.service_info and self.zeroconf:
                await asyncio.get_event_loop().run_in_executor(
                    None, self.zeroconf.unregister_service, self.service_info
                )
            
            if self.zeroconf:
                await asyncio.get_event_loop().run_in_executor(
                    None, self.zeroconf.close
                )
            
            logger.info("Stopped mDNS discovery")
            
        except Exception as e:
            logger.error(f"Error stopping mDNS: {e}")
    
    def _get_local_ips(self) -> List[str]:
        """Get local IP addresses suitable for mDNS"""
        local_ips = []
        
        try:
            # Get all network interfaces
            for interface in socket.getaddrinfo(socket.gethostname(), None):
                ip = interface[4][0]
                ip_obj = ipaddress.ip_address(ip)
                
                # Only include private IPs suitable for mDNS
                if isinstance(ip_obj, ipaddress.IPv4Address):
                    if ip_obj.is_private and not ip_obj.is_loopback:
                        local_ips.append(str(ip_obj))
            
            # Fallback to common private subnets
            if not local_ips:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                try:
                    # This doesn't actually send anything
                    s.connect(('10.255.255.255', 1))
                    local_ips.append(s.getsockname()[0])
                except:
                    pass
                finally:
                    s.close()
            
        except Exception as e:
            logger.error(f"Error getting local IPs: {e}")
        
        return local_ips
    
    # ServiceBrowser callbacks for mDNS
    def add_service(self, zeroconf: Zeroconf, service_type: str, name: str):
        """Called when a new mDNS service is discovered"""
        asyncio.create_task(self._handle_mdns_service(zeroconf, service_type, name))
    
    def remove_service(self, zeroconf: Zeroconf, service_type: str, name: str):
        """Called when an mDNS service is removed"""
        logger.info(f"mDNS service removed: {name}")
    
    def update_service(self, zeroconf: Zeroconf, service_type: str, name: str):
        """Called when an mDNS service is updated"""
        pass
    
    async def _handle_mdns_service(self, zeroconf: Zeroconf, service_type: str, name: str):
        """Handle discovered mDNS service"""
        try:
            # Get service info
            info = await asyncio.get_event_loop().run_in_executor(
                None, zeroconf.get_service_info, service_type, name
            )
            
            if not info:
                return
            
            # Extract node information
            properties = info.properties
            node_id_str = properties.get(b'node_id', b'').decode('utf-8')
            
            if not node_id_str or node_id_str == self.node.node_id.to_base58():
                return  # Skip self
            
            # Build multiaddr
            addresses = []
            for addr_bytes in info.addresses:
                ip = socket.inet_ntoa(addr_bytes)
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
                'capabilities': properties.get(b'capabilities', b'').decode('utf-8').split(',')
            }
            
            logger.info(f"Discovered peer via mDNS: {node_id_str[:16]}...")
            
            # Notify discovery handler
            if self.on_peer_discovered:
                await self.on_peer_discovered(peer_info)
            
        except Exception as e:
            logger.error(f"Error handling mDNS service {name}: {e}")
    
    async def _connect_bootstrap_nodes(self):
        """Connect to configured bootstrap nodes"""
        logger.info(f"Connecting to {len(self.config.bootstrap_nodes)} bootstrap nodes...")
        
        tasks = []
        for bootstrap_addr in self.config.bootstrap_nodes:
            tasks.append(self._connect_bootstrap(bootstrap_addr))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        connected = sum(1 for r in results if r and not isinstance(r, Exception))
        logger.info(f"Successfully bootstrapped from {connected}/{len(self.config.bootstrap_nodes)} nodes")
    
    async def _connect_bootstrap(self, address: str) -> bool:
        """Connect to a single bootstrap node"""
        try:
            # Parse multiaddr
            # Format: /ip4/1.2.3.4/tcp/4001/p2p/QmNodeID
            parts = address.strip('/').split('/')
            
            if len(parts) < 6:
                logger.error(f"Invalid bootstrap address: {address}")
                return False
            
            peer_info = {
                'node_id': parts[5],  # p2p ID
                'addresses': [address],
                'source': 'bootstrap',
                'peer_type': PeerType.BOOTSTRAP
            }
            
            # Notify discovery handler
            if self.on_peer_discovered:
                await self.on_peer_discovered(peer_info)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to bootstrap {address}: {e}")
            return False

    async def _fetch_bootstrap_api(self) -> List[str]:
        """Retrieve bootstrap nodes from REST API"""
        try:
            import requests
            resp = await asyncio.to_thread(requests.get, self.config.bootstrap_api_url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                return [str(n) for n in data]
            if isinstance(data, dict):
                return [str(n) for n in data.get("bootstrap_nodes", [])]
        except Exception as e:
            logger.error(f"Bootstrap API fetch failed: {e}")
        return []

    async def _fetch_dns_seed(self) -> List[str]:
        """Resolve TXT records for dnsaddr seeds"""
        try:
            import dns.resolver  # type: ignore
        except Exception:
            logger.warning("dnspython not available; skipping DNS seed lookup")
            return []

        try:
            resolver = dns.resolver.Resolver()
            answers = await asyncio.get_event_loop().run_in_executor(
                None, resolver.resolve, self.config.dns_seed_domain, "TXT"
            )
            nodes: List[str] = []
            for rdata in answers:
                for txt in rdata.strings:
                    entry = txt.decode()
                    if entry.startswith("dnsaddr="):
                        nodes.append(entry.split("=", 1)[1])
            return nodes
        except Exception as e:
            logger.error(f"DNS seed lookup failed: {e}")
            return []
    
    async def find_peers_dht(self, count: int = 10) -> List[Dict[str, any]]:
        """Find random peers using DHT"""
        if not self.node.dht:
            return []
        
        try:
            logger.info(f"Searching for {count} peers via DHT...")
            
            # Generate random target IDs for random walk
            discovered = []
            
            for _ in range(count):
                # Random 32-byte target
                target = NodeID(
                    raw_id=os.urandom(32),
                    public_key=None  # Not needed for search
                )
                
                # Find closest peers to target
                peers = await self.node.dht.find_closest_peers(target, k=3)
                
                for peer in peers:
                    async with self._lock:
                        if peer['node_id'] not in self.discovered_peers:
                            self.discovered_peers.add(peer['node_id'])
                            discovered.append(peer)
                
                if len(discovered) >= count:
                    break
            
            logger.info(f"Found {len(discovered)} new peers via DHT")
            return discovered[:count]
            
        except Exception as e:
            logger.error(f"DHT peer discovery failed: {e}")
            return []
    
    async def announce_self(self):
        """Announce our presence to the network"""
        if not self.node.dht:
            return
        
        try:
            # Announce on our own node ID
            await self.node.dht.announce(
                self.node.node_id.raw_id,
                self.node.config.p2p.listen_port
            )
            
            # Also announce capabilities
            capabilities = ['mesh', 'dns', 'routing', 'quantum']
            for cap in capabilities:
                key = f"capability:{cap}".encode()
                await self.node.dht.announce(key, self.node.config.p2p.listen_port)
            
            logger.info("Announced presence to DHT")
            
        except Exception as e:
            logger.error(f"Failed to announce to DHT: {e}")
    
    async def find_peers_by_capability(self, capability: str) -> List[Dict[str, any]]:
        """Find peers with specific capability"""
        if not self.node.dht:
            return []
        
        try:
            key = f"capability:{capability}".encode()
            providers = await self.node.dht.find_providers(key)
            
            peers = []
            for provider in providers:
                if provider['node_id'] != self.node.node_id.to_base58():
                    peers.append({
                        'node_id': provider['node_id'],
                        'addresses': provider['addresses'],
                        'source': 'dht_capability',
                        'capabilities': [capability]
                    })
            
            logger.info(f"Found {len(peers)} peers with capability '{capability}'")
            return peers
            
        except Exception as e:
            logger.error(f"Failed to find peers by capability: {e}")
            return []


class PeerExchange:
    """Peer exchange protocol for discovering peers through other peers"""
    
    def __init__(self, node: NetworkNode):
        self.node = node
        self.exchange_interval = 300  # 5 minutes
        self._exchange_task = None
    
    async def start(self):
        """Start peer exchange protocol"""
        self._exchange_task = asyncio.create_task(self._exchange_loop())
        logger.info("Started peer exchange protocol")
    
    async def stop(self):
        """Stop peer exchange"""
        if self._exchange_task:
            self._exchange_task.cancel()
            await asyncio.gather(self._exchange_task, return_exceptions=True)
    
    async def _exchange_loop(self):
        """Periodically exchange peer lists"""
        await asyncio.sleep(60)  # Initial delay
        
        while True:
            try:
                await self._perform_exchange()
                await asyncio.sleep(self.exchange_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in peer exchange: {e}")
                await asyncio.sleep(60)
    
    async def _perform_exchange(self):
        """Perform peer exchange with random peers"""
        # Select random connected peers
        connected_peers = list(self.node.connections.keys())
        if len(connected_peers) < 2:
            return
        
        # Select up to 3 random peers
        import random
        selected = random.sample(connected_peers, min(3, len(connected_peers)))
        
        for peer_id in selected:
            await self._exchange_with_peer(peer_id)
    
    async def _exchange_with_peer(self, peer_id: NodeID):
        """Exchange peer lists with a specific peer"""
        try:
            # Prepare our peer list (limited selection)
            our_peers = []
            for pid, pinfo in list(self.node.peers.items())[:20]:
                if pid != peer_id:  # Don't send the peer its own info
                    our_peers.append({
                        'node_id': pid.to_base58(),
                        'addresses': pinfo.addresses,
                        'peer_type': pinfo.peer_type.value,
                        'latency_ms': pinfo.latency_ms
                    })
            
            # Send peer exchange request
            await self.node.send_message(peer_id, {
                'type': 'peer_exchange_request',
                'peers': our_peers
            })
            
            logger.debug(f"Sent peer exchange to {peer_id.to_base58()[:16]}...")
            
        except Exception as e:
            logger.error(f"Peer exchange failed with {peer_id.to_base58()[:16]}: {e}")
    
    async def handle_peer_exchange(self, from_peer: NodeID, peers: List[Dict]):
        """Handle incoming peer exchange"""
        new_peers = 0
        
        for peer_data in peers:
            try:
                node_id_str = peer_data['node_id']
                
                # Skip if already known or self
                if (node_id_str == self.node.node_id.to_base58() or
                    any(p.node_id.to_base58() == node_id_str for p in self.node.peers.values())):
                    continue
                
                # Notify discovery system
                if hasattr(self.node, 'discovery') and self.node.discovery.on_peer_discovered:
                    await self.node.discovery.on_peer_discovered({
                        'node_id': node_id_str,
                        'addresses': peer_data['addresses'],
                        'source': 'peer_exchange',
                        'peer_type': peer_data.get('peer_type', 'regular')
                    })
                
                new_peers += 1
                
            except Exception as e:
                logger.error(f"Error processing exchanged peer: {e}")
        
        if new_peers > 0:
            logger.info(f"Discovered {new_peers} new peers via exchange from {from_peer.to_base58()[:16]}")
        
        # Send our peers in response
        await self._exchange_with_peer(from_peer)