# enhanced_csp/network/__init__.py
"""
Enhanced CSP Network Stack
=========================

Complete P2P networking implementation with:
- Hybrid discovery (mDNS + Bootstrap + DHT)
- NAT traversal (STUN/TURN/hole-punching)
- QUIC transport with TCP fallback
- Dynamic mesh topology
- B.A.T.M.A.N.-inspired routing
- DNS overlay (.web4ai)
- Adaptive routing with ML prediction
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from pathlib import Path

from enhanced_csp.network.core import NetworkConfig, NodeID
from enhanced_csp.network.core.node import NetworkNode
from enhanced_csp.config import settings

# Heavy modules are loaded lazily inside EnhancedCSPNetwork.start()


logger = logging.getLogger(__name__)


class EnhancedCSPNetwork:
    """Main network stack interface"""
    
    def __init__(self, config: Optional[NetworkConfig] = None):
        """Initialize network stack with configuration"""
        self.config = config or NetworkConfig()

        if not self.config.p2p.bootstrap_api_url:
            self.config.p2p.bootstrap_api_url = settings.peoplesai_boot_api
        if not self.config.p2p.dns_seed_domain:
            self.config.p2p.dns_seed_domain = settings.peoplesai_boot_dns
        
        # Core components
        self.node: Optional[NetworkNode] = None
        self.discovery: Optional[HybridDiscovery] = None
        self.dht: Optional[KademliaDHT] = None
        self.nat: Optional[NATTraversal] = None
        self.transport: Optional[MultiProtocolTransport] = None
        self.topology: Optional[MeshTopologyManager] = None
        self.routing: Optional[BatmanRouting] = None
        self.dns: Optional[DNSOverlay] = None
        self.adaptive_routing: Optional[AdaptiveRoutingEngine] = None
        
        # Additional services
        self.peer_exchange: Optional[PeerExchange] = None
        
        # State
        self.is_running = False
    
    async def start(self):
        """Start the network stack"""
        if self.is_running:
            logger.warning("Network already running")
            return
        
        logger.info("Starting Enhanced CSP Network Stack...")
        
        try:
            # Import heavy modules lazily
            from .p2p.transport import MultiProtocolTransport
            from .p2p.nat import NATTraversal
            from .p2p.dht import KademliaDHT
            from .p2p.discovery import HybridDiscovery, PeerExchange
            from .mesh.topology import MeshTopologyManager
            from .mesh.routing import BatmanRouting
            from .dns.overlay import DNSOverlay
            from .routing.adaptive import AdaptiveRoutingEngine

            # Initialize node
            self.node = NetworkNode(self.config)
            await self.node.start()
            
            # Initialize transport layer
            self.transport = MultiProtocolTransport(self.node)
            await self.transport.start()
            self.node.transport = self.transport
            
            # Initialize NAT traversal
            self.nat = NATTraversal(self.config.p2p)
            nat_info = await self.nat.detect_nat(self.config.p2p.listen_port)
            self.node.nat_info = nat_info
            
            # Initialize DHT
            self.dht = KademliaDHT(self.node)
            await self.dht.start(
                listen_addr=self.config.p2p.listen_address,
                port=self.config.p2p.listen_port
            )
            self.node.dht = self.dht
            
            # Initialize discovery
            self.discovery = HybridDiscovery(self.node, self.config.p2p)
            self.discovery.on_peer_discovered = self._on_peer_discovered
            await self.discovery.start()
            self.node.discovery = self.discovery
            
            # Initialize mesh topology
            self.topology = MeshTopologyManager(self.node, self.config.mesh)
            await self.topology.start()
            self.node.topology = self.topology
            
            # Initialize routing
            self.routing = BatmanRouting(self.node, self.topology)
            await self.routing.start()
            self.node.routing = self.routing
            
            # Initialize DNS overlay
            self.dns = DNSOverlay(self.node, self.config.dns)
            await self.dns.start(self.dht)
            self.node.dns = self.dns
            
            # Initialize adaptive routing
            self.adaptive_routing = AdaptiveRoutingEngine(
                self.node, self.config.routing, self.routing
            )
            await self.adaptive_routing.start()
            self.node.adaptive_routing = self.adaptive_routing
            
            # Start peer exchange
            self.peer_exchange = PeerExchange(self.node)
            await self.peer_exchange.start()
            
            # Set connection handler
            self.transport.on_connection = self._on_connection
            
            self.is_running = True
            logger.info(f"Network started - Node ID: {self.node.node_id.to_base58()}")
            
            # Log network info
            self._log_network_info()
            
        except Exception as e:
            logger.error(f"Failed to start network: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the network stack"""
        logger.info("Stopping Enhanced CSP Network Stack...")
        
        self.is_running = False
        
        # Stop components in reverse order
        if self.peer_exchange:
            await self.peer_exchange.stop()
        
        if self.adaptive_routing:
            await self.adaptive_routing.stop()
        
        if self.dns:
            await self.dns.stop()
        
        if self.routing:
            await self.routing.stop()
        
        if self.topology:
            await self.topology.stop()
        
        if self.discovery:
            await self.discovery.stop()
        
        if self.dht:
            await self.dht.stop()
        
        if self.transport:
            await self.transport.stop()
        
        if self.node:
            await self.node.stop()
        
        logger.info("Network stopped")
    
    async def _on_peer_discovered(self, peer_info: Dict[str, Any]):
        """Handle discovered peer"""
        try:
            # Connect to peer
            addresses = peer_info.get('addresses', [])
            
            for address in addresses:
                peer_connection = await self.node.connect_to_peer(address)
                if peer_connection:
                    logger.info(f"Connected to peer: {peer_info['node_id'][:16]}...")
                    
                    # Add to mesh topology
                    await self.topology.add_peer(peer_connection)
                    break
                    
        except Exception as e:
            logger.error(f"Failed to connect to discovered peer: {e}")
    
    async def _on_connection(self, connection):
        """Handle new connection"""
        try:
            # Add peer to topology
            if connection.remote_peer:
                await self.topology.add_peer(connection.remote_peer)

            # Route incoming messages to the node
            connection.message_handler = self.node.handle_raw_message

        except Exception as e:
            logger.error(f"Error handling new connection: {e}")
    
    def _log_network_info(self):
        """Log network information"""
        info = self.get_network_info()
        
        logger.info("=" * 60)
        logger.info("Network Information:")
        logger.info(f"  Node ID: {info['node_id']}")
        logger.info(f"  NAT Type: {info['nat_type']}")
        logger.info(f"  External Address: {info['external_address']}")
        logger.info(f"  DNS Name: {info['dns_name']}")
        logger.info(f"  Transport: QUIC + TCP")
        logger.info(f"  Topology: {info['topology_type']}")
        logger.info(f"  Routing: B.A.T.M.A.N.-inspired")
        logger.info("=" * 60)
    
    def get_network_info(self) -> Dict[str, Any]:
        """Get current network information"""
        info = {
            'node_id': self.node.node_id.to_base58() if self.node else None,
            'status': 'running' if self.is_running else 'stopped'
        }
        
        if self.nat and self.nat.nat_info:
            info.update({
                'nat_type': self.nat.nat_info.nat_type.value,
                'external_address': f"{self.nat.nat_info.external_ip}:{self.nat.nat_info.external_port}",
                'internal_address': f"{self.nat.nat_info.internal_ip}:{self.nat.nat_info.internal_port}"
            })
        
        if self.node:
            info['dns_name'] = f"{self.node.node_id.to_base58()[:8]}.web4ai"
        
        if self.topology:
            topology_info = self.topology.get_network_view()
            info.update({
                'topology_type': topology_info['topology_type'],
                'peers': topology_info['nodes'] - 1,  # Exclude self
                'connections': topology_info['edges'],
                'super_peers': len(topology_info['super_peers'])
            })
        
        if self.routing:
            routing_info = self.routing.get_routing_metrics()
            info.update({
                'routes': routing_info['total_routes'],
                'avg_tq': routing_info['average_tq']
            })
        
        if self.adaptive_routing:
            adaptive_info = self.adaptive_routing.get_routing_stats()
            info.update({
                'active_flows': adaptive_info['active_flows'],
                'ml_routing': adaptive_info['ml_enabled']
            })
        
        return info
    
    async def connect_to_peer(self, address: str) -> bool:
        """Connect to a specific peer"""
        if not self.node:
            raise RuntimeError("Network not started")
        
        peer_info = await self.node.connect_to_peer(address)
        if peer_info:
            await self.topology.add_peer(peer_info)
            return True
        
        return False
    
    async def resolve_dns(self, name: str) -> Optional[str]:
        """Resolve .web4ai DNS name"""
        if not self.dns:
            raise RuntimeError("DNS not initialized")
        
        from .dns.overlay import DNSRecordType
        response = await self.dns.resolve(name, DNSRecordType.A)
        
        if response and response.records:
            return response.records[0].value
        
        return None
    
    async def send_message(self, destination: str, message: Dict[str, Any]):
        """Send message to destination (node ID or DNS name)"""
        if not self.node:
            raise RuntimeError("Network not started")
        
        # Resolve DNS if needed
        if destination.endswith('.web4ai'):
            # Resolve to node ID
            # For now, extract node ID from DNS name
            node_id_str = destination.split('.')[0]
            # TODO: Proper DNS resolution
        else:
            node_id_str = destination
        
        # Get route
        # TODO: Convert string to NodeID properly
        route = await self.adaptive_routing.get_best_route(node_id_str)
        
        if route:
            await self.node.send_message(route.next_hop, message)
        else:
            logger.error(f"No route to {destination}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive network statistics"""
        stats = {
            'node': self.node.get_stats().__dict__ if self.node else {},
            'transport': self.transport.get_stats().__dict__ if self.transport else {},
            'topology': self.topology.metrics.__dict__ if self.topology else {},
            'routing': self.routing.get_routing_metrics() if self.routing else {},
            'dns': self.dns.get_stats() if self.dns else {},
            'adaptive': self.adaptive_routing.get_routing_stats() if self.adaptive_routing else {}
        }
        
        return stats


# Convenience function for quick network creation
async def create_network(config: Optional[NetworkConfig] = None) -> EnhancedCSPNetwork:
    """Create and start a network instance"""
    network = EnhancedCSPNetwork(config)
    await network.start()
    return network

