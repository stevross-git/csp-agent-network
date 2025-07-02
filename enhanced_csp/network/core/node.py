# network/core/node.py
"""
Enhanced CSP Network Node Implementation
Complete implementation with all required functionality
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime

from .types import (
    NodeID, NodeCapabilities, PeerInfo, NetworkMessage, 
    MessageType, NetworkConfig
)
from ..p2p.transport import P2PTransport
from ..p2p.discovery import HybridDiscovery
from ..p2p.dht import KademliaDHT
from ..p2p.nat import NATTraversal
from ..mesh.topology import MeshTopologyManager
from ..mesh.routing import BatmanRouting
from ..dns.overlay import DNSOverlay
from ..routing.adaptive import AdaptiveRoutingEngine

logger = logging.getLogger(__name__)


class NetworkNode:
    """
    Core network node implementation for Enhanced CSP.
    Manages P2P communication, discovery, and routing.
    """
    
    def __init__(self, config: Optional[NetworkConfig] = None):
        """Initialize network node with configuration."""
        self.config = config or NetworkConfig()
        self.node_id = NodeID.generate()
        self.capabilities = NodeCapabilities(
            relay=True,
            storage=self.config.enable_storage,
            compute=self.config.enable_compute,
            quantum=self.config.enable_quantum,
            blockchain=self.config.enable_blockchain,
            dns=self.config.enable_dns,
            bootstrap=False
        )
        
        # Core components
        self.transport: Optional[P2PTransport] = None
        self.discovery: Optional[HybridDiscovery] = None
        self.dht: Optional[KademliaDHT] = None
        self.nat: Optional[NATTraversal] = None
        self.topology: Optional[MeshTopologyManager] = None
        self.routing: Optional[BatmanRouting] = None
        self.dns: Optional[DNSOverlay] = None
        self.adaptive_routing: Optional[AdaptiveRoutingEngine] = None
        
        # State management
        self.peers: Dict[NodeID, PeerInfo] = {}
        self.is_running = False
        self._message_handlers: Dict[MessageType, List[Callable]] = {}
        self._background_tasks: List[asyncio.Task] = []
        
        # Statistics
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
            "peers_connected": 0,
            "start_time": None
        }
        
    async def start(self) -> bool:
        """
        Start the network node and all components.
        Returns True if successful, False otherwise.
        """
        if self.is_running:
            logger.warning(f"Node {self.node_id} already running")
            return True
            
        try:
            logger.info(f"Starting network node {self.node_id}")
            self.stats["start_time"] = datetime.utcnow()
            
            # Initialize transport layer
            self.transport = P2PTransport(self.config.p2p)
            await self.transport.start()
            
            # Initialize discovery mechanism
            self.discovery = HybridDiscovery(self.config.p2p, self.node_id)
            self.discovery.on_peer_discovered = self._handle_peer_discovered
            await self.discovery.start()
            
            # Initialize DHT
            self.dht = KademliaDHT(self.node_id, self.transport)
            await self.dht.start()
            
            # Initialize NAT traversal
            self.nat = NATTraversal(self.config.p2p)
            await self.nat.start()
            
            # Initialize mesh topology
            self.topology = MeshTopologyManager(
                self.node_id,
                self.config.mesh,
                self.send_message
            )
            await self.topology.start()
            
            # Initialize routing
            self.routing = BatmanRouting(
                self.node_id,
                self.topology,
                self.send_message
            )
            await self.routing.start()
            
            # Initialize DNS overlay if enabled
            if self.config.enable_dns:
                self.dns = DNSOverlay(
                    self.node_id,
                    self.config.dns,
                    self.dht
                )
                await self.dns.start()
            
            # Initialize adaptive routing
            self.adaptive_routing = AdaptiveRoutingEngine(
                self.node_id,
                self.routing,
                self.config.routing
            )
            await self.adaptive_routing.start()
            
            # Start background tasks
            self._start_background_tasks()
            
            self.is_running = True
            logger.info(f"Network node {self.node_id} started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start network node: {e}")
            await self.stop()
            return False
    
    async def stop(self) -> bool:
        """
        Stop the network node and cleanup resources.
        Returns True if successful, False otherwise.
        """
        if not self.is_running:
            return True
            
        logger.info(f"Stopping network node {self.node_id}")
        
        try:
            # Cancel background tasks
            for task in self._background_tasks:
                task.cancel()
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()
            
            # Stop components in reverse order
            if self.adaptive_routing:
                await self.adaptive_routing.stop()
                
            if self.dns:
                await self.dns.stop()
                
            if self.routing:
                await self.routing.stop()
                
            if self.topology:
                await self.topology.stop()
                
            if self.nat:
                await self.nat.stop()
                
            if self.dht:
                await self.dht.stop()
                
            if self.discovery:
                await self.discovery.stop()
                
            if self.transport:
                await self.transport.stop()
            
            self.is_running = False
            logger.info(f"Network node {self.node_id} stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping network node: {e}")
            return False
    
    async def send_message(
        self, 
        recipient: NodeID, 
        message: Any,
        message_type: MessageType = MessageType.DATA,
        ttl: int = 64
    ) -> bool:
        """
        Send a message to another node.
        
        Args:
            recipient: Target node ID
            message: Message payload
            message_type: Type of message
            ttl: Time-to-live for routing
            
        Returns:
            True if message was sent successfully
        """
        if not self.is_running:
            logger.error("Cannot send message: node not running")
            return False
            
        try:
            # Create network message
            net_msg = NetworkMessage.create(
                msg_type=message_type,
                sender=self.node_id,
                payload=message,
                recipient=recipient,
                ttl=ttl
            )
            
            # Get peer info
            peer_info = self.peers.get(recipient)
            if not peer_info:
                # Try to discover peer
                logger.debug(f"Peer {recipient} not found, attempting discovery")
                peer_info = await self._discover_peer(recipient)
                if not peer_info:
                    logger.error(f"Failed to find peer {recipient}")
                    return False
            
            # Send via transport
            success = await self.transport.send(peer_info.address, net_msg)
            
            if success:
                self.stats["messages_sent"] += 1
                self.stats["bytes_sent"] += len(str(message))
                
            return success
            
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False
    
    async def broadcast_message(
        self,
        message: Any,
        message_type: MessageType = MessageType.DATA,
        ttl: int = 64
    ) -> int:
        """
        Broadcast a message to all known peers.
        
        Returns:
            Number of peers message was sent to
        """
        sent_count = 0
        
        for peer_id in list(self.peers.keys()):
            if await self.send_message(peer_id, message, message_type, ttl):
                sent_count += 1
                
        return sent_count
    
    def register_handler(
        self,
        message_type: MessageType,
        handler: Callable[[NetworkMessage], Any]
    ):
        """Register a handler for a specific message type."""
        if message_type not in self._message_handlers:
            self._message_handlers[message_type] = []
        self._message_handlers[message_type].append(handler)
    
    def unregister_handler(
        self,
        message_type: MessageType,
        handler: Callable[[NetworkMessage], Any]
    ):
        """Unregister a message handler."""
        if message_type in self._message_handlers:
            self._message_handlers[message_type].remove(handler)
    
    async def connect_to_peer(self, address: str) -> bool:
        """
        Manually connect to a peer at the given address.
        
        Args:
            address: Multiaddr or host:port format
            
        Returns:
            True if connection successful
        """
        try:
            success = await self.transport.connect(address)
            if success:
                self.stats["peers_connected"] += 1
            return success
        except Exception as e:
            logger.error(f"Failed to connect to peer {address}: {e}")
            return False
    
    def get_peers(self) -> List[PeerInfo]:
        """Get list of connected peers."""
        return list(self.peers.values())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get node statistics."""
        stats = self.stats.copy()
        stats["uptime"] = (
            (datetime.utcnow() - self.stats["start_time"]).total_seconds()
            if self.stats["start_time"] else 0
        )
        stats["peer_count"] = len(self.peers)
        return stats
    
    # Private methods
    
    async def _handle_peer_discovered(self, peer_info: Dict[str, Any]):
        """Handle newly discovered peer."""
        try:
            node_id = NodeID.from_string(peer_info["node_id"])
            
            # Create PeerInfo object
            peer = PeerInfo(
                id=node_id,
                address=peer_info["addresses"][0] if peer_info["addresses"] else "",
                port=0,  # Extract from address
                capabilities=NodeCapabilities(),
                last_seen=datetime.utcnow(),
                metadata=peer_info
            )
            
            # Add to peers
            self.peers[node_id] = peer
            logger.info(f"Added peer {node_id}")
            
            # Notify topology manager
            if self.topology:
                await self.topology.add_peer(peer)
                
        except Exception as e:
            logger.error(f"Error handling discovered peer: {e}")
    
    async def _discover_peer(self, peer_id: NodeID) -> Optional[PeerInfo]:
        """Attempt to discover a specific peer."""
        # Try DHT lookup
        if self.dht:
            result = await self.dht.find_node(peer_id)
            if result:
                return result
                
        # Try DNS lookup
        if self.dns:
            result = await self.dns.resolve(str(peer_id))
            if result:
                return result
                
        return None
    
    def _start_background_tasks(self):
        """Start background maintenance tasks."""
        self._background_tasks.append(
            asyncio.create_task(self._peer_maintenance_loop())
        )
        self._background_tasks.append(
            asyncio.create_task(self._stats_reporting_loop())
        )
    
    async def _peer_maintenance_loop(self):
        """Maintain peer connections."""
        while self.is_running:
            try:
                # Remove stale peers
                now = datetime.utcnow()
                stale_peers = []
                
                for peer_id, peer_info in self.peers.items():
                    if (now - peer_info.last_seen).total_seconds() > 300:  # 5 minutes
                        stale_peers.append(peer_id)
                
                for peer_id in stale_peers:
                    del self.peers[peer_id]
                    logger.debug(f"Removed stale peer {peer_id}")
                
                # Maintain minimum peers
                if len(self.peers) < self.config.p2p.min_peers:
                    await self.discovery.find_peers()
                
                await asyncio.sleep(30)  # Run every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in peer maintenance: {e}")
    
    async def _stats_reporting_loop(self):
        """Periodically log statistics."""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Log every minute
                
                stats = self.get_stats()
                logger.info(
                    f"Node stats - Peers: {stats['peer_count']}, "
                    f"Messages sent: {stats['messages_sent']}, "
                    f"Messages received: {stats['messages_received']}, "
                    f"Uptime: {stats['uptime']:.0f}s"
                )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in stats reporting: {e}")


class EnhancedCSPNetwork:
    """
    High-level network interface for Enhanced CSP.
    Manages multiple nodes and provides simplified API.
    """
    
    def __init__(self, config: Optional[NetworkConfig] = None):
        """Initialize Enhanced CSP Network."""
        self.config = config or NetworkConfig()
        self.nodes: Dict[str, NetworkNode] = {}
        
    async def create_node(self, name: str = "default") -> NetworkNode:
        """Create and start a new network node."""
        node = NetworkNode(self.config)
        
        if await node.start():
            self.nodes[name] = node
            return node
        else:
            raise RuntimeError("Failed to start network node")
    
    async def stop_node(self, name: str = "default") -> bool:
        """Stop and remove a network node."""
        if name in self.nodes:
            success = await self.nodes[name].stop()
            if success:
                del self.nodes[name]
            return success
        return False
    
    async def stop_all(self):
        """Stop all network nodes."""
        for name in list(self.nodes.keys()):
            await self.stop_node(name)
    
    def get_node(self, name: str = "default") -> Optional[NetworkNode]:
        """Get a network node by name."""
        return self.nodes.get(name)