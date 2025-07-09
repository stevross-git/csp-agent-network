# network/core/node.py
"""
Enhanced CSP Network Node Implementation
Complete implementation with all required functionality
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime


from .types import (
    NodeID, NodeCapabilities, PeerInfo, NetworkMessage, 
    MessageType
)
from .config import NetworkConfig

# Import concrete implementation instead of abstract class
from ..p2p.transport import MultiProtocolTransport
from ..p2p.discovery import HybridDiscovery
from ..p2p.dht import KademliaDHT
from ..p2p.nat import NATTraversal
from ..mesh.topology import MeshTopologyManager
from ..dns.overlay import DNSOverlay
from ..routing.adaptive import AdaptiveRoutingEngine

logger = logging.getLogger(__name__)

class SimpleRoutingStub:
    """Simple routing stub to prevent import errors"""
    
    def __init__(self, node=None, topology=None):
        self.node = node
        self.topology = topology
        self.routing_table = {}
        self.is_running = False
    
    async def start(self):
        self.is_running = True
        logging.info("Simple routing stub started")
        return True
    
    async def stop(self):
        self.is_running = False
        logging.info("Simple routing stub stopped")
    
    def get_route(self, destination):
        """Get route to destination"""
        return self.routing_table.get(destination)
    
    def get_all_routes(self, destination):
        """Get all routes to destination"""
        route = self.routing_table.get(destination)
        return [route] if route else []


class NetworkNode:
    """
    Core network node implementation for Enhanced CSP.
    Manages P2P communication, discovery, and routing.
    """
    
    def __init__(self, config: Optional[NetworkConfig] = None):
        """Initialize network node with configuration."""
        self.config = config or NetworkConfig()
        self.node_id = NodeID.generate()
        self._event_handlers = {}
        self.capabilities = NodeCapabilities(
            relay=True,
            storage=self.config.enable_storage,
            compute=self.config.enable_compute,
            quantum=self.config.enable_quantum,
            blockchain=self.config.enable_blockchain,
            dns=self.config.enable_dns,
            bootstrap=False
        )
        
        # Core components - initialized as None to avoid import issues
        self.transport: Optional['MultiProtocolTransport'] = None
        self.discovery: Optional['HybridDiscovery'] = None
        self.dht: Optional['KademliaDHT'] = None
        self.nat: Optional['NATTraversal'] = None
        self.topology: Optional['MeshTopologyManager'] = None
        self.routing: Optional['SimpleRoutingStub'] = None
        self.dns: Optional['DNSOverlay'] = None
        self.adaptive_routing: Optional['AdaptiveRoutingEngine'] = None
        
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
        
        # Add metrics attribute to prevent AttributeError
        self.metrics = {
            'messages_sent': 0,
            'messages_received': 0,
            'peers_connected': 0,
            'bandwidth_in': 0,
            'bandwidth_out': 0,
            'routing_table_size': 0,
            'last_updated': time.time()
        }
        
    async def start(self) -> bool:
        """
        Start the network node and all components.
        Returns True if successful, False otherwise.
        """
        if self.is_running:
            logger.warning("Node is already running")
            return True
        
        try:
            logger.info(f"Starting network node {self.node_id}")
            
            # Initialize components with late imports
            await self._initialize_components()
            
            # Start components in order
            if self.transport and not await self._start_transport():
                return False
            
            if self.discovery and not await self._start_discovery():
                return False
            
            if self.dht and not await self._start_dht():
                return False
            
            if self.nat and not await self._start_nat():
                return False
            
            if self.topology and not await self._start_topology():
                return False
            
            if self.routing and not await self._start_routing():
                return False
            
            if self.dns and not await self._start_dns():
                return False
            
            if self.adaptive_routing and not await self._start_adaptive_routing():
                return False
            
            # Start background tasks
            self._start_background_tasks()
            
            self.is_running = True
            self.stats["start_time"] = time.time()
            
            logger.info(f"Network node {self.node_id} started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start network node: {e}")
            await self.stop()
            return False
    
    async def _initialize_components(self):
        """Initialize all components with late imports to avoid circular dependencies"""
        try:
            # Import components here to avoid circular imports
            from ..p2p.transport import MultiProtocolTransport
            from ..p2p.discovery import HybridDiscovery
            from ..p2p.dht import KademliaDHT
            from ..p2p.nat import NATTraversal
            from ..mesh.topology import MeshTopologyManager
            from ..dns.overlay import DNSOverlay
            from ..routing.adaptive import AdaptiveRoutingEngine
            
            # Initialize transport - FIXED: Use concrete MultiProtocolTransport
            self.transport = MultiProtocolTransport(self.config.p2p)
            
            # Initialize discovery - FIXED: correct parameter order
            self.discovery = HybridDiscovery(self.config.p2p, self.node_id)
            
            # Initialize DHT
            if self.config.enable_dht:
                self.dht = KademliaDHT(self.node_id, self.transport)
            
            # Initialize NAT traversal
            self.nat = NATTraversal(self.config.p2p)
            
            # Initialize mesh topology
            if self.config.enable_mesh:
                # Create a send message function for the topology manager
                async def topology_send_message(recipient: str, message: Any) -> bool:
                    """Send message function for topology manager"""
                    if self.transport:
                        return await self.transport.send(recipient, message)
                    return False
                
                self.topology = MeshTopologyManager(self.node_id, self.config.mesh, topology_send_message)
            
            # Initialize routing - FIXED: correct import path
            if getattr(self.config, 'enable_routing', True) and self.topology:
                try:
                    # Try to import BatmanRouting
                    from ..mesh.routing import BatmanRouting
                    self.routing = BatmanRouting(self, self.topology)
                    logging.info("BatmanRouting initialized successfully")
                except ImportError as e:
                    logging.warning(f"BatmanRouting not available: {e}")
                    # Fallback to simple routing stub
                    self.routing = SimpleRoutingStub(self, self.topology)
                    logging.info("Using SimpleRoutingStub as fallback")
                except Exception as e:
                    logging.error(f"Error initializing BatmanRouting: {e}")
                    # Fallback to simple routing stub
                    self.routing = SimpleRoutingStub(self, self.topology)
                    logging.info("Using SimpleRoutingStub as fallback due to error")
            
            # Initialize DNS overlay
            if self.config.enable_dns:
                self.dns = DNSOverlay(self.node_id, self.dht, self.config.dns)
            
            # Initialize adaptive routing
            if self.config.enable_adaptive_routing and self.routing:
                self.adaptive_routing = AdaptiveRoutingEngine(
                    self, 
                    self.config.routing, 
                    self.routing
                )
            
        except ImportError as e:
            logger.error(f"Failed to import required components: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    async def _start_transport(self) -> bool:
        """Start P2P transport"""
        try:
            if self.transport:
                return await self.transport.start()
            return False
        except Exception as e:
            logger.error(f"Failed to start transport: {e}")
            return False
    
    async def _start_discovery(self) -> bool:
        """Start peer discovery"""
        try:
            if self.discovery:
                await self.discovery.start()
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to start discovery: {e}")
            return False
    
    async def _start_dht(self) -> bool:
        """Start DHT"""
        try:
            if self.dht:
                await self.dht.start()
                return True
            return True  # DHT is optional
        except Exception as e:
            logger.error(f"Failed to start DHT: {e}")
            return False
    
    async def _start_nat(self) -> bool:
        """Start NAT traversal"""
        try:
            if self.nat:
                await self.nat.start()
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to start NAT traversal: {e}")
            return False
    
    async def _start_topology(self) -> bool:
        """Start mesh topology manager"""
        try:
            if self.topology:
                await self.topology.start()
                return True
            return True  # Topology is optional
        except Exception as e:
            logger.error(f"Failed to start topology manager: {e}")
            return False
    
    async def _start_routing(self) -> bool:
        """Start routing protocol"""
        try:
            if self.routing:
                return await self.routing.start()
            return True  # Routing is optional
        except Exception as e:
            logger.error(f"Failed to start routing: {e}")
            return False
    
    async def _start_dns(self) -> bool:
        """Start DNS overlay"""
        try:
            if self.dns:
                await self.dns.start()
                return True
            return True  # DNS is optional
        except Exception as e:
            logger.error(f"Failed to start DNS overlay: {e}")
            return False
    
    async def _start_adaptive_routing(self) -> bool:
        """Start adaptive routing engine"""
        try:
            if self.adaptive_routing:
                await self.adaptive_routing.start()
                return True
            return True  # Adaptive routing is optional
        except Exception as e:
            logger.error(f"Failed to start adaptive routing: {e}")
            return False
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        self._background_tasks = [
            asyncio.create_task(self._peer_maintenance_loop()),
            asyncio.create_task(self._stats_update_loop()),
            asyncio.create_task(self._health_check_loop())
        ]
    
    async def stop(self) -> bool:
        """Stop the network node and all components."""
        if not self.is_running:
            logger.warning("Node is not running")
            return True
        
        try:
            logger.info(f"Stopping network node {self.node_id}")
            
            # Cancel background tasks
            for task in self._background_tasks:
                if not task.done():
                    task.cancel()
            
            if self._background_tasks:
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
    
    async def send_message(self, message: NetworkMessage) -> bool:
        """Send a message to another node."""
        if not self.is_running or not self.transport:
            return False
        
        try:
            # Convert NetworkMessage to dict for transport
            message_dict = {
                'type': message.type.value if hasattr(message.type, 'value') else str(message.type),
                'payload': message.payload,
                'sender': str(message.sender),
                'recipient': str(message.recipient) if message.recipient else None,
                'timestamp': message.timestamp.isoformat() if hasattr(message.timestamp, 'isoformat') else str(message.timestamp)
            }
            
            # Send via transport
            success = await self.transport.send(str(message.recipient), message_dict)
            
            if success:
                self.stats["messages_sent"] += 1
                self.stats["bytes_sent"] += len(str(message_dict))
                # Update metrics
                self.metrics["messages_sent"] += 1
                self.metrics["last_updated"] = time.time()
            return success
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False
    
    async def broadcast_message(self, message: NetworkMessage) -> int:
        """Broadcast a message to all connected peers."""
        if not self.is_running or not self.transport:
            return 0
        
        try:
            count = 0
            # Convert NetworkMessage to dict
            message_dict = {
                'type': message.type.value if hasattr(message.type, 'value') else str(message.type),
                'payload': message.payload,
                'sender': str(message.sender),
                'timestamp': message.timestamp.isoformat() if hasattr(message.timestamp, 'isoformat') else str(message.timestamp)
            }
            
            # Send to all connected peers
            for peer_id in self.peers:
                if await self.transport.send(str(peer_id), message_dict):
                    count += 1
            
            if count > 0:
                self.stats["messages_sent"] += count
                self.stats["bytes_sent"] += len(str(message_dict)) * count
                # Update metrics
                self.metrics["messages_sent"] += count
                self.metrics["last_updated"] = time.time()
            
            return count
        except Exception as e:
            logger.error(f"Failed to broadcast message: {e}")
            return 0
    
    def add_message_handler(self, message_type: MessageType, handler: Callable):
        """Add a message handler for a specific message type."""
        if message_type not in self._message_handlers:
            self._message_handlers[message_type] = []
        self._message_handlers[message_type].append(handler)
        
        # Register with transport if available
        if self.transport:
            self.transport.register_handler(str(message_type), handler)
    
    def remove_message_handler(self, message_type: MessageType, handler: Callable):
        """Remove a message handler."""
        if message_type in self._message_handlers:
            try:
                self._message_handlers[message_type].remove(handler)
            except ValueError:
                pass
    
    async def _handle_message(self, message: NetworkMessage):
        """Handle an incoming message."""
        self.stats["messages_received"] += 1
        self.stats["bytes_received"] += len(str(message))
        # Update metrics
        self.metrics["messages_received"] += 1
        self.metrics["last_updated"] = time.time()
        
        # Call registered handlers
        if message.type in self._message_handlers:
            for handler in self._message_handlers[message.type]:
                try:
                    await handler(message)
                except Exception as e:
                    logger.error(f"Error in message handler: {e}")
    
    async def _peer_maintenance_loop(self):
        """Maintain peer connections and clean up stale ones."""
        while self.is_running:
            try:
                current_time = datetime.utcnow()
                stale_peers = []
                
                for peer_id, peer_info in self.peers.items():
                    # Check if peer is stale (no activity for 5 minutes)
                    if hasattr(peer_info, 'last_seen') and (current_time - peer_info.last_seen).total_seconds() > 300:
                        stale_peers.append(peer_id)
                
                # Remove stale peers
                for peer_id in stale_peers:
                    del self.peers[peer_id]
                    logger.debug(f"Removed stale peer {peer_id}")
                
                # Update peer count
                peer_count = len(self.peers)
                self.stats["peers_connected"] = peer_count
                self.metrics["peers_connected"] = peer_count
                
                # Update routing table size if routing is available
                if self.routing and hasattr(self.routing, 'routing_table'):
                    self.metrics["routing_table_size"] = len(self.routing.routing_table)
                
                await asyncio.sleep(60)  # Run every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in peer maintenance loop: {e}")
                await asyncio.sleep(60)
    
    async def _stats_update_loop(self):
        """Update and log statistics."""
        while self.is_running:
            try:
                # Log stats periodically
                if self.stats["start_time"]:
                    uptime = time.time() - self.stats["start_time"]
                    logger.info(f"Node stats - Uptime: {uptime:.0f}s, "
                              f"Messages sent: {self.stats['messages_sent']}, "
                              f"Messages received: {self.stats['messages_received']}, "
                              f"Peers: {self.stats['peers_connected']}")
                
                await asyncio.sleep(300)  # Log every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in stats update loop: {e}")
                await asyncio.sleep(300)
    
    async def _health_check_loop(self):
        """Perform health checks on all components."""
        while self.is_running:
            try:
                # Check component health
                healthy = True
                
                if self.transport and not await self._check_transport_health():
                    healthy = False
                
                if self.discovery and not await self._check_discovery_health():
                    healthy = False
                
                if not healthy:
                    logger.warning("Node health check failed")
                
                await asyncio.sleep(120)  # Check every 2 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(120)
    
    async def _check_transport_health(self) -> bool:
        """Check if transport is healthy."""
        try:
            # Check if transport is running
            return self.transport.is_running if self.transport else False
        except Exception:
            return False
    
    async def _check_discovery_health(self) -> bool:
        """Check if discovery is healthy."""
        try:
            # Check if discovery has the is_running attribute
            if self.discovery and hasattr(self.discovery, 'is_running'):
                return self.discovery.is_running
            # If no is_running attribute, assume healthy if discovery exists
            return self.discovery is not None
        except Exception:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get node statistics."""
        return self.stats.copy()
    
    def get_peer_info(self, peer_id: NodeID) -> Optional[PeerInfo]:
        """Get information about a peer."""
        return self.peers.get(peer_id)
    
    def get_all_peers(self) -> Dict[NodeID, PeerInfo]:
        """Get all peer information."""
        return self.peers.copy()
    
    def on_event(self, event_type: str, handler: Callable):
        """Register an event handler."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
    
    async def emit_event(self, event_type: str, data: Any):
        """Emit an event to registered handlers."""
        handlers = self._event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")


class EnhancedCSPNetwork:
    """
    Enhanced CSP Network - high-level interface.
    Manages multiple nodes and provides simplified API.
    """
    
    def __init__(self, config: Optional[NetworkConfig] = None):
        """Initialize Enhanced CSP Network."""
        self.config = config or NetworkConfig()
        self.nodes: Dict[str, NetworkNode] = {}
        self.node_id = NodeID.generate()
        self.is_running = False
        
        # Add metrics attribute to prevent AttributeError
        self.metrics = {
            'nodes_active': 0,
            'total_messages': 0,
            'network_health': 100.0,
            'last_updated': time.time()
        }
        
    async def start(self) -> bool:
        """Start the Enhanced CSP Network."""
        try:
            # Create and start the default node
            default_node = await self.create_node("default")
            self.is_running = True
            
            # Update metrics
            self.metrics['nodes_active'] = len(self.nodes)
            self.metrics['last_updated'] = time.time()
            
            logger.info(f"Enhanced CSP Network started with node ID: {self.node_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to start Enhanced CSP Network: {e}")
            self.is_running = False
            return False
    
    async def stop(self) -> bool:
        """Stop the Enhanced CSP Network."""
        try:
            await self.stop_all()
            self.is_running = False
            
            # Update metrics
            self.metrics['nodes_active'] = 0
            self.metrics['last_updated'] = time.time()
            
            logger.info("Enhanced CSP Network stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to stop Enhanced CSP Network: {e}")
            return False
    
    async def create_node(self, name: str = "default") -> NetworkNode:
        """Create and start a new network node."""
        node = NetworkNode(self.config)
        
        if await node.start():
            self.nodes[name] = node
            
            # Update metrics
            self.metrics['nodes_active'] = len(self.nodes)
            self.metrics['last_updated'] = time.time()
            
            return node
        else:
            raise RuntimeError("Failed to start network node")
    
    async def stop_node(self, name: str = "default") -> bool:
        """Stop and remove a network node."""
        if name in self.nodes:
            success = await self.nodes[name].stop()
            if success:
                del self.nodes[name]
                # Update metrics
                self.metrics['nodes_active'] = len(self.nodes)
                self.metrics['last_updated'] = time.time()
            return success
        return False
    
    async def stop_all(self):
        """Stop all network nodes."""
        for name in list(self.nodes.keys()):
            await self.stop_node(name)
    
    def get_node(self, name: str = "default") -> Optional[NetworkNode]:
        """Get a network node by name."""
        return self.nodes.get(name)

    async def get_metrics(self) -> Dict[str, Any]:
        """Return network metrics including per-node details."""
        # Base metrics for the overall network
        metrics_snapshot = self.metrics.copy()

        # Include metrics for each managed node
        node_metrics = {}
        for name, node in self.nodes.items():
            # Each NetworkNode stores metrics as a dictionary
            node_metrics[name] = node.metrics.copy()

        metrics_snapshot["nodes"] = node_metrics
        return metrics_snapshot