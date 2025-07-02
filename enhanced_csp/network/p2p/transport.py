# network/p2p/transport.py
"""
Multi-protocol transport layer for Enhanced CSP.
Supports TCP, QUIC, and WebSocket transports.
"""

import asyncio
import logging
from typing import Dict, Optional, Any, Callable, Tuple
from dataclasses import dataclass
import json

from ..core.types import P2PConfig, NetworkMessage
from ..protocol_optimizer import BinaryProtocol, MessageType as BinaryMessageType

logger = logging.getLogger(__name__)


@dataclass
class Connection:
    """Represents a network connection."""
    address: str
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    protocol: str = "tcp"
    
    async def close(self):
        """Close the connection."""
        self.writer.close()
        await self.writer.wait_closed()


class MultiProtocolTransport(P2PTransport):
    """
    Enhanced transport supporting multiple protocols.
    Primary implementation uses TCP with optional QUIC support.
    """
    
    def __init__(self, config: P2PConfig):
        super().__init__(config)
        self.protocol = BinaryProtocol()
        self.message_handlers: Dict[str, List[Callable]] = {}
        
    async def start(self):
        """Start transport services."""
        await super().start()
        
        # Start additional protocol servers if configured
        if self.config.enable_quic:
            logger.info("QUIC support enabled but not implemented in this version")
            
    async def send(self, address: str, message: Any) -> bool:
        """
        Send a message to a peer.
        
        Args:
            address: Peer address (host:port or multiaddr)
            message: Message to send
            
        Returns:
            True if sent successfully
        """
        try:
            # Get or create connection
            if address not in self.connections:
                if not await self.connect(address):
                    return False
                    
            conn = self.connections.get(address)
            if not conn:
                return False
            
            # Determine message type
            msg_type = BinaryMessageType.DATA
            if isinstance(message, dict):
                if message.get('type') == 'ping':
                    msg_type = BinaryMessageType.PING
                elif message.get('type') == 'pong':
                    msg_type = BinaryMessageType.PONG
                elif message.get('type') in ['control', 'peer_exchange']:
                    msg_type = BinaryMessageType.CONTROL
            
            # Encode message
            encoded = self.protocol.encode_message(message, msg_type)
            
            # Send over connection
            conn.writer.write(encoded)
            await conn.writer.drain()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message to {address}: {e}")
            # Remove failed connection
            if address in self.connections:
                await self.connections[address].close()
                del self.connections[address]
            return False
    
    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle incoming connection."""
        addr = writer.get_extra_info('peername')
        address = f"{addr[0]}:{addr[1]}" if addr else "unknown"
        
        logger.info(f"New connection from {address}")
        
        # Store connection
        conn = Connection(address, reader, writer)
        self.connections[address] = conn
        
        try:
            # Read messages
            while True:
                # Read header first
                header_data = await reader.read(self.protocol.HEADER_SIZE)
                if not header_data:
                    break
                    
                # Decode header to get message length
                try:
                    _, _, _, length = self.protocol.decode_header_only(header_data)
                except Exception as e:
                    logger.error(f"Invalid header from {address}: {e}")
                    break
                
                # Read full message
                payload_data = await reader.read(length)
                if len(payload_data) < length:
                    logger.error(f"Incomplete message from {address}")
                    break
                
                # Decode message
                try:
                    message, msg_type, flags = self.protocol.decode_message(
                        header_data + payload_data
                    )
                    
                    # Dispatch to handlers
                    await self._dispatch_message(address, message, msg_type)
                    
                except Exception as e:
                    logger.error(f"Failed to decode message from {address}: {e}")
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Connection error with {address}: {e}")
        finally:
            # Clean up connection
            await conn.close()
            if address in self.connections:
                del self.connections[address]
            logger.info(f"Connection closed: {address}")
    
    def register_handler(self, message_type: str, handler: Callable):
        """Register a message handler."""
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        self.message_handlers[message_type].append(handler)
    
    async def _dispatch_message(self, sender: str, message: Any, msg_type: BinaryMessageType):
        """Dispatch message to registered handlers."""
        # Map binary message type to handler key
        handler_key = msg_type.name
        
        if handler_key in self.message_handlers:
            for handler in self.message_handlers[handler_key]:
                try:
                    await handler(sender, message)
                except Exception as e:
                    logger.error(f"Handler error for {handler_key}: {e}")


# network/routing/adaptive.py
"""
Adaptive routing engine with ML-based path prediction.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import statistics

from ..core.types import NodeID, RoutingConfig
from ..mesh.routing import BatmanRouting

logger = logging.getLogger(__name__)


@dataclass
class PathMetrics:
    """Metrics for a network path."""
    latency_ms: float
    packet_loss: float
    bandwidth_mbps: float
    jitter_ms: float
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def score(self) -> float:
        """Calculate path quality score (0-100)."""
        # Lower latency is better
        latency_score = max(0, 100 - (self.latency_ms / 10))
        
        # Lower packet loss is better
        loss_score = max(0, 100 - (self.packet_loss * 100))
        
        # Higher bandwidth is better
        bandwidth_score = min(100, self.bandwidth_mbps)
        
        # Lower jitter is better
        jitter_score = max(0, 100 - (self.jitter_ms / 5))
        
        # Weighted average
        return (
            latency_score * 0.4 +
            loss_score * 0.3 +
            bandwidth_score * 0.2 +
            jitter_score * 0.1
        )


class AdaptiveRoutingEngine:
    """
    Adaptive routing with machine learning predictions.
    Optimizes paths based on real-time network conditions.
    """
    
    def __init__(self, node_id: NodeID, base_routing: BatmanRouting, config: RoutingConfig):
        """Initialize adaptive routing engine."""
        self.node_id = node_id
        self.base_routing = base_routing
        self.config = config
        
        # Path metrics cache
        self.path_metrics: Dict[Tuple[NodeID, NodeID], List[PathMetrics]] = {}
        
        # ML predictor (simplified for now)
        self.enable_ml = config.enable_ml_predictor
        
        # Background tasks
        self._monitor_task: Optional[asyncio.Task] = None
        self._optimization_task: Optional[asyncio.Task] = None
        
        self.is_running = False
        
    async def start(self):
        """Start adaptive routing engine."""
        if self.is_running:
            return
            
        logger.info("Starting adaptive routing engine")
        self.is_running = True
        
        # Start background tasks
        self._monitor_task = asyncio.create_task(self._monitor_paths())
        self._optimization_task = asyncio.create_task(self._optimize_routes())
        
    async def stop(self):
        """Stop adaptive routing engine."""
        if not self.is_running:
            return
            
        self.is_running = False
        
        # Cancel tasks
        for task in [self._monitor_task, self._optimization_task]:
            if task:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
    
    async def get_best_path(self, destination: NodeID) -> Optional[List[NodeID]]:
        """
        Get the best path to a destination.
        
        Args:
            destination: Target node ID
            
        Returns:
            List of node IDs representing the path, or None if no path exists
        """
        # Get base routing path
        base_path = await self.base_routing.get_path(destination)
        
        if not base_path or not self.config.enable_multipath:
            return base_path
        
        # Check for alternative paths
        alternatives = await self._find_alternative_paths(destination)
        
        if not alternatives:
            return base_path
        
        # Score all paths
        best_path = base_path
        best_score = await self._score_path(base_path)
        
        for path in alternatives:
            score = await self._score_path(path)
            if score > best_score:
                best_score = score
                best_path = path
        
        return best_path
    
    async def report_path_metrics(self, path: List[NodeID], metrics: PathMetrics):
        """Report observed metrics for a path."""
        if len(path) < 2:
            return
            
        # Store metrics for each hop
        for i in range(len(path) - 1):
            hop = (path[i], path[i + 1])
            
            if hop not in self.path_metrics:
                self.path_metrics[hop] = []
            
            self.path_metrics[hop].append(metrics)
            
            # Keep only recent metrics (last 100)
            if len(self.path_metrics[hop]) > 100:
                self.path_metrics[hop] = self.path_metrics[hop][-100:]
    
    async def _score_path(self, path: List[NodeID]) -> float:
        """Calculate quality score for a path."""
        if len(path) < 2:
            return 0.0
        
        total_score = 0.0
        hop_count = 0
        
        for i in range(len(path) - 1):
            hop = (path[i], path[i + 1])
            
            if hop in self.path_metrics and self.path_metrics[hop]:
                # Use recent metrics
                recent_metrics = self.path_metrics[hop][-10:]
                
                # Calculate average metrics
                avg_latency = statistics.mean(m.latency_ms for m in recent_metrics)
                avg_loss = statistics.mean(m.packet_loss for m in recent_metrics)
                avg_bandwidth = statistics.mean(m.bandwidth_mbps for m in recent_metrics)
                avg_jitter = statistics.mean(m.jitter_ms for m in recent_metrics)
                
                hop_metrics = PathMetrics(
                    latency_ms=avg_latency,
                    packet_loss=avg_loss,
                    bandwidth_mbps=avg_bandwidth,
                    jitter_ms=avg_jitter
                )
                
                total_score += hop_metrics.score()
                hop_count += 1
            else:
                # No metrics available, use default score
                total_score += 50.0
                hop_count += 1
        
        return total_score / hop_count if hop_count > 0 else 0.0
    
    async def _find_alternative_paths(self, destination: NodeID) -> List[List[NodeID]]:
        """Find alternative paths to a destination."""
        # For now, just get paths from base routing
        # In a full implementation, this would use k-shortest paths algorithm
        return await self.base_routing.get_alternative_paths(
            destination, 
            self.config.max_paths_per_destination
        )
    
    async def _predict_path_quality(self, path: List[NodeID], future_time: datetime) -> float:
        """
        Use ML to predict future path quality.
        Simplified implementation - real version would use trained model.
        """
        if not self.enable_ml:
            return await self._score_path(path)
        
        # Simple time-based prediction
        current_score = await self._score_path(path)
        
        # Degrade score based on time (paths tend to degrade)
        time_diff = (future_time - datetime.utcnow()).total_seconds()
        degradation = min(time_diff / 3600, 0.5)  # Max 50% degradation per hour
        
        return current_score * (1 - degradation)
    
    async def _monitor_paths(self):
        """Monitor path quality in background."""
        while self.is_running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Get active destinations from base routing
                destinations = await self.base_routing.get_active_destinations()
                
                for dest in destinations:
                    # Probe path quality
                    path = await self.get_best_path(dest)
                    if path:
                        # In real implementation, would send probe packets
                        # For now, simulate metrics
                        metrics = PathMetrics(
                            latency_ms=20.0,
                            packet_loss=0.01,
                            bandwidth_mbps=100.0,
                            jitter_ms=2.0
                        )
                        await self.report_path_metrics(path, metrics)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in path monitoring: {e}")
    
    async def _optimize_routes(self):
        """Periodically optimize routing tables."""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Optimize every minute
                
                # Clean old metrics
                now = datetime.utcnow()
                for hop, metrics_list in list(self.path_metrics.items()):
                    # Remove metrics older than 5 minutes
                    self.path_metrics[hop] = [
                        m for m in metrics_list
                        if (now - m.last_updated).total_seconds() < 300
                    ]
                    
                    if not self.path_metrics[hop]:
                        del self.path_metrics[hop]
                
                # Update routing preferences based on current metrics
                if self.config.enable_multipath:
                    await self._update_multipath_weights()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in route optimization: {e}")
    
    async def _update_multipath_weights(self):
        """Update multipath routing weights based on path quality."""
        destinations = await self.base_routing.get_active_destinations()
        
        for dest in destinations:
            paths = await self._find_alternative_paths(dest)
            if len(paths) <= 1:
                continue
            
            # Calculate weights based on path scores
            path_weights = []
            for path in paths:
                score = await self._score_path(path)
                path_weights.append((path, score))
            
            # Normalize weights
            total_score = sum(w for _, w in path_weights)
            if total_score > 0:
                normalized_weights = [
                    (path, score / total_score)
                    for path, score in path_weights
                ]
                
                # Update routing table with weights
                await self.base_routing.set_multipath_weights(dest, normalized_weights)


# network/mesh/topology.py
"""
Mesh network topology management.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Set, Callable, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from ..core.types import NodeID, PeerInfo, MeshConfig

logger = logging.getLogger(__name__)


class TopologyType(Enum):
    """Types of mesh topologies."""
    FULL_MESH = "full_mesh"
    PARTIAL_MESH = "partial_mesh"
    DYNAMIC_PARTIAL = "dynamic_partial"
    HIERARCHICAL = "hierarchical"


@dataclass
class MeshLink:
    """Represents a link in the mesh network."""
    local_node: NodeID
    remote_node: NodeID
    quality: float  # 0.0 to 1.0
    latency_ms: float
    established: datetime
    last_active: datetime
    
    def is_active(self) -> bool:
        """Check if link is active."""
        age = (datetime.utcnow() - self.last_active).total_seconds()
        return age < 60  # Active if used in last minute


class MeshTopologyManager:
    """
    Manages mesh network topology.
    Handles peer connections and topology optimization.
    """
    
    def __init__(self, node_id: NodeID, config: MeshConfig, send_message_fn: Callable):
        """Initialize topology manager."""
        self.node_id = node_id
        self.config = config
        self.send_message = send_message_fn
        
        # Topology state
        self.peers: Dict[NodeID, PeerInfo] = {}
        self.mesh_links: Dict[Tuple[NodeID, NodeID], MeshLink] = {}
        self.super_peers: Set[NodeID] = set()
        
        # Topology type
        self.topology_type = TopologyType(config.topology_type)
        
        # Background tasks
        self._maintenance_task: Optional[asyncio.Task] = None
        
        self.is_running = False
        
    async def start(self):
        """Start topology manager."""
        if self.is_running:
            return
            
        logger.info(f"Starting mesh topology manager with {self.topology_type.value}")
        self.is_running = True
        
        # Start maintenance task
        self._maintenance_task = asyncio.create_task(self._topology_maintenance())
        
    async def stop(self):
        """Stop topology manager."""
        if not self.is_running:
            return
            
        self.is_running = False
        
        if self._maintenance_task:
            self._maintenance_task.cancel()
            await asyncio.gather(self._maintenance_task, return_exceptions=True)
    
    async def add_peer(self, peer_info: PeerInfo) -> bool:
        """
        Add a peer to the mesh network.
        
        Args:
            peer_info: Information about the peer
            
        Returns:
            True if peer was added successfully
        """
        if peer_info.id == self.node_id:
            return False
        
        # Check if we should accept this peer
        if not await self._should_accept_peer(peer_info):
            logger.debug(f"Rejecting peer {peer_info.id} based on topology rules")
            return False
        
        # Add to peers
        self.peers[peer_info.id] = peer_info
        
        # Create mesh link
        link = MeshLink(
            local_node=self.node_id,
            remote_node=peer_info.id,
            quality=1.0,
            latency_ms=0.0,
            established=datetime.utcnow(),
            last_active=datetime.utcnow()
        )
        
        self.mesh_links[(self.node_id, peer_info.id)] = link
        
        # Check if peer is a super peer
        if self.config.enable_super_peers and self._is_super_peer(peer_info):
            self.super_peers.add(peer_info.id)
            logger.info(f"Added super peer {peer_info.id}")
        
        logger.info(f"Added peer {peer_info.id} to mesh network")
        return True
    
    async def remove_peer(self, peer_id: NodeID):
        """Remove a peer from the mesh network."""
        if peer_id in self.peers:
            del self.peers[peer_id]
            
        # Remove mesh links
        links_to_remove = [
            key for key in self.mesh_links.keys()
            if peer_id in key
        ]
        
        for key in links_to_remove:
            del self.mesh_links[key]
        
        # Remove from super peers
        self.super_peers.discard(peer_id)
        
        logger.info(f"Removed peer {peer_id} from mesh network")
    
    def get_mesh_neighbors(self) -> List[NodeID]:
        """Get direct mesh neighbors."""
        neighbors = []
        
        for (local, remote), link in self.mesh_links.items():
            if local == self.node_id and link.is_active():
                neighbors.append(remote)
        
        return neighbors
    
    async def update_link_quality(self, peer_id: NodeID, quality: float, latency_ms: float):
        """Update quality metrics for a mesh link."""
        key = (self.node_id, peer_id)
        if key in self.mesh_links:
            link = self.mesh_links[key]
            link.quality = quality
            link.latency_ms = latency_ms
            link.last_active = datetime.utcnow()
    
    async def _should_accept_peer(self, peer_info: PeerInfo) -> bool:
        """Determine if we should accept a peer based on topology rules."""
        current_peer_count = len(self.peers)
        
        if self.topology_type == TopologyType.FULL_MESH:
            # Accept all peers in full mesh
            return True
            
        elif self.topology_type == TopologyType.PARTIAL_MESH:
            # Accept up to max_peers
            return current_peer_count < self.config.max_peers
            
        elif self.topology_type == TopologyType.DYNAMIC_PARTIAL:
            # Accept based on peer quality and current load
            if current_peer_count >= self.config.max_peers:
                # Replace lowest quality peer if new peer is better
                lowest_quality_peer = self._find_lowest_quality_peer()
                if lowest_quality_peer:
                    # For now, accept if we have room
                    return True
                return False
            return True
            
        elif self.topology_type == TopologyType.HIERARCHICAL:
            # Accept based on hierarchy rules
            if self._is_super_peer_candidate():
                # Super peers accept more connections
                return current_peer_count < self.config.max_peers * 2
            else:
                # Regular nodes prefer super peers
                return (peer_info.id in self.super_peers or 
                        current_peer_count < self.config.max_peers // 2)
        
        return True
    
    def _is_super_peer(self, peer_info: PeerInfo) -> bool:
        """Check if a peer qualifies as a super peer."""
        # Check capabilities
        if not peer_info.capabilities.relay:
            return False
        
        # Check capacity (simplified - would check actual bandwidth/resources)
        return True
    
    def _is_super_peer_candidate(self) -> bool:
        """Check if this node can be a super peer."""
        # Simplified check - in reality would measure resources
        return len(self.peers) > 10
    
    def _find_lowest_quality_peer(self) -> Optional[NodeID]:
        """Find the peer with lowest link quality."""
        lowest_quality = 1.0
        lowest_peer = None
        
        for (local, remote), link in self.mesh_links.items():
            if local == self.node_id and link.quality < lowest_quality:
                lowest_quality = link.quality
                lowest_peer = remote
        
        return lowest_peer
    
    async def _topology_maintenance(self):
        """Periodic topology maintenance."""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.routing_update_interval)
                
                # Remove inactive links
                now = datetime.utcnow()
                inactive_peers = []
                
                for peer_id, peer_info in self.peers.items():
                    key = (self.node_id, peer_id)
                    if key in self.mesh_links:
                        link = self.mesh_links[key]
                        if not link.is_active():
                            inactive_peers.append(peer_id)
                
                for peer_id in inactive_peers:
                    await self.remove_peer(peer_id)
                
                # Optimize topology if dynamic
                if self.topology_type == TopologyType.DYNAMIC_PARTIAL:
                    await self._optimize_topology()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in topology maintenance: {e}")
    
    async def _optimize_topology(self):
        """Optimize mesh topology based on current conditions."""
        # Simple optimization - prefer high quality links
        if len(self.peers) > self.config.max_peers:
            # Sort peers by link quality
            peer_qualities = []
            
            for peer_id in self.peers:
                key = (self.node_id, peer_id)
                if key in self.mesh_links:
                    link = self.mesh_links[key]
                    peer_qualities.append((peer_id, link.quality))
            
            # Sort by quality
            peer_qualities.sort(key=lambda x: x[1], reverse=True)
            
            # Keep only top peers
            peers_to_keep = set(pid for pid, _ in peer_qualities[:self.config.max_peers])
            peers_to_remove = set(self.peers.keys()) - peers_to_keep
            
            for peer_id in peers_to_remove:
                await self.remove_peer(peer_id)


# network/mesh/routing.py
"""
B.A.T.M.A.N.-inspired mesh routing implementation.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime

from ..core.types import NodeID, MessageType

logger = logging.getLogger(__name__)


@dataclass
class RoutingEntry:
    """Entry in the routing table."""
    destination: NodeID
    next_hop: NodeID
    metric: float  # Lower is better
    sequence_number: int
    last_updated: datetime
    path: List[NodeID] = field(default_factory=list)
    
    def is_stale(self) -> bool:
        """Check if routing entry is stale."""
        age = (datetime.utcnow() - self.last_updated).total_seconds()
        return age > 300  # 5 minutes


class BatmanRouting:
    """
    Better Approach To Mobile Ad-hoc Networking (B.A.T.M.A.N.) inspired routing.
    Implements a proactive routing protocol for mesh networks.
    """
    
    def __init__(self, node_id: NodeID, topology_manager: 'MeshTopologyManager', send_message_fn: Callable):
        """Initialize BATMAN routing."""
        self.node_id = node_id
        self.topology = topology_manager
        self.send_message = send_message_fn
        
        # Routing table
        self.routing_table: Dict[NodeID, RoutingEntry] = {}
        
        # Sequence number for originator messages
        self.sequence_number = 0
        
        # Multipath routing support
        self.alternative_routes: Dict[NodeID, List[RoutingEntry]] = {}
        self.multipath_weights: Dict[NodeID, Dict[Tuple[NodeID, ...], float]] = {}
        
        # Background tasks
        self._ogm_task: Optional[asyncio.Task] = None
        self._maintenance_task: Optional[asyncio.Task] = None
        
        self.is_running = False
        
    async def start(self):
        """Start routing protocol."""
        if self.is_running:
            return
            
        logger.info("Starting BATMAN routing protocol")
        self.is_running = True
        
        # Start sending Originator Messages (OGMs)
        self._ogm_task = asyncio.create_task(self._send_ogm_loop())
        self._maintenance_task = asyncio.create_task(self._routing_maintenance())
        
    async def stop(self):
        """Stop routing protocol."""
        if not self.is_running:
            return
            
        self.is_running = False
        
        for task in [self._ogm_task, self._maintenance_task]:
            if task:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
    
    async def get_path(self, destination: NodeID) -> Optional[List[NodeID]]:
        """
        Get path to destination.
        
        Args:
            destination: Target node ID
            
        Returns:
            List of node IDs representing the path, or None if no route exists
        """
        if destination == self.node_id:
            return [self.node_id]
        
        if destination in self.routing_table:
            entry = self.routing_table[destination]
            if not entry.is_stale():
                return entry.path
        
        return None
    
    async def get_next_hop(self, destination: NodeID) -> Optional[NodeID]:
        """Get next hop for a destination."""
        if destination in self.routing_table:
            entry = self.routing_table[destination]
            if not entry.is_stale():
                return entry.next_hop
        
        return None
    
    async def get_alternative_paths(self, destination: NodeID, max_paths: int = 3) -> List[List[NodeID]]:
        """Get alternative paths to a destination."""
        if destination not in self.alternative_routes:
            return []
        
        paths = []
        for entry in self.alternative_routes[destination][:max_paths]:
            if not entry.is_stale():
                paths.append(entry.path)
        
        return paths
    
    async def set_multipath_weights(self, destination: NodeID, weights: List[Tuple[List[NodeID], float]]):
        """Set weights for multipath routing."""
        self.multipath_weights[destination] = {
            tuple(path): weight
            for path, weight in weights
        }
    
    async def get_active_destinations(self) -> List[NodeID]:
        """Get list of destinations with active routes."""
        active = []
        
        for dest, entry in self.routing_table.items():
            if not entry.is_stale():
                active.append(dest)
        
        return active
    
    async def handle_routing_message(self, sender: NodeID, message: Dict):
        """Handle incoming routing protocol message."""
        msg_type = message.get('type')
        
        if msg_type == 'ogm':
            await self._handle_ogm(sender, message)
        elif msg_type == 'route_error':
            await self._handle_route_error(sender, message)
    
    async def _handle_ogm(self, sender: NodeID, ogm: Dict):
        """Handle Originator Message."""
        originator = NodeID(ogm['originator'])
        sequence = ogm['sequence']
        metric = ogm['metric']
        path = [NodeID(n) for n in ogm.get('path', [])]
        
        # Add sender to path
        path.append(sender)
        
        # Calculate new metric (add link cost)
        link_cost = 1.0  # In real implementation, based on link quality
        new_metric = metric + link_cost
        
        # Check if this is a better route
        should_update = False
        
        if originator not in self.routing_table:
            should_update = True
        else:
            current_entry = self.routing_table[originator]
            
            # Update if newer sequence number or better metric
            if sequence > current_entry.sequence_number:
                should_update = True
            elif sequence == current_entry.sequence_number and new_metric < current_entry.metric:
                should_update = True
        
        if should_update:
            # Update routing table
            entry = RoutingEntry(
                destination=originator,
                next_hop=sender,
                metric=new_metric,
                sequence_number=sequence,
                last_updated=datetime.utcnow(),
                path=[self.node_id] + path + [originator]
            )
            
            self.routing_table[originator] = entry
            
            # Store as alternative route
            if originator not in self.alternative_routes:
                self.alternative_routes[originator] = []
            
            # Keep sorted by metric
            self.alternative_routes[originator].append(entry)
            self.alternative_routes[originator].sort(key=lambda e: e.metric)
            self.alternative_routes[originator] = self.alternative_routes[originator][:5]  # Keep top 5
            
            # Forward OGM to neighbors (except sender)
            await self._forward_ogm(ogm, sender, new_metric, path)
    
    async def _forward_ogm(self, ogm: Dict, exclude: NodeID, metric: float, path: List[NodeID]):
        """Forward OGM to neighbors."""
        neighbors = self.topology.get_mesh_neighbors()
        
        forward_ogm = {
            'type': 'ogm',
            'originator': ogm['originator'],
            'sequence': ogm['sequence'],
            'metric': metric,
            'path': [n.value for n in path]
        }
        
        for neighbor in neighbors:
            if neighbor != exclude and neighbor not in path:  # Avoid loops
                await self.send_message(neighbor, forward_ogm, MessageType.CONTROL)
    
    async def _handle_route_error(self, sender: NodeID, error: Dict):
        """Handle route error message."""
        destination = NodeID(error['destination'])
        
        # Remove failed route
        if destination in self.routing_table:
            entry = self.routing_table[destination]
            if entry.next_hop == sender:
                del self.routing_table[destination]
                logger.info(f"Removed route to {destination} due to error")
                
                # Try alternative route
                if destination in self.alternative_routes:
                    for alt_entry in self.alternative_routes[destination]:
                        if alt_entry.next_hop != sender and not alt_entry.is_stale():
                            self.routing_table[destination] = alt_entry
                            logger.info(f"Switched to alternative route for {destination}")
                            break
    
    async def _send_ogm_loop(self):
        """Periodically send Originator Messages."""
        while self.is_running:
            try:
                # Increment sequence number
                self.sequence_number += 1
                
                # Create OGM
                ogm = {
                    'type': 'ogm',
                    'originator': self.node_id.value,
                    'sequence': self.sequence_number,
                    'metric': 0.0,  # Initial metric
                    'path': []
                }
                
                # Send to all neighbors
                neighbors = self.topology.get_mesh_neighbors()
                for neighbor in neighbors:
                    await self.send_message(neighbor, ogm, MessageType.CONTROL)
                
                # Wait for next interval (1-2 seconds with jitter)
                import random
                interval = 1.0 + random.random()
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in OGM loop: {e}")
    
    async def _routing_maintenance(self):
        """Maintain routing table."""
        while self.is_running:
            try:
                await asyncio.sleep(30)  # Run every 30 seconds
                
                # Remove stale entries
                stale_destinations = []
                
                for dest, entry in self.routing_table.items():
                    if entry.is_stale():
                        stale_destinations.append(dest)
                
                for dest in stale_destinations:
                    del self.routing_table[dest]
                    logger.debug(f"Removed stale route to {dest}")
                
                # Clean alternative routes
                for dest in list(self.alternative_routes.keys()):
                    self.alternative_routes[dest] = [
                        e for e in self.alternative_routes[dest]
                        if not e.is_stale()
                    ]
                    
                    if not self.alternative_routes[dest]:
                        del self.alternative_routes[dest]
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in routing maintenance: {e}")


# network/dns/overlay.py
"""
DNS overlay for the mesh network.
Provides .web4ai domain resolution.
"""

import asyncio
import logging
from typing import Dict, Optional, List, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from ..core.types import NodeID, DNSConfig
from ..p2p.dht import KademliaDHT

logger = logging.getLogger(__name__)


class DNSRecordType(Enum):
    """DNS record types."""
    A = "A"  # Node address
    TXT = "TXT"  # Text data
    SRV = "SRV"  # Service
    CNAME = "CNAME"  # Alias


@dataclass
class DNSRecord:
    """DNS record in the overlay."""
    name: str
    record_type: DNSRecordType
    value: Any
    ttl: int
    created: datetime
    signature: Optional[bytes] = None
    
    def is_expired(self) -> bool:
        """Check if record has expired."""
        age = (datetime.utcnow() - self.created).total_seconds()
        return age > self.ttl


class DNSOverlay:
    """
    DNS overlay network for decentralized name resolution.
    Maps .web4ai names to node IDs and services.
    """
    
    def __init__(self, node_id: NodeID, config: DNSConfig, dht: KademliaDHT):
        """Initialize DNS overlay."""
        self.node_id = node_id
        self.config = config
        self.dht = dht
        
        # Local DNS cache
        self.cache: Dict[str, DNSRecord] = {}
        
        # Registered names for this node
        self.registered_names: Set[str] = set()
        
        # Background tasks
        self._maintenance_task: Optional[asyncio.Task] = None
        
        self.is_running = False
        
    async def start(self):
        """Start DNS overlay."""
        if self.is_running:
            return
            
        logger.info(f"Starting DNS overlay with root domain {self.config.root_domain}")
        self.is_running = True
        
        # Start maintenance
        self._maintenance_task = asyncio.create_task(self._dns_maintenance())
        
    async def stop(self):
        """Stop DNS overlay."""
        if not self.is_running:
            return
            
        self.is_running = False
        
        if self._maintenance_task:
            self._maintenance_task.cancel()
            await asyncio.gather(self._maintenance_task, return_exceptions=True)
    
    async def register(self, name: str, record_type: DNSRecordType = DNSRecordType.A, 
                      value: Optional[Any] = None, ttl: Optional[int] = None) -> bool:
        """
        Register a DNS name.
        
        Args:
            name: DNS name (e.g., "mynode.web4ai")
            record_type: Type of DNS record
            value: Record value (defaults to node ID for A records)
            ttl: Time-to-live in seconds
            
        Returns:
            True if registered successfully
        """
        if not name.endswith(self.config.root_domain):
            name = f"{name}{self.config.root_domain}"
        
        # Default values
        if value is None and record_type == DNSRecordType.A:
            value = self.node_id.value
        
        if ttl is None:
            ttl = self.config.default_ttl
        
        # Create DNS record
        record = DNSRecord(
            name=name,
            record_type=record_type,
            value=value,
            ttl=ttl,
            created=datetime.utcnow()
        )
        
        # Sign record if DNSSEC enabled
        if self.config.enable_dnssec:
            record.signature = self._sign_record(record)
        
        # Store in DHT
        key = f"dns:{name}:{record_type.value}"
        success = await self.dht.store(key, {
            'name': record.name,
            'type': record.record_type.value,
            'value': record.value,
            'ttl': record.ttl,
            'created': record.created.isoformat(),
            'signature': record.signature.hex() if record.signature else None,
            'owner': self.node_id.value
        }, ttl=ttl)
        
        if success:
            self.registered_names.add(name)
            self.cache[key] = record
            logger.info(f"Registered DNS name: {name}")
        
        return success
    
    async def resolve(self, name: str, record_type: DNSRecordType = DNSRecordType.A) -> Optional[Any]:
        """
        Resolve a DNS name.
        
        Args:
            name: DNS name to resolve
            record_type: Type of record to retrieve
            
        Returns:
            Record value if found, None otherwise
        """
        if not name.endswith(self.config.root_domain):
            name = f"{name}{self.config.root_domain}"
        
        key = f"dns:{name}:{record_type.value}"
        
        # Check cache first
        if key in self.cache:
            record = self.cache[key]
            if not record.is_expired():
                return record.value
            else:
                del self.cache[key]
        
        # Query DHT
        result = await self.dht.get(key)
        
        if result:
            # Verify DNSSEC signature if enabled
            if self.config.enable_dnssec and result.get('signature'):
                if not self._verify_signature(result):
                    logger.warning(f"Invalid DNSSEC signature for {name}")
                    return None
            
            # Create record from result
            record = DNSRecord(
                name=result['name'],
                record_type=DNSRecordType(result['type']),
                value=result['value'],
                ttl=result['ttl'],
                created=datetime.fromisoformat(result['created']),
                signature=bytes.fromhex(result['signature']) if result.get('signature') else None
            )
            
            # Cache the record
            self.cache[key] = record
            
            return record.value
        
        return None
    
    async def list_services(self, service_type: str) -> List[str]:
        """
        List all nodes providing a specific service.
        
        Args:
            service_type: Type of service (e.g., "storage", "compute")
            
        Returns:
            List of node names providing the service
        """
        # Query for SRV records
        service_name = f"_{service_type}._tcp{self.config.root_domain}"
        srv_records = []
        
        # This would require DHT range query support
        # For now, return empty list
        return srv_records
    
    def _sign_record(self, record: DNSRecord) -> bytes:
        """Sign a DNS record for DNSSEC."""
        # Simplified - real implementation would use proper DNSSEC
        import hashlib
        data = f"{record.name}:{record.record_type.value}:{record.value}:{record.ttl}"
        return hashlib.sha256(data.encode()).digest()
    
    def _verify_signature(self, record_data: Dict[str, Any]) -> bool:
        """Verify DNSSEC signature."""
        # Simplified verification
        expected = self._sign_record(DNSRecord(
            name=record_data['name'],
            record_type=DNSRecordType(record_data['type']),
            value=record_data['value'],
            ttl=record_data['ttl'],
            created=datetime.fromisoformat(record_data['created'])
        ))
        
        actual = bytes.fromhex(record_data['signature'])
        return expected == actual
    
    async def _dns_maintenance(self):
        """Periodic DNS maintenance."""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                # Clean expired cache entries
                expired = []
                for key, record in self.cache.items():
                    if record.is_expired():
                        expired.append(key)
                
                for key in expired:
                    del self.cache[key]
                
                # Refresh registered names
                for name in self.registered_names:
                    # Re-register to prevent expiration
                    await self.register(name)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in DNS maintenance: {e}")