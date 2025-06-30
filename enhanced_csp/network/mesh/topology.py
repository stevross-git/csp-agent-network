# enhanced_csp/network/mesh/topology.py
"""
Dynamic mesh topology management with adaptive super-peer election
"""

import asyncio
import logging
import random
import time
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
import numpy as np

from ..core.types import (
    NodeID, PeerInfo, PeerType, MeshConfig, NetworkStats
)
from ..core.node import NetworkNode


logger = logging.getLogger(__name__)


class TopologyType(Enum):
    """Mesh topology types"""
    FULL_MESH = "full_mesh"
    PARTIAL_MESH = "partial_mesh"
    HIERARCHICAL = "hierarchical"
    SMALL_WORLD = "small_world"
    SCALE_FREE = "scale_free"
    DYNAMIC_ADAPTIVE = "dynamic_adaptive"


@dataclass
class TopologyMetrics:
    """Metrics for topology optimization"""
    avg_path_length: float = 0.0
    clustering_coefficient: float = 0.0
    connectivity: float = 0.0
    diameter: int = 0
    resilience_score: float = 0.0
    efficiency: float = 0.0
    betweenness_centrality: Dict[NodeID, float] = field(default_factory=dict)


@dataclass
class SuperPeerCandidate:
    """Candidate for super-peer role"""
    node_id: NodeID
    capacity_score: float
    uptime: float
    connection_count: int
    geographic_diversity: float
    reliability_score: float
    
    @property
    def election_score(self) -> float:
        """Calculate overall election score"""
        return (
            self.capacity_score * 0.4 +
            self.reliability_score * 0.3 +
            self.connection_count * 0.2 +
            self.geographic_diversity * 0.1
        )


class MeshTopologyManager:
    """Manages dynamic mesh network topology"""
    
    def __init__(self, node: NetworkNode, config: MeshConfig):
        self.node = node
        self.config = config
        
        # Network graph
        self.graph = nx.Graph()
        self.graph.add_node(node.node_id)
        
        # Topology state
        self.topology_type = TopologyType(config.topology_type)
        self.metrics = TopologyMetrics()
        self.super_peers: Set[NodeID] = set()
        self.peer_connections: Dict[NodeID, Set[NodeID]] = {
            node.node_id: set()
        }
        
        # Optimization state
        self.last_optimization = time.time()
        self.optimization_interval = 60  # seconds
        self.target_connections = self._calculate_target_connections()
        
        # Tasks
        self._tasks: List[asyncio.Task] = []
    
    def _calculate_target_connections(self) -> int:
        """Calculate optimal number of connections based on network size"""
        # Use sqrt(n) * log(n) for good connectivity with reasonable overhead
        estimated_size = self.config.max_peers
        return min(
            int(np.sqrt(estimated_size) * np.log(estimated_size)),
            self.config.max_peers // 2
        )
    
    async def start(self):
        """Start topology management"""
        logger.info(f"Starting mesh topology manager ({self.topology_type.value})")
        
        # Start maintenance tasks
        self._tasks.extend([
            asyncio.create_task(self._topology_maintenance_loop()),
            asyncio.create_task(self._super_peer_election_loop()),
            asyncio.create_task(self._metrics_calculation_loop())
        ])
    
    async def stop(self):
        """Stop topology management"""
        for task in self._tasks:
            task.cancel()
        
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
    
    async def add_peer(self, peer_info: PeerInfo) -> bool:
        """Add peer to mesh topology"""
        peer_id = peer_info.node_id
        
        # Add to graph
        self.graph.add_node(peer_id, **{
            'peer_type': peer_info.peer_type,
            'latency': peer_info.latency_ms,
            'capacity': peer_info.bandwidth_mbps
        })
        
        # Initialize connection set
        if peer_id not in self.peer_connections:
            self.peer_connections[peer_id] = set()
        
        # Decide if we should connect based on topology
        should_connect = await self._should_connect_to_peer(peer_info)
        
        if should_connect:
            await self._establish_mesh_connection(peer_info)
        
        return should_connect
    
    async def remove_peer(self, peer_id: NodeID):
        """Remove peer from mesh topology"""
        # Remove from graph
        if self.graph.has_node(peer_id):
            self.graph.remove_node(peer_id)
        
        # Remove from connections
        self.peer_connections.pop(peer_id, None)
        
        # Remove from other peers' connections
        for connections in self.peer_connections.values():
            connections.discard(peer_id)
        
        # Remove from super peers
        self.super_peers.discard(peer_id)
        
        # Check if topology needs rebalancing
        await self._check_topology_health()
    
    async def _should_connect_to_peer(self, peer_info: PeerInfo) -> bool:
        """Decide if we should establish connection with peer"""
        our_connections = len(self.peer_connections.get(self.node.node_id, set()))
        
        # Always connect to super peers
        if peer_info.peer_type == PeerType.SUPER_PEER:
            return True
        
        # Check if we need more connections
        if our_connections < self.target_connections:
            # Use selection criteria
            return self._evaluate_peer_quality(peer_info) > 0.5
        
        # Replace existing connection if this peer is better
        return await self._should_replace_connection(peer_info)
    
    def _evaluate_peer_quality(self, peer_info: PeerInfo) -> float:
        """Evaluate peer quality for connection (0-1 score)"""
        weights = self.config.peer_selection_weights
        
        # Normalize metrics
        latency_score = 1.0 - min(peer_info.latency_ms / 500.0, 1.0) if peer_info.latency_ms else 0.5
        capacity_score = min(peer_info.bandwidth_mbps / 100.0, 1.0) if peer_info.bandwidth_mbps else 0.5
        
        # Weighted score
        score = (
            latency_score * weights.get('latency', 0.7) +
            capacity_score * weights.get('capacity', 0.3)
        )
        
        # Boost score for peers that increase diversity
        diversity_bonus = self._calculate_diversity_bonus(peer_info)
        
        return min(score + diversity_bonus, 1.0)
    
    def _calculate_diversity_bonus(self, peer_info: PeerInfo) -> float:
        """Calculate bonus for geographic/network diversity"""
        # In production, would use actual geographic/AS information
        # For now, use node ID for pseudo-diversity
        existing_ids = list(self.peer_connections.keys())
        if not existing_ids:
            return 0.2
        
        # Simple diversity metric based on ID distance
        min_distance = min(
            self._id_distance(peer_info.node_id, existing)
            for existing in existing_ids
        )
        
        # More distance = more diversity
        return min(min_distance / 128.0, 0.2)  # Max 0.2 bonus
    
    def _id_distance(self, id1: NodeID, id2: NodeID) -> int:
        """Calculate XOR distance between node IDs"""
        # XOR distance in ID space
        xor = int.from_bytes(id1.raw_id, 'big') ^ int.from_bytes(id2.raw_id, 'big')
        return xor.bit_length()
    
    async def _should_replace_connection(self, new_peer: PeerInfo) -> bool:
        """Check if we should replace an existing connection"""
        our_connections = self.peer_connections.get(self.node.node_id, set())
        if len(our_connections) < self.target_connections:
            return True
        
        # Find worst performing connection
        worst_peer = None
        worst_score = 1.0
        
        for peer_id in our_connections:
            if peer_id in self.node.peers:
                peer_info = self.node.peers[peer_id]
                score = self._evaluate_peer_quality(peer_info)
                if score < worst_score:
                    worst_score = score
                    worst_peer = peer_id
        
        # Replace if new peer is significantly better
        new_score = self._evaluate_peer_quality(new_peer)
        if worst_peer and new_score > worst_score * 1.2:  # 20% better
            await self._disconnect_peer(worst_peer)
            return True
        
        return False
    
    async def _establish_mesh_connection(self, peer_info: PeerInfo):
        """Establish mesh connection with peer"""
        peer_id = peer_info.node_id
        our_id = self.node.node_id
        
        # Add edge to graph
        self.graph.add_edge(our_id, peer_id, weight=peer_info.routing_cost)
        
        # Update connection tracking
        self.peer_connections[our_id].add(peer_id)
        if peer_id in self.peer_connections:
            self.peer_connections[peer_id].add(our_id)
        
        logger.info(f"Established mesh connection with {peer_id.to_base58()[:16]}...")
    
    async def _disconnect_peer(self, peer_id: NodeID):
        """Disconnect from a peer"""
        our_id = self.node.node_id
        
        # Remove edge from graph
        if self.graph.has_edge(our_id, peer_id):
            self.graph.remove_edge(our_id, peer_id)
        
        # Update connection tracking
        self.peer_connections[our_id].discard(peer_id)
        if peer_id in self.peer_connections:
            self.peer_connections[peer_id].discard(our_id)
        
        # Notify node to disconnect
        await self.node.disconnect_peer(peer_id)
    
    async def _check_topology_health(self):
        """Check and repair topology if needed"""
        our_connections = len(self.peer_connections.get(self.node.node_id, set()))
        
        # Too few connections - find new peers
        if our_connections < self.target_connections * 0.5:
            logger.warning(f"Low connectivity ({our_connections} connections), "
                         f"seeking new peers...")
            # Trigger peer discovery
            if hasattr(self.node, 'discovery'):
                await self.node.discovery.find_peers_dht(
                    count=self.target_connections - our_connections
                )
        
        # Check if graph is still connected
        if not nx.is_connected(self.graph):
            logger.warning("Mesh topology disconnected, attempting repair...")
            await self._repair_partitioned_topology()
    
    async def _repair_partitioned_topology(self):
        """Repair partitioned topology"""
        # Find connected components
        components = list(nx.connected_components(self.graph))
        
        if len(components) <= 1:
            return
        
        logger.info(f"Found {len(components)} partitions, bridging...")
        
        # Try to bridge partitions
        for i in range(len(components) - 1):
            comp1 = components[i]
            comp2 = components[i + 1]
            
            # Find best nodes to bridge
            node1 = self._find_bridge_node(comp1)
            node2 = self._find_bridge_node(comp2)
            
            if node1 and node2:
                # Request connection
                # This would trigger actual connection establishment
                logger.info(f"Bridging partitions via {node1} <-> {node2}")
    
    def _find_bridge_node(self, component: Set[NodeID]) -> Optional[NodeID]:
        """Find best node in component for bridging"""
        # Prefer super peers or high-capacity nodes
        candidates = []
        
        for node_id in component:
            if node_id == self.node.node_id:
                return node_id  # We can always bridge
            
            if node_id in self.super_peers:
                candidates.append((node_id, 1.0))
            elif node_id in self.node.peers:
                peer = self.node.peers[node_id]
                score = self._evaluate_peer_quality(peer)
                candidates.append((node_id, score))
        
        if candidates:
            # Return highest scoring node
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
        
        return None
    
    async def optimize_topology(self):
        """Optimize mesh topology for better performance"""
        logger.info("Optimizing mesh topology...")
        
        current_metrics = self._calculate_metrics()
        
        # Different optimization strategies based on topology type
        if self.topology_type == TopologyType.DYNAMIC_ADAPTIVE:
            await self._optimize_adaptive_topology(current_metrics)
        elif self.topology_type == TopologyType.SMALL_WORLD:
            await self._optimize_small_world(current_metrics)
        elif self.topology_type == TopologyType.SCALE_FREE:
            await self._optimize_scale_free(current_metrics)
        
        # Update metrics
        self.metrics = self._calculate_metrics()
        
        logger.info(f"Topology optimized - Avg path length: {self.metrics.avg_path_length:.2f}, "
                   f"Clustering: {self.metrics.clustering_coefficient:.3f}")
    
    async def _optimize_adaptive_topology(self, current_metrics: TopologyMetrics):
        """Optimize adaptive topology based on current conditions"""
        our_id = self.node.node_id
        our_connections = self.peer_connections.get(our_id, set())
        
        # Add shortcuts if path length is too high
        if current_metrics.avg_path_length > 3.0:
            await self._add_shortcuts()
        
        # Improve clustering if too low
        if current_metrics.clustering_coefficient < 0.3:
            await self._improve_clustering()
        
        # Balance connections
        if len(our_connections) > self.target_connections * 1.5:
            await self._prune_connections()
    
    async def _add_shortcuts(self):
        """Add shortcut connections to reduce path length"""
        # Find distant nodes that we frequently communicate with
        # For now, add random shortcuts
        
        our_id = self.node.node_id
        candidates = []
        
        for node_id in self.graph.nodes():
            if node_id != our_id and not self.graph.has_edge(our_id, node_id):
                try:
                    distance = nx.shortest_path_length(self.graph, our_id, node_id)
                    if distance > 3:
                        candidates.append((node_id, distance))
                except nx.NetworkXNoPath:
                    pass
        
        # Add shortcuts to most distant nodes
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        for node_id, _ in candidates[:2]:  # Add up to 2 shortcuts
            if node_id in self.node.peers:
                peer_info = self.node.peers[node_id]
                await self._establish_mesh_connection(peer_info)
    
    async def _improve_clustering(self):
        """Improve local clustering coefficient"""
        our_id = self.node.node_id
        our_neighbors = list(self.graph.neighbors(our_id))
        
        # Find neighbors that should be connected
        for i in range(len(our_neighbors)):
            for j in range(i + 1, len(our_neighbors)):
                node1, node2 = our_neighbors[i], our_neighbors[j]
                
                # If neighbors aren't connected, suggest connection
                if not self.graph.has_edge(node1, node2):
                    # Send introduction message
                    await self._introduce_peers(node1, node2)
    
    async def _introduce_peers(self, peer1: NodeID, peer2: NodeID):
        """Introduce two peers to each other"""
        intro_msg = {
            'type': 'peer_introduction',
            'peer1': peer1.to_base58(),
            'peer2': peer2.to_base58(),
            'reason': 'clustering_improvement'
        }
        
        # Send to both peers
        await self.node.send_message(peer1, intro_msg)
        await self.node.send_message(peer2, intro_msg)
    
    async def _prune_connections(self):
        """Remove excess connections"""
        our_id = self.node.node_id
        our_connections = list(self.peer_connections.get(our_id, set()))
        
        if len(our_connections) <= self.target_connections:
            return
        
        # Score all connections
        connection_scores = []
        for peer_id in our_connections:
            if peer_id in self.node.peers:
                peer = self.node.peers[peer_id]
                score = self._evaluate_peer_quality(peer)
                
                # Boost score for important structural positions
                if peer_id in self.metrics.betweenness_centrality:
                    centrality = self.metrics.betweenness_centrality[peer_id]
                    score += centrality * 0.1
                
                connection_scores.append((peer_id, score))
        
        # Sort by score and remove lowest scoring
        connection_scores.sort(key=lambda x: x[1])
        
        to_remove = len(our_connections) - self.target_connections
        for peer_id, _ in connection_scores[:to_remove]:
            await self._disconnect_peer(peer_id)
    
    def _calculate_metrics(self) -> TopologyMetrics:
        """Calculate topology metrics"""
        metrics = TopologyMetrics()
        
        if len(self.graph) < 2:
            return metrics
        
        try:
            # Basic metrics
            if nx.is_connected(self.graph):
                metrics.avg_path_length = nx.average_shortest_path_length(self.graph)
                metrics.diameter = nx.diameter(self.graph)
            
            metrics.clustering_coefficient = nx.average_clustering(self.graph)
            metrics.connectivity = nx.node_connectivity(self.graph)
            
            # Betweenness centrality
            centrality = nx.betweenness_centrality(self.graph)
            metrics.betweenness_centrality = {
                NodeID(k): v for k, v in centrality.items()
            }
            
            # Efficiency (how well connected the network is)
            metrics.efficiency = nx.global_efficiency(self.graph)
            
            # Resilience score (resistance to node failures)
            metrics.resilience_score = self._calculate_resilience()
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
        
        return metrics
    
    def _calculate_resilience(self) -> float:
        """Calculate network resilience to node failures"""
        if len(self.graph) < 10:
            return 1.0  # Small networks are considered fully resilient
        
        # Simulate random node failures
        original_connectivity = nx.node_connectivity(self.graph)
        
        resilience_scores = []
        for _ in range(min(5, len(self.graph) // 10)):
            # Copy graph and remove random node
            test_graph = self.graph.copy()
            node_to_remove = random.choice(list(test_graph.nodes()))
            test_graph.remove_node(node_to_remove)
            
            # Check connectivity after failure
            if nx.is_connected(test_graph):
                remaining_connectivity = nx.node_connectivity(test_graph)
                resilience_scores.append(remaining_connectivity / original_connectivity)
            else:
                resilience_scores.append(0.0)
        
        return sum(resilience_scores) / len(resilience_scores) if resilience_scores else 0.0
    
    async def _topology_maintenance_loop(self):
        """Periodic topology maintenance"""
        while True:
            try:
                await asyncio.sleep(self.optimization_interval)
                
                # Check if optimization is needed
                if time.time() - self.last_optimization > self.optimization_interval:
                    await self.optimize_topology()
                    self.last_optimization = time.time()
                
                # Check topology health
                await self._check_topology_health()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in topology maintenance: {e}")
    
    async def _super_peer_election_loop(self):
        """Periodic super-peer election"""
        if not self.config.enable_super_peers:
            return
        
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                await self.elect_super_peers()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in super-peer election: {e}")
    
    async def _metrics_calculation_loop(self):
        """Periodic metrics calculation"""
        while True:
            try:
                await asyncio.sleep(30)  # Every 30 seconds
                self.metrics = self._calculate_metrics()
                
                # Log important metrics
                if self.metrics.avg_path_length > 4.0:
                    logger.warning(f"High average path length: {self.metrics.avg_path_length}")
                
                if self.metrics.resilience_score < 0.5:
                    logger.warning(f"Low resilience score: {self.metrics.resilience_score}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error calculating metrics: {e}")
    
    async def elect_super_peers(self):
        """Elect super-peers based on capacity and reliability"""
        logger.info("Running super-peer election...")
        
        candidates = []
        
        # Evaluate all peers as candidates
        for peer_id, peer_info in self.node.peers.items():
            candidate = self._evaluate_super_peer_candidate(peer_id, peer_info)
            if candidate.capacity_score >= self.config.super_peer_capacity_threshold:
                candidates.append(candidate)
        
        # Include ourselves if qualified
        our_candidate = self._evaluate_self_as_super_peer()
        if our_candidate.capacity_score >= self.config.super_peer_capacity_threshold:
            candidates.append(our_candidate)
        
        # Sort by election score
        candidates.sort(key=lambda c: c.election_score, reverse=True)
        
        # Determine number of super-peers (sqrt of network size)
        network_size = len(self.graph)
        target_super_peers = max(3, int(np.sqrt(network_size)))
        
        # Elect top candidates
        new_super_peers = set()
        for candidate in candidates[:target_super_peers]:
            new_super_peers.add(candidate.node_id)
        
        # Check for changes
        if new_super_peers != self.super_peers:
            added = new_super_peers - self.super_peers
            removed = self.super_peers - new_super_peers
            
            logger.info(f"Super-peer changes - Added: {len(added)}, Removed: {len(removed)}")
            
            # Update super-peer set
            self.super_peers = new_super_peers
            
            # Notify network of changes
            await self._announce_super_peer_changes(added, removed)
            
            # Update our own status if changed
            if self.node.node_id in added:
                logger.info("We have been elected as super-peer!")
                await self._assume_super_peer_role()
            elif self.node.node_id in removed:
                logger.info("We are no longer a super-peer")
                await self._relinquish_super_peer_role()
    
    def _evaluate_super_peer_candidate(self, peer_id: NodeID, 
                                     peer_info: PeerInfo) -> SuperPeerCandidate:
        """Evaluate a peer as super-peer candidate"""
        # Calculate capacity score
        capacity_score = peer_info.bandwidth_mbps or 0.0
        
        # Calculate uptime (estimated from last_seen)
        uptime = (time.time() - peer_info.last_seen.timestamp()) / 3600  # hours
        
        # Connection count
        connection_count = len(self.peer_connections.get(peer_id, set()))
        
        # Geographic diversity (using ID distance as proxy)
        geographic_diversity = self._calculate_geographic_diversity(peer_id)
        
        # Reliability score based on packet loss and uptime
        reliability_score = (1.0 - peer_info.packet_loss) * min(uptime / 24, 1.0)
        
        return SuperPeerCandidate(
            node_id=peer_id,
            capacity_score=capacity_score,
            uptime=uptime,
            connection_count=connection_count,
            geographic_diversity=geographic_diversity,
            reliability_score=reliability_score
        )
    
    def _evaluate_self_as_super_peer(self) -> SuperPeerCandidate:
        """Evaluate ourselves as super-peer candidate"""
        # Get our stats
        stats = self.node.get_stats()
        
        # Estimate our capacity
        capacity_score = 100.0  # TODO: Measure actual bandwidth
        
        # Our uptime
        uptime = stats.uptime_seconds / 3600
        
        # Our connections
        connection_count = len(self.peer_connections.get(self.node.node_id, set()))
        
        # We provide geographic diversity by definition
        geographic_diversity = 1.0
        
        # Our reliability
        reliability_score = 0.95  # High self-confidence
        
        return SuperPeerCandidate(
            node_id=self.node.node_id,
            capacity_score=capacity_score,
            uptime=uptime,
            connection_count=connection_count,
            geographic_diversity=geographic_diversity,
            reliability_score=reliability_score
        )
    
    def _calculate_geographic_diversity(self, peer_id: NodeID) -> float:
        """Calculate geographic diversity score for a peer"""
        # In production, would use actual geographic data
        # For now, use ID-based clustering
        
        if not self.super_peers:
            return 1.0
        
        # Find minimum distance to existing super-peers
        min_distance = min(
            self._id_distance(peer_id, sp_id)
            for sp_id in self.super_peers
        )
        
        # Normalize to 0-1 range
        return min(min_distance / 64.0, 1.0)
    
    async def _announce_super_peer_changes(self, added: Set[NodeID], 
                                         removed: Set[NodeID]):
        """Announce super-peer changes to network"""
        announcement = {
            'type': 'super_peer_announcement',
            'added': [node_id.to_base58() for node_id in added],
            'removed': [node_id.to_base58() for node_id in removed],
            'current_super_peers': [node_id.to_base58() for node_id in self.super_peers],
            'timestamp': time.time()
        }
        
        # Broadcast to all peers
        await self.node.broadcast_message(announcement)
    
    async def _assume_super_peer_role(self):
        """Assume super-peer responsibilities"""
        # Increase our connection limit
        self.target_connections = min(
            self.target_connections * 2,
            self.config.max_peers
        )
        
        # Enable additional services
        # - Act as relay for NAT traversal
        # - Provide DHT bootstrap service
        # - Cache popular content
        
        # Update our peer info
        if hasattr(self.node, 'local_peer_info'):
            self.node.local_peer_info.peer_type = PeerType.SUPER_PEER
    
    async def _relinquish_super_peer_role(self):
        """Give up super-peer responsibilities"""
        # Reduce connection limit back to normal
        self.target_connections = self._calculate_target_connections()
        
        # Disable super-peer services
        
        # Update our peer info
        if hasattr(self.node, 'local_peer_info'):
            self.node.local_peer_info.peer_type = PeerType.REGULAR
        
        # Prune excess connections gracefully
        await self._prune_connections()
    
    def get_shortest_path(self, target: NodeID) -> Optional[List[NodeID]]:
        """Get shortest path to target node"""
        try:
            path = nx.shortest_path(self.graph, self.node.node_id, target)
            return [NodeID(node_id) for node_id in path]
        except nx.NetworkXNoPath:
            return None
    
    def get_all_paths(self, target: NodeID, cutoff: int = 5) -> List[List[NodeID]]:
        """Get all paths to target within cutoff length"""
        try:
            paths = nx.all_simple_paths(
                self.graph, 
                self.node.node_id, 
                target,
                cutoff=cutoff
            )
            return [[NodeID(node_id) for node_id in path] for path in paths]
        except nx.NetworkXNoPath:
            return []
    
    def get_network_view(self) -> Dict[str, Any]:
        """Get current view of network topology"""
        return {
            'topology_type': self.topology_type.value,
            'nodes': len(self.graph),
            'edges': self.graph.number_of_edges(),
            'super_peers': [sp.to_base58() for sp in self.super_peers],
            'metrics': {
                'avg_path_length': self.metrics.avg_path_length,
                'clustering': self.metrics.clustering_coefficient,
                'diameter': self.metrics.diameter,
                'resilience': self.metrics.resilience_score,
                'efficiency': self.metrics.efficiency
            },
            'our_connections': [
                peer_id.to_base58() 
                for peer_id in self.peer_connections.get(self.node.node_id, set())
            ]
        }