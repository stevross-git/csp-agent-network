# enhanced_csp/network/routing/multipath.py
"""
Multipath routing and load balancing implementation
Provides path diversity and traffic distribution
"""

import asyncio
import logging
import time
import hashlib
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

from ..core.types import NodeID, RoutingEntry, RoutingConfig


logger = logging.getLogger(__name__)


@dataclass
class PathDiversity:
    """Metrics for path diversity"""
    edge_overlap: float  # 0-1, lower is better
    node_overlap: float  # 0-1, lower is better
    geographic_diversity: float  # 0-1, higher is better
    reliability_diversity: float  # 0-1, higher is better
    
    @property
    def diversity_score(self) -> float:
        """Combined diversity score (higher is better)"""
        return (
            (1.0 - self.edge_overlap) * 0.3 +
            (1.0 - self.node_overlap) * 0.3 +
            self.geographic_diversity * 0.2 +
            self.reliability_diversity * 0.2
        )


@dataclass 
class PathPerformance:
    """Performance tracking for a path"""
    path_id: str
    path: List[NodeID]
    bytes_sent: int = 0
    packets_sent: int = 0
    packets_lost: int = 0
    total_rtt: float = 0.0
    last_used: float = field(default_factory=time.time)
    congestion_events: int = 0
    
    @property
    def loss_rate(self) -> float:
        """Calculate packet loss rate"""
        if self.packets_sent == 0:
            return 0.0
        return self.packets_lost / self.packets_sent
    
    @property 
    def avg_rtt(self) -> float:
        """Calculate average RTT"""
        if self.packets_sent == 0:
            return 0.0
        return self.total_rtt / self.packets_sent


class MultipathManager:
    """Manages multipath routing and load balancing"""
    
    def __init__(self, config: RoutingConfig):
        self.config = config
        
        # Path performance tracking
        self.path_performance: Dict[str, PathPerformance] = {}
        
        # Flow to paths mapping
        self.flow_paths: Dict[str, List[str]] = defaultdict(list)
        
        # Congestion detection
        self.congestion_threshold = 0.1  # 10% loss
        self.rtt_threshold = 500.0  # 500ms
        
        # Path selection state
        self.path_weights: Dict[str, float] = {}
        self.last_rebalance = time.time()
        self.rebalance_interval = 10.0  # seconds
    
    def select_diverse_paths(self, routes: List[Tuple[RoutingEntry, float]], 
                           max_paths: int = 3) -> List[RoutingEntry]:
        """Select diverse paths from available routes"""
        if not routes:
            return []
        
        # Sort by cost
        routes.sort(key=lambda x: x[1])
        
        selected = [routes[0][0]]  # Always include best path
        
        # Select additional paths based on diversity
        for route, cost in routes[1:]:
            if len(selected) >= max_paths:
                break
            
            # Calculate diversity with existing paths
            diversity = self._calculate_path_diversity(
                route.path, 
                [r.path for r in selected]
            )
            
            # Select if diverse enough
            if diversity.diversity_score > 0.5:
                selected.append(route)
        
        # If not enough diverse paths, add next best
        while len(selected) < min(max_paths, len(routes)):
            for route, _ in routes:
                if route not in selected:
                    selected.append(route)
                    break
        
        return selected
    
    def _calculate_path_diversity(self, new_path: List[NodeID], 
                                existing_paths: List[List[NodeID]]) -> PathDiversity:
        """Calculate diversity between new path and existing paths"""
        if not existing_paths:
            return PathDiversity(0.0, 0.0, 1.0, 1.0)
        
        # Convert paths to sets for comparison
        new_nodes = set(new_path)
        new_edges = set(self._path_to_edges(new_path))
        
        total_node_overlap = 0.0
        total_edge_overlap = 0.0
        
        for existing_path in existing_paths:
            existing_nodes = set(existing_path)
            existing_edges = set(self._path_to_edges(existing_path))
            
            # Calculate overlaps
            node_overlap = len(new_nodes & existing_nodes) / len(new_nodes)
            edge_overlap = len(new_edges & existing_edges) / max(len(new_edges), 1)
            
            total_node_overlap += node_overlap
            total_edge_overlap += edge_overlap
        
        # Average overlaps
        avg_node_overlap = total_node_overlap / len(existing_paths)
        avg_edge_overlap = total_edge_overlap / len(existing_paths)
        
        # Geographic diversity (simplified - based on node ID distribution)
        geographic_diversity = self._calculate_geographic_diversity(
            new_path, existing_paths
        )
        
        # Reliability diversity (different first hops provide redundancy)
        first_hops = set(path[1] if len(path) > 1 else path[0] 
                        for path in existing_paths)
        new_first_hop = new_path[1] if len(new_path) > 1 else new_path[0]
        reliability_diversity = 1.0 if new_first_hop not in first_hops else 0.0
        
        return PathDiversity(
            edge_overlap=avg_edge_overlap,
            node_overlap=avg_node_overlap,
            geographic_diversity=geographic_diversity,
            reliability_diversity=reliability_diversity
        )
    
    def _path_to_edges(self, path: List[NodeID]) -> List[Tuple[NodeID, NodeID]]:
        """Convert path to list of edges"""
        edges = []
        for i in range(len(path) - 1):
            edges.append((path[i], path[i + 1]))
        return edges
    
    def _calculate_geographic_diversity(self, new_path: List[NodeID],
                                      existing_paths: List[List[NodeID]]) -> float:
        """Estimate geographic diversity based on node IDs"""
        # In production, would use actual geographic data
        # For now, use ID-based heuristic
        
        def path_hash(path):
            # Create a hash representing the path's "region"
            combined = b''.join(node.raw_id[:4] for node in path)
            return hashlib.sha256(combined).digest()[:8]
        
        new_hash = path_hash(new_path)
        existing_hashes = [path_hash(p) for p in existing_paths]
        
        # Calculate average distance
        distances = []
        for existing_hash in existing_hashes:
            # XOR distance as proxy for geographic distance
            distance = sum(a ^ b for a, b in zip(new_hash, existing_hash))
            distances.append(distance)
        
        avg_distance = sum(distances) / len(distances) if distances else 0
        
        # Normalize to 0-1 range
        return min(avg_distance / 1000.0, 1.0)
    
    def assign_flow_to_paths(self, flow_id: str, paths: List[RoutingEntry]) -> Dict[str, float]:
        """Assign flow to multiple paths with initial weights"""
        path_ids = []
        
        for i, route in enumerate(paths):
            path_id = self._get_path_id(route.path)
            path_ids.append(path_id)
            
            # Initialize performance tracking
            if path_id not in self.path_performance:
                self.path_performance[path_id] = PathPerformance(
                    path_id=path_id,
                    path=route.path
                )
        
        # Store flow mapping
        self.flow_paths[flow_id] = path_ids
        
        # Calculate initial weights (equal distribution)
        weights = {}
        for path_id in path_ids:
            weights[path_id] = 1.0 / len(path_ids)
        
        self.path_weights.update(weights)
        
        return {path_id: weights[path_id] for path_id in path_ids}
    
    def _get_path_id(self, path: List[NodeID]) -> str:
        """Generate unique ID for path"""
        path_bytes = b''.join(node.raw_id for node in path)
        return hashlib.sha256(path_bytes).hexdigest()[:16]
    
    def select_path_for_packet(self, flow_id: str, packet_size: int) -> Optional[str]:
        """Select path for packet using current weights"""
        if flow_id not in self.flow_paths:
            return None
        
        path_ids = self.flow_paths[flow_id]
        if not path_ids:
            return None
        
        # Check if rebalancing needed
        if time.time() - self.last_rebalance > self.rebalance_interval:
            self._rebalance_paths(flow_id)
        
        # Weighted random selection
        weights = [self.path_weights.get(pid, 1.0) for pid in path_ids]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return path_ids[0]  # Fallback to first path
        
        # Normalize weights
        weights = [w / total_weight for w in weights]
        
        # Select path
        selected_idx = np.random.choice(len(path_ids), p=weights)
        selected_path = path_ids[selected_idx]
        
        # Update statistics
        perf = self.path_performance[selected_path]
        perf.bytes_sent += packet_size
        perf.packets_sent += 1
        perf.last_used = time.time()
        
        return selected_path
    
    def _rebalance_paths(self, flow_id: str):
        """Rebalance path weights based on performance"""
        path_ids = self.flow_paths.get(flow_id, [])
        if len(path_ids) <= 1:
            return
        
        # Calculate performance scores
        scores = {}
        for path_id in path_ids:
            perf = self.path_performance[path_id]
            
            # Check for congestion
            if perf.loss_rate > self.congestion_threshold:
                scores[path_id] = 0.1  # Heavily reduce weight
            elif perf.avg_rtt > self.rtt_threshold:
                scores[path_id] = 0.5  # Moderately reduce weight  
            else:
                # Performance-based score
                loss_factor = 1.0 - perf.loss_rate
                rtt_factor = 100.0 / max(perf.avg_rtt, 100.0)  # Normalize to 100ms
                
                scores[path_id] = loss_factor * 0.7 + rtt_factor * 0.3
        
        # Convert scores to weights
        total_score = sum(scores.values())
        if total_score > 0:
            for path_id in path_ids:
                self.path_weights[path_id] = scores.get(path_id, 0.0) / total_score
        
        self.last_rebalance = time.time()
        
        logger.debug(f"Rebalanced flow {flow_id}: {self.path_weights}")
    
    def update_path_performance(self, path_id: str, rtt: float, lost: bool):
        """Update path performance metrics"""
        if path_id not in self.path_performance:
            return
        
        perf = self.path_performance[path_id]
        perf.total_rtt += rtt
        
        if lost:
            perf.packets_lost += 1
            perf.congestion_events += 1
    
    def handle_path_failure(self, path_id: str):
        """Handle path failure by removing from active paths"""
        logger.warning(f"Path {path_id} failed")
        
        # Set weight to 0
        self.path_weights[path_id] = 0.0
        
        # Mark performance
        if path_id in self.path_performance:
            perf = self.path_performance[path_id]
            perf.congestion_events += 10  # Heavy penalty
        
        # Remove from affected flows
        for flow_id, paths in list(self.flow_paths.items()):
            if path_id in paths:
                paths.remove(path_id)
                if not paths:
                    # No paths left for flow
                    del self.flow_paths[flow_id]
    
    def get_flow_stats(self, flow_id: str) -> Dict[str, Any]:
        """Get statistics for a flow"""
        path_ids = self.flow_paths.get(flow_id, [])
        
        stats = {
            'path_count': len(path_ids),
            'paths': []
        }
        
        for path_id in path_ids:
            perf = self.path_performance.get(path_id)
            if perf:
                stats['paths'].append({
                    'path_id': path_id,
                    'weight': self.path_weights.get(path_id, 0.0),
                    'bytes_sent': perf.bytes_sent,
                    'packets_sent': perf.packets_sent,
                    'loss_rate': perf.loss_rate,
                    'avg_rtt': perf.avg_rtt,
                    'congestion_events': perf.congestion_events
                })
        
        return stats
    
    def get_overall_stats(self) -> Dict[str, Any]:
        """Get overall multipath statistics"""
        total_flows = len(self.flow_paths)
        total_paths = len(self.path_performance)
        active_paths = sum(1 for p in self.path_performance.values()
                          if time.time() - p.last_used < 60)
        
        total_bytes = sum(p.bytes_sent for p in self.path_performance.values())
        total_packets = sum(p.packets_sent for p in self.path_performance.values())
        total_lost = sum(p.packets_lost for p in self.path_performance.values())
        
        return {
            'total_flows': total_flows,
            'total_paths': total_paths,
            'active_paths': active_paths,
            'total_bytes_sent': total_bytes,
            'total_packets_sent': total_packets,
            'overall_loss_rate': total_lost / total_packets if total_packets > 0 else 0.0,
            'rebalance_interval': self.rebalance_interval
        }
    
    def cleanup_inactive_flows(self, timeout: float = 300.0):
        """Clean up inactive flows and paths"""
        current_time = time.time()
        
        # Find inactive paths
        inactive_paths = set()
        for path_id, perf in list(self.path_performance.items()):
            if current_time - perf.last_used > timeout:
                inactive_paths.add(path_id)
                del self.path_performance[path_id]
                self.path_weights.pop(path_id, None)
        
        # Remove from flows
        for flow_id, paths in list(self.flow_paths.items()):
            self.flow_paths[flow_id] = [p for p in paths if p not in inactive_paths]
            if not self.flow_paths[flow_id]:
                del self.flow_paths[flow_id]
        
        if inactive_paths:
            logger.info(f"Cleaned up {len(inactive_paths)} inactive paths")