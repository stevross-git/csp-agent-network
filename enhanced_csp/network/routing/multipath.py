# enhanced_csp/network/routing/multipath.py
"""
Multipath routing manager for load balancing and redundancy
"""

import logging
from typing import List, Tuple, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass
import random

logger = logging.getLogger(__name__)


@dataclass
class PathDiversity:
    """Measures diversity between paths"""
    node_overlap: float  # Percentage of overlapping nodes
    link_overlap: float  # Percentage of overlapping links
    geographic_distance: float  # Geographic diversity (if available)
    
    @property
    def diversity_score(self) -> float:
        """Calculate overall diversity score (0-1, higher is more diverse)"""
        return 1.0 - (self.node_overlap * 0.6 + self.link_overlap * 0.4)


class MultipathManager:
    """Manages multipath routing for load balancing and redundancy"""
    
    def __init__(self, max_paths: int = 3):
        self.max_paths = max_paths
        self.logger = logging.getLogger(__name__)
    
    def select_diverse_paths(self, routes_with_costs: List[Tuple[Any, float]], 
                           max_paths: int = None) -> List[Any]:
        """
        Select diverse paths from available routes
        
        Args:
            routes_with_costs: List of (route, cost) tuples
            max_paths: Maximum number of paths to select
            
        Returns:
            List of selected routes
        """
        if not routes_with_costs:
            return []
        
        max_paths = max_paths or self.max_paths
        
        # Sort by cost (ascending - lower cost is better)
        sorted_routes = sorted(routes_with_costs, key=lambda x: x[1])
        
        if len(sorted_routes) <= max_paths:
            return [route for route, _ in sorted_routes]
        
        # Always include the best route
        selected = [sorted_routes[0][0]]
        remaining = sorted_routes[1:]
        
        # Select diverse paths
        while len(selected) < max_paths and remaining:
            best_candidate = None
            best_diversity = -1
            
            for route, cost in remaining:
                # Calculate diversity with already selected paths
                diversity = self._calculate_path_diversity(route, selected)
                
                # Score combining cost and diversity
                # Lower cost and higher diversity are better
                normalized_cost = cost / (sorted_routes[0][1] + 1e-6)  # Normalize to best route
                score = diversity - (normalized_cost * 0.3)  # Prefer diversity over cost
                
                if score > best_diversity:
                    best_diversity = score
                    best_candidate = (route, cost)
            
            if best_candidate:
                selected.append(best_candidate[0])
                remaining.remove(best_candidate)
            else:
                break
        
        return selected
    
    def _calculate_path_diversity(self, candidate_route: Any, selected_routes: List[Any]) -> float:
        """Calculate diversity of candidate route compared to selected routes"""
        if not selected_routes:
            return 1.0
        
        diversities = []
        
        for selected_route in selected_routes:
            diversity = self._calculate_pairwise_diversity(candidate_route, selected_route)
            diversities.append(diversity)
        
        # Return minimum diversity (most conservative)
        return min(diversities) if diversities else 1.0
    
    def _calculate_pairwise_diversity(self, route1: Any, route2: Any) -> float:
        """Calculate diversity between two routes"""
        try:
            # Try to extract path information
            path1 = getattr(route1, 'path', [])
            path2 = getattr(route2, 'path', [])
            
            if not path1 or not path2:
                # If we can't get path info, use next hop diversity
                next_hop1 = getattr(route1, 'next_hop', None)
                next_hop2 = getattr(route2, 'next_hop', None)
                
                if next_hop1 and next_hop2:
                    return 1.0 if next_hop1 != next_hop2 else 0.0
                else:
                    return 0.5  # Unknown diversity
            
            # Calculate node overlap
            set1 = set(path1)
            set2 = set(path2)
            
            if not set1 or not set2:
                return 0.5
            
            overlap = len(set1.intersection(set2))
            total_unique = len(set1.union(set2))
            
            if total_unique == 0:
                return 0.0
            
            node_overlap = overlap / total_unique
            diversity = 1.0 - node_overlap
            
            return diversity
            
        except Exception as e:
            self.logger.warning(f"Error calculating path diversity: {e}")
            return 0.5  # Default diversity if calculation fails
    
    def calculate_load_distribution(self, paths: List[Any], 
                                  path_costs: List[float]) -> List[float]:
        """
        Calculate load distribution weights for paths
        
        Args:
            paths: List of available paths
            path_costs: List of costs for each path
            
        Returns:
            List of weights for load distribution
        """
        if not paths or not path_costs or len(paths) != len(path_costs):
            return []
        
        # Convert costs to weights (inverse relationship)
        weights = []
        for cost in path_costs:
            if cost <= 0:
                weights.append(1.0)
            else:
                weights.append(1.0 / cost)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            # Equal distribution if all weights are zero
            weights = [1.0 / len(paths)] * len(paths)
        
        return weights
    
    def select_path_for_packet(self, paths: List[Any], 
                              weights: List[float]) -> Optional[Any]:
        """
        Select a path for a packet using weighted random selection
        
        Args:
            paths: Available paths
            weights: Weights for each path
            
        Returns:
            Selected path or None if no paths available
        """
        if not paths or not weights or len(paths) != len(weights):
            return None
        
        # Weighted random selection
        try:
            selected_path = random.choices(paths, weights=weights, k=1)[0]
            return selected_path
        except (ValueError, IndexError):
            # Fallback to first path if random selection fails
            return paths[0] if paths else None
    
    def get_path_statistics(self, paths: List[Any]) -> dict:
        """Get statistics about the path set"""
        if not paths:
            return {"count": 0, "diversity": 0.0}
        
        stats = {
            "count": len(paths),
            "diversity": 0.0,
            "avg_hop_count": 0.0,
            "min_hop_count": float('inf'),
            "max_hop_count": 0
        }
        
        try:
            # Calculate path statistics
            hop_counts = []
            
            for path in paths:
                path_list = getattr(path, 'path', [])
                if path_list:
                    hop_count = len(path_list)
                    hop_counts.append(hop_count)
                    stats["min_hop_count"] = min(stats["min_hop_count"], hop_count)
                    stats["max_hop_count"] = max(stats["max_hop_count"], hop_count)
            
            if hop_counts:
                stats["avg_hop_count"] = sum(hop_counts) / len(hop_counts)
                
                # Calculate diversity as variance in hop counts
                if len(hop_counts) > 1:
                    mean_hops = stats["avg_hop_count"]
                    variance = sum((h - mean_hops) ** 2 for h in hop_counts) / len(hop_counts)
                    stats["diversity"] = min(variance / mean_hops, 1.0) if mean_hops > 0 else 0.0
                else:
                    stats["diversity"] = 0.0
            
            # Fix min_hop_count if no valid paths
            if stats["min_hop_count"] == float('inf'):
                stats["min_hop_count"] = 0
        
        except Exception as e:
            self.logger.warning(f"Error calculating path statistics: {e}")
        
        return stats