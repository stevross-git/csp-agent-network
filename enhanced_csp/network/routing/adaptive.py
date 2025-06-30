# enhanced_csp/network/routing/adaptive.py
"""
Adaptive routing engine with real-time optimization and ML prediction
"""

import asyncio
import logging
import time
import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from collections import deque, defaultdict
import heapq

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("scikit-learn not available, ML prediction disabled")

from ..core.types import NodeID, RoutingEntry, RoutingConfig
from ..core.node import NetworkNode
from ..mesh.routing import BatmanRouting
from .metrics import MetricsCollector
from .multipath import MultipathManager


logger = logging.getLogger(__name__)


@dataclass
class RouteMetrics:
    """Detailed metrics for a route"""
    destination: NodeID
    next_hop: NodeID
    rtt_ms: float
    bandwidth_mbps: float
    packet_loss: float
    jitter_ms: float
    hop_count: int
    security_score: float
    last_updated: float = field(default_factory=time.time)
    samples: int = 0
    
    @property
    def cost(self) -> float:
        """Calculate composite routing cost"""
        # Normalize metrics
        latency_factor = self.rtt_ms / 10.0  # 10ms baseline
        bandwidth_factor = 100.0 / max(self.bandwidth_mbps, 1.0)  # 100Mbps baseline
        loss_factor = 1.0 + (self.packet_loss * 10)  # Heavy penalty for loss
        security_factor = 2.0 - self.security_score  # Lower security = higher cost
        
        # Composite cost
        return (
            latency_factor * 0.4 +
            bandwidth_factor * 0.3 +
            loss_factor * 0.2 +
            security_factor * 0.1
        ) * self.hop_count
    
    def update(self, rtt: float, bandwidth: float, loss: float, jitter: float):
        """Update metrics with exponential moving average"""
        alpha = 0.3  # EMA factor
        
        if self.samples == 0:
            # First sample
            self.rtt_ms = rtt
            self.bandwidth_mbps = bandwidth
            self.packet_loss = loss
            self.jitter_ms = jitter
        else:
            # Exponential moving average
            self.rtt_ms = alpha * rtt + (1 - alpha) * self.rtt_ms
            self.bandwidth_mbps = alpha * bandwidth + (1 - alpha) * self.bandwidth_mbps
            self.packet_loss = alpha * loss + (1 - alpha) * self.packet_loss
            self.jitter_ms = alpha * jitter + (1 - alpha) * self.jitter_ms
        
        self.samples += 1
        self.last_updated = time.time()


@dataclass
class FlowState:
    """State for a network flow"""
    flow_id: str
    source: NodeID
    destination: NodeID
    paths: List[List[NodeID]]
    path_weights: List[float]
    bytes_sent: Dict[int, int] = field(default_factory=dict)
    last_path_index: int = 0
    created: float = field(default_factory=time.time)


class AdaptiveRoutingEngine:
    """Adaptive routing with ML-based prediction"""
    
    def __init__(self, node: NetworkNode, config: RoutingConfig,
                 batman_routing: BatmanRouting):
        self.node = node
        self.config = config
        self.batman = batman_routing
        
        # Route metrics
        self.route_metrics: Dict[Tuple[NodeID, NodeID], RouteMetrics] = {}
        
        # Metrics collector
        self.metrics_collector = MetricsCollector(node)
        
        # Multipath manager
        self.multipath = MultipathManager(config)
        
        # Active flows
        self.flows: Dict[str, FlowState] = {}
        
        # ML predictor
        self.ml_predictor = None
        if ML_AVAILABLE and config.enable_ml_predictor:
            self._init_ml_predictor()
        
        # Historical data for ML
        self.metric_history: deque = deque(maxlen=1000)
        
        # Route flapping detection
        self.route_changes: Dict[NodeID, deque] = defaultdict(
            lambda: deque(maxlen=10)
        )
        
        # Tasks
        self._tasks: List[asyncio.Task] = []
    
    def _init_ml_predictor(self):
        """Initialize ML predictor for route optimization"""
        self.ml_predictor = RoutePredictor()
        self.scaler = StandardScaler()
        logger.info("Initialized ML route predictor")
    
    async def start(self):
        """Start adaptive routing engine"""
        logger.info("Starting adaptive routing engine")
        
        # Start components
        await self.metrics_collector.start()
        
        # Start tasks
        self._tasks.extend([
            asyncio.create_task(self._metrics_update_loop()),
            asyncio.create_task(self._route_optimization_loop()),
            asyncio.create_task(self._ml_training_loop())
        ])
    
    async def stop(self):
        """Stop adaptive routing engine"""
        await self.metrics_collector.stop()
        
        for task in self._tasks:
            task.cancel()
        
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
    
    async def get_best_route(self, destination: NodeID) -> Optional[RoutingEntry]:
        """Get best route considering real-time metrics"""
        # Get base routes from B.A.T.M.A.N.
        routes = self.batman.get_all_routes(destination)
        
        if not routes:
            return None
        
        # Score routes based on current metrics
        best_route = None
        best_cost = float('inf')
        
        for route in routes:
            # Get metrics for this route
            metrics_key = (destination, route.next_hop)
            
            if metrics_key in self.route_metrics:
                metrics = self.route_metrics[metrics_key]
                cost = metrics.cost
                
                # Apply ML prediction if available
                if self.ml_predictor and self._has_sufficient_history(destination):
                    predicted_cost = await self._predict_route_cost(
                        destination, route.next_hop
                    )
                    # Blend current and predicted
                    cost = 0.7 * cost + 0.3 * predicted_cost
            else:
                # Use B.A.T.M.A.N. metric as fallback
                cost = route.metric
            
            if cost < best_cost:
                best_cost = cost
                best_route = route
        
        # Update route metric
        if best_route:
            best_route.metric = best_cost
        
        return best_route
    
    async def get_multipath_routes(self, destination: NodeID, 
                                 flow_id: Optional[str] = None) -> List[RoutingEntry]:
        """Get multiple paths for load balancing"""
        if not self.config.enable_multipath:
            route = await self.get_best_route(destination)
            return [route] if route else []
        
        # Get all available routes
        all_routes = self.batman.get_all_routes(destination)
        
        if not all_routes:
            return []
        
        # Score and sort routes
        scored_routes = []
        for route in all_routes:
            metrics_key = (destination, route.next_hop)
            
            if metrics_key in self.route_metrics:
                cost = self.route_metrics[metrics_key].cost
            else:
                cost = route.metric
            
            scored_routes.append((cost, route))
        
        scored_routes.sort(key=lambda x: x[0])
        
        # Select diverse paths
        selected = self.multipath.select_diverse_paths(
            [(r, c) for c, r in scored_routes],
            max_paths=self.config.max_paths_per_destination
        )
        
        # Create or update flow state
        if flow_id:
            if flow_id not in self.flows:
                self.flows[flow_id] = FlowState(
                    flow_id=flow_id,
                    source=self.node.node_id,
                    destination=destination,
                    paths=[r.path for r in selected],
                    path_weights=[1.0 / len(selected)] * len(selected)
                )
            else:
                # Update paths if changed
                flow = self.flows[flow_id]
                flow.paths = [r.path for r in selected]
                # Adjust weights based on performance
                flow.path_weights = await self._calculate_path_weights(flow)
        
        return selected
    
    async def route_packet(self, destination: NodeID, packet_size: int,
                          flow_id: Optional[str] = None) -> Optional[NodeID]:
        """Select next hop for packet routing"""
        # Fast path for single route
        if not self.config.enable_multipath or not flow_id:
            route = await self.get_best_route(destination)
            return route.next_hop if route else None
        
        # Multipath routing
        if flow_id not in self.flows:
            routes = await self.get_multipath_routes(destination, flow_id)
            if not routes:
                return None
        
        flow = self.flows[flow_id]
        
        # Select path using weighted round-robin or other algorithm
        if self.config.load_balance_algorithm == "weighted_round_robin":
            path_index = self._weighted_round_robin(flow)
        elif self.config.load_balance_algorithm == "least_loaded":
            path_index = self._select_least_loaded_path(flow)
        else:
            path_index = 0  # Default to first path
        
        # Record usage
        if path_index not in flow.bytes_sent:
            flow.bytes_sent[path_index] = 0
        flow.bytes_sent[path_index] += packet_size
        
        # Get next hop from selected path
        if path_index < len(flow.paths) and len(flow.paths[path_index]) > 1:
            return flow.paths[path_index][1]  # Second node in path
        
        return None
    
    def _weighted_round_robin(self, flow: FlowState) -> int:
        """Select path using weighted round-robin"""
        # Simple implementation - can be optimized
        rand = np.random.random()
        cumsum = 0.0
        
        for i, weight in enumerate(flow.path_weights):
            cumsum += weight
            if rand < cumsum:
                return i
        
        return len(flow.path_weights) - 1
    
    def _select_least_loaded_path(self, flow: FlowState) -> int:
        """Select path with least bytes sent"""
        min_bytes = float('inf')
        best_path = 0
        
        for i in range(len(flow.paths)):
            bytes_sent = flow.bytes_sent.get(i, 0)
            if bytes_sent < min_bytes:
                min_bytes = bytes_sent
                best_path = i
        
        return best_path
    
    async def _calculate_path_weights(self, flow: FlowState) -> List[float]:
        """Calculate path weights based on performance"""
        weights = []
        total_cost = 0.0
        
        for i, path in enumerate(flow.paths):
            if len(path) < 2:
                weights.append(0.0)
                continue
            
            # Get metrics for first hop
            metrics_key = (flow.destination, path[1])
            
            if metrics_key in self.route_metrics:
                cost = self.route_metrics[metrics_key].cost
            else:
                cost = 1.0
            
            # Inverse cost for weight (better routes get more traffic)
            weight = 1.0 / max(cost, 0.1)
            weights.append(weight)
            total_cost += weight
        
        # Normalize weights
        if total_cost > 0:
            weights = [w / total_cost for w in weights]
        else:
            weights = [1.0 / len(weights)] * len(weights)
        
        return weights
    
    async def _metrics_update_loop(self):
        """Continuously update route metrics"""
        while True:
            try:
                await asyncio.sleep(self.config.metric_update_interval)
                
                # Measure metrics to all known destinations
                destinations = set()
                for route in self.batman.routing_table.values():
                    destinations.add(route.destination)
                
                # Update metrics for each destination
                for dest in destinations:
                    await self._update_route_metrics(dest)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics update: {e}")
    
    async def _update_route_metrics(self, destination: NodeID):
        """Update metrics for routes to destination"""
        routes = self.batman.get_all_routes(destination)
        
        for route in routes:
            try:
                # Measure metrics via this route
                metrics = await self.metrics_collector.measure_path(
                    route.path,
                    route.next_hop
                )
                
                if metrics:
                    # Create or update RouteMetrics
                    metrics_key = (destination, route.next_hop)
                    
                    if metrics_key not in self.route_metrics:
                        self.route_metrics[metrics_key] = RouteMetrics(
                            destination=destination,
                            next_hop=route.next_hop,
                            rtt_ms=metrics['rtt'],
                            bandwidth_mbps=metrics['bandwidth'],
                            packet_loss=metrics['loss'],
                            jitter_ms=metrics['jitter'],
                            hop_count=len(route.path) - 1,
                            security_score=self._calculate_security_score(route)
                        )
                    else:
                        self.route_metrics[metrics_key].update(
                            metrics['rtt'],
                            metrics['bandwidth'],
                            metrics['loss'],
                            metrics['jitter']
                        )
                    
                    # Store in history for ML
                    self._store_metric_history(destination, route.next_hop, metrics)
                    
            except Exception as e:
                logger.error(f"Failed to update metrics for {destination}: {e}")
    
    def _calculate_security_score(self, route: RoutingEntry) -> float:
        """Calculate security score for route"""
        # Factors:
        # - Hop count (fewer = better)
        # - Super-peer presence
        # - Known/trusted nodes
        
        score = 1.0
        
        # Penalize long paths
        score -= min(len(route.path) * 0.1, 0.5)
        
        # Bonus for super-peers in path
        for node_id in route.path:
            if node_id in self.node.topology.super_peers:
                score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _store_metric_history(self, destination: NodeID, next_hop: NodeID,
                            metrics: Dict[str, float]):
        """Store metrics in history for ML training"""
        # Add time-based features
        hour = time.localtime().tm_hour
        day_of_week = time.localtime().tm_wday
        
        history_entry = {
            'timestamp': time.time(),
            'destination': destination.to_base58(),
            'next_hop': next_hop.to_base58(),
            'hour': hour,
            'day_of_week': day_of_week,
            'rtt': metrics['rtt'],
            'bandwidth': metrics['bandwidth'],
            'loss': metrics['loss'],
            'jitter': metrics['jitter']
        }
        
        self.metric_history.append(history_entry)
    
    async def _route_optimization_loop(self):
        """Periodically optimize routes"""
        while True:
            try:
                await asyncio.sleep(30)  # Every 30 seconds
                
                # Check for route flapping
                await self._detect_route_flapping()
                
                # Optimize flow assignments
                await self._optimize_flows()
                
                # Clean up old flows
                await self._cleanup_flows()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in route optimization: {e}")
    
    async def _detect_route_flapping(self):
        """Detect and mitigate route flapping"""
        current_time = time.time()
        
        for dest, changes in self.route_changes.items():
            if len(changes) < 5:
                continue
            
            # Check if too many changes in short time
            recent_changes = [t for t in changes if current_time - t < 60]
            
            if len(recent_changes) > 5:
                logger.warning(f"Route flapping detected for {dest.to_base58()[:16]}")
                # Increase route stability threshold
                # This would dampen route changes
    
    async def _optimize_flows(self):
        """Optimize active flow assignments"""
        for flow_id, flow in list(self.flows.items()):
            # Skip recently created flows
            if time.time() - flow.created < 10:
                continue
            
            # Re-evaluate paths
            new_routes = await self.get_multipath_routes(flow.destination, flow_id)
            
            if new_routes:
                # Check if paths changed significantly
                new_paths = [r.path for r in new_routes]
                if new_paths != flow.paths:
                    logger.info(f"Optimizing flow {flow_id} paths")
                    flow.paths = new_paths
                    flow.path_weights = await self._calculate_path_weights(flow)
    
    async def _cleanup_flows(self):
        """Remove inactive flows"""
        current_time = time.time()
        timeout = 300  # 5 minutes
        
        expired = []
        for flow_id, flow in self.flows.items():
            # Check last activity
            if current_time - flow.created > timeout:
                # Check if any recent traffic
                total_bytes = sum(flow.bytes_sent.values())
                if total_bytes == 0:
                    expired.append(flow_id)
        
        for flow_id in expired:
            del self.flows[flow_id]
            logger.debug(f"Cleaned up inactive flow {flow_id}")
    
    async def _ml_training_loop(self):
        """Periodically train ML predictor"""
        if not self.ml_predictor:
            return
        
        while True:
            try:
                await asyncio.sleep(self.config.ml_update_interval)
                
                # Check if enough data
                if len(self.metric_history) < 100:
                    continue
                
                # Train predictor
                await self._train_ml_predictor()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in ML training: {e}")
    
    async def _train_ml_predictor(self):
        """Train ML predictor on historical data"""
        logger.info("Training ML route predictor...")
        
        # Prepare training data
        X = []
        y = []
        
        for entry in self.metric_history:
            # Features
            features = [
                entry['hour'],
                entry['day_of_week'],
                entry['rtt'],
                entry['bandwidth'],
                entry['loss'],
                entry['jitter']
            ]
            X.append(features)
            
            # Target: composite cost
            cost = (
                entry['rtt'] / 10.0 * 0.4 +
                100.0 / max(entry['bandwidth'], 1.0) * 0.3 +
                (1.0 + entry['loss'] * 10) * 0.2 +
                0.1  # Fixed security component
            )
            y.append(cost)
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        await asyncio.get_event_loop().run_in_executor(
            None, self.ml_predictor.train, X_scaled, y
        )
        
        logger.info("ML predictor training completed")
    
    async def _predict_route_cost(self, destination: NodeID, 
                                next_hop: NodeID) -> float:
        """Predict future route cost using ML"""
        if not self.ml_predictor:
            return 1.0
        
        # Get current metrics
        metrics_key = (destination, next_hop)
        if metrics_key not in self.route_metrics:
            return 1.0
        
        metrics = self.route_metrics[metrics_key]
        
        # Prepare features
        hour = time.localtime().tm_hour
        day_of_week = time.localtime().tm_wday
        
        features = np.array([[
            hour,
            day_of_week,
            metrics.rtt_ms,
            metrics.bandwidth_mbps,
            metrics.packet_loss,
            metrics.jitter_ms
        ]])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        try:
            prediction = await asyncio.get_event_loop().run_in_executor(
                None, self.ml_predictor.predict, features_scaled
            )
            return float(prediction[0])
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return metrics.cost
    
    def _has_sufficient_history(self, destination: NodeID) -> bool:
        """Check if we have enough history for ML prediction"""
        dest_str = destination.to_base58()
        count = sum(1 for entry in self.metric_history 
                   if entry['destination'] == dest_str)
        return count >= 10
    
    async def handle_route_error(self, destination: NodeID, next_hop: NodeID,
                               error_type: str):
        """Handle route error notification"""
        logger.warning(f"Route error to {destination.to_base58()[:16]} "
                      f"via {next_hop.to_base58()[:16]}: {error_type}")
        
        # Update metrics to reflect error
        metrics_key = (destination, next_hop)
        if metrics_key in self.route_metrics:
            metrics = self.route_metrics[metrics_key]
            
            if error_type == "timeout":
                metrics.rtt_ms = min(metrics.rtt_ms * 2, 10000)
                metrics.packet_loss = min(metrics.packet_loss + 0.1, 1.0)
            elif error_type == "unreachable":
                metrics.packet_loss = 1.0
        
        # Record route change
        self.route_changes[destination].append(time.time())
        
        # Trigger immediate route recalculation
        await self.get_best_route(destination)
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get adaptive routing statistics"""
        return {
            'total_routes': len(self.route_metrics),
            'active_flows': len(self.flows),
            'ml_enabled': self.ml_predictor is not None,
            'history_size': len(self.metric_history),
            'multipath_enabled': self.config.enable_multipath,
            'average_metrics': self._calculate_average_metrics()
        }
    
    def _calculate_average_metrics(self) -> Dict[str, float]:
        """Calculate average metrics across all routes"""
        if not self.route_metrics:
            return {}
        
        total_rtt = 0.0
        total_bandwidth = 0.0
        total_loss = 0.0
        count = 0
        
        for metrics in self.route_metrics.values():
            total_rtt += metrics.rtt_ms
            total_bandwidth += metrics.bandwidth_mbps
            total_loss += metrics.packet_loss
            count += 1
        
        return {
            'avg_rtt_ms': total_rtt / count,
            'avg_bandwidth_mbps': total_bandwidth / count,
            'avg_packet_loss': total_loss / count
        }


class RoutePredictor:
    """ML predictor for route costs"""
    
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            random_state=42
        )
        self.is_trained = False
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the predictor"""
        self.model.fit(X, y)
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict route costs"""
        if not self.is_trained:
            return np.ones(X.shape[0])
        
        return self.model.predict(X)