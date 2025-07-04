"""
RouterAgent and LocalAgent Implementation
========================================

RouterAgent: Intelligent request routing with multi-strategy load balancing
LocalAgent: Ollama integration for local model hosting

Features:
- Multiple load balancing strategies (Round Robin, Least Connections, Latency-Aware)
- Consistent hashing for request distribution
- Real-time health monitoring and failover
- Ollama API integration with async support
- Automatic model lifecycle management
- Performance optimization and caching
"""

import asyncio
import logging
import time
import uuid
import json
import hashlib
import statistics
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import aiohttp
import consistent_hashing
import socket
import subprocess
import sys

# Import our distributed AI components
from distributed_ai_core import (
    AIRequest, AIResponse, NodeMetrics, LoadBalanceStrategy, 
    SemanticCache, CircuitBreaker, ShardAgent
)

logger = logging.getLogger(__name__)

# ============================================================================
# ROUTER AGENT - INTELLIGENT REQUEST ROUTING
# ============================================================================

class RouterAgent:
    """Intelligent request routing with advanced load balancing"""
    
    def __init__(self, router_id: str, config: Dict[str, Any]):
        self.router_id = router_id
        self.config = config
        self.nodes = {}  # node_id -> NodeInfo
        self.load_balancer = LoadBalancer(config.get("load_balance_strategy", LoadBalanceStrategy.LATENCY_AWARE))
        self.semantic_cache = SemanticCache(
            cache_size=config.get("cache_size", 10000),
            similarity_threshold=config.get("cache_similarity", 0.95)
        )
        self.circuit_breakers = {}  # node_id -> CircuitBreaker
        self.health_monitor = HealthMonitor(self)
        self.request_queue = asyncio.Queue()
        self.metrics = RouterMetrics()
        self.fallback_strategies = config.get("fallback_strategies", ["retry", "redirect", "local"])
        
        # Start background tasks
        self.background_tasks = []
    
    async def start(self):
        """Start the router agent"""
        logger.info(f"Starting RouterAgent {self.router_id}")
        
        # Start health monitoring
        self.background_tasks.append(
            asyncio.create_task(self.health_monitor.start())
        )
        
        # Start request processing
        self.background_tasks.append(
            asyncio.create_task(self._process_request_queue())
        )
        
        # Start metrics collection
        self.background_tasks.append(
            asyncio.create_task(self._collect_metrics())
        )
        
        logger.info(f"RouterAgent {self.router_id} started successfully")
    
    async def stop(self):
        """Stop the router agent"""
        logger.info(f"Stopping RouterAgent {self.router_id}")
        
        # Cancel all background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
    
    async def register_node(self, node_info: 'NodeInfo'):
        """Register a new node with the router"""
        self.nodes[node_info.node_id] = node_info
        self.circuit_breakers[node_info.node_id] = CircuitBreaker()
        
        logger.info(f"Registered node {node_info.node_id} with capabilities: {node_info.capabilities}")
    
    async def route_request(self, request: AIRequest) -> AIResponse:
        """Route request to appropriate node"""
        start_time = time.time()
        
        try:
            # Check semantic cache first
            cached_response = await self.semantic_cache.get(request)
            if cached_response:
                self.metrics.cache_hits += 1
                logger.info(f"Cache hit for request {request.request_id}")
                return cached_response
            
            # Select target node
            target_node = await self.load_balancer.select_node(request, self.nodes)
            if not target_node:
                raise Exception("No available nodes for request")
            
            # Route request with circuit breaker protection
            circuit_breaker = self.circuit_breakers[target_node.node_id]
            response = await circuit_breaker.call(
                self._execute_request, 
                request, 
                target_node
            )
            
            # Cache successful response
            await self.semantic_cache.set(request, response)
            
            # Update metrics
            execution_time = time.time() - start_time
            self.metrics.update_request_metrics(target_node.node_id, execution_time, True)
            
            return response
            
        except Exception as e:
            # Try fallback strategies
            response = await self._handle_request_failure(request, str(e))
            
            execution_time = time.time() - start_time
            self.metrics.update_request_metrics("fallback", execution_time, False)
            
            return response
    
    async def _execute_request(self, request: AIRequest, target_node: 'NodeInfo') -> AIResponse:
        """Execute request on target node"""
        logger.info(f"Routing request {request.request_id} to node {target_node.node_id}")
        
        # Simulate request execution
        if target_node.node_type == "shard":
            return await self._execute_shard_request(request, target_node)
        elif target_node.node_type == "local":
            return await self._execute_local_request(request, target_node)
        else:
            raise Exception(f"Unknown node type: {target_node.node_type}")
    
    async def _execute_shard_request(self, request: AIRequest, target_node: 'NodeInfo') -> AIResponse:
        """Execute request on shard node"""
        # In real implementation, this would make HTTP/gRPC call to shard node
        await asyncio.sleep(0.1)  # Simulate network latency
        
        return AIResponse(
            request_id=request.request_id,
            content=f"[SHARD:{target_node.node_id}] Response to: {request.prompt[:50]}...",
            model_name=request.model_name,
            execution_time=0.1,
            tokens_used=len(request.prompt.split()),
            metadata={"routed_to": target_node.node_id, "node_type": "shard"}
        )
    
    async def _execute_local_request(self, request: AIRequest, target_node: 'NodeInfo') -> AIResponse:
        """Execute request on local node"""
        # In real implementation, this would communicate with LocalAgent
        await asyncio.sleep(0.05)  # Simulate local execution
        
        return AIResponse(
            request_id=request.request_id,
            content=f"[LOCAL:{target_node.node_id}] Response to: {request.prompt[:50]}...",
            model_name=request.model_name,
            execution_time=0.05,
            tokens_used=len(request.prompt.split()),
            metadata={"routed_to": target_node.node_id, "node_type": "local"}
        )
    
    async def _handle_request_failure(self, request: AIRequest, error: str) -> AIResponse:
        """Handle request failure with fallback strategies"""
        logger.warning(f"Request {request.request_id} failed: {error}")
        
        for strategy in self.fallback_strategies:
            try:
                if strategy == "retry":
                    # Retry with different node
                    available_nodes = [n for n in self.nodes.values() if n.status == "healthy"]
                    if available_nodes:
                        fallback_node = available_nodes[0]  # Simple fallback selection
                        return await self._execute_request(request, fallback_node)
                
                elif strategy == "redirect":
                    # Redirect to backup router (if available)
                    # This would be implemented with actual backup router communication
                    pass
                
                elif strategy == "local":
                    # Fallback to local processing
                    return AIResponse(
                        request_id=request.request_id,
                        content=f"[FALLBACK] Basic response to: {request.prompt[:50]}...",
                        model_name=request.model_name,
                        execution_time=0.01,
                        tokens_used=len(request.prompt.split()),
                        metadata={"fallback_strategy": strategy}
                    )
                    
            except Exception as fallback_error:
                logger.warning(f"Fallback strategy {strategy} failed: {fallback_error}")
                continue
        
        # If all fallback strategies fail, return error response
        return AIResponse(
            request_id=request.request_id,
            content="",
            model_name=request.model_name,
            execution_time=0.0,
            tokens_used=0,
            error=f"Request failed: {error}"
        )
    
    async def _process_request_queue(self):
        """Process queued requests"""
        while True:
            try:
                request = await self.request_queue.get()
                response = await self.route_request(request)
                # In real implementation, this would send response back to client
                logger.info(f"Processed request {request.request_id}")
            except Exception as e:
                logger.error(f"Error processing request: {e}")
            await asyncio.sleep(0.001)  # Small delay to prevent busy waiting
    
    async def _collect_metrics(self):
        """Collect and update metrics"""
        while True:
            try:
                # Update node metrics
                for node_id, node_info in self.nodes.items():
                    metrics = await self._get_node_metrics(node_info)
                    self.metrics.update_node_metrics(node_id, metrics)
                
                # Update cache metrics
                cache_stats = self.semantic_cache.get_stats()
                self.metrics.cache_metrics = cache_stats
                
                await asyncio.sleep(10)  # Collect metrics every 10 seconds
                
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                await asyncio.sleep(5)
    
    async def _get_node_metrics(self, node_info: 'NodeInfo') -> NodeMetrics:
        """Get metrics from a node"""
        # In real implementation, this would query the actual node
        return NodeMetrics(
            node_id=node_info.node_id,
            cpu_usage=50.0 + (hash(node_info.node_id) % 40),  # Simulate metrics
            memory_usage=60.0 + (hash(node_info.node_id) % 30),
            gpu_usage=70.0 + (hash(node_info.node_id) % 25),
            active_connections=10 + (hash(node_info.node_id) % 20),
            requests_per_second=100.0 + (hash(node_info.node_id) % 50),
            average_latency=0.1 + (hash(node_info.node_id) % 10) / 100,
            error_rate=0.01 + (hash(node_info.node_id) % 5) / 1000
        )
    
    def get_router_metrics(self) -> Dict[str, Any]:
        """Get comprehensive router metrics"""
        return {
            "router_id": self.router_id,
            "registered_nodes": len(self.nodes),
            "healthy_nodes": len([n for n in self.nodes.values() if n.status == "healthy"]),
            "load_balance_strategy": self.load_balancer.strategy.value,
            "cache_metrics": self.semantic_cache.get_stats(),
            "request_metrics": self.metrics.get_request_metrics(),
            "node_metrics": self.metrics.get_node_metrics(),
            "circuit_breaker_states": {
                node_id: cb.state.value 
                for node_id, cb in self.circuit_breakers.items()
            }
        }

# ============================================================================
# LOAD BALANCER
# ============================================================================

class LoadBalancer:
    """Advanced load balancing with multiple strategies"""
    
    def __init__(self, strategy: LoadBalanceStrategy):
        self.strategy = strategy
        self.round_robin_counter = 0
        self.consistent_hash_ring = {}
        self.latency_history = defaultdict(deque)
        self.connection_counts = defaultdict(int)
    
    async def select_node(self, request: AIRequest, nodes: Dict[str, 'NodeInfo']) -> Optional['NodeInfo']:
        """Select best node based on load balancing strategy"""
        
        # Filter healthy nodes that can handle the request
        available_nodes = [
            node for node in nodes.values()
            if node.status == "healthy" and request.model_name in node.capabilities
        ]
        
        if not available_nodes:
            return None
        
        if self.strategy == LoadBalanceStrategy.ROUND_ROBIN:
            return self._round_robin_select(available_nodes)
        elif self.strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
            return self._least_connections_select(available_nodes)
        elif self.strategy == LoadBalanceStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_select(available_nodes)
        elif self.strategy == LoadBalanceStrategy.CONSISTENT_HASHING:
            return self._consistent_hashing_select(request, available_nodes)
        elif self.strategy == LoadBalanceStrategy.LATENCY_AWARE:
            return self._latency_aware_select(available_nodes)
        elif self.strategy == LoadBalanceStrategy.RESOURCE_AWARE:
            return self._resource_aware_select(available_nodes)
        else:
            return available_nodes[0]  # Default to first available
    
    def _round_robin_select(self, nodes: List['NodeInfo']) -> 'NodeInfo':
        """Round robin selection"""
        selected = nodes[self.round_robin_counter % len(nodes)]
        self.round_robin_counter += 1
        return selected
    
    def _least_connections_select(self, nodes: List['NodeInfo']) -> 'NodeInfo':
        """Select node with least active connections"""
        return min(nodes, key=lambda n: self.connection_counts[n.node_id])
    
    def _weighted_round_robin_select(self, nodes: List['NodeInfo']) -> 'NodeInfo':
        """Weighted round robin based on node capacity"""
        # Weight by compute capacity (higher capacity = more weight)
        weights = [node.compute_capacity for node in nodes]
        total_weight = sum(weights)
        
        # Select based on weight
        selection_point = (self.round_robin_counter * 100) % total_weight
        current_weight = 0
        
        for i, node in enumerate(nodes):
            current_weight += weights[i]
            if current_weight >= selection_point:
                self.round_robin_counter += 1
                return node
        
        return nodes[0]  # Fallback
    
    def _consistent_hashing_select(self, request: AIRequest, nodes: List['NodeInfo']) -> 'NodeInfo':
        """Consistent hashing for request distribution"""
        # Create hash ring if not exists
        if not self.consistent_hash_ring:
            self._build_hash_ring(nodes)
        
        # Hash the request
        request_hash = hashlib.md5(f"{request.model_name}:{request.prompt}".encode()).hexdigest()
        
        # Find node in hash ring
        # Simplified consistent hashing implementation
        node_hashes = sorted(self.consistent_hash_ring.keys())
        for node_hash in node_hashes:
            if request_hash <= node_hash:
                return self.consistent_hash_ring[node_hash]
        
        return self.consistent_hash_ring[node_hashes[0]]  # Wrap around
    
    def _latency_aware_select(self, nodes: List['NodeInfo']) -> 'NodeInfo':
        """Select node with lowest average latency"""
        best_node = None
        best_latency = float('inf')
        
        for node in nodes:
            latency_history = self.latency_history[node.node_id]
            if latency_history:
                avg_latency = statistics.mean(latency_history)
                if avg_latency < best_latency:
                    best_latency = avg_latency
                    best_node = node
        
        return best_node or nodes[0]  # Fallback to first if no history
    
    def _resource_aware_select(self, nodes: List['NodeInfo']) -> 'NodeInfo':
        """Select node with best resource availability"""
        best_node = None
        best_score = float('-inf')
        
        for node in nodes:
            # Calculate resource availability score
            cpu_availability = 100 - node.cpu_usage
            memory_availability = 100 - node.memory_usage
            gpu_availability = 100 - node.gpu_usage
            
            # Weighted score
            score = (cpu_availability * 0.3 + 
                    memory_availability * 0.4 + 
                    gpu_availability * 0.3)
            
            if score > best_score:
                best_score = score
                best_node = node
        
        return best_node or nodes[0]
    
    def _build_hash_ring(self, nodes: List['NodeInfo']):
        """Build consistent hash ring"""
        self.consistent_hash_ring = {}
        
        for node in nodes:
            # Create multiple hash points for better distribution
            for i in range(100):  # 100 virtual nodes per physical node
                hash_key = hashlib.md5(f"{node.node_id}:{i}".encode()).hexdigest()
                self.consistent_hash_ring[hash_key] = node
    
    def update_latency(self, node_id: str, latency: float):
        """Update latency history for a node"""
        history = self.latency_history[node_id]
        history.append(latency)
        
        # Keep only recent history
        if len(history) > 100:
            history.popleft()
    
    def update_connection_count(self, node_id: str, delta: int):
        """Update connection count for a node"""
        self.connection_counts[node_id] += delta
        if self.connection_counts[node_id] < 0:
            self.connection_counts[node_id] = 0

# ============================================================================
# LOCAL AGENT - OLLAMA INTEGRATION
# ============================================================================

class LocalAgent:
    """Local model hosting with Ollama integration"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.config = config
        self.ollama_base_url = config.get("ollama_url", "http://localhost:11434")
        self.loaded_models = {}
        self.model_cache = {}
        self.performance_metrics = defaultdict(list)
        self.circuit_breaker = CircuitBreaker()
        
        # Resource management
        self.max_concurrent_requests = config.get("max_concurrent", 10)
        self.request_semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        
        # Model lifecycle management
        self.model_unload_timeout = config.get("model_unload_timeout", 300)  # 5 minutes
        self.model_access_times = {}
        
        # Initialize HTTP session
        self.http_session = None
    
    async def start(self):
        """Start the local agent"""
        logger.info(f"Starting LocalAgent {self.agent_id}")
        
        # Initialize HTTP session
        self.http_session = aiohttp.ClientSession()
        
        # Check Ollama availability
        if not await self._check_ollama_health():
            logger.warning("Ollama is not available. Some features may not work.")
        
        # Start model lifecycle management
        asyncio.create_task(self._manage_model_lifecycle())
        
        logger.info(f"LocalAgent {self.agent_id} started successfully")
    
    async def stop(self):
        """Stop the local agent"""
        logger.info(f"Stopping LocalAgent {self.agent_id}")
        
        # Close HTTP session
        if self.http_session:
            await self.http_session.close()
    
    async def _check_ollama_health(self) -> bool:
        """Check if Ollama is healthy"""
        try:
            async with self.http_session.get(f"{self.ollama_base_url}/api/tags") as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False
    
    async def ensure_model_loaded(self, model_name: str) -> bool:
        """Ensure model is loaded and ready"""
        if model_name in self.loaded_models:
            self.model_access_times[model_name] = time.time()
            return True
        
        try:
            # Load model
            logger.info(f"Loading model {model_name}")
            
            # Check if model exists
            if not await self._model_exists(model_name):
                logger.error(f"Model {model_name} not found")
                return False
            
            # Pull model if not already available
            await self._pull_model(model_name)
            
            # Mark as loaded
            self.loaded_models[model_name] = {
                "loaded_at": time.time(),
                "status": "ready"
            }
            self.model_access_times[model_name] = time.time()
            
            logger.info(f"Model {model_name} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False
    
    async def _model_exists(self, model_name: str) -> bool:
        """Check if model exists in Ollama"""
        try:
            async with self.http_session.get(f"{self.ollama_base_url}/api/tags") as response:
                if response.status == 200:
                    models_data = await response.json()
                    model_names = [model['name'] for model in models_data.get('models', [])]
                    return model_name in model_names
                return False
        except Exception as e:
            logger.error(f"Error checking model existence: {e}")
            return False
    
    async def _pull_model(self, model_name: str):
        """Pull model from Ollama registry"""
        try:
            payload = {"name": model_name}
            async with self.http_session.post(
                f"{self.ollama_base_url}/api/pull", 
                json=payload
            ) as response:
                if response.status != 200:
                    raise Exception(f"Failed to pull model: {response.status}")
                
                # Stream the response to track progress
                async for line in response.content:
                    if line:
                        progress_data = json.loads(line.decode())
                        if progress_data.get('status') == 'success':
                            logger.info(f"Model {model_name} pulled successfully")
                            break
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            raise
    
    async def process_request(self, request: AIRequest) -> AIResponse:
        """Process inference request locally"""
        start_time = time.time()
        
        async with self.request_semaphore:
            try:
                # Ensure model is loaded
                if not await self.ensure_model_loaded(request.model_name):
                    raise Exception(f"Failed to load model {request.model_name}")
                
                # Execute inference through circuit breaker
                response = await self.circuit_breaker.call(
                    self._execute_ollama_inference,
                    request
                )
                
                execution_time = time.time() - start_time
                
                # Update metrics
                self.performance_metrics[request.model_name].append({
                    "execution_time": execution_time,
                    "timestamp": time.time(),
                    "tokens": len(request.prompt.split()),
                    "success": True
                })
                
                return response
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Local inference failed: {e}")
                
                # Update error metrics
                self.performance_metrics[request.model_name].append({
                    "execution_time": execution_time,
                    "timestamp": time.time(),
                    "tokens": len(request.prompt.split()),
                    "success": False,
                    "error": str(e)
                })
                
                return AIResponse(
                    request_id=request.request_id,
                    content="",
                    model_name=request.model_name,
                    execution_time=execution_time,
                    tokens_used=0,
                    error=str(e)
                )
    
    async def _execute_ollama_inference(self, request: AIRequest) -> AIResponse:
        """Execute inference using Ollama API"""
        logger.info(f"Executing local inference for {request.model_name}")
        
        # Prepare request payload
        payload = {
            "model": request.model_name,
            "prompt": request.prompt,
            "stream": False,
            "options": {
                "temperature": request.parameters.get("temperature", 0.7),
                "top_p": request.parameters.get("top_p", 0.9),
                "max_tokens": request.parameters.get("max_tokens", 100)
            }
        }
        
        try:
            async with self.http_session.post(
                f"{self.ollama_base_url}/api/generate",
                json=payload
            ) as response:
                if response.status != 200:
                    raise Exception(f"Ollama API error: {response.status}")
                
                result = await response.json()
                
                return AIResponse(
                    request_id=request.request_id,
                    content=result.get("response", ""),
                    model_name=request.model_name,
                    execution_time=time.time() - request.created_at,
                    tokens_used=result.get("eval_count", 0),
                    metadata={
                        "local_agent": self.agent_id,
                        "model_info": result.get("model", ""),
                        "eval_duration": result.get("eval_duration", 0)
                    }
                )
                
        except Exception as e:
            logger.error(f"Ollama inference failed: {e}")
            raise
    
    async def _manage_model_lifecycle(self):
        """Manage model loading and unloading"""
        while True:
            try:
                current_time = time.time()
                models_to_unload = []
                
                # Check for models that haven't been accessed recently
                for model_name, last_access in self.model_access_times.items():
                    if current_time - last_access > self.model_unload_timeout:
                        models_to_unload.append(model_name)
                
                # Unload old models
                for model_name in models_to_unload:
                    await self._unload_model(model_name)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in model lifecycle management: {e}")
                await asyncio.sleep(30)
    
    async def _unload_model(self, model_name: str):
        """Unload model from memory"""
        try:
            logger.info(f"Unloading model {model_name}")
            
            # Remove from loaded models
            if model_name in self.loaded_models:
                del self.loaded_models[model_name]
            
            if model_name in self.model_access_times:
                del self.model_access_times[model_name]
            
            # In a real implementation, you might call an Ollama API to unload
            # For now, we just remove from our tracking
            
            logger.info(f"Model {model_name} unloaded successfully")
            
        except Exception as e:
            logger.error(f"Error unloading model {model_name}: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get local agent metrics"""
        return {
            "agent_id": self.agent_id,
            "loaded_models": list(self.loaded_models.keys()),
            "model_count": len(self.loaded_models),
            "performance_metrics": dict(self.performance_metrics),
            "circuit_breaker_state": self.circuit_breaker.state.value,
            "resource_usage": {
                "max_concurrent": self.max_concurrent_requests,
                "current_requests": self.max_concurrent_requests - self.request_semaphore._value
            }
        }

# ============================================================================
# SUPPORTING CLASSES
# ============================================================================

@dataclass
class NodeInfo:
    """Information about a node in the network"""
    node_id: str
    node_type: str  # "shard", "local", "router"
    endpoint: str
    capabilities: List[str]  # List of supported models
    status: str = "healthy"
    compute_capacity: float = 1.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0
    last_heartbeat: float = field(default_factory=time.time)

class HealthMonitor:
    """Monitor health of nodes in the network"""
    
    def __init__(self, router_agent: RouterAgent):
        self.router_agent = router_agent
        self.health_check_interval = 30  # seconds
    
    async def start(self):
        """Start health monitoring"""
        while True:
            try:
                await self._check_all_nodes()
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def _check_all_nodes(self):
        """Check health of all registered nodes"""
        for node_id, node_info in self.router_agent.nodes.items():
            try:
                is_healthy = await self._check_node_health(node_info)
                node_info.status = "healthy" if is_healthy else "unhealthy"
                node_info.last_heartbeat = time.time()
            except Exception as e:
                logger.warning(f"Health check failed for node {node_id}: {e}")
                node_info.status = "unhealthy"
    
    async def _check_node_health(self, node_info: NodeInfo) -> bool:
        """Check health of a specific node"""
        try:
            # In real implementation, this would make actual health check requests
            # For now, simulate with random health status
            import random
            return random.random() > 0.1  # 90% uptime simulation
        except Exception:
            return False

class RouterMetrics:
    """Metrics collection for router"""
    
    def __init__(self):
        self.request_counts = defaultdict(int)
        self.request_latencies = defaultdict(list)
        self.error_counts = defaultdict(int)
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_metrics = {}
        self.node_metrics = {}
    
    def update_request_metrics(self, node_id: str, latency: float, success: bool):
        """Update request metrics"""
        self.request_counts[node_id] += 1
        self.request_latencies[node_id].append(latency)
        
        if not success:
            self.error_counts[node_id] += 1
        
        # Keep only recent latencies
        if len(self.request_latencies[node_id]) > 1000:
            self.request_latencies[node_id] = self.request_latencies[node_id][-1000:]
    
    def update_node_metrics(self, node_id: str, metrics: NodeMetrics):
        """Update node metrics"""
        self.node_metrics[node_id] = metrics
    
    def get_request_metrics(self) -> Dict[str, Any]:
        """Get request metrics summary"""
        return {
            "total_requests": sum(self.request_counts.values()),
            "request_counts_by_node": dict(self.request_counts),
            "average_latencies": {
                node_id: statistics.mean(latencies) if latencies else 0
                for node_id, latencies in self.request_latencies.items()
            },
            "error_counts": dict(self.error_counts),
            "cache_hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        }
    
    def get_node_metrics(self) -> Dict[str, Any]:
        """Get node metrics summary"""
        return {node_id: {
            "cpu_usage": metrics.cpu_usage,
            "memory_usage": metrics.memory_usage,
            "gpu_usage": metrics.gpu_usage,
            "active_connections": metrics.active_connections,
            "requests_per_second": metrics.requests_per_second,
            "average_latency": metrics.average_latency,
            "error_rate": metrics.error_rate
        } for node_id, metrics in self.node_metrics.items()}

# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

async def test_router_and_local_agents():
    """Test RouterAgent and LocalAgent integration"""
    
    # Create RouterAgent
    router_config = {
        "load_balance_strategy": LoadBalanceStrategy.LATENCY_AWARE,
        "cache_size": 1000,
        "cache_similarity": 0.95,
        "fallback_strategies": ["retry", "local"]
    }
    
    router = RouterAgent("main_router", router_config)
    await router.start()
    
    # Create LocalAgent
    local_config = {
        "ollama_url": "http://localhost:11434",
        "max_concurrent": 5,
        "model_unload_timeout": 300
    }
    
    local_agent = LocalAgent("local_001", local_config)
    await local_agent.start()
    
    # Register nodes with router
    shard_node = NodeInfo(
        node_id="shard_001",
        node_type="shard",
        endpoint="http://shard001:8000",
        capabilities=["llama2-7b", "llama2-13b"]
    )
    
    local_node = NodeInfo(
        node_id="local_001",
        node_type="local",
        endpoint="http://localhost:8001",
        capabilities=["llama2-7b", "codellama"]
    )
    
    await router.register_node(shard_node)
    await router.register_node(local_node)
    
    # Test inference requests
    requests = [
        AIRequest(
            request_id=str(uuid.uuid4()),
            model_name="llama2-7b",
            prompt="What is artificial intelligence?",
            parameters={"max_tokens": 100}
        ),
        AIRequest(
            request_id=str(uuid.uuid4()),
            model_name="codellama",
            prompt="Write a Python function to calculate fibonacci numbers",
            parameters={"max_tokens": 200}
        )
    ]
    
    # Process requests
    for request in requests:
        response = await router.route_request(request)
        print(f"Request {request.request_id}: {response.content[:100]}...")
    
    # Test local agent directly
    local_request = AIRequest(
        request_id=str(uuid.uuid4()),
        model_name="llama2-7b",
        prompt="Explain quantum computing",
        parameters={"max_tokens": 150}
    )
    
    local_response = await local_agent.process_request(local_request)
    print(f"Local response: {local_response.content[:100]}...")
    
    # Get metrics
    router_metrics = router.get_router_metrics()
    local_metrics = local_agent.get_metrics()
    
    print(f"Router metrics: {json.dumps(router_metrics, indent=2)}")
    print(f"Local agent metrics: {json.dumps(local_metrics, indent=2)}")
    
    # Cleanup
    await router.stop()
    await local_agent.stop()

if __name__ == "__main__":
    asyncio.run(test_router_and_local_agents())
