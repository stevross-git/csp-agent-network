"""
CSP Integration Layer & Configuration Management
===============================================

Seamless integration with CSP Agent Network and comprehensive configuration management.

Features:
- Channel-based communication with CSP network
- Temporal consistency and formal verification
- YAML-based configuration management
- Environment-specific configurations
- Dynamic configuration updates
- Monitoring and observability integration
"""

import asyncio
import logging
import yaml
import json
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum
import time
from datetime import datetime
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry

# CSP and distributed AI imports
from core.advanced_csp_core import AdvancedCSPEngine, Channel, Event, Process, ProcessContext
from ai_integration.csp_ai_integration import AIAgent, CollaborativeAIProcess
from distributed_ai_core import ShardAgent, AIRequest, AIResponse, ModelShard, ShardingStrategy
from router_local_agents import RouterAgent, LocalAgent, NodeInfo, LoadBalanceStrategy

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================

class DeploymentEnvironment(Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

@dataclass
class NetworkConfig:
    """Network configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    max_connections: int = 1000
    connection_timeout: float = 30.0
    keepalive_timeout: float = 60.0
    enable_tls: bool = True
    tls_cert_file: Optional[str] = None
    tls_key_file: Optional[str] = None

@dataclass
class ShardConfig:
    """Shard configuration"""
    strategy: ShardingStrategy = ShardingStrategy.TENSOR_PARALLEL
    num_shards: int = 4
    shard_rank: int = 0
    device_mapping: Dict[str, str] = field(default_factory=dict)
    memory_limit_gb: float = 16.0
    enable_zero_copy: bool = True
    enable_flash_attention: bool = True
    quantization_type: Optional[str] = None

@dataclass
class RouterConfig:
    """Router configuration"""
    load_balance_strategy: LoadBalanceStrategy = LoadBalanceStrategy.LATENCY_AWARE
    cache_size: int = 10000
    cache_similarity_threshold: float = 0.95
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    fallback_strategies: List[str] = field(default_factory=lambda: ["retry", "redirect", "local"])
    health_check_interval: float = 30.0

@dataclass
class LocalAgentConfig:
    """Local agent configuration"""
    ollama_url: str = "http://localhost:11434"
    max_concurrent_requests: int = 10
    model_unload_timeout: float = 300.0
    auto_pull_models: bool = True
    model_cache_size: int = 5
    enable_gpu: bool = True

@dataclass
class CSPConfig:
    """CSP engine configuration"""
    namespace: str = "distributed_ai"
    enable_quantum_channels: bool = True
    enable_temporal_consistency: bool = True
    enable_formal_verification: bool = True
    max_processes: int = 1000
    process_timeout: float = 300.0
    channel_buffer_size: int = 1000

@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    enable_prometheus: bool = True
    prometheus_port: int = 9090
    metrics_collection_interval: float = 10.0
    enable_distributed_tracing: bool = True
    jaeger_endpoint: Optional[str] = None
    log_level: str = "INFO"

@dataclass
class SecurityConfig:
    """Security configuration"""
    enable_authentication: bool = True
    api_key_header: str = "X-API-Key"
    allowed_origins: List[str] = field(default_factory=list)
    rate_limit_requests: int = 1000
    rate_limit_window: int = 3600
    enable_encryption: bool = True
    encryption_key: Optional[str] = None

@dataclass
class DistributedAIConfig:
    """Main distributed AI configuration"""
    environment: DeploymentEnvironment = DeploymentEnvironment.DEVELOPMENT
    cluster_name: str = "distributed_ai_cluster"
    node_id: str = "node_001"
    
    # Component configurations
    network: NetworkConfig = field(default_factory=NetworkConfig)
    shard: ShardConfig = field(default_factory=ShardConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    local_agent: LocalAgentConfig = field(default_factory=LocalAgentConfig)
    csp: CSPConfig = field(default_factory=CSPConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # Model configurations
    models: Dict[str, Any] = field(default_factory=dict)
    
    # Additional settings
    debug: bool = False
    auto_scaling: bool = True
    backup_enabled: bool = True
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'DistributedAIConfig':
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return cls.from_dict(config_data)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DistributedAIConfig':
        """Create configuration from dictionary"""
        # Handle nested configurations
        config_dict = config_dict.copy()
        
        # Convert string enums
        if 'environment' in config_dict:
            config_dict['environment'] = DeploymentEnvironment(config_dict['environment'])
        
        # Handle nested dataclasses
        if 'network' in config_dict:
            config_dict['network'] = NetworkConfig(**config_dict['network'])
        
        if 'shard' in config_dict:
            shard_config = config_dict['shard']
            if 'strategy' in shard_config:
                shard_config['strategy'] = ShardingStrategy(shard_config['strategy'])
            config_dict['shard'] = ShardConfig(**shard_config)
        
        if 'router' in config_dict:
            router_config = config_dict['router']
            if 'load_balance_strategy' in router_config:
                router_config['load_balance_strategy'] = LoadBalanceStrategy(router_config['load_balance_strategy'])
            config_dict['router'] = RouterConfig(**router_config)
        
        if 'local_agent' in config_dict:
            config_dict['local_agent'] = LocalAgentConfig(**config_dict['local_agent'])
        
        if 'csp' in config_dict:
            config_dict['csp'] = CSPConfig(**config_dict['csp'])
        
        if 'monitoring' in config_dict:
            config_dict['monitoring'] = MonitoringConfig(**config_dict['monitoring'])
        
        if 'security' in config_dict:
            config_dict['security'] = SecurityConfig(**config_dict['security'])
        
        return cls(**config_dict)
    
    def to_yaml(self, output_path: str):
        """Save configuration to YAML file"""
        config_dict = self.to_dict()
        
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        config_dict = asdict(self)
        
        # Convert enums to strings
        config_dict['environment'] = self.environment.value
        config_dict['shard']['strategy'] = self.shard.strategy.value
        config_dict['router']['load_balance_strategy'] = self.router.load_balance_strategy.value
        
        return config_dict
    
    def get_config_hash(self) -> str:
        """Get hash of configuration for change detection"""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()

# ============================================================================
# CONFIGURATION MANAGER
# ============================================================================

class ConfigurationManager:
    """Manages configuration loading, validation, and updates"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_cache = {}
        self.config_watchers = {}
        self.config_change_callbacks = []
        
        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def load_config(self, environment: DeploymentEnvironment = DeploymentEnvironment.DEVELOPMENT) -> DistributedAIConfig:
        """Load configuration for specific environment"""
        config_file = self.config_dir / f"{environment.value}.yaml"
        
        if not config_file.exists():
            logger.warning(f"Config file {config_file} not found, creating default")
            self.create_default_config(environment)
        
        try:
            config = DistributedAIConfig.from_yaml(str(config_file))
            self.config_cache[environment] = config
            
            # Start watching for changes
            self.start_config_watcher(environment)
            
            logger.info(f"Loaded configuration for {environment.value}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load config for {environment.value}: {e}")
            raise
    
    def create_default_config(self, environment: DeploymentEnvironment):
        """Create default configuration file"""
        config = DistributedAIConfig(environment=environment)
        
        # Environment-specific defaults
        if environment == DeploymentEnvironment.PRODUCTION:
            config.network.host = "0.0.0.0"
            config.network.port = 8000
            config.network.enable_tls = True
            config.security.enable_authentication = True
            config.monitoring.enable_prometheus = True
            config.debug = False
            config.auto_scaling = True
            
        elif environment == DeploymentEnvironment.DEVELOPMENT:
            config.network.host = "127.0.0.1"
            config.network.port = 8001
            config.network.enable_tls = False
            config.security.enable_authentication = False
            config.debug = True
            config.auto_scaling = False
            
        elif environment == DeploymentEnvironment.TESTING:
            config.network.port = 8002
            config.debug = True
            config.monitoring.enable_prometheus = False
        
        # Save default config
        config_file = self.config_dir / f"{environment.value}.yaml"
        config.to_yaml(str(config_file))
        
        logger.info(f"Created default configuration for {environment.value}")
    
    def start_config_watcher(self, environment: DeploymentEnvironment):
        """Start watching configuration file for changes"""
        if environment in self.config_watchers:
            return
        
        config_file = self.config_dir / f"{environment.value}.yaml"
        
        # Simple file watcher implementation
        def watch_config():
            last_modified = config_file.stat().st_mtime
            
            while True:
                try:
                    current_modified = config_file.stat().st_mtime
                    if current_modified != last_modified:
                        logger.info(f"Configuration file changed: {config_file}")
                        self.reload_config(environment)
                        last_modified = current_modified
                    
                    time.sleep(5)  # Check every 5 seconds
                    
                except Exception as e:
                    logger.error(f"Error watching config file: {e}")
                    time.sleep(10)
        
        watcher_thread = threading.Thread(target=watch_config, daemon=True)
        watcher_thread.start()
        
        self.config_watchers[environment] = watcher_thread
    
    def reload_config(self, environment: DeploymentEnvironment):
        """Reload configuration from file"""
        try:
            old_config = self.config_cache.get(environment)
            new_config = self.load_config(environment)
            
            # Notify callbacks about config change
            for callback in self.config_change_callbacks:
                try:
                    callback(environment, old_config, new_config)
                except Exception as e:
                    logger.error(f"Error in config change callback: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to reload config for {environment.value}: {e}")
    
    def add_config_change_callback(self, callback):
        """Add callback for configuration changes"""
        self.config_change_callbacks.append(callback)
    
    def validate_config(self, config: DistributedAIConfig) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        # Network validation
        if config.network.port < 1024 or config.network.port > 65535:
            errors.append("Network port must be between 1024 and 65535")
        
        # Shard validation
        if config.shard.num_shards < 1:
            errors.append("Number of shards must be at least 1")
        
        if config.shard.shard_rank >= config.shard.num_shards:
            errors.append("Shard rank must be less than number of shards")
        
        # Router validation
        if config.router.cache_size < 0:
            errors.append("Cache size must be non-negative")
        
        if not (0.0 <= config.router.cache_similarity_threshold <= 1.0):
            errors.append("Cache similarity threshold must be between 0.0 and 1.0")
        
        # Security validation
        if config.security.enable_authentication and not config.security.api_key_header:
            errors.append("API key header required when authentication is enabled")
        
        return errors

# ============================================================================
# CSP INTEGRATION LAYER
# ============================================================================

class CSPIntegrationLayer:
    """Seamless integration with CSP Agent Network"""
    
    def __init__(self, config: DistributedAIConfig):
        self.config = config
        self.csp_engine = AdvancedCSPEngine()
        self.channels = {}
        self.processes = {}
        self.event_handlers = {}
        self.metrics_collector = None
        
        # Initialize components
        self.shard_agents = {}
        self.router_agents = {}
        self.local_agents = {}
        
        # CSP-specific channels
        self.setup_csp_channels()
    
    def setup_csp_channels(self):
        """Setup CSP channels for distributed AI communication"""
        namespace = self.config.csp.namespace
        
        # Create semantic channels for AI communication
        self.channels["ai_requests"] = self.csp_engine.create_channel(
            f"{namespace}_ai_requests",
            channel_type="semantic"
        )
        
        self.channels["ai_responses"] = self.csp_engine.create_channel(
            f"{namespace}_ai_responses",
            channel_type="semantic"
        )
        
        # Create synchronous channels for coordination
        self.channels["coordination"] = self.csp_engine.create_channel(
            f"{namespace}_coordination",
            channel_type="synchronous"
        )
        
        # Create temporal channels for consistency
        if self.config.csp.enable_temporal_consistency:
            self.channels["temporal_sync"] = self.csp_engine.create_channel(
                f"{namespace}_temporal_sync",
                channel_type="temporal"
            )
        
        # Create quantum channels for advanced features
        if self.config.csp.enable_quantum_channels:
            self.channels["quantum_comm"] = self.csp_engine.create_channel(
                f"{namespace}_quantum_comm",
                channel_type="quantum"
            )
        
        logger.info(f"Created {len(self.channels)} CSP channels")
    
    async def start(self):
        """Start the CSP integration layer"""
        logger.info("Starting CSP Integration Layer")
        
        # Start CSP engine
        await self.csp_engine.start()
        
        # Initialize and start components
        await self.initialize_components()
        
        # Start event processing
        await self.start_event_processing()
        
        # Start metrics collection
        if self.config.monitoring.enable_prometheus:
            await self.start_metrics_collection()
        
        logger.info("CSP Integration Layer started successfully")
    
    async def stop(self):
        """Stop the CSP integration layer"""
        logger.info("Stopping CSP Integration Layer")
        
        # Stop components
        await self.stop_components()
        
        # Stop CSP engine
        await self.csp_engine.stop()
        
        logger.info("CSP Integration Layer stopped")
    
    async def initialize_components(self):
        """Initialize distributed AI components"""
        node_id = self.config.node_id
        
        # Initialize ShardAgent if configured
        if hasattr(self.config, 'shard') and self.config.shard.num_shards > 1:
            shard_agent = ShardAgent(
                node_id=f"{node_id}_shard",
                config=asdict(self.config.shard),
                csp_engine=self.csp_engine
            )
            
            # Create model shards based on configuration
            for model_name, model_config in self.config.models.items():
                shard = ModelShard(
                    shard_id=f"{model_name}_shard_{self.config.shard.shard_rank}",
                    model_name=model_name,
                    shard_rank=self.config.shard.shard_rank,
                    total_shards=self.config.shard.num_shards,
                    strategy=self.config.shard.strategy,
                    device_id=self.config.shard.device_mapping.get(model_name, "cuda:0"),
                    memory_usage=model_config.get('memory_usage', 8.0)
                )
                
                await shard_agent.initialize_shard(shard)
            
            self.shard_agents[node_id] = shard_agent
        
        # Initialize RouterAgent
        router_agent = RouterAgent(
            router_id=f"{node_id}_router",
            config=asdict(self.config.router)
        )
        await router_agent.start()
        self.router_agents[node_id] = router_agent
        
        # Initialize LocalAgent
        local_agent = LocalAgent(
            agent_id=f"{node_id}_local",
            config=asdict(self.config.local_agent)
        )
        await local_agent.start()
        self.local_agents[node_id] = local_agent
        
        # Register nodes with router
        await self.register_nodes_with_router()
        
        logger.info(f"Initialized {len(self.shard_agents)} shard agents, {len(self.router_agents)} router agents, {len(self.local_agents)} local agents")
    
    async def register_nodes_with_router(self):
        """Register all nodes with the router"""
        node_id = self.config.node_id
        router = self.router_agents[node_id]
        
        # Register shard nodes
        for shard_id, shard_agent in self.shard_agents.items():
            node_info = NodeInfo(
                node_id=shard_id,
                node_type="shard",
                endpoint=f"http://{self.config.network.host}:{self.config.network.port}",
                capabilities=list(self.config.models.keys())
            )
            await router.register_node(node_info)
        
        # Register local nodes
        for local_id, local_agent in self.local_agents.items():
            node_info = NodeInfo(
                node_id=local_id,
                node_type="local",
                endpoint=f"http://{self.config.network.host}:{self.config.network.port + 1}",
                capabilities=list(self.config.models.keys())
            )
            await router.register_node(node_info)
    
    async def start_event_processing(self):
        """Start processing CSP events"""
        # Register event handlers
        self.event_handlers["ai_request"] = self.handle_ai_request
        self.event_handlers["ai_response"] = self.handle_ai_response
        self.event_handlers["coordination"] = self.handle_coordination
        
        # Start event processing loops
        for channel_name, channel in self.channels.items():
            asyncio.create_task(self.process_channel_events(channel_name, channel))
    
    async def process_channel_events(self, channel_name: str, channel: Channel):
        """Process events from a specific channel"""
        logger.info(f"Starting event processing for channel: {channel_name}")
        
        while True:
            try:
                # Wait for events on this channel
                event = await channel.receive(self.config.node_id)
                
                # Handle event based on type
                handler = self.event_handlers.get(event.event_type)
                if handler:
                    await handler(event)
                else:
                    logger.warning(f"No handler for event type: {event.event_type}")
                    
            except Exception as e:
                logger.error(f"Error processing event on channel {channel_name}: {e}")
                await asyncio.sleep(1)
    
    async def handle_ai_request(self, event: Event):
        """Handle AI request event"""
        try:
            request_data = event.data
            request = AIRequest(**request_data)
            
            # Route through the router
            router = list(self.router_agents.values())[0]
            response = await router.route_request(request)
            
            # Send response back through CSP
            response_event = Event(
                event_type="ai_response",
                channel_id=self.channels["ai_responses"].channel_id,
                data=asdict(response)
            )
            
            await self.channels["ai_responses"].send(response_event, self.config.node_id)
            
        except Exception as e:
            logger.error(f"Error handling AI request: {e}")
    
    async def handle_ai_response(self, event: Event):
        """Handle AI response event"""
        try:
            response_data = event.data
            response = AIResponse(**response_data)
            
            # Process response (could be forwarded to client, cached, etc.)
            logger.info(f"Received AI response for request {response.request_id}")
            
        except Exception as e:
            logger.error(f"Error handling AI response: {e}")
    
    async def handle_coordination(self, event: Event):
        """Handle coordination event"""
        try:
            coordination_data = event.data
            coordination_type = coordination_data.get("type")
            
            if coordination_type == "node_health_check":
                await self.handle_health_check(coordination_data)
            elif coordination_type == "load_balancing_update":
                await self.handle_load_balancing_update(coordination_data)
            elif coordination_type == "model_deployment":
                await self.handle_model_deployment(coordination_data)
            
        except Exception as e:
            logger.error(f"Error handling coordination event: {e}")
    
    async def handle_health_check(self, data: Dict[str, Any]):
        """Handle health check coordination"""
        node_id = data.get("node_id")
        
        # Get health status from appropriate component
        health_status = {
            "node_id": self.config.node_id,
            "timestamp": time.time(),
            "healthy": True,
            "components": {}
        }
        
        # Check shard agents
        for shard_id, shard_agent in self.shard_agents.items():
            metrics = shard_agent.get_metrics()
            health_status["components"][shard_id] = {
                "type": "shard",
                "healthy": metrics.get("circuit_breaker_state") != "open",
                "metrics": metrics
            }
        
        # Check router agents
        for router_id, router_agent in self.router_agents.items():
            metrics = router_agent.get_router_metrics()
            health_status["components"][router_id] = {
                "type": "router",
                "healthy": metrics.get("healthy_nodes", 0) > 0,
                "metrics": metrics
            }
        
        # Check local agents
        for local_id, local_agent in self.local_agents.items():
            metrics = local_agent.get_metrics()
            health_status["components"][local_id] = {
                "type": "local",
                "healthy": metrics.get("circuit_breaker_state") != "open",
                "metrics": metrics
            }
        
        # Send health status back
        health_event = Event(
            event_type="health_status",
            channel_id=self.channels["coordination"].channel_id,
            data=health_status
        )
        
        await self.channels["coordination"].send(health_event, self.config.node_id)
    
    async def handle_load_balancing_update(self, data: Dict[str, Any]):
        """Handle load balancing update"""
        # Update load balancing configuration
        new_strategy = data.get("strategy")
        if new_strategy:
            for router_agent in self.router_agents.values():
                router_agent.load_balancer.strategy = LoadBalanceStrategy(new_strategy)
                logger.info(f"Updated load balancing strategy to {new_strategy}")
    
    async def handle_model_deployment(self, data: Dict[str, Any]):
        """Handle model deployment request"""
        model_name = data.get("model_name")
        deployment_config = data.get("config", {})
        
        # Deploy model to appropriate agents
        for shard_agent in self.shard_agents.values():
            # Create new shard for the model
            shard = ModelShard(
                shard_id=f"{model_name}_shard_{int(time.time())}",
                model_name=model_name,
                shard_rank=0,
                total_shards=1,
                strategy=ShardingStrategy.TENSOR_PARALLEL,
                device_id="cuda:0",
                memory_usage=deployment_config.get("memory_usage", 8.0)
            )
            
            success = await shard_agent.initialize_shard(shard)
            if success:
                logger.info(f"Successfully deployed model {model_name}")
            else:
                logger.error(f"Failed to deploy model {model_name}")
    
    async def start_metrics_collection(self):
        """Start Prometheus metrics collection"""
        if not self.config.monitoring.enable_prometheus:
            return
        
        # Create metrics collector
        self.metrics_collector = PrometheusMetricsCollector(
            self.config.monitoring.prometheus_port,
            self.config.monitoring.metrics_collection_interval
        )
        
        # Register metrics sources
        for shard_agent in self.shard_agents.values():
            self.metrics_collector.register_source("shard", shard_agent.get_metrics)
        
        for router_agent in self.router_agents.values():
            self.metrics_collector.register_source("router", router_agent.get_router_metrics)
        
        for local_agent in self.local_agents.values():
            self.metrics_collector.register_source("local", local_agent.get_metrics)
        
        # Start metrics collection
        await self.metrics_collector.start()
        
        logger.info(f"Started Prometheus metrics collection on port {self.config.monitoring.prometheus_port}")
    
    async def stop_components(self):
        """Stop all components"""
        # Stop router agents
        for router_agent in self.router_agents.values():
            await router_agent.stop()
        
        # Stop local agents
        for local_agent in self.local_agents.values():
            await local_agent.stop()
        
        # Stop metrics collector
        if self.metrics_collector:
            await self.metrics_collector.stop()
    
    async def process_distributed_request(self, request: AIRequest) -> AIResponse:
        """Process a distributed AI request through CSP"""
        # Send request through CSP channel
        request_event = Event(
            event_type="ai_request",
            channel_id=self.channels["ai_requests"].channel_id,
            data=asdict(request)
        )
        
        await self.channels["ai_requests"].send(request_event, self.config.node_id)
        
        # Wait for response (in real implementation, this would be more sophisticated)
        # For now, route directly through router
        router = list(self.router_agents.values())[0]
        response = await router.route_request(request)
        
        return response
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        metrics = {
            "node_id": self.config.node_id,
            "cluster_name": self.config.cluster_name,
            "environment": self.config.environment.value,
            "timestamp": time.time(),
            "components": {
                "shard_agents": {},
                "router_agents": {},
                "local_agents": {}
            },
            "csp_channels": {
                name: {
                    "channel_id": channel.channel_id,
                    "channel_type": getattr(channel, 'channel_type', 'unknown')
                }
                for name, channel in self.channels.items()
            }
        }
        
        # Collect metrics from all components
        for shard_id, shard_agent in self.shard_agents.items():
            metrics["components"]["shard_agents"][shard_id] = shard_agent.get_metrics()
        
        for router_id, router_agent in self.router_agents.items():
            metrics["components"]["router_agents"][router_id] = router_agent.get_router_metrics()
        
        for local_id, local_agent in self.local_agents.items():
            metrics["components"]["local_agents"][local_id] = local_agent.get_metrics()
        
        return metrics

# ============================================================================
# PROMETHEUS METRICS COLLECTOR
# ============================================================================

class PrometheusMetricsCollector:
    """Prometheus metrics collection and export"""
    
    def __init__(self, port: int, collection_interval: float):
        self.port = port
        self.collection_interval = collection_interval
        self.registry = CollectorRegistry()
        self.metrics_sources = {}
        self.running = False
        
        # Initialize metrics
        self.setup_metrics()
    
    def setup_metrics(self):
        """Setup Prometheus metrics"""
        self.request_counter = Counter(
            'distributed_ai_requests_total',
            'Total number of AI requests',
            ['component', 'model', 'status'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'distributed_ai_request_duration_seconds',
            'AI request duration',
            ['component', 'model'],
            registry=self.registry
        )
        
        self.active_connections = Gauge(
            'distributed_ai_active_connections',
            'Number of active connections',
            ['component'],
            registry=self.registry
        )
        
        self.resource_usage = Gauge(
            'distributed_ai_resource_usage_percent',
            'Resource usage percentage',
            ['component', 'resource'],
            registry=self.registry
        )
        
        self.cache_hit_rate = Gauge(
            'distributed_ai_cache_hit_rate',
            'Cache hit rate',
            ['component'],
            registry=self.registry
        )
    
    def register_source(self, component_type: str, metrics_func):
        """Register a metrics source"""
        self.metrics_sources[component_type] = metrics_func
    
    async def start(self):
        """Start metrics collection"""
        self.running = True
        
        # Start Prometheus HTTP server
        prometheus_client.start_http_server(self.port, registry=self.registry)
        
        # Start metrics collection loop
        asyncio.create_task(self.collect_metrics_loop())
    
    async def stop(self):
        """Stop metrics collection"""
        self.running = False
    
    async def collect_metrics_loop(self):
        """Main metrics collection loop"""
        while self.running:
            try:
                await self.collect_metrics()
                await asyncio.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def collect_metrics(self):
        """Collect metrics from all sources"""
        for component_type, metrics_func in self.metrics_sources.items():
            try:
                metrics = metrics_func()
                self.update_prometheus_metrics(component_type, metrics)
            except Exception as e:
                logger.error(f"Error collecting metrics from {component_type}: {e}")
    
    def update_prometheus_metrics(self, component_type: str, metrics: Dict[str, Any]):
        """Update Prometheus metrics with collected data"""
        # Update request metrics
        if 'performance_metrics' in metrics:
            for model_name, model_metrics in metrics['performance_metrics'].items():
                for metric in model_metrics:
                    if metric.get('success', True):
                        self.request_counter.labels(
                            component=component_type,
                            model=model_name,
                            status='success'
                        ).inc()
                        
                        self.request_duration.labels(
                            component=component_type,
                            model=model_name
                        ).observe(metric.get('execution_time', 0))
                    else:
                        self.request_counter.labels(
                            component=component_type,
                            model=model_name,
                            status='error'
                        ).inc()
        
        # Update resource usage
        if 'resource_usage' in metrics:
            resource_usage = metrics['resource_usage']
            for resource, usage in resource_usage.items():
                self.resource_usage.labels(
                    component=component_type,
                    resource=resource
                ).set(usage)
        
        # Update cache metrics
        if 'cache_metrics' in metrics:
            cache_metrics = metrics['cache_metrics']
            hit_rate = cache_metrics.get('hit_rate', 0)
            self.cache_hit_rate.labels(component=component_type).set(hit_rate)
        
        # Update active connections
        if 'active_connections' in metrics:
            self.active_connections.labels(component=component_type).set(
                metrics['active_connections']
            )

# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

async def test_csp_integration():
    """Test the CSP integration layer"""
    
    # Create configuration
    config = DistributedAIConfig(
        environment=DeploymentEnvironment.DEVELOPMENT,
        node_id="test_node_001",
        cluster_name="test_cluster"
    )
    
    # Add some test models
    config.models = {
        "llama2-7b": {
            "memory_usage": 8.0,
            "quantization": "int8"
        },
        "codellama": {
            "memory_usage": 12.0,
            "quantization": "fp8"
        }
    }
    
    # Create and start CSP integration layer
    csp_layer = CSPIntegrationLayer(config)
    await csp_layer.start()
    
    # Test distributed request processing
    request = AIRequest(
        request_id="test_request_001",
        model_name="llama2-7b",
        prompt="What is the future of distributed AI?",
        parameters={"max_tokens": 100}
    )
    
    response = await csp_layer.process_distributed_request(request)
    print(f"Response: {response.content}")
    
    # Get system metrics
    metrics = csp_layer.get_system_metrics()
    print(f"System metrics: {json.dumps(metrics, indent=2, default=str)}")
    
    # Test configuration management
    config_manager = ConfigurationManager()
    
    # Create and load configuration
    test_config = config_manager.load_config(DeploymentEnvironment.DEVELOPMENT)
    print(f"Loaded config: {test_config.cluster_name}")
    
    # Validate configuration
    errors = config_manager.validate_config(test_config)
    if errors:
        print(f"Configuration errors: {errors}")
    else:
        print("Configuration is valid")
    
    # Stop the system
    await csp_layer.stop()

if __name__ == "__main__":
    asyncio.run(test_csp_integration())