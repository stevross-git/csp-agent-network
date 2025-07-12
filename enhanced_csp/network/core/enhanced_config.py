# enhanced_csp/network/core/enhanced_config.py
"""
Enhanced configuration classes for the optimized CSP network.
Extends the base configuration with performance optimization settings.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path

# Import base configurations
try:
    from .config import P2PConfig, NetworkConfig
except ImportError:
    # Fallback for direct execution
    from config import P2PConfig, NetworkConfig


@dataclass 
class QUICConfig:
    """QUIC-specific configuration options"""
    enable_0rtt: bool = True                    # Enable 0-RTT for faster reconnections
    enable_connection_migration: bool = True    # Enable connection migration
    congestion_control: str = "bbr"            # BBR, cubic, or reno
    initial_rtt_ms: int = 100                  # Initial RTT estimate
    max_stream_data: int = 1024 * 1024         # 1MB per stream
    max_connection_data: int = 10 * 1024 * 1024 # 10MB per connection
    idle_timeout_ms: int = 30000               # 30 second idle timeout
    keep_alive_interval_ms: int = 5000         # 5 second keep-alive


@dataclass
class BatchConfig:
    """Intelligent batching configuration"""
    max_batch_size: int = 50                   # Maximum messages per batch
    max_wait_time_ms: int = 10                 # Maximum wait time for batching
    max_batch_bytes: int = 1024 * 1024         # 1MB maximum batch size
    enable_priority_bypass: bool = True        # Allow high priority to bypass batching
    adaptive_sizing: bool = True               # Adapt batch size to network conditions
    deadline_enforcement: bool = True          # Enforce message deadlines
    enable_compression: bool = True            # Compress large batches


@dataclass
class CompressionConfig:
    """Adaptive compression configuration"""
    min_compress_bytes: int = 512              # Minimum size to trigger compression
    default_algorithm: str = "zstd"            # zstd, lz4, gzip, or snappy
    compression_level: int = 3                 # Compression level (1-9)
    enable_adaptive_selection: bool = True     # Auto-select best algorithm
    dictionary_training: bool = True           # Train compression dictionaries
    max_dictionary_size: int = 64 * 1024       # 64KB dictionary size
    enable_streaming: bool = True              # Enable streaming compression


@dataclass
class ConnectionPoolConfig:
    """Connection pool configuration"""
    max_connections_per_host: int = 20         # Max connections per endpoint
    keep_alive_timeout: int = 300              # 5 minutes keep-alive
    health_check_interval: int = 60            # 1 minute health checks
    load_balance_algorithm: str = "least_loaded" # round_robin, least_loaded, fastest_response
    enable_multiplexing: bool = True           # Enable HTTP/2 style multiplexing
    max_concurrent_streams: int = 100          # Max concurrent streams per connection
    connection_timeout: int = 30               # Connection timeout in seconds


@dataclass
class VectorizedIOConfig:
    """Vectorized I/O configuration"""
    enable_zero_copy: bool = True              # Enable zero-copy operations
    ring_buffer_size: int = 50 * 1024 * 1024  # 50MB ring buffer
    batch_timeout_ms: int = 10                 # 10ms batch timeout
    max_vectorized_ops: int = 1000             # Max operations per vector call
    enable_sendmsg: bool = True                # Use sendmsg for vectorized sends
    socket_buffer_size: int = 2 * 1024 * 1024 # 2MB socket buffers


@dataclass 
class TopologyConfig:
    """Topology optimization configuration"""
    update_interval: int = 30                  # Optimization interval in seconds
    measurement_interval: int = 10             # Measurement interval in seconds
    stability_threshold: float = 0.8           # Network stability threshold
    max_alternative_routes: int = 3            # Max alternative routes to consider
    enable_adaptive_routing: bool = True       # Enable adaptive route optimization
    bottleneck_threshold: float = 0.6          # Bottleneck detection threshold
    route_cache_size: int = 1000              # Route cache size


@dataclass
class SerializationConfig:
    """Fast serialization configuration"""
    enable_format_caching: bool = True         # Cache format decisions
    cache_max_size: int = 1000                # Format cache size
    json_threshold_bytes: int = 1024           # JSON threshold
    binary_data_threshold: float = 0.1        # Binary content threshold
    enable_orjson: bool = True                 # Enable orjson if available
    enable_msgpack: bool = True               # Enable msgpack if available
    enable_pickle: bool = False               # Enable pickle (security risk)


@dataclass
class SpeedOptimizedConfig:
    """
    Complete speed optimization configuration.
    Combines all performance enhancement settings.
    """
    # Transport layer options
    enable_quic: bool = True
    enable_tcp: bool = True                    # Keep TCP as fallback
    enable_zero_copy: bool = True
    local_mesh: bool = True                    # Disable TLS for local networks
    
    # Connection management
    max_connections: int = 200
    connection_timeout: int = 30
    max_message_size: int = 10 * 1024 * 1024  # 10MB
    
    # Batching and compression
    max_batch_size: int = 100
    max_wait_time_ms: int = 5                  # Aggressive batching
    max_batch_bytes: int = 2 * 1024 * 1024    # 2MB batches
    enable_priority_bypass: bool = True
    adaptive_sizing: bool = True
    
    # Compression settings
    min_compress_bytes: int = 256              # Compress smaller payloads
    default_algorithm: str = "lz4"             # Fast compression
    enable_adaptive_selection: bool = True
    dictionary_training: bool = True
    
    # Topology optimization  
    enable_adaptive_routing: bool = True
    route_optimization_interval: int = 15      # More frequent optimization
    
    # Performance tuning
    enable_vectorized_io: bool = True
    enable_connection_pooling: bool = True
    enable_fast_serialization: bool = True
    
    # Sub-configurations
    quic: QUICConfig = field(default_factory=QUICConfig)
    batching: BatchConfig = field(default_factory=BatchConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    connection_pool: ConnectionPoolConfig = field(default_factory=ConnectionPoolConfig)
    vectorized_io: VectorizedIOConfig = field(default_factory=VectorizedIOConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    serialization: SerializationConfig = field(default_factory=SerializationConfig)
    
    def __post_init__(self):
        """Configure sub-components based on main settings"""
        # Update sub-configs based on main settings
        self.batching.max_batch_size = self.max_batch_size
        self.batching.max_wait_time_ms = self.max_wait_time_ms
        self.batching.max_batch_bytes = self.max_batch_bytes
        
        self.compression.min_compress_bytes = self.min_compress_bytes
        self.compression.default_algorithm = self.default_algorithm
        self.compression.enable_adaptive_selection = self.enable_adaptive_selection
        
        self.topology.enable_adaptive_routing = self.enable_adaptive_routing
        self.topology.update_interval = self.route_optimization_interval
        
        self.vectorized_io.enable_zero_copy = self.enable_zero_copy
        
        # Adjust settings for maximum performance profile
        if hasattr(self, '_profile') and self._profile == 'maximum_performance':
            self._configure_maximum_performance()
    
    def _configure_maximum_performance(self):
        """Configure for maximum performance"""
        # Aggressive batching
        self.max_batch_size = 200
        self.max_wait_time_ms = 2
        self.batching.max_batch_size = 200
        self.batching.max_wait_time_ms = 2
        
        # Fast compression
        self.default_algorithm = "lz4"
        self.compression.default_algorithm = "lz4"
        self.compression.compression_level = 1
        
        # Frequent optimization
        self.route_optimization_interval = 10
        self.topology.update_interval = 10
        self.topology.measurement_interval = 5
        
        # Large buffers
        self.vectorized_io.ring_buffer_size = 100 * 1024 * 1024  # 100MB
        self.vectorized_io.batch_timeout_ms = 5
        
        # More connections
        self.max_connections = 500
        self.connection_pool.max_connections_per_host = 50


@dataclass
class EnhancedP2PConfig(P2PConfig):
    """
    Enhanced P2P configuration with optimization settings.
    Extends the base P2PConfig with performance features.
    """
    # QUIC-specific settings
    enable_quic: bool = True
    local_mesh: bool = True
    quic_congestion_control: str = "bbr"
    quic_initial_rtt_ms: int = 100
    quic_enable_0rtt: bool = True
    
    # Zero-copy and vectorized I/O
    enable_zero_copy: bool = True
    enable_vectorized_io: bool = True
    max_concurrent_streams: int = 20
    
    # Connection pooling
    enable_connection_pooling: bool = True
    max_connections_per_host: int = 20
    
    # Performance optimizations
    enable_fast_serialization: bool = True
    enable_adaptive_compression: bool = True
    enable_intelligent_batching: bool = True
    
    # Network optimization
    enable_topology_optimization: bool = True
    enable_adaptive_routing: bool = True


# Predefined speed profiles
SPEED_PROFILES = {
    'balanced': SpeedOptimizedConfig(),
    
    'maximum_performance': SpeedOptimizedConfig(
        # Aggressive settings for maximum speed
        max_batch_size=200,
        max_wait_time_ms=2,
        min_compress_bytes=128,
        default_algorithm="lz4",
        route_optimization_interval=10,
        max_connections=500,
        _profile='maximum_performance'
    ),
    
    'low_latency': SpeedOptimizedConfig(
        # Optimized for lowest latency
        max_batch_size=10,
        max_wait_time_ms=1,
        enable_adaptive_routing=True,
        route_optimization_interval=5,
        default_algorithm="snappy",
        min_compress_bytes=1024,  # Less compression for speed
    ),
    
    'high_throughput': SpeedOptimizedConfig(
        # Optimized for maximum throughput
        max_batch_size=500,
        max_wait_time_ms=20,
        max_batch_bytes=5 * 1024 * 1024,  # 5MB batches
        default_algorithm="zstd",
        min_compress_bytes=64,
        enable_adaptive_selection=True,
    ),
    
    'bandwidth_optimized': SpeedOptimizedConfig(
        # Optimized for limited bandwidth
        max_batch_size=100,
        max_wait_time_ms=50,
        default_algorithm="zstd",
        min_compress_bytes=128,
        enable_adaptive_selection=True,
        dictionary_training=True,
    )
}


def get_speed_profile(profile_name: str) -> SpeedOptimizedConfig:
    """Get a predefined speed optimization profile"""
    if profile_name not in SPEED_PROFILES:
        raise ValueError(f"Unknown profile: {profile_name}. Available: {list(SPEED_PROFILES.keys())}")
    return SPEED_PROFILES[profile_name]


def create_custom_profile(**kwargs) -> SpeedOptimizedConfig:
    """Create a custom speed optimization profile"""
    base_config = SPEED_PROFILES['balanced']
    
    # Update with custom settings
    for key, value in kwargs.items():
        if hasattr(base_config, key):
            setattr(base_config, key, value)
        else:
            raise ValueError(f"Unknown configuration option: {key}")
    
    return base_config