# enhanced_csp/network/p2p/transport_factory.py
"""
Transport factory for creating optimized transport instances.
Handles automatic fallbacks and configuration selection.
"""

import logging
from typing import Optional, Dict, Any
from ..core.config import P2PConfig
from ..core.enhanced_config import SpeedOptimizedConfig
from .transport import P2PTransport, MultiProtocolTransport

logger = logging.getLogger(__name__)


def create_transport(config: P2PConfig) -> P2PTransport:
    """
    Factory function to create appropriate transport based on configuration.
    
    Args:
        config: P2P configuration
        
    Returns:
        Configured transport instance
    """
    # Check if QUIC is enabled and available
    if getattr(config, 'enable_quic', False):
        try:
            from .quic_transport import QUICTransport, create_quic_transport
            logger.info("Creating QUIC transport")
            return create_quic_transport(config)
        except ImportError:
            logger.warning("QUIC not available (aioquic not installed), falling back to TCP")
        except Exception as e:
            logger.warning(f"QUIC transport creation failed: {e}, falling back to TCP")
    
    # Fallback to multi-protocol transport (TCP-based)
    logger.info("Creating TCP-based multi-protocol transport")
    return MultiProtocolTransport(config)


def create_optimized_transport(config: SpeedOptimizedConfig) -> P2PTransport:
    """
    Create an optimized transport stack based on speed configuration.
    
    Args:
        config: Speed optimization configuration
        
    Returns:
        Optimized transport instance
    """
    from ..optimized_channel import OptimizedTransportStack
    
    logger.info(f"Creating optimized transport stack with profile: {getattr(config, '_profile', 'custom')}")
    return OptimizedTransportStack(config)


def create_transport_with_fallback(config: P2PConfig, 
                                 preferred_protocols: Optional[list] = None) -> P2PTransport:
    """
    Create transport with protocol fallback chain.
    
    Args:
        config: P2P configuration
        preferred_protocols: List of preferred protocols in order ['quic', 'tcp', 'websocket']
        
    Returns:
        Transport instance with best available protocol
    """
    if preferred_protocols is None:
        preferred_protocols = ['quic', 'tcp']
    
    for protocol in preferred_protocols:
        try:
            if protocol == 'quic':
                from .quic_transport import create_quic_transport
                transport = create_quic_transport(config)
                logger.info(f"Successfully created {protocol} transport")
                return transport
                
            elif protocol == 'tcp':
                transport = MultiProtocolTransport(config)
                logger.info(f"Successfully created {protocol} transport")
                return transport
                
            elif protocol == 'websocket':
                # WebSocket transport could be implemented here
                logger.warning("WebSocket transport not implemented, skipping")
                continue
                
        except ImportError as e:
            logger.warning(f"Protocol {protocol} not available: {e}")
            continue
        except Exception as e:
            logger.error(f"Failed to create {protocol} transport: {e}")
            continue
    
    # Final fallback to basic TCP
    logger.warning("All preferred protocols failed, using basic TCP transport")
    return MultiProtocolTransport(config)


def get_available_protocols() -> Dict[str, bool]:
    """
    Check which transport protocols are available.
    
    Returns:
        Dictionary mapping protocol names to availability
    """
    protocols = {
        'tcp': True,  # Always available
        'quic': False,
        'websocket': False
    }
    
    # Check QUIC availability
    try:
        import aioquic
        protocols['quic'] = True
    except ImportError:
        pass
    
    # Check WebSocket availability (if implemented)
    try:
        import websockets
        protocols['websocket'] = True
    except ImportError:
        pass
    
    return protocols


def recommend_transport_config(network_conditions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Recommend transport configuration based on network conditions.
    
    Args:
        network_conditions: Dictionary with network condition metrics
        
    Returns:
        Recommended configuration parameters
    """
    if network_conditions is None:
        network_conditions = {}
    
    recommendations = {
        'enable_quic': True,
        'enable_tcp': True,  # Keep as fallback
        'connection_timeout': 30,
        'max_connections': 100
    }
    
    # Analyze network conditions
    latency = network_conditions.get('avg_latency_ms', 50)
    packet_loss = network_conditions.get('packet_loss_ratio', 0.01)
    bandwidth_mbps = network_conditions.get('bandwidth_mbps', 100)
    is_mobile = network_conditions.get('is_mobile_network', False)
    
    # High latency networks
    if latency > 100:
        recommendations.update({
            'connection_timeout': 60,
            'enable_connection_migration': True,
            'enable_0rtt': True
        })
    
    # High packet loss
    if packet_loss > 0.05:  # 5% packet loss
        recommendations.update({
            'congestion_control': 'bbr',
            'enable_redundancy': True
        })
    
    # Low bandwidth
    if bandwidth_mbps < 10:
        recommendations.update({
            'enable_compression': True,
            'compression_algorithm': 'zstd',
            'max_batch_size': 50
        })
    
    # Mobile networks
    if is_mobile:
        recommendations.update({
            'enable_connection_migration': True,
            'keep_alive_interval': 30,
            'adaptive_timeout': True
        })
    
    return recommendations


def create_development_transport() -> P2PTransport:
    """Create transport optimized for development environment"""
    config = P2PConfig(
        listen_port=30300,
        enable_quic=False,  # Simpler for development
        connection_timeout=10,
        max_message_size=1024 * 1024  # 1MB
    )
    
    return MultiProtocolTransport(config)


def create_production_transport(enable_tls: bool = True) -> P2PTransport:
    """Create transport optimized for production environment"""
    config = P2PConfig(
        listen_port=30300,
        enable_quic=True,
        connection_timeout=30,
        max_message_size=10 * 1024 * 1024,  # 10MB
        enable_tls=enable_tls
    )
    
    return create_transport_with_fallback(config, ['quic', 'tcp'])


def create_high_performance_transport() -> P2PTransport:
    """Create transport optimized for maximum performance"""
    from ..core.enhanced_config import get_speed_profile
    
    config = get_speed_profile('maximum_performance')
    return create_optimized_transport(config)


def create_low_latency_transport() -> P2PTransport:
    """Create transport optimized for low latency"""
    from ..core.enhanced_config import get_speed_profile
    
    config = get_speed_profile('low_latency')
    return create_optimized_transport(config)


def auto_configure_transport(target_use_case: str = "general") -> P2PTransport:
    """
    Automatically configure transport based on use case.
    
    Args:
        target_use_case: Use case ('general', 'gaming', 'file_transfer', 'chat', 'streaming')
        
    Returns:
        Configured transport instance
    """
    use_case_configs = {
        'general': {
            'profile': 'balanced',
            'enable_quic': True,
            'max_batch_size': 50
        },
        'gaming': {
            'profile': 'low_latency',
            'enable_quic': True,
            'max_wait_time_ms': 1
        },
        'file_transfer': {
            'profile': 'high_throughput',
            'enable_compression': True,
            'max_batch_size': 200
        },
        'chat': {
            'profile': 'balanced',
            'enable_quic': True,
            'max_batch_size': 20
        },
        'streaming': {
            'profile': 'high_throughput',
            'enable_quic': True,
            'enable_zero_copy': True
        }
    }
    
    config_params = use_case_configs.get(target_use_case, use_case_configs['general'])
    
    if 'profile' in config_params:
        from ..core.enhanced_config import get_speed_profile
        config = get_speed_profile(config_params['profile'])
        
        # Apply additional customizations
        for key, value in config_params.items():
            if key != 'profile' and hasattr(config, key):
                setattr(config, key, value)
        
        return create_optimized_transport(config)
    else:
        # Fallback to basic configuration
        p2p_config = P2PConfig()
        for key, value in config_params.items():
            if hasattr(p2p_config, key):
                setattr(p2p_config, key, value)
        
        return create_transport(p2p_config)


def validate_transport_config(config: P2PConfig) -> Dict[str, Any]:
    """
    Validate transport configuration and return recommendations.
    
    Args:
        config: Transport configuration to validate
        
    Returns:
        Validation results and recommendations
    """
    results = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'recommendations': []
    }
    
    # Check port availability
    from ..utils import validate_port_number, is_port_available
    
    if not validate_port_number(config.listen_port):
        results['errors'].append(f"Invalid port number: {config.listen_port}")
        results['valid'] = False
    
    if not is_port_available(config.listen_address, config.listen_port):
        results['warnings'].append(f"Port {config.listen_port} may already be in use")
    
    # Check message size limits
    if config.max_message_size > 100 * 1024 * 1024:  # 100MB
        results['warnings'].append("Very large max message size may impact performance")
    
    # Check timeout values
    if config.connection_timeout < 5:
        results['warnings'].append("Very short connection timeout may cause issues on slow networks")
    
    # Check QUIC requirements
    if getattr(config, 'enable_quic', False):
        try:
            import aioquic
        except ImportError:
            results['warnings'].append("QUIC enabled but aioquic not installed")
            results['recommendations'].append("Install aioquic: pip install aioquic")
    
    return results


# Export commonly used factory functions
__all__ = [
    'create_transport',
    'create_optimized_transport', 
    'create_transport_with_fallback',
    'get_available_protocols',
    'recommend_transport_config',
    'create_development_transport',
    'create_production_transport',
    'create_high_performance_transport',
    'create_low_latency_transport',
    'auto_configure_transport',
    'validate_transport_config'
]