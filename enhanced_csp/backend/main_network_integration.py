# Add this to the imports section after WebSocket dependencies:

# Network Integration
try:
    from backend.network_integration import (
        initialize_network_service,
        shutdown_network_service,
        network_service,
        integrate_websocket_with_network
    )
    NETWORK_AVAILABLE = True
except ImportError:
    logger.warning("Network integration not available")
    NETWORK_AVAILABLE = False
    network_service = None

# Add this in the lifespan function after component registry initialization:
        
        # Initialize network service
        if NETWORK_AVAILABLE:
            await initialize_network_service(app)
            
            # Integrate WebSocket with network
            if WEBSOCKET_AVAILABLE and network_service and network_service.is_initialized:
                integrate_websocket_with_network(connection_manager, network_service)

# Add this in the shutdown section after the shutdown log message:
        
        # Shutdown network service
        if NETWORK_AVAILABLE and network_service:
            await shutdown_network_service()

# Add this to the features dict in the root endpoint:
            "network": {
                "enabled": NETWORK_AVAILABLE and network_service and network_service.is_initialized,
                "node_id": network_service.network.node_id.to_base58() if (network_service and network_service.is_initialized) else None,
                "peers": len(network_service.node_registry) - 1 if (network_service and network_service.is_initialized) else 0
            },
