#!/bin/bash
# Script to integrate Enhanced CSP Network into the Backend

set -e

echo "🔧 Integrating Enhanced CSP Network into Backend"
echo "=============================================="

# 1. Create the network integration module
echo "📝 Creating backend/network_integration.py..."
cp backend_network_integration.py enhanced_csp/backend/network_integration.py

# 2. Update backend/main.py to include network integration
echo "📝 Updating backend/main.py..."
cat > enhanced_csp/backend/main_network_update.py << 'EOF'
# This file contains the updates needed for backend/main.py
# Add these imports to the import section:

from backend.network_integration import (
    initialize_network_service,
    shutdown_network_service,
    network_service,
    integrate_websocket_with_network,
    NetworkStatusResponse
)

# Add this to the lifespan function after database initialization:
async def lifespan_with_network(app: FastAPI):
    """Application lifespan manager with network integration"""
    # Startup
    logger.info("🚀 Starting Enhanced CSP Visual Designer Backend with Network Support")
    
    try:
        # Initialize database connections if available
        if DATABASE_AVAILABLE:
            await startup_database()
            
            # Initialize WebSocket manager with Redis
            redis_client = await db_manager.get_redis()
            if WEBSOCKET_AVAILABLE:
                await init_websocket_manager(redis_client)
        
        # Initialize network service
        await initialize_network_service(app)
        
        # Integrate WebSocket with network
        if WEBSOCKET_AVAILABLE and network_service.is_initialized:
            integrate_websocket_with_network(connection_manager, network_service)
        
        # Initialize component registry if available
        if COMPONENTS_AVAILABLE:
            component_registry = await get_component_registry()
        
        logger.info("✅ All systems initialized successfully")
        
        yield
        
    finally:
        # Shutdown
        logger.info("🛑 Shutting down Enhanced CSP Backend")
        
        # Shutdown network service
        await shutdown_network_service()
        
        # Shutdown WebSocket manager
        if WEBSOCKET_AVAILABLE:
            await shutdown_websocket_manager()
        
        # Close database connections
        if DATABASE_AVAILABLE:
            await shutdown_database()
        
        logger.info("👋 Shutdown complete")
EOF

# 3. Create a patch file for main.py
echo "📝 Creating patch for backend/main.py..."
cat > enhanced_csp/backend/main_network.patch << 'EOF'
--- a/backend/main.py
+++ b/backend/main.py
@@ -85,6 +85,13 @@ try:
 except ImportError:
     EXECUTION_AVAILABLE = False
 
+# Network Integration
+from backend.network_integration import (
+    initialize_network_service,
+    shutdown_network_service,
+    network_service,
+    integrate_websocket_with_network
+)
+
 # WebSocket dependencies
 try:
     from backend.realtime.websocket_manager import (
@@ -290,6 +297,12 @@ async def lifespan(app: FastAPI):
         if COMPONENTS_AVAILABLE:
             component_registry = await get_component_registry()
         
+        # Initialize network service
+        await initialize_network_service(app)
+        
+        # Integrate WebSocket with network
+        if WEBSOCKET_AVAILABLE and network_service.is_initialized:
+            integrate_websocket_with_network(connection_manager, network_service)
+        
         # Initialize AI services if available
         if AI_INTEGRATION_AVAILABLE:
             await init_ai_services()
@@ -307,6 +320,9 @@ async def lifespan(app: FastAPI):
         # Shutdown
         logger.info("🛑 Shutting down Enhanced CSP Visual Designer Backend")
         
+        # Shutdown network service
+        await shutdown_network_service()
+        
         # Shutdown WebSocket manager
         if WEBSOCKET_AVAILABLE:
             await shutdown_websocket_manager()
@@ -573,6 +589,11 @@ async def root():
             "real_time_collaboration": WEBSOCKET_AVAILABLE,
             "ai_integration": True,
             "component_registry": COMPONENTS_AVAILABLE,
+            "network": {
+                "enabled": network_service.is_initialized,
+                "node_id": network_service.network.node_id.to_base58() if network_service.is_initialized else None,
+                "peers": len(network_service.node_registry) - 1 if network_service.is_initialized else 0
+            },
             "execution_engine": EXECUTION_AVAILABLE,
             "performance_monitoring": True,
             "database": DATABASE_AVAILABLE
EOF

# 4. Create WebSocket integration update
echo "📝 Creating WebSocket network integration..."
cat > enhanced_csp/backend/realtime/websocket_network_integration.py << 'EOF'
"""
WebSocket Network Integration
============================

Extensions to WebSocket manager for network-aware collaboration.
"""

from typing import Dict, Any, Optional, Set
from uuid import UUID
from datetime import datetime
import logging

from backend.realtime.websocket_manager import WebSocketEvent, EventType

logger = logging.getLogger(__name__)


class NetworkAwareWebSocketManager:
    """Extension for network-aware WebSocket management."""
    
    def __init__(self, base_manager, network_service):
        self.base_manager = base_manager
        self.network_service = network_service
        self.remote_sessions: Dict[str, Dict[str, Any]] = {}  # Track remote users
        
    async def handle_network_design_update(self, update: Dict[str, Any]):
        """Handle design update from network node."""
        design_id = UUID(update["design_id"])
        event_data = update["event"]
        sender_node = update.get("sender")
        
        # Create WebSocket event from network update
        event = WebSocketEvent(
            type=EventType(event_data["type"]),
            data=event_data["data"],
            user_id=f"network:{sender_node[:8]}" if sender_node else "network:unknown",
            timestamp=datetime.fromisoformat(event_data["timestamp"])
        )
        
        # Broadcast to local clients only (avoid echo)
        await self.base_manager.broadcast_to_design(
            design_id, 
            event,
            exclude_network=True  # Custom flag to prevent network re-broadcast
        )
        
    async def register_remote_user(self, design_id: str, node_id: str, user_info: Dict[str, Any]):
        """Register a remote user from another node."""
        session_key = f"{node_id}:{user_info['user_id']}"
        
        self.remote_sessions[session_key] = {
            "design_id": design_id,
            "node_id": node_id,
            "user_info": user_info,
            "last_seen": datetime.utcnow()
        }
        
        # Notify local users
        await self.base_manager.broadcast_to_design(
            UUID(design_id),
            WebSocketEvent(
                type=EventType.USER_JOIN,
                data={
                    "user_id": session_key,
                    "display_name": f"{user_info.get('name', 'Remote User')} (Network)",
                    "is_remote": True,
                    "node_id": node_id[:8]
                }
            )
        )
        
    async def get_all_design_users(self, design_id: UUID) -> List[Dict[str, Any]]:
        """Get all users including remote network users."""
        # Get local users
        local_users = await self.base_manager.get_design_users(design_id)
        
        # Add remote users
        remote_users = []
        design_id_str = str(design_id)
        
        for session_key, session in self.remote_sessions.items():
            if session["design_id"] == design_id_str:
                remote_users.append({
                    "user_id": session_key,
                    "display_name": session["user_info"].get("name", "Remote User"),
                    "is_remote": True,
                    "node_id": session["node_id"][:8],
                    "last_activity": session["last_seen"].isoformat()
                })
                
        return local_users + remote_users
EOF

# 5. Create environment configuration
echo "📝 Creating network configuration..."
cat >> enhanced_csp/backend/.env.example << 'EOF'

# Network Configuration
NETWORK_ENABLED=true
NETWORK_PORT=30300
BOOTSTRAP_NODES=seed1.web4ai:30300,seed2.web4ai:30300
ENABLE_QUANTUM=false
ENABLE_BLOCKCHAIN=false
NETWORK_CAPABILITIES=execution,storage,relay,collaboration
EOF

# 6. Create API documentation
echo "📝 Creating network API documentation..."
cat > enhanced_csp/backend/api/network_api.md << 'EOF'
# Network API Documentation

## Overview

The Enhanced CSP Backend now includes P2P networking capabilities for distributed execution and real-time collaboration across network nodes.

## Endpoints

### GET /api/network/status
Get current network status and node information.

**Response:**
```json
{
  "node_id": "QmXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
  "status": "online",
  "peers": 5,
  "designs": 10,
  "executions": 3,
  "capabilities": {
    "execution": true,
    "storage": true,
    "relay": true,
    "collaboration": true
  },
  "network_info": {
    "nat_type": "full_cone",
    "external_address": "1.2.3.4:30300",
    "topology_type": "dynamic_partial"
  }
}
```

### GET /api/network/nodes
List all known network nodes.

**Response:**
```json
{
  "nodes": [
    {
      "node_id": "QmXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
      "address": "1.2.3.4:30300",
      "status": "online",
      "capabilities": {},
      "designs": ["design_123"],
      "last_seen": "2024-01-01T00:00:00Z"
    }
  ],
  "count": 1
}
```

### GET /api/network/designs/{design_id}/nodes
Find nodes that have a specific design.

### POST /api/network/execute
Execute a design on the distributed network.

**Request:**
```json
{
  "design_id": "design_123",
  "execution_config": {
    "timeout": 300,
    "priority": "high"
  },
  "preferred_node": "QmXXXX",  // Optional
  "requirements": {
    "min_memory": 4096,
    "capabilities": ["quantum"]
  }
}
```

### POST /api/network/broadcast
Broadcast a message to all network nodes (admin only).

## WebSocket Integration

The WebSocket system is now network-aware. Design updates are automatically synchronized across all nodes in the network.

### Network Events

- `network:peer_joined` - A new peer joined the network
- `network:peer_left` - A peer left the network
- `network:design_sync` - Design synchronized from network
- `network:execution_update` - Execution update from remote node

## Configuration

Set these environment variables:

- `NETWORK_ENABLED` - Enable/disable network features
- `NETWORK_PORT` - P2P listening port (default: 30300)
- `BOOTSTRAP_NODES` - Comma-separated list of bootstrap nodes
- `ENABLE_QUANTUM` - Enable quantum features
- `ENABLE_BLOCKCHAIN` - Enable blockchain integration

## Security

All network communication is encrypted using TLS 1.3 with optional post-quantum cryptography.
EOF

# 7. Create integration test
echo "📝 Creating integration test..."
cat > enhanced_csp/backend/tests/test_network_integration.py << 'EOF'
"""
Test Network Integration
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from backend.network_integration import CSPNetworkService, NetworkNode
from enhanced_csp.network import NetworkConfig


@pytest.fixture
async def network_service():
    """Create test network service."""
    service = CSPNetworkService()
    
    # Mock network
    service.network = Mock()
    service.network.node_id = Mock()
    service.network.node_id.to_base58.return_value = "QmTest123"
    service.network.topology = Mock()
    service.network.topology.get_peers.return_value = []
    service.network.broadcast = AsyncMock()
    service.network.send_message = AsyncMock()
    
    service.is_initialized = True
    
    yield service
    
    service.is_initialized = False


@pytest.mark.asyncio
async def test_register_design(network_service):
    """Test design registration."""
    design_id = "design_123"
    
    await network_service.register_design(design_id)
    
    assert design_id in network_service.design_nodes
    assert "QmTest123" in network_service.design_nodes[design_id]
    
    # Should broadcast
    network_service.network.broadcast.assert_called_once()
    

@pytest.mark.asyncio
async def test_find_design_nodes(network_service):
    """Test finding nodes with a design."""
    design_id = "design_123"
    
    # Register on multiple nodes
    network_service.design_nodes[design_id] = {"node1", "node2", "node3"}
    
    nodes = await network_service.find_design_nodes(design_id)
    
    assert len(nodes) == 3
    assert "node1" in nodes
    

@pytest.mark.asyncio
async def test_execute_on_node(network_service):
    """Test distributed execution."""
    node_id = "QmRemoteNode"
    design_id = "design_123"
    config = {"timeout": 300}
    
    execution_id = await network_service.execute_on_node(node_id, design_id, config)
    
    assert execution_id is not None
    assert execution_id in network_service.execution_nodes
    assert network_service.execution_nodes[execution_id] == node_id
    
    # Should send message to node
    network_service.network.send_message.assert_called_once()


@pytest.mark.asyncio
async def test_message_handling(network_service):
    """Test network message handling."""
    # Test capability query
    await network_service._handle_network_message("sender123", {
        "type": "capability_query"
    })
    
    network_service.network.send_message.assert_called_with("sender123", {
        "type": "capability_response",
        "capabilities": {},
        "designs": [],
        "load": {"cpu": 0.5, "memory": 0.6, "executions": 0},
        "sender": "QmTest123"
    })
    
    # Test capability response
    await network_service._handle_network_message("sender123", {
        "type": "capability_response",
        "capabilities": {"execution": True},
        "designs": ["design_456"]
    })
    
    assert "sender123" in network_service.node_registry
    node = network_service.node_registry["sender123"]
    assert node.capabilities == {"execution": True}
    assert "design_456" in node.designs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
EOF

# 8. Create example usage script
echo "📝 Creating example usage script..."
cat > enhanced_csp/backend/examples/distributed_execution.py << 'EOF'
#!/usr/bin/env python3
"""
Example: Distributed Design Execution
====================================

This example shows how to use the network features for distributed execution.
"""

import asyncio
import os
from pathlib import Path
import sys

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.network_integration import CSPNetworkService
from enhanced_csp.network import NetworkConfig, P2PConfig


async def main():
    """Run distributed execution example."""
    # Configure network
    config = NetworkConfig(
        p2p=P2PConfig(
            listen_port=30301,  # Different port for testing
            bootstrap_nodes=["localhost:30300"],  # Connect to main node
            enable_dht=True
        ),
        node_capabilities={
            "execution": True,
            "storage": True,
            "collaboration": True
        }
    )
    
    # Create service
    service = CSPNetworkService()
    await service.initialize(config)
    
    print(f"Node ID: {service.network.node_id.to_base58()}")
    
    # Wait for peer discovery
    await asyncio.sleep(5)
    
    print(f"Connected peers: {len(service.node_registry) - 1}")
    
    # Register a design
    design_id = "example_design_123"
    await service.register_design(design_id)
    print(f"Registered design: {design_id}")
    
    # Find nodes with the design
    nodes = await service.find_design_nodes(design_id)
    print(f"Nodes with design: {nodes}")
    
    # Execute on a node
    if len(nodes) > 1:  # Execute on a different node
        target_node = [n for n in nodes if n != service.network.node_id.to_base58()][0]
        
        execution_id = await service.execute_on_node(
            target_node,
            design_id,
            {"timeout": 60, "mode": "distributed"}
        )
        
        print(f"Started execution {execution_id} on node {target_node[:8]}")
    
    # Keep running
    try:
        await asyncio.sleep(300)
    except KeyboardInterrupt:
        pass
    
    # Shutdown
    await service.shutdown()
    print("Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
EOF

# 9. Create systemd service file
echo "📝 Creating systemd service configuration..."
cat > enhanced_csp/backend/enhanced-csp-backend.service << 'EOF'
[Unit]
Description=Enhanced CSP Backend with Network Support
After=network.target postgresql.service redis.service

[Service]
Type=notify
User=csp
WorkingDirectory=/opt/enhanced-csp
Environment="NETWORK_ENABLED=true"
Environment="NETWORK_PORT=30300"
Environment="BOOTSTRAP_NODES=seed1.web4ai:30300,seed2.web4ai:30300"
ExecStart=/usr/bin/python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# 10. Summary
echo ""
echo "✅ Network Integration Complete!"
echo "================================"
echo ""
echo "The backend is now integrated with the Enhanced CSP Network stack."
echo ""
echo "Features added:"
echo "- P2P network service for distributed execution"
echo "- Network-aware WebSocket collaboration"
echo "- API endpoints for network management"
echo "- Distributed design execution"
echo "- Real-time synchronization across nodes"
echo ""
echo "To use the network features:"
echo ""
echo "1. Apply the changes:"
echo "   - Copy backend/network_integration.py to your backend"
echo "   - Apply the patch to backend/main.py"
echo "   - Update your .env file with network settings"
echo ""
echo "2. Start with network enabled:"
echo "   NETWORK_ENABLED=true python -m backend.main"
echo ""
echo "3. Test distributed execution:"
echo "   python backend/examples/distributed_execution.py"
echo ""
echo "4. Access network API:"
echo "   GET http://localhost:8000/api/network/status"
echo ""
echo "The backend now supports:"
echo "- Distributed design execution across network nodes"
echo "- Real-time collaboration between distributed users"
echo "- Load balancing and fault tolerance"
echo "- Network-wide design registry"
echo "- P2P communication with TLS encryption"
