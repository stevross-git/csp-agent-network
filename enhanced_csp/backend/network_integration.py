#!/usr/bin/env python3
"""
Backend Network Integration Script
==================================

This script integrates the Enhanced CSP Network stack into the backend,
providing P2P capabilities for distributed design execution and real-time
collaboration across network nodes.
"""

import asyncio
import logging
import os
from typing import Dict, Any, Optional, List, Set
from uuid import UUID, uuid4
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, WebSocket, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Enhanced CSP Network imports
from enhanced_csp.network import (
    NetworkConfig,
    EnhancedCSPNetwork,
    NodeID,
    P2PConfig,
    MeshConfig,
    DNSConfig
)

# Backend imports
from backend.auth.auth_system import UnifiedUserInfo, require_permission_unified
from backend.realtime.websocket_manager import ConnectionManager, WebSocketEvent, EventType
from backend.database.connection import DatabaseManager
from backend.schemas.api_schemas import BaseResponse

logger = logging.getLogger(__name__)


# =============================================================================
# NETWORK SERVICE
# =============================================================================

@dataclass
class NetworkNode:
    """Represents a network node in the CSP system."""
    node_id: str
    address: str
    capabilities: Dict[str, bool]
    last_seen: datetime
    status: str = "online"
    designs: Set[str] = field(default_factory=set)
    executions: Set[str] = field(default_factory=set)


class CSPNetworkService:
    """
    Service that integrates the Enhanced CSP Network with the backend.
    Provides P2P capabilities for distributed execution and collaboration.
    """
    
    def __init__(self):
        self.network: Optional[EnhancedCSPNetwork] = None
        self.node_registry: Dict[str, NetworkNode] = {}
        self.design_nodes: Dict[str, Set[str]] = {}  # design_id -> set of node_ids
        self.execution_nodes: Dict[str, str] = {}  # execution_id -> node_id
        self.is_initialized = False
        
    async def initialize(self, config: Optional[NetworkConfig] = None):
        """Initialize the network service."""
        if self.is_initialized:
            return
            
        # Create network configuration
        if config is None:
            config = NetworkConfig(
                p2p=P2PConfig(
                    listen_port=int(os.getenv("NETWORK_PORT", "30300")),
                    enable_dht=True,
                    enable_nat_traversal=True,
                    bootstrap_nodes=os.getenv("BOOTSTRAP_NODES", "").split(",") if os.getenv("BOOTSTRAP_NODES") else []
                ),
                mesh=MeshConfig(
                    topology_type="dynamic_partial",
                    enable_super_peers=True
                ),
                dns=DNSConfig(
                    root_domain=".web4ai",
                    enable_dnssec=True
                ),
                enable_quantum=os.getenv("ENABLE_QUANTUM", "false").lower() == "true",
                enable_blockchain=os.getenv("ENABLE_BLOCKCHAIN", "false").lower() == "true",
                node_capabilities={
                    "execution": True,
                    "storage": True,
                    "relay": True,
                    "collaboration": True
                }
            )
            
        # Create and start network
        self.network = EnhancedCSPNetwork(config)
        await self.network.start()
        
        # Register ourselves
        self.node_registry[self.network.node_id.to_base58()] = NetworkNode(
            node_id=self.network.node_id.to_base58(),
            address=f"{config.p2p.listen_address}:{config.p2p.listen_port}",
            capabilities=config.node_capabilities,
            last_seen=datetime.utcnow()
        )
        
        # Start background tasks
        asyncio.create_task(self._peer_discovery_loop())
        asyncio.create_task(self._message_handler_loop())
        
        self.is_initialized = True
        logger.info(f"Network service initialized with node ID: {self.network.node_id.to_base58()}")
        
    async def shutdown(self):
        """Shutdown the network service."""
        if self.network:
            await self.network.stop()
        self.is_initialized = False
        
    async def register_design(self, design_id: str, node_id: Optional[str] = None):
        """Register a design with the network."""
        if node_id is None:
            node_id = self.network.node_id.to_base58()
            
        if design_id not in self.design_nodes:
            self.design_nodes[design_id] = set()
        self.design_nodes[design_id].add(node_id)
        
        # Broadcast design availability
        await self.broadcast_message({
            "type": "design_registered",
            "design_id": design_id,
            "node_id": node_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    async def find_design_nodes(self, design_id: str) -> List[str]:
        """Find nodes that have a specific design."""
        return list(self.design_nodes.get(design_id, set()))
        
    async def execute_on_node(self, node_id: str, design_id: str, execution_config: Dict[str, Any]) -> str:
        """Execute a design on a specific network node."""
        execution_id = str(uuid4())
        
        # Send execution request to node
        await self.send_to_node(node_id, {
            "type": "execution_request",
            "execution_id": execution_id,
            "design_id": design_id,
            "config": execution_config,
            "requester": self.network.node_id.to_base58()
        })
        
        self.execution_nodes[execution_id] = node_id
        return execution_id
        
    async def broadcast_message(self, message: Dict[str, Any]):
        """Broadcast a message to all connected nodes."""
        if not self.network:
            return
            
        # Add sender info
        message["sender"] = self.network.node_id.to_base58()
        
        # Broadcast via network
        await self.network.broadcast(message)
        
    async def send_to_node(self, node_id: str, message: Dict[str, Any]):
        """Send a message to a specific node."""
        if not self.network:
            return
            
        # Add sender info
        message["sender"] = self.network.node_id.to_base58()
        
        # Send via network
        await self.network.send_message(node_id, message)
        
    async def _peer_discovery_loop(self):
        """Continuously discover and register peers."""
        while self.is_initialized:
            try:
                # Get current peers
                peers = self.network.topology.get_peers()
                
                for peer in peers:
                    peer_id = peer.id.to_base58()
                    if peer_id not in self.node_registry:
                        # Query peer capabilities
                        await self.send_to_node(peer_id, {
                            "type": "capability_query"
                        })
                        
                # Clean up offline nodes
                now = datetime.utcnow()
                for node_id, node in list(self.node_registry.items()):
                    if (now - node.last_seen).seconds > 300:  # 5 minutes
                        node.status = "offline"
                        
            except Exception as e:
                logger.error(f"Error in peer discovery: {e}")
                
            await asyncio.sleep(30)  # Run every 30 seconds
            
    async def _message_handler_loop(self):
        """Handle incoming network messages."""
        if not self.network or not self.network.transport:
            return
            
        # Set message handler
        self.network.transport.on_message = self._handle_network_message
        
    async def _handle_network_message(self, sender_id: str, message: Dict[str, Any]):
        """Handle an incoming network message."""
        try:
            msg_type = message.get("type")
            
            if msg_type == "capability_query":
                # Respond with our capabilities
                await self.send_to_node(sender_id, {
                    "type": "capability_response",
                    "capabilities": self.network.config.node_capabilities,
                    "designs": list(self.design_nodes.keys()),
                    "load": self._get_node_load()
                })
                
            elif msg_type == "capability_response":
                # Register peer node
                self.node_registry[sender_id] = NetworkNode(
                    node_id=sender_id,
                    address=message.get("address", "unknown"),
                    capabilities=message.get("capabilities", {}),
                    last_seen=datetime.utcnow(),
                    designs=set(message.get("designs", []))
                )
                
            elif msg_type == "design_registered":
                # Update design registry
                design_id = message["design_id"]
                node_id = message["node_id"]
                if design_id not in self.design_nodes:
                    self.design_nodes[design_id] = set()
                self.design_nodes[design_id].add(node_id)
                
            elif msg_type == "execution_request":
                # Handle execution request
                await self._handle_execution_request(sender_id, message)
                
            elif msg_type == "execution_update":
                # Forward execution updates to WebSocket clients
                await self._forward_execution_update(message)
                
        except Exception as e:
            logger.error(f"Error handling network message: {e}")
            
    async def _handle_execution_request(self, requester_id: str, message: Dict[str, Any]):
        """Handle an execution request from another node."""
        execution_id = message["execution_id"]
        design_id = message["design_id"]
        config = message["config"]
        
        # TODO: Integrate with execution engine
        # For now, just acknowledge
        await self.send_to_node(requester_id, {
            "type": "execution_started",
            "execution_id": execution_id,
            "design_id": design_id,
            "node_id": self.network.node_id.to_base58()
        })
        
    async def _forward_execution_update(self, message: Dict[str, Any]):
        """Forward execution updates to WebSocket clients."""
        # TODO: Integrate with WebSocket manager
        pass
        
    def _get_node_load(self) -> Dict[str, float]:
        """Get current node load metrics."""
        # TODO: Implement actual load metrics
        return {
            "cpu": 0.5,
            "memory": 0.6,
            "executions": len(self.execution_nodes)
        }


# =============================================================================
# NETWORK API ENDPOINTS
# =============================================================================

# Create network service instance
network_service = CSPNetworkService()


class NetworkStatusResponse(BaseModel):
    """Network status response model."""
    node_id: str
    status: str
    peers: int
    designs: int
    executions: int
    capabilities: Dict[str, bool]
    network_info: Dict[str, Any]


class DistributedExecutionRequest(BaseModel):
    """Request model for distributed execution."""
    design_id: str
    execution_config: Dict[str, Any] = {}
    preferred_node: Optional[str] = None
    requirements: Dict[str, Any] = {}


def create_network_router(app: FastAPI) -> None:
    """Create and configure network-related API endpoints."""
    
    @app.get("/api/network/status", response_model=NetworkStatusResponse, tags=["network"])
    async def get_network_status(
        current_user: UnifiedUserInfo = Depends(require_permission_unified("read"))
    ):
        """Get network status and information."""
        if not network_service.is_initialized:
            raise HTTPException(status_code=503, detail="Network service not initialized")
            
        network_info = network_service.network.get_network_info()
        
        return NetworkStatusResponse(
            node_id=network_service.network.node_id.to_base58(),
            status="online" if network_service.is_initialized else "offline",
            peers=len(network_service.node_registry) - 1,
            designs=len(network_service.design_nodes),
            executions=len(network_service.execution_nodes),
            capabilities=network_service.network.config.node_capabilities,
            network_info=network_info
        )
        
    @app.get("/api/network/nodes", tags=["network"])
    async def list_network_nodes(
        current_user: UnifiedUserInfo = Depends(require_permission_unified("read"))
    ):
        """List all known network nodes."""
        nodes = []
        for node_id, node in network_service.node_registry.items():
            nodes.append({
                "node_id": node_id,
                "address": node.address,
                "status": node.status,
                "capabilities": node.capabilities,
                "designs": list(node.designs),
                "last_seen": node.last_seen.isoformat()
            })
        return {"nodes": nodes, "count": len(nodes)}
        
    @app.get("/api/network/designs/{design_id}/nodes", tags=["network"])
    async def find_design_nodes(
        design_id: str,
        current_user: UnifiedUserInfo = Depends(require_permission_unified("read"))
    ):
        """Find nodes that have a specific design."""
        nodes = await network_service.find_design_nodes(design_id)
        return {
            "design_id": design_id,
            "nodes": nodes,
            "count": len(nodes)
        }
        
    @app.post("/api/network/execute", tags=["network"])
    async def execute_distributed(
        request: DistributedExecutionRequest,
        background_tasks: BackgroundTasks,
        current_user: UnifiedUserInfo = Depends(require_permission_unified("execute"))
    ):
        """Execute a design on the distributed network."""
        # Find suitable node
        if request.preferred_node:
            node_id = request.preferred_node
        else:
            # Find node with the design
            nodes = await network_service.find_design_nodes(request.design_id)
            if not nodes:
                raise HTTPException(status_code=404, detail="No nodes found with this design")
            
            # TODO: Implement load balancing
            node_id = nodes[0]
            
        # Execute on selected node
        execution_id = await network_service.execute_on_node(
            node_id,
            request.design_id,
            request.execution_config
        )
        
        return {
            "execution_id": execution_id,
            "node_id": node_id,
            "status": "started",
            "message": f"Execution started on node {node_id[:8]}..."
        }
        
    @app.post("/api/network/broadcast", tags=["network"])
    async def broadcast_to_network(
        message: Dict[str, Any],
        current_user: UnifiedUserInfo = Depends(require_permission_unified("admin"))
    ):
        """Broadcast a message to all network nodes."""
        await network_service.broadcast_message(message)
        return {"success": True, "message": "Message broadcast to network"}


# =============================================================================
# WEBSOCKET NETWORK INTEGRATION
# =============================================================================

def integrate_websocket_with_network(
    connection_manager: ConnectionManager,
    network_service: CSPNetworkService
):
    """Integrate WebSocket manager with network service for distributed collaboration."""
    
    # Override broadcast method to include network broadcast
    original_broadcast = connection_manager.broadcast_to_design
    
    async def network_aware_broadcast(design_id: UUID, event: WebSocketEvent, exclude_session: Optional[str] = None):
        """Broadcast to both local WebSocket clients and network nodes."""
        # Local broadcast
        await original_broadcast(design_id, event, exclude_session)
        
        # Network broadcast for collaborative events
        if event.type in [EventType.NODE_ADDED, EventType.NODE_REMOVED, 
                         EventType.CONNECTION_ADDED, EventType.CONNECTION_REMOVED,
                         EventType.NODE_UPDATED]:
            await network_service.broadcast_message({
                "type": "design_update",
                "design_id": str(design_id),
                "event": event.to_dict(),
                "timestamp": datetime.utcnow().isoformat()
            })
    
    connection_manager.broadcast_to_design = network_aware_broadcast


# =============================================================================
# BACKEND INTEGRATION
# =============================================================================

async def initialize_network_service(app: FastAPI):
    """Initialize network service during app startup."""
    try:
        # Check if network is enabled
        if os.getenv("NETWORK_ENABLED", "false").lower() != "true":
            logger.info("Network service disabled")
            return
            
        await network_service.initialize()
        logger.info("Network service initialized successfully")
        
        # Register network API endpoints
        create_network_router(app)
        
    except Exception as e:
        logger.error(f"Failed to initialize network service: {e}")
        # Continue without network features


async def shutdown_network_service():
    """Shutdown network service during app shutdown."""
    if network_service.is_initialized:
        await network_service.shutdown()
        logger.info("Network service shutdown complete")
