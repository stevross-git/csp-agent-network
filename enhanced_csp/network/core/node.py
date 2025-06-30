# enhanced_csp/network/core/node.py
"""
Base network node implementation for Enhanced CSP
"""

import asyncio
import logging
from typing import Dict, List, Optional, Set, Any, Callable
from datetime import datetime
import json
import os
import time
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from ...security.key_storage import SecureKeyManager
from cryptography.hazmat.backends import default_backend

from .types import (
    NodeID, NodeStatus, PeerInfo, NetworkStats, NetworkConfig,
    Transport, Connection, DHT, RoutingEntry, PeerType
)


logger = logging.getLogger(__name__)


class NetworkNode:
    """Base network node with P2P, mesh, DNS, and routing capabilities"""
    
    def __init__(self, config: NetworkConfig):
        self.config = config
        self.node_id: Optional[NodeID] = None
        self.status = NodeStatus.STARTING
        self.start_time = datetime.now()
        
        # Core components (will be initialized in start())
        self.transport: Optional[Transport] = None
        self.dht: Optional[DHT] = None
        self.routing_table: Dict[NodeID, RoutingEntry] = {}
        self.peers: Dict[NodeID, PeerInfo] = {}
        self.connections: Dict[NodeID, Connection] = {}

        # Registered agents and channels
        self.agents: Dict[str, Any] = {}
        self.channels: Dict[str, Any] = {}
        
        # Statistics
        self.stats = NetworkStats()
        
        # Event handlers
        self._event_handlers: Dict[str, List[Callable]] = {}
        
        # Background tasks
        self._tasks: List[asyncio.Task] = []
        
        # Initialize data directory
        self.data_dir = Path(config.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or generate node identity
        self._load_or_generate_identity()
    
    def _load_or_generate_identity(self):
        """Load existing node identity or generate new one"""
        key_id = self.config.node_key_path or "default"
        manager = SecureKeyManager()
        stored = manager.load_node_key(key_id)

        if stored and not manager.should_rotate(key_id):
            private_key = ed25519.Ed25519PrivateKey.from_private_bytes(stored)
            logger.info("Loaded node identity from secure storage")
        else:
            private_key = ed25519.Ed25519PrivateKey.generate()
            key_bytes = private_key.private_bytes(
                serialization.Encoding.Raw,
                serialization.PrivateFormat.Raw,
                serialization.NoEncryption()
            )
            manager.store_node_key(key_id, key_bytes)
            if stored:
                logger.info("Rotated node identity and stored securely")
            else:
                logger.info("Generated new node identity and stored securely")
        
        self.private_key = private_key
        self.public_key = private_key.public_key()
        self.node_id = NodeID.from_public_key(self.public_key)

        self.logger = logger.getChild(self.node_id.to_base58()[:8])
        self.logger.info(f"Node ID: {self.node_id.to_base58()}")
    
    async def start(self):
        """Start the network node"""
        logger.info("Starting network node...")
        
        try:
            self.status = NodeStatus.STARTING
            
            # Initialize transport layer
            await self._init_transport()
            
            # Initialize DHT
            await self._init_dht()
            
            # Start discovery
            self.status = NodeStatus.DISCOVERING
            await self._start_discovery()
            
            # Initialize mesh topology
            await self._init_mesh()
            
            # Initialize DNS overlay
            await self._init_dns()
            
            # Initialize adaptive routing
            await self._init_routing()
            
            # Start background tasks
            self._start_background_tasks()
            
            self.status = NodeStatus.CONNECTED
            logger.info("Network node started successfully")
            
            # Emit started event
            await self.emit_event("node_started", {
                "node_id": self.node_id.to_base58(),
                "status": self.status.name
            })
            
        except Exception as e:
            self.status = NodeStatus.ERROR
            logger.error(f"Failed to start network node: {e}")
            raise
    
    async def stop(self):
        """Stop the network node"""
        logger.info("Stopping network node...")
        
        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
        
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        
        # Close all connections
        for conn in list(self.connections.values()):
            await conn.close()
        
        # Stop transport
        if self.transport:
            await self.transport.close()
        
        self.status = NodeStatus.DISCONNECTED
        logger.info("Network node stopped")
    
    async def _init_transport(self):
        """Initialize transport layer (QUIC with TCP fallback)"""
        logger.info("Initializing transport layer...")
        raise NotImplementedError
    
    async def _init_dht(self):
        """Initialize Kademlia DHT"""
        logger.info("Initializing DHT...")
        raise NotImplementedError
    
    async def _start_discovery(self):
        """Start peer discovery (mDNS + bootstrap + DHT)"""
        logger.info("Starting peer discovery...")
        raise NotImplementedError
    
    async def _init_mesh(self):
        """Initialize mesh topology"""
        logger.info("Initializing mesh topology...")
        raise NotImplementedError
    
    async def _init_dns(self):
        """Initialize DNS overlay"""
        logger.info("Initializing DNS overlay...")
        raise NotImplementedError
    
    async def _init_routing(self):
        """Initialize adaptive routing"""
        logger.info("Initializing adaptive routing...")
        raise NotImplementedError
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        self._tasks.extend([
            asyncio.create_task(self._peer_maintenance_loop()),
            asyncio.create_task(self._metrics_collection_loop()),
            asyncio.create_task(self._routing_update_loop()),
        ])
    
    async def _peer_maintenance_loop(self):
        """Maintain peer connections and discover new peers"""
        while True:
            try:
                await asyncio.sleep(30)  # Run every 30 seconds
                
                # Remove stale peers
                now = time.monotonic()
                stale_peers = [
                    peer_id for peer_id, peer in self.peers.items()
                    if now - peer.last_seen.timestamp() > 300
                ]
                
                for peer_id in stale_peers:
                    await self.disconnect_peer(peer_id)
                
                # Discover new peers if below max
                if len(self.peers) < self.config.p2p.max_peers:
                    await self._discover_dht()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in peer maintenance: {e}")
    
    async def _metrics_collection_loop(self):
        """Collect network metrics"""
        while True:
            try:
                await asyncio.sleep(self.config.routing.metric_update_interval)
                
                # Update peer metrics
                for peer_id, peer in self.peers.items():
                    if peer_id in self.connections:
                        # Measure latency, bandwidth, packet loss
                        # This will be implemented with actual measurements
                        pass
                
                # Update statistics
                self.stats.active_connections = len(self.connections)
                self.stats.total_peers = len(self.peers)
                self.stats.super_peers = sum(
                    1 for p in self.peers.values() 
                    if p.peer_type == PeerType.SUPER_PEER
                )
                self.stats.uptime_seconds = (
                    datetime.now() - self.start_time
                ).total_seconds()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
    
    async def _routing_update_loop(self):
        """Update routing table periodically"""
        while True:
            try:
                await asyncio.sleep(self.config.mesh.routing_update_interval)
                
                # Update routing entries
                # This will be implemented by the routing module
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in routing update: {e}")
    
    async def connect_to_peer(self, address: str) -> Optional[PeerInfo]:
        """Connect to a peer by address"""
        try:
            # Parse multiaddr and establish connection
            # This will be implemented by the transport module
            logger.info(f"Connecting to peer at {address}")
            
            # Placeholder return
            return None
            
        except Exception as e:
            logger.error(f"Failed to connect to peer {address}: {e}")
            return None
    
    async def disconnect_peer(self, peer_id: NodeID):
        """Disconnect from a peer"""
        if peer_id in self.connections:
            await self.connections[peer_id].close()
            del self.connections[peer_id]
        
        if peer_id in self.peers:
            del self.peers[peer_id]
        
        logger.info(f"Disconnected from peer {peer_id.to_base58()}")
    
    async def send_message(self, peer_id: NodeID, message: Dict[str, Any]):
        """Send a message to a peer"""
        if peer_id not in self.connections:
            logger.error(f"No connection to peer {peer_id.to_base58()}")
            return

        try:
            data = self._serialize_message(message)
            if len(data) > self.config.max_message_size:
                raise ValueError("message too large")
            await self.connections[peer_id].send(data)
            
            self.stats.messages_sent += 1
            self.stats.bytes_sent += len(data)
            
        except Exception as e:
            logger.error(f"Failed to send message to {peer_id.to_base58()}: {e}")
    
    async def broadcast_message(self, message: Dict[str, Any],
                              exclude_peers: Optional[Set[NodeID]] = None):
        """Broadcast a message to all connected peers"""
        exclude_peers = exclude_peers or set()
        
        tasks = []
        for peer_id in self.connections:
            if peer_id not in exclude_peers:
                tasks.append(self.send_message(peer_id, message))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _serialize_message(self, message: Dict[str, Any]) -> bytes:
        """Serialize message to bytes"""
        return json.dumps(message).encode()

    def _deserialize_message(self, data: bytes) -> Dict[str, Any]:
        """Deserialize bytes back into a message."""
        try:
            return json.loads(data.decode())
        except Exception:
            return {}

    def register_agent(self, agent_id: str, agent: Any) -> None:
        """Register an agent for local dispatch."""
        self.agents[agent_id] = agent

    def register_channel(self, channel_id: str, channel: Any) -> None:
        """Register a communication channel."""
        self.channels[channel_id] = channel

    async def handle_raw_message(self, data: bytes) -> None:
        """Handle raw incoming data."""
        message = self._deserialize_message(data)
        await self._dispatch_message(message)

    async def _dispatch_message(self, message: Dict[str, Any]) -> None:
        recipient = message.get("recipient")
        if recipient in self.agents:
            agent = self.agents[recipient]
            if hasattr(agent, "receive_csp_message"):
                response = agent.receive_csp_message(message)
                if isinstance(response, dict):
                    await self._dispatch_message(response)
        else:
            await self.emit_event("message_received", message)
    
    def on_event(self, event_type: str, handler: Callable):
        """Register an event handler"""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
    
    async def emit_event(self, event_type: str, data: Any):
        """Emit an event to all registered handlers"""
        if event_type in self._event_handlers:
            for handler in self._event_handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data)
                    else:
                        handler(data)
                except Exception as e:
                    logger.error(f"Error in event handler for {event_type}: {e}")
    
    def get_stats(self) -> NetworkStats:
        """Get current network statistics"""
        return self.stats
    
    def get_peers(self) -> List[PeerInfo]:
        """Get list of connected peers"""
        return list(self.peers.values())
    
    def get_routing_table(self) -> List[RoutingEntry]:
        """Get current routing table"""
        return list(self.routing_table.values())
    
    # Placeholder methods for discovery
    async def _discover_mdns(self):
        """Discover peers using mDNS"""
        logger.info("Starting mDNS discovery...")
        raise NotImplementedError
    
    async def _discover_dht(self):
        """Discover peers using DHT"""
        logger.info("Starting DHT discovery...")
        raise NotImplementedError
    
    async def _connect_to_peer(self, address: str):
        """Connect to a peer"""
        logger.info(f"Connecting to peer: {address}")
        raise NotImplementedError
