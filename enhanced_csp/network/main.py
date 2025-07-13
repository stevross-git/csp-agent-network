#!/usr/bin/env python3
"""
Enhanced CSP Network - Unified Implementation
Production entry script with integrated core network node implementation.
Supports genesis nodes, peer discovery, security, and extensible components.
"""
import inspect
import asyncio
import signal
import sys
import os
import json
import logging
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Union
import argparse
from datetime import datetime, timedelta
import ipaddress
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Third-party imports
try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.prompt import Prompt
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    from aiohttp import web, ClientSession
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

# Enhanced CSP imports - use relative imports when running directly
try:
    # Try absolute imports first (for installed package)
    from enhanced_csp.network.core.config import (
        NetworkConfig, SecurityConfig, P2PConfig, MeshConfig, 
        DNSConfig, RoutingConfig
    )
    from enhanced_csp.network.core.types import (
        NodeID, NodeCapabilities, PeerInfo, NetworkMessage, MessageType
    )
    from enhanced_csp.network.security.security_hardening import SecurityOrchestrator
    from enhanced_csp.quantum_csp_engine import QuantumCSPEngine
    from enhanced_csp.blockchain_csp_network import BlockchainCSPNetwork
except ImportError:
    # Fall back to relative imports (for direct execution)
    sys.path.insert(0, str(Path(__file__).parent))
    from core.config import (
        NetworkConfig, SecurityConfig, P2PConfig, MeshConfig, 
        DNSConfig, RoutingConfig
    )
    from core.types import (
        NodeID, NodeCapabilities, PeerInfo, NetworkMessage, MessageType
    )
    sys.path.insert(0, str(Path(__file__).parent.parent))
    try:
        from security.security_hardening import SecurityOrchestrator
        from quantum_csp_engine import QuantumCSPEngine
        from blockchain_csp_network import BlockchainCSPNetwork
    except ImportError:
        # These modules may not exist, create placeholder classes
        class SecurityOrchestrator:
            def __init__(self, *args, **kwargs):
                pass
            async def initialize(self):
                pass
            async def shutdown(self):
                pass
            async def monitor_threats(self):
                pass
            async def rotate_tls_certificates(self):
                pass
                
        class QuantumCSPEngine:
            def __init__(self, *args, **kwargs):
                pass
            async def initialize(self):
                pass
            async def shutdown(self):
                pass
                
        class BlockchainCSPNetwork:
            def __init__(self, *args, **kwargs):
                pass
            async def initialize(self):
                pass
            async def shutdown(self):
                pass

# Constants
DEFAULT_BOOTSTRAP_NODES = []  # Empty for genesis node
DEFAULT_STUN_SERVERS = ["stun:stun.l.google.com:19302", "stun:stun.cloudflare.com:3478"]
DEFAULT_TURN_SERVERS = []
DEFAULT_LISTEN_ADDRESS = "0.0.0.0"
DEFAULT_LISTEN_PORT = 30300
DEFAULT_STATUS_PORT = 6969
LOG_ROTATION_DAYS = 7
TLS_ROTATION_DAYS = 30
GENESIS_DNS_RECORDS = {
    # Original .web4ai domains
    "seed1.web4ai": None,
    "seed2.web4ai": None,
    "bootstrap.web4ai": None,
    "genesis.web4ai": None,
    
    # New peoplesainetwork.com domains
    "genesis.peoplesainetwork.com": None,
    "bootstrap.peoplesainetwork.com": None,
    "seed1.peoplesainetwork.com": None,
    "seed2.peoplesainetwork.com": None,
    
    # Additional bootstrap points
    "boot1.peoplesainetwork.com": None,
    "boot2.peoplesainetwork.com": None,
}

# Global console for rich output
console = Console() if RICH_AVAILABLE else None

logger = logging.getLogger(__name__)


class SimpleRoutingStub:
    """Fallback routing stub so the node can still start if BatmanRouting is missing."""

    def __init__(self, node=None, topology=None):
        self.node = node
        self.topology = topology
        self.routing_table: Dict[str, Any] = {}
        self.is_running = False

    async def start(self):
        self.is_running = True
        logger.info("Simple routing stub started")
        return True

    async def stop(self):
        self.is_running = False
        logger.info("Simple routing stub stopped")

    def get_route(self, destination):
        return self.routing_table.get(destination)

    def get_all_routes(self, destination):
        route = self.routing_table.get(destination)
        return [route] if route else []


class EnhancedCSPNetwork:
    """
    Unified Enhanced CSP Network node combining core networking with production features.
    Integrates transport, discovery, routing, security, and optional quantum/blockchain components.
    """

    def __init__(self, config: Optional[NetworkConfig] = None):
        self.config = config or NetworkConfig()
        self.node_id = NodeID.generate()
        self._event_handlers: Dict[str, List[Callable]] = {}

        # Safely access config attributes with fallbacks
        self.capabilities = NodeCapabilities(
            relay=True,
            storage=getattr(self.config, 'enable_storage', False),
            compute=getattr(self.config, 'enable_compute', False),
            quantum=getattr(self.config, 'enable_quantum', False),
            blockchain=getattr(self.config, 'enable_blockchain', False),
            dns=getattr(self.config, 'enable_dns', False),
            bootstrap=False,
        )

        # Core components
        self.transport: Optional[Any] = None
        self.discovery: Optional[Any] = None
        self.dht: Optional[Any] = None
        self.nat: Optional[Any] = None
        self.topology: Optional[Any] = None
        self.routing: Optional[Any] = None
        self.dns_overlay: Optional[Any] = None
        self.adaptive_routing: Optional[Any] = None

        # Runtime state
        self.peers: Dict[NodeID, PeerInfo] = {}
        self.is_running = False
        self._message_handlers: Dict[MessageType, List[Callable]] = {}
        self._background_tasks: List[asyncio.Task] = []

        # Stats/metrics
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
            "peers_connected": 0,
            "start_time": None,
        }
        self.metrics = {
            "messages_sent": 0,
            "messages_received": 0,
            "peers_connected": 0,
            "bandwidth_in": 0,
            "bandwidth_out": 0,
            "routing_table_size": 0,
            "uptime": 0,
            "last_updated": time.time(),
        }

    # Event-bus helpers
    def on_event(self, name: str, handler: Callable[[dict], Any]) -> None:
        self._event_handlers.setdefault(name, []).append(handler)

    async def _dispatch_event(self, name: str, payload: dict) -> None:
        for fn in self._event_handlers.get(name, []):
            try:
                if asyncio.iscoroutinefunction(fn):
                    await fn(payload)
                else:
                    fn(payload)
            except Exception:
                logger.exception("Unhandled exception in %s handler", name)

    # Polymorphic send_message – supports both routing-level and API-level
    async def send_message(self, *args) -> bool:
        """Send either a raw dict to a peer or a high-level NetworkMessage.
        
        * send_message(peer_id: NodeID, data: dict) – low-level, used by routing layer
        * send_message(message: NetworkMessage) – high-level application API
        """
        if not self.transport or not self.is_running:
            logger.error("Transport not initialised or node not running")
            return False

        try:
            # High-level form – single NetworkMessage argument
            if len(args) == 1 and isinstance(args[0], NetworkMessage):
                nm: NetworkMessage = args[0]
                packet = {
                    "type": nm.type.value if hasattr(nm.type, "value") else str(nm.type),
                    "payload": nm.payload,
                    "sender": str(nm.sender),
                    "recipient": str(nm.recipient) if nm.recipient else None,
                    "timestamp": nm.timestamp.isoformat()
                    if hasattr(nm.timestamp, "isoformat")
                    else str(nm.timestamp),
                }
                peer = str(nm.recipient)
                success = await self.transport.send(peer, packet)
            # Low-level form – peer_id + raw dict
            elif len(args) == 2:
                peer, packet = args
                success = await self.transport.send(str(peer), packet)
            else:
                raise ValueError("Invalid arguments for send_message")

            if success:
                self.stats["messages_sent"] += 1
                self.stats["bytes_sent"] += len(str(packet))
                self.metrics["messages_sent"] += 1
                self.metrics["last_updated"] = time.time()
            return success
        except Exception as exc:
            logger.exception("Failed to send message: %s", exc)
            return False

    async def start(self) -> bool:
        """Start the network node and all subsystems."""
        if self.is_running:
            logger.warning("Node is already running")
            return True

        try:
            logger.info("Starting Enhanced CSP network node %s", self.node_id)
            await self._initialize_components()

            # Start components in order
            if self.transport and not await self._start_transport():
                return False
            if self.discovery and not await self._start_discovery():
                return False
            if self.dht and not await self._start_dht():
                return False
            if self.nat and not await self._start_nat():
                return False
            if self.topology and not await self._start_topology():
                return False
            if self.routing and not await self._start_routing():
                return False
            if self.dns_overlay and not await self._start_dns():
                return False
            if self.adaptive_routing and not await self._start_adaptive_routing():
                return False

            self._start_background_tasks()
            self.is_running = True
            self.stats["start_time"] = time.time()
            logger.info("Enhanced CSP network node %s started successfully", self.node_id)
            return True
        except Exception:
            logger.exception("Failed to start network node")
            await self.stop()
            return False

    async def _initialize_components(self):
        """Lazy-import and construct components."""
        try:
            # Transport
            from ..p2p.transport import MultiProtocolTransport
            self.transport = MultiProtocolTransport(self.config.p2p, self.config.security)

            # Discovery
            from ..p2p.discovery import HybridDiscovery
            self.discovery = HybridDiscovery(self.config.p2p, self.node_id)

            # DHT
            if self.config.enable_dht:
                from ..p2p.dht import KademliaDHT
                self.dht = KademliaDHT(self.node_id, self.transport)

            # NAT traversal
            from ..p2p.nat import NATTraversal
            self.nat = NATTraversal(self.config.p2p)

            # Mesh topology
            if self.config.enable_mesh:
                from ..mesh.topology import MeshTopologyManager
                async def _send(peer: str, pkt: Any) -> bool:
                    return await self.transport.send(peer, pkt)
                self.topology = MeshTopologyManager(self.node_id, self.config.mesh, _send)

            # Routing
            if getattr(self.config, "enable_routing", True) and self.topology:
                try:
                    from ..mesh.routing import BatmanRouting
                    self.routing = BatmanRouting(self, self.topology)
                    logger.info("BatmanRouting initialised successfully")
                except Exception as exc:
                    logger.warning("BatmanRouting unavailable (%s) – using stub", exc)
                    self.routing = SimpleRoutingStub(self, self.topology)

            # DNS overlay
            if self.config.enable_dns:
                from ..dns.overlay import DNSOverlay
                self.dns_overlay = DNSOverlay(self)

            # Adaptive routing (ML-based)
            if self.config.enable_adaptive_routing and self.routing:
                from ..routing.adaptive import AdaptiveRoutingEngine
                self.adaptive_routing = AdaptiveRoutingEngine(self, self.config.routing, self.routing)

        except ImportError as e:
            logger.warning("Some components unavailable: %s. Using stubs where possible.", e)
            # Create minimal stubs for missing components
            if not self.transport:
                class StubTransport:
                    def __init__(self):
                        self.is_running = True
                        self.handlers = {}
                    
                    async def start(self):
                        logger.info("Stub transport started")
                        return True
                    
                    async def stop(self):
                        logger.info("Stub transport stopped")
                        return True
                    
                    async def send(self, peer, data):
                        logger.debug(f"Stub transport: would send to {peer}: {data}")
                        return False
                    
                    def register_handler(self, event, handler):
                        self.handlers[event] = handler
                        logger.debug(f"Registered handler for {event}")
                
                self.transport = StubTransport()
            
            # Create stubs for other missing components
            if not self.discovery:
                class StubDiscovery:
                    def __init__(self):
                        self.is_running = True
                    
                    async def start(self):
                        logger.info("Stub discovery started")
                        return True
                    
                    async def stop(self):
                        logger.info("Stub discovery stopped")
                        return True
                
                self.discovery = StubDiscovery()
            
            if not self.nat:
                class StubNAT:
                    def __init__(self):
                        self.is_running = True
                    
                    async def start(self):
                        logger.info("Stub NAT traversal started")
                        return True
                    
                    async def stop(self):
                        logger.info("Stub NAT traversal stopped")
                        return True
                
                self.nat = StubNAT()
            
            if not self.topology:
                class StubTopology:
                    def __init__(self):
                        self.is_running = True
                    
                    async def start(self):
                        logger.info("Stub topology manager started")
                        return True
                    
                    async def stop(self):
                        logger.info("Stub topology manager stopped")
                        return True
                
                self.topology = StubTopology()
            
            if not self.routing:
                self.routing = SimpleRoutingStub(self, self.topology)
            
            if not self.dns_overlay:
                class StubDNS:
                    def __init__(self):
                        self.is_running = True
                        self.records = {}
                    
                    async def start(self):
                        logger.info("Stub DNS overlay started")
                        return True
                    
                    async def stop(self):
                        logger.info("Stub DNS overlay stopped")
                        return True
                    
                    async def register(self, domain, address):
                        self.records[domain] = address
                        logger.info(f"Registered DNS: {domain} -> {address}")
                        return True
                    
                    async def resolve(self, domain):
                        result = self.records.get(domain)
                        if result:
                            return result
                        raise Exception(f"Domain {domain} not found")
                    
                    async def list_records(self):
                        return self.records.copy()
                
                self.dns_overlay = StubDNS()

    async def _start_transport(self) -> bool:
        if not self.transport:
            logger.warning("No transport available")
            return False
        try:
            if hasattr(self.transport, 'start'):
                result = await self.transport.start()
                if result and hasattr(self.transport, 'register_handler'):
                    self.transport.register_handler("__raw__", self._raw_incoming)
                return result
            return True
        except Exception as e:
            logger.exception("Failed to start transport: %s", e)
            return False

    async def _raw_incoming(self, peer: str, packet: dict):
        """Handle incoming packets and dispatch to event bus."""
        await self._dispatch_event(packet.get("type", "unknown"), {**packet, "sender_id": peer})
        self.stats["messages_received"] += 1
        self.stats["bytes_received"] += len(str(packet))
        self.metrics["messages_received"] += 1
        self.metrics["last_updated"] = time.time()

    async def _start_discovery(self) -> bool:
        if not self.discovery:
            return True
        try:
            if hasattr(self.discovery, 'start'):
                await self.discovery.start()
            return True
        except Exception as e:
            logger.exception("Failed to start discovery: %s", e)
            return False

    async def _start_dht(self) -> bool:
        if not self.dht:
            return True
        try:
            if hasattr(self.dht, 'start'):
                await self.dht.start()
            return True
        except Exception as e:
            logger.exception("Failed to start DHT: %s", e)
            return False

    async def _start_nat(self) -> bool:
        if not self.nat:
            return True
        try:
            if hasattr(self.nat, 'start'):
                await self.nat.start()
            return True
        except Exception as e:
            logger.exception("Failed to start NAT traversal: %s", e)
            return False

    async def _start_topology(self) -> bool:
        if not self.topology:
            return True
        try:
            if hasattr(self.topology, 'start'):
                await self.topology.start()
            return True
        except Exception as e:
            logger.exception("Failed to start topology manager: %s", e)
            return False

    async def _start_routing(self) -> bool:
        if not self.routing:
            return True
        try:
            if hasattr(self.routing, 'start'):
                return await self.routing.start()
            return True
        except Exception as e:
            logger.exception("Failed to start routing layer: %s", e)
            return False

    async def _start_dns(self) -> bool:
        if not self.dns_overlay:
            return True
        try:
            if hasattr(self.dns_overlay, 'start'):
                await self.dns_overlay.start()
            return True
        except Exception as e:
            logger.exception("Failed to start DNS overlay: %s", e)
            return False

    async def _start_adaptive_routing(self) -> bool:
        if not self.adaptive_routing:
            return True
        try:
            if hasattr(self.adaptive_routing, 'start'):
                await self.adaptive_routing.start()
            return True
        except Exception as e:
            logger.exception("Failed to start adaptive routing: %s", e)
            return False

    async def stop(self) -> bool:
        """Stop the network node and all subsystems."""
        if not self.is_running:
            return True

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        await asyncio.gather(*self._background_tasks, return_exceptions=True)

        # Stop components in reverse order
        for component in (
            self.adaptive_routing,
            self.dns_overlay,
            self.routing,
            self.topology,
            self.nat,
            self.dht,
            self.discovery,
            self.transport,
        ):
            if component and hasattr(component, "stop"):
                try:
                    await component.stop()
                except Exception:
                    logger.exception("Error stopping %s", component)

        self.is_running = False
        logger.info("Enhanced CSP network node %s stopped", self.node_id)
        return True

    async def broadcast_message(self, message: NetworkMessage) -> int:
        """Broadcast a message to all connected peers."""
        if not self.is_running or not self.transport:
            return 0
        packet = {
            "type": message.type.value if hasattr(message.type, "value") else str(message.type),
            "payload": message.payload,
            "sender": str(message.sender),
            "timestamp": message.timestamp.isoformat() if hasattr(message.timestamp, "isoformat") else str(message.timestamp),
        }
        count = 0
        for pid in self.peers:
            if hasattr(self.transport, 'send') and await self.transport.send(str(pid), packet):
                count += 1
        if count:
            self.stats["messages_sent"] += count
            self.stats["bytes_sent"] += len(str(packet)) * count
            self.metrics["messages_sent"] += count
            self.metrics["last_updated"] = time.time()
        return count

    def _start_background_tasks(self):
        """Start background maintenance tasks."""
        self._background_tasks = [
            asyncio.create_task(self._peer_maintenance_loop()),
            asyncio.create_task(self._stats_update_loop()),
            asyncio.create_task(self._health_check_loop()),
        ]

    async def _peer_maintenance_loop(self):
        """Maintain peer connections and cleanup stale entries."""
        while self.is_running:
            try:
                now = datetime.utcnow()
                stale = [
                    pid for pid, info in self.peers.items() 
                    if hasattr(info, 'last_seen') and (now - info.last_seen).total_seconds() > 300
                ]
                for pid in stale:
                    del self.peers[pid]
                
                self.stats["peers_connected"] = len(self.peers)
                self.metrics["peers_connected"] = len(self.peers)
                self.metrics["routing_table_size"] = len(self.routing.routing_table) if self.routing and hasattr(self.routing, 'routing_table') else 0
                self.metrics["uptime"] = time.time() - self.stats["start_time"] if self.stats["start_time"] else 0
                self.metrics["last_updated"] = time.time()
                
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Peer maintenance error")
                await asyncio.sleep(60)

    async def _stats_update_loop(self):
        """Periodically log statistics."""
        while self.is_running:
            try:
                if self.stats["start_time"]:
                    uptime = time.time() - self.stats["start_time"]
                    logger.info(
                        "Node stats – Uptime: %.0fs, Sent: %d, Received: %d, Peers: %d",
                        uptime,
                        self.stats["messages_sent"],
                        self.stats["messages_received"],
                        self.stats["peers_connected"],
                    )
                await asyncio.sleep(300)  # Every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Stats loop error")
                await asyncio.sleep(300)

    async def _health_check_loop(self):
        """Monitor component health."""
        while self.is_running:
            try:
                healthy = True
                if self.transport and not getattr(self.transport, "is_running", True):
                    healthy = False
                if self.discovery and not getattr(self.discovery, "is_running", True):
                    healthy = False
                if not healthy:
                    logger.warning("Node health check failed")
                await asyncio.sleep(120)  # Every 2 minutes
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Health check error")
                await asyncio.sleep(120)

    # Utility methods
    def add_message_handler(self, message_type: MessageType, handler: Callable):
        """Add a message handler for a specific message type."""
        self._message_handlers.setdefault(message_type, []).append(handler)
        if self.transport and hasattr(self.transport, 'register_handler'):
            self.transport.register_handler(str(message_type), handler)

    def remove_message_handler(self, message_type: MessageType, handler: Callable):
        """Remove a message handler."""
        self._message_handlers.get(message_type, []).remove(handler)

    def get_stats(self) -> Dict[str, Any]:
        """Get current node statistics."""
        return self.stats.copy()

    def get_metrics(self) -> Dict[str, Any]:
        """Get current node metrics."""
        return self.metrics.copy()

    def get_peer_info(self, peer_id: NodeID) -> Optional[PeerInfo]:
        """Get information about a specific peer."""
        return self.peers.get(peer_id)

    def get_all_peers(self) -> Dict[NodeID, PeerInfo]:
        """Get information about all connected peers."""
        return self.peers.copy()

    def get_peers(self) -> List[PeerInfo]:
        """Get list of connected peers (compatibility method)."""
        return list(self.peers.values())


class NodeManager:
    """Manages the lifecycle of an Enhanced CSP node with production features."""
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.logger = self._setup_logging()
        self.config = self._build_config()
        self.network: Optional[EnhancedCSPNetwork] = None
        self.security_orchestrator: Optional[SecurityOrchestrator] = None
        self.quantum_engine: Optional[QuantumCSPEngine] = None
        self.blockchain: Optional[BlockchainCSPNetwork] = None
        self.shutdown_event = asyncio.Event()
        self.tasks: List[asyncio.Task] = []
        self.is_genesis = args.genesis
        
    async def initialize(self):
        """Initialize the node and all subsystems."""
        self.logger.info("Initializing Enhanced CSP Node...")
        
        if self.is_genesis:
            self.logger.info("Starting as GENESIS node - First bootstrap and DNS seed")
            await self._initialize_genesis_node()
            
        # Create network instance
        self.network = EnhancedCSPNetwork(self.config)
        
        # Attach security orchestrator
        self.security_orchestrator = SecurityOrchestrator(self.config.security)
        self.logger.info("Initializing security orchestrator")
        await self.security_orchestrator.initialize()
        
        # Attach quantum engine if enabled
        if self.args.enable_quantum:
            self.quantum_engine = QuantumCSPEngine(self.network)
            await self.quantum_engine.initialize()
            
        # Attach blockchain if enabled
        if self.args.enable_blockchain:
            self.blockchain = BlockchainCSPNetwork(self.network)
            await self.blockchain.initialize()
            
        # Start the network
        await self.network.start()
        
        self.logger.info(f"Node started with ID: {self.network.node_id}")
        
        # If genesis node, set up initial DNS records
        if self.is_genesis:
            await self._setup_genesis_dns()
            
        # Start background tasks
        self._start_background_tasks()
        
    async def _initialize_genesis_node(self):
        """Initialize special configuration for genesis node."""
        self.logger.info("Configuring genesis node settings...")
        
        # Ensure we're a super peer
        self.config.is_super_peer = True
        self.config.max_peers = 1000  # Higher limit for genesis
        
        # Disable bootstrap attempts since we ARE the bootstrap
        self.config.p2p.bootstrap_nodes = []
        
        # Enable all capabilities for genesis node
        self.config.node_capabilities = ["relay", "storage", "compute", "dns", "bootstrap"]
        
        # Set up genesis-specific security
        self.config.security.enable_ca_mode = getattr(self.config.security, 'enable_ca_mode', True)
        self.config.security.trust_anchors = getattr(self.config.security, 'trust_anchors', ["self"])
        
    async def _setup_genesis_dns(self):
        """Set up initial DNS records for the network."""
        self.logger.info("Setting up genesis DNS records...")
        
        try:
            # Get our public address
            public_ip = await self._get_public_ip()
            node_multiaddr = f"/ip4/{public_ip}/tcp/{self.config.p2p.listen_port}/p2p/{self.network.node_id}"
            
            # Register all genesis DNS names if DNS overlay is available
            if hasattr(self.network, 'dns_overlay') and self.network.dns_overlay:
                for domain, _ in GENESIS_DNS_RECORDS.items():
                    try:
                        if hasattr(self.network.dns_overlay, 'register'):
                            await self.network.dns_overlay.register(domain, node_multiaddr)
                            self.logger.info(f"Registered DNS: {domain} -> {node_multiaddr}")
                        else:
                            self.logger.warning(f"DNS overlay doesn't support registration for {domain}")
                    except Exception as e:
                        self.logger.error(f"Failed to register {domain}: {e}")
                
                # Also register our node ID as a DNS name
                short_id = str(self.network.node_id)[:16]
                try:
                    if hasattr(self.network.dns_overlay, 'register'):
                        await self.network.dns_overlay.register(f"{short_id}.web4ai", node_multiaddr)
                except Exception as e:
                    self.logger.error(f"Failed to register node ID DNS: {e}")
            else:
                self.logger.warning("DNS overlay not available, skipping DNS registration")
        except Exception as e:
            self.logger.error(f"Failed to setup genesis DNS: {e}")
            # Don't fail the entire startup for DNS issues
            pass
        
    async def _get_public_ip(self) -> str:
        """Get public IP address of this node."""
        # Try STUN first
        if self.config.p2p.stun_servers:
            try:
                import aiostun
                stun_client = aiostun.Client(self.config.p2p.stun_servers[0])
                response = await stun_client.get_external_address()
                return response['external_ip']
            except:
                pass
        
        # Fallback to external service
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get('https://api.ipify.org') as resp:
                    return await resp.text()
        except:
            # Last resort - use configured listen address
            if self.config.p2p.listen_address != "0.0.0.0":
                return self.config.p2p.listen_address
            return "127.0.0.1"  # Localhost fallback
    
    def _start_background_tasks(self):
        """Start background maintenance tasks."""
        # TLS key rotation - safely check if the attribute exists
        if hasattr(self.config, 'security') and hasattr(self.config.security, 'tls_rotation_interval') and self.config.security.tls_rotation_interval:
            self.tasks.append(asyncio.create_task(self._tls_rotation_task()))
            self.logger.info("TLS rotation task started")
        else:
            self.logger.info("TLS rotation disabled (no tls_rotation_interval)")
        
        # Metrics collection
        self.tasks.append(asyncio.create_task(self._metrics_collection_task()))
        
        # Security monitoring
        if self.security_orchestrator:
            self.tasks.append(asyncio.create_task(
                self.security_orchestrator.monitor_threats()
            ))
            self.logger.info("Security monitoring task started")
        
        # Genesis node maintenance
        if self.is_genesis:
            self.tasks.append(asyncio.create_task(self._genesis_maintenance_task()))
            self.logger.info("Genesis maintenance task started")
    
    async def _genesis_maintenance_task(self):
        """Maintenance tasks specific to genesis node."""
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                # Update DNS records if our IP changed
                if hasattr(self.network, 'dns_overlay') and self.network.dns_overlay:
                    current_ip = await self._get_public_ip()
                    node_multiaddr = f"/ip4/{current_ip}/tcp/{self.config.p2p.listen_port}/p2p/{self.network.node_id}"
                    
                    for domain in GENESIS_DNS_RECORDS.keys():
                        try:
                            if hasattr(self.network.dns_overlay, 'resolve'):
                                existing = await self.network.dns_overlay.resolve(domain)
                                if existing != node_multiaddr:
                                    await self.network.dns_overlay.register(domain, node_multiaddr)
                                    self.logger.info(f"Updated DNS record: {domain}")
                        except:
                            # Skip if DNS operations fail
                            pass
                
                # Log network statistics
                stats = await self.collect_metrics()
                self.logger.info(f"Genesis node stats: {stats.get('peers_connected', 0)} peers connected")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Genesis maintenance error: {e}")
    
    async def _tls_rotation_task(self):
        """Rotate TLS certificates periodically."""
        rotation_interval = timedelta(days=self.args.tls_rotation_days)
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(rotation_interval.total_seconds())
                self.logger.info("Rotating TLS certificates...")
                await self.security_orchestrator.rotate_tls_certificates()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"TLS rotation failed: {e}")
    
    async def _metrics_collection_task(self):
        """Collect metrics periodically."""
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Every minute
                metrics = await self.collect_metrics()
                
                # Log key metrics
                self.logger.debug(f"Metrics: peers={metrics.get('peers_connected', 0)}, "
                                f"messages_sent={metrics.get('messages_sent', 0)}, "
                                f"uptime={metrics.get('uptime', 0):.0f}s")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Metrics collection failed: {e}")
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect current node metrics."""
        if self.network:
            return self.network.get_metrics()
        
        # Default metrics if network not available
        return {
            "peers_connected": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "bandwidth_in": 0,
            "bandwidth_out": 0,
            "uptime": 0
        }
    
    async def shutdown(self):
        """Gracefully shutdown the node."""
        self.logger.info("Shutting down Enhanced CSP Node...")
        
        # Cancel background tasks
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Shutdown subsystems
        if self.blockchain:
            await self.blockchain.shutdown()
        if self.quantum_engine:
            await self.quantum_engine.shutdown()
        if self.security_orchestrator:
            self.logger.info("Shutting down security orchestrator")
            await self.security_orchestrator.shutdown()
        if self.network:
            await self.network.stop()
            
        self.logger.info("Node shutdown complete")
    
    def _build_config(self) -> NetworkConfig:
        """Build network configuration from command line arguments."""
        
        # Build security config - only include parameters that SecurityConfig actually supports
        try:
            # Try to create SecurityConfig with all parameters first
            security = SecurityConfig(
                enable_tls=not self.args.no_tls,
                enable_mtls=self.args.mtls,
                enable_pq_crypto=self.args.pq_crypto,
                enable_zero_trust=self.args.zero_trust,
                tls_cert_path=self.args.tls_cert,
                tls_key_path=self.args.tls_key,
                ca_cert_path=self.args.ca_cert,
                audit_log_path=Path(self.args.audit_log) if self.args.audit_log else None,
                enable_threat_detection=not self.args.no_threat_detection,
                enable_intrusion_prevention=self.args.ips,
                enable_compliance_mode=self.args.compliance,
                compliance_standards=self.args.compliance_standards.split(',') if self.args.compliance_standards else [],
                tls_rotation_interval=self.args.tls_rotation_days * 86400,
            )
        except TypeError as e:
            # If that fails, create with minimal parameters that are likely supported
            self.logger.warning(f"SecurityConfig initialization failed with full parameters: {e}")
            self.logger.info("Creating SecurityConfig with minimal parameters")
            
            # Create with just the basic parameters that are most likely to exist
            security = SecurityConfig(
                enable_tls=not self.args.no_tls,
            )
            
            # Try to set additional attributes if they exist
            if hasattr(security, 'enable_mtls'):
                security.enable_mtls = self.args.mtls
            if hasattr(security, 'enable_pq_crypto'):
                security.enable_pq_crypto = self.args.pq_crypto
            if hasattr(security, 'enable_zero_trust'):
                security.enable_zero_trust = self.args.zero_trust
            if hasattr(security, 'tls_cert_path'):
                security.tls_cert_path = self.args.tls_cert
            if hasattr(security, 'tls_key_path'):
                security.tls_key_path = self.args.tls_key
            if hasattr(security, 'ca_cert_path'):
                security.ca_cert_path = self.args.ca_cert
            if hasattr(security, 'audit_log_path'):
                security.audit_log_path = Path(self.args.audit_log) if self.args.audit_log else None
            if hasattr(security, 'enable_threat_detection'):
                security.enable_threat_detection = not self.args.no_threat_detection
            if hasattr(security, 'enable_intrusion_prevention'):
                security.enable_intrusion_prevention = self.args.ips
            if hasattr(security, 'enable_compliance_mode'):
                security.enable_compliance_mode = self.args.compliance
            if hasattr(security, 'compliance_standards'):
                security.compliance_standards = self.args.compliance_standards.split(',') if self.args.compliance_standards else []
            if hasattr(security, 'tls_rotation_interval'):
                security.tls_rotation_interval = self.args.tls_rotation_days * 86400
        
        # Build P2P config with similar error handling
        try:
            p2p = P2PConfig(
                listen_address=self.args.listen_address,
                listen_port=self.args.listen_port,
                bootstrap_nodes=self.args.bootstrap if not self.args.genesis else [],
                stun_servers=self.args.stun_servers or DEFAULT_STUN_SERVERS,
                turn_servers=self.args.turn_servers or [],
                max_peers=self.args.max_peers,
                enable_mdns=not self.args.no_mdns,
            )
        except TypeError as e:
            self.logger.warning(f"P2PConfig initialization failed: {e}")
            # Create minimal P2P config
            p2p = P2PConfig()
            if hasattr(p2p, 'listen_address'):
                p2p.listen_address = self.args.listen_address
            if hasattr(p2p, 'listen_port'):
                p2p.listen_port = self.args.listen_port
            if hasattr(p2p, 'bootstrap_nodes'):
                p2p.bootstrap_nodes = self.args.bootstrap if not self.args.genesis else []
            if hasattr(p2p, 'stun_servers'):
                p2p.stun_servers = self.args.stun_servers or DEFAULT_STUN_SERVERS
            if hasattr(p2p, 'turn_servers'):
                p2p.turn_servers = self.args.turn_servers or []
            if hasattr(p2p, 'max_peers'):
                p2p.max_peers = self.args.max_peers
            if hasattr(p2p, 'enable_mdns'):
                p2p.enable_mdns = not self.args.no_mdns
        
        # Build mesh config with error handling
        try:
            mesh = MeshConfig(
                max_peers=self.args.max_peers,
            )
        except TypeError as e:
            self.logger.warning(f"MeshConfig initialization failed: {e}")
            mesh = MeshConfig()
            if hasattr(mesh, 'max_peers'):
                mesh.max_peers = self.args.max_peers
        
        # Build DNS config with error handling
        try:
            dns = DNSConfig(
                root_domain=".web4ai",
            )
        except TypeError as e:
            self.logger.warning(f"DNSConfig initialization failed: {e}")
            dns = DNSConfig()
            if hasattr(dns, 'root_domain'):
                dns.root_domain = ".web4ai"
        
        # Build routing config with error handling
        try:
            routing = RoutingConfig(
                enable_qos=self.args.qos,
            )
        except TypeError as e:
            self.logger.warning(f"RoutingConfig initialization failed: {e}")
            routing = RoutingConfig()
            if hasattr(routing, 'enable_qos'):
                routing.enable_qos = self.args.qos
        
        # Build main network config
        try:
            config = NetworkConfig(
                # Core network settings
                network_id=self.args.network_id,
                listen_address=self.args.listen_address,
                listen_port=self.args.listen_port,
                node_capabilities=["relay", "storage"] if not self.args.genesis else ["relay", "storage", "compute", "dns", "bootstrap"],
                
                # Sub-configurations
                security=security,
                p2p=p2p,
                mesh=mesh,
                dns=dns,
                routing=routing,
                
                # Feature flags
                enable_discovery=True,
                enable_dht=not self.args.no_dht,
                enable_nat_traversal=not self.args.no_nat,
                enable_mesh=True,
                enable_dns=self.args.enable_dns or self.args.genesis,
                enable_adaptive_routing=True,
                enable_metrics=not self.args.no_metrics,
                enable_compression=not self.args.no_compression,
                enable_storage=self.args.enable_storage,
                enable_quantum=self.args.enable_quantum,
                enable_blockchain=self.args.enable_blockchain,
                enable_compute=self.args.enable_compute,
            )
        except TypeError as e:
            self.logger.warning(f"NetworkConfig initialization failed: {e}")
            # Create minimal network config
            config = NetworkConfig()
            
            # Set basic attributes that are most likely to exist
            if hasattr(config, 'network_id'):
                config.network_id = self.args.network_id
            if hasattr(config, 'listen_address'):
                config.listen_address = self.args.listen_address
            if hasattr(config, 'listen_port'):
                config.listen_port = self.args.listen_port
            if hasattr(config, 'node_capabilities'):
                config.node_capabilities = ["relay", "storage"] if not self.args.genesis else ["relay", "storage", "compute", "dns", "bootstrap"]
            if hasattr(config, 'security'):
                config.security = security
            if hasattr(config, 'p2p'):
                config.p2p = p2p
            if hasattr(config, 'mesh'):
                config.mesh = mesh
            if hasattr(config, 'dns'):
                config.dns = dns
            if hasattr(config, 'routing'):
                config.routing = routing
            if hasattr(config, 'enable_discovery'):
                config.enable_discovery = True
            if hasattr(config, 'enable_dht'):
                config.enable_dht = not self.args.no_dht
            if hasattr(config, 'enable_nat_traversal'):
                config.enable_nat_traversal = not self.args.no_nat
            if hasattr(config, 'enable_mesh'):
                config.enable_mesh = True
            if hasattr(config, 'enable_dns'):
                config.enable_dns = self.args.enable_dns or self.args.genesis
            if hasattr(config, 'enable_adaptive_routing'):
                config.enable_adaptive_routing = True
            if hasattr(config, 'enable_metrics'):
                config.enable_metrics = not self.args.no_metrics
            if hasattr(config, 'enable_compression'):
                config.enable_compression = not self.args.no_compression
            if hasattr(config, 'enable_storage'):
                config.enable_storage = self.args.enable_storage
            if hasattr(config, 'enable_quantum'):
                config.enable_quantum = self.args.enable_quantum
            if hasattr(config, 'enable_blockchain'):
                config.enable_blockchain = self.args.enable_blockchain
            if hasattr(config, 'enable_compute'):
                config.enable_compute = self.args.enable_compute
        
        return config


    def _build_config_safe(self) -> NetworkConfig:
        """Build network configuration with safe parameter checking using introspection."""
        
        # Import inspect module for parameter introspection
        import inspect
        
        # Helper function to get supported parameters for any class
        def get_supported_params(cls):
            try:
                sig = inspect.signature(cls.__init__)
                return list(sig.parameters.keys())
            except Exception as e:
                self.logger.warning(f"Could not inspect {cls.__name__}: {e}")
                return ['self']  # Minimal fallback
        
        # Get supported parameters for each config class
        security_params = get_supported_params(SecurityConfig)
        p2p_params = get_supported_params(P2PConfig)
        mesh_params = get_supported_params(MeshConfig)
        dns_params = get_supported_params(DNSConfig)
        routing_params = get_supported_params(RoutingConfig)
        network_params = get_supported_params(NetworkConfig)
        
        self.logger.debug(f"SecurityConfig accepts: {security_params}")
        self.logger.debug(f"P2PConfig accepts: {p2p_params}")
        self.logger.debug(f"NetworkConfig accepts: {network_params}")
        
        # Build security config with only supported parameters
        security_kwargs = {}
        
        if 'enable_tls' in security_params:
            security_kwargs['enable_tls'] = not self.args.no_tls
        if 'enable_mtls' in security_params:
            security_kwargs['enable_mtls'] = self.args.mtls
        if 'enable_pq_crypto' in security_params:
            security_kwargs['enable_pq_crypto'] = self.args.pq_crypto
        if 'enable_zero_trust' in security_params:
            security_kwargs['enable_zero_trust'] = self.args.zero_trust
        if 'tls_cert_path' in security_params:
            security_kwargs['tls_cert_path'] = self.args.tls_cert
        if 'tls_key_path' in security_params:
            security_kwargs['tls_key_path'] = self.args.tls_key
        if 'ca_cert_path' in security_params:
            security_kwargs['ca_cert_path'] = self.args.ca_cert
        if 'audit_log_path' in security_params:
            security_kwargs['audit_log_path'] = Path(self.args.audit_log) if self.args.audit_log else None
        if 'enable_threat_detection' in security_params:
            security_kwargs['enable_threat_detection'] = not self.args.no_threat_detection
        if 'enable_intrusion_prevention' in security_params:
            security_kwargs['enable_intrusion_prevention'] = self.args.ips
        if 'enable_compliance_mode' in security_params:
            security_kwargs['enable_compliance_mode'] = self.args.compliance
        if 'compliance_standards' in security_params:
            security_kwargs['compliance_standards'] = self.args.compliance_standards.split(',') if self.args.compliance_standards else []
        if 'tls_rotation_interval' in security_params:
            security_kwargs['tls_rotation_interval'] = self.args.tls_rotation_days * 86400
        
        # Create security config with only supported parameters
        security = SecurityConfig(**security_kwargs)
        
        # Build P2P config with only supported parameters
        p2p_kwargs = {}
        
        if 'listen_address' in p2p_params:
            p2p_kwargs['listen_address'] = self.args.listen_address
        if 'listen_port' in p2p_params:
            p2p_kwargs['listen_port'] = self.args.listen_port
        if 'bootstrap_nodes' in p2p_params:
            p2p_kwargs['bootstrap_nodes'] = self.args.bootstrap if not self.args.genesis else []
        if 'stun_servers' in p2p_params:
            p2p_kwargs['stun_servers'] = self.args.stun_servers or DEFAULT_STUN_SERVERS
        if 'turn_servers' in p2p_params:
            p2p_kwargs['turn_servers'] = self.args.turn_servers or []
        if 'max_peers' in p2p_params:
            p2p_kwargs['max_peers'] = self.args.max_peers
        if 'enable_mdns' in p2p_params:
            p2p_kwargs['enable_mdns'] = not self.args.no_mdns
        
        p2p = P2PConfig(**p2p_kwargs)
        
        # Build mesh config with only supported parameters
        mesh_kwargs = {}
        
        if 'max_peers' in mesh_params:
            mesh_kwargs['max_peers'] = self.args.max_peers
        
        mesh = MeshConfig(**mesh_kwargs)
        
        # Build DNS config with only supported parameters
        dns_kwargs = {}
        
        if 'root_domain' in dns_params:
            dns_kwargs['root_domain'] = ".web4ai"
        
        dns = DNSConfig(**dns_kwargs)
        
        # Build routing config with only supported parameters
        routing_kwargs = {}
        
        if 'enable_qos' in routing_params:
            routing_kwargs['enable_qos'] = self.args.qos
        
        routing = RoutingConfig(**routing_kwargs)
        
        # Build main network config with only supported parameters
        network_kwargs = {}
        
        if 'network_id' in network_params:
            network_kwargs['network_id'] = self.args.network_id
        if 'listen_address' in network_params:
            network_kwargs['listen_address'] = self.args.listen_address
        if 'listen_port' in network_params:
            network_kwargs['listen_port'] = self.args.listen_port
        if 'node_capabilities' in network_params:
            network_kwargs['node_capabilities'] = ["relay", "storage"] if not self.args.genesis else ["relay", "storage", "compute", "dns", "bootstrap"]
        if 'security' in network_params:
            network_kwargs['security'] = security
        if 'p2p' in network_params:
            network_kwargs['p2p'] = p2p
        if 'mesh' in network_params:
            network_kwargs['mesh'] = mesh
        if 'dns' in network_params:
            network_kwargs['dns'] = dns
        if 'routing' in network_params:
            network_kwargs['routing'] = routing
        if 'enable_discovery' in network_params:
            network_kwargs['enable_discovery'] = True
        if 'enable_dht' in network_params:
            network_kwargs['enable_dht'] = not self.args.no_dht
        if 'enable_nat_traversal' in network_params:
            network_kwargs['enable_nat_traversal'] = not self.args.no_nat
        if 'enable_mesh' in network_params:
            network_kwargs['enable_mesh'] = True
        if 'enable_dns' in network_params:
            network_kwargs['enable_dns'] = self.args.enable_dns or self.args.genesis
        if 'enable_adaptive_routing' in network_params:
            network_kwargs['enable_adaptive_routing'] = True
        if 'enable_metrics' in network_params:
            network_kwargs['enable_metrics'] = not self.args.no_metrics
        if 'enable_compression' in network_params:
            network_kwargs['enable_compression'] = not self.args.no_compression
        if 'enable_storage' in network_params:
            network_kwargs['enable_storage'] = self.args.enable_storage
        if 'enable_quantum' in network_params:
            network_kwargs['enable_quantum'] = self.args.enable_quantum
        if 'enable_blockchain' in network_params:
            network_kwargs['enable_blockchain'] = self.args.enable_blockchain
        if 'enable_compute' in network_params:
            network_kwargs['enable_compute'] = self.args.enable_compute
        
        config = NetworkConfig(**network_kwargs)
        
        return config
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        log_level = getattr(logging, self.args.log_level.upper())
        
        if RICH_AVAILABLE and not self.args.no_shell:
            # Rich logging handler
            logging.basicConfig(
                level=log_level,
                format="%(message)s",
                handlers=[
                    RichHandler(
                        rich_tracebacks=True,
                        tracebacks_show_locals=self.args.debug,
                    )
                ]
            )
        else:
            # Standard logging
            logging.basicConfig(
                level=log_level,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        return logging.getLogger("enhanced_csp.main")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Enhanced CSP Network Node",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
# Start as genesis node (first in network)
%(prog)s --genesis

# Join existing network via bootstrap
%(prog)s --bootstrap /ip4/1.2.3.4/tcp/30300/p2p/Qm...

# Join via DNS seed
%(prog)s --bootstrap genesis.web4ai

# Enable all features
%(prog)s --enable-quantum --enable-blockchain --super-peer
"""
    )
    
    # Network options
    network = parser.add_argument_group('network')
    network.add_argument('--genesis', action='store_true',
                        help='Start as genesis node (first bootstrap)')
    network.add_argument('--bootstrap', nargs='*', default=[],
                        help='Bootstrap nodes (multiaddr or .web4ai domain)')
    network.add_argument('--listen-address', default=DEFAULT_LISTEN_ADDRESS,
                        help='Listen address (default: %(default)s)')
    network.add_argument('--listen-port', type=int, default=DEFAULT_LISTEN_PORT,
                        help='Listen port (default: %(default)s)')
    network.add_argument('--network-id', default='enhanced-csp',
                        help='Network identifier (default: %(default)s)')
    network.add_argument('--super-peer', action='store_true',
                        help='Run as super peer with higher capacity')
    network.add_argument('--max-peers', type=int, default=100,
                        help='Maximum peer connections (default: %(default)s)')
    
    # NAT traversal
    nat = parser.add_argument_group('NAT traversal')
    nat.add_argument('--stun-servers', nargs='*',
                    help='STUN servers for NAT detection')
    nat.add_argument('--turn-servers', nargs='*',
                    help='TURN servers for relay')
    nat.add_argument('--no-nat', action='store_true',
                    help='Disable NAT traversal')
    nat.add_argument('--no-upnp', action='store_true',
                    help='Disable UPnP port mapping')
    
    # Security options
    security = parser.add_argument_group('security')
    security.add_argument('--no-tls', action='store_true',
                        help='Disable TLS encryption')
    security.add_argument('--mtls', action='store_true',
                        help='Enable mutual TLS')
    security.add_argument('--tls-cert', help='TLS certificate path')
    security.add_argument('--tls-key', help='TLS private key path')
    security.add_argument('--ca-cert', help='CA certificate path')
    security.add_argument('--pq-crypto', action='store_true',
                        help='Enable post-quantum cryptography')
    security.add_argument('--zero-trust', action='store_true',
                        help='Enable zero-trust security model')
    security.add_argument('--audit-log', help='Audit log file path')
    security.add_argument('--no-threat-detection', action='store_true',
                        help='Disable threat detection')
    security.add_argument('--ips', action='store_true',
                        help='Enable intrusion prevention')
    security.add_argument('--compliance', action='store_true',
                        help='Enable compliance mode')
    security.add_argument('--compliance-standards',
                        help='Comma-separated compliance standards')
    security.add_argument('--tls-rotation-days', type=int, default=TLS_ROTATION_DAYS,
                        help='TLS certificate rotation interval (default: %(default)s)')
    
    # Features
    features = parser.add_argument_group('features')
    features.add_argument('--enable-quantum', action='store_true',
                        help='Enable quantum CSP engine')
    features.add_argument('--enable-blockchain', action='store_true',
                        help='Enable blockchain integration')
    features.add_argument('--enable-storage', action='store_true',
                        help='Enable distributed storage')
    features.add_argument('--enable-compute', action='store_true',
                        help='Enable distributed compute')
    features.add_argument('--enable-dns', action='store_true',
                        help='Enable DNS overlay service')
    features.add_argument('--no-relay', action='store_true',
                        help='Disable relay functionality')
    features.add_argument('--no-dht', action='store_true',
                        help='Disable DHT')
    features.add_argument('--no-mdns', action='store_true',
                        help='Disable mDNS discovery')
    
    # Performance
    perf = parser.add_argument_group('performance')
    perf.add_argument('--no-compression', action='store_true',
                    help='Disable message compression')
    perf.add_argument('--no-encryption', action='store_true',
                    help='Disable message encryption')
    perf.add_argument('--qos', action='store_true',
                    help='Enable QoS traffic shaping')
    perf.add_argument('--bandwidth-limit', type=int, default=0,
                    help='Bandwidth limit in KB/s (0=unlimited)')
    perf.add_argument('--routing', default='batman-adv',
                    choices=['batman-adv', 'babel', 'olsr'],
                    help='Routing algorithm (default: %(default)s)')
    
    # Monitoring
    monitor = parser.add_argument_group('monitoring')
    monitor.add_argument('--status-port', type=int, default=DEFAULT_STATUS_PORT,
                        help='Status HTTP server port (default: %(default)s)')
    monitor.add_argument('--no-status', action='store_true',
                        help='Disable status server')
    monitor.add_argument('--no-metrics', action='store_true',
                        help='Disable metrics collection')
    monitor.add_argument('--metrics-interval', type=int, default=60,
                        help='Metrics collection interval (default: %(default)s)')
    
    # DNS/DHT
    discovery = parser.add_argument_group('discovery')
    discovery.add_argument('--dns-seeds', nargs='*',
                        help='DNS seed domains for discovery')
    discovery.add_argument('--dht-bootstrap', nargs='*',
                        help='DHT bootstrap nodes')
    discovery.add_argument('--no-ipv6', action='store_true',
                        help='Disable IPv6 support')
    
    # Logging
    parser.add_argument('--log-level', default='INFO',
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                    help='Log level (default: %(default)s)')
    parser.add_argument('--debug', action='store_true',
                    help='Enable debug mode')
    parser.add_argument('--no-shell', action='store_true',
                    help='Disable interactive shell')
    
    return parser.parse_args()


class InteractiveShell:
    """Interactive command shell for the node."""
    
    def __init__(self, manager: NodeManager):
        self.manager = manager
        self.commands = {
            'help': self.cmd_help,
            'peers': self.cmd_peers,
            'dns': self.cmd_dns,
            'send': self.cmd_send,
            'stats': self.cmd_stats,
            'loglevel': self.cmd_loglevel,
            'quit': self.cmd_quit,
        }
    
    async def run(self):
        """Run the interactive shell."""
        if RICH_AVAILABLE:
            console.print("\n[bold cyan]Enhanced CSP Node Interactive Shell[/bold cyan]")
            console.print("Type 'help' for available commands\n")
        else:
            print("\nEnhanced CSP Node Interactive Shell")
            print("Type 'help' for available commands\n")
        
        while not self.manager.shutdown_event.is_set():
            try:
                if RICH_AVAILABLE:
                    command = await asyncio.get_event_loop().run_in_executor(
                        None, Prompt.ask, "[bold]csp>[/bold]"
                    )
                else:
                    command = await asyncio.get_event_loop().run_in_executor(
                        None, input, "csp>: "
                    )
                
                if not command:
                    continue
                
                parts = command.strip().split()
                if not parts:
                    continue
                
                cmd = parts[0].lower()
                args = parts[1:]
                
                if cmd in self.commands:
                    await self.commands[cmd](args)
                else:
                    print(f"Unknown command: {cmd}")
                    
            except (EOFError, KeyboardInterrupt):
                print("\nUse 'quit' to exit")
            except Exception as e:
                print(f"Error: {e}")
    
    async def cmd_help(self, args: List[str]):
        """Show help message."""
        help_text = """
Available commands:
help              - Show this help message
peers             - List connected peers
dns <name>        - Resolve .web4ai domain
send <peer> <msg> - Send message to peer
stats             - Show node statistics
loglevel <level>  - Set logging level
quit              - Exit the shell
"""
        print(help_text)
    
    async def cmd_peers(self, args: List[str]):
        """List connected peers."""
        try:
            peers = self.manager.network.get_peers() if hasattr(self.manager.network, 'get_peers') else []
        except:
            peers = []
            
        if not peers:
            print("No connected peers")
            return
        
        if RICH_AVAILABLE:
            table = Table(title="Connected Peers")
            table.add_column("Peer ID", style="cyan")
            table.add_column("Address", style="green")
            table.add_column("Latency", style="yellow")
            table.add_column("Reputation", style="blue")
            
            for peer in peers:
                table.add_row(
                    str(getattr(peer, 'id', 'unknown'))[:16] + "...",
                    f"{getattr(peer, 'address', 'unknown')}:{getattr(peer, 'port', 0)}",
                    f"{getattr(peer, 'latency', 0):.2f}ms" if getattr(peer, 'latency', None) else "N/A",
                    f"{getattr(peer, 'reputation', 0):.2f}"
                )
            console.print(table)
        else:
            print(f"\nConnected peers ({len(peers)}):")
            for peer in peers:
                print(f"  {getattr(peer, 'id', 'unknown')}: {getattr(peer, 'address', 'unknown')}:{getattr(peer, 'port', 0)}")
    
    async def cmd_dns(self, args: List[str]):
        """DNS operations."""
        if not args:
            print("Usage: dns <name>")
            print("       dns list              - List all DNS records (genesis only)")
            print("       dns register <name> <addr> - Register DNS name (genesis only)")
            return
        
        if not hasattr(self.manager.network, 'dns_overlay') or not self.manager.network.dns_overlay:
            print("DNS overlay not available")
            return
        
        if args[0] == "list" and self.manager.is_genesis:
            try:
                if hasattr(self.manager.network.dns_overlay, 'list_records'):
                    records = await self.manager.network.dns_overlay.list_records()
                else:
                    records = getattr(self.manager.network.dns_overlay, 'records', {})
                    
                if RICH_AVAILABLE:
                    table = Table(title="DNS Records")
                    table.add_column("Domain", style="cyan")
                    table.add_column("Address", style="green")
                    for domain, addr in records.items():
                        table.add_row(domain, addr)
                    console.print(table)
                else:
                    print("\nDNS Records:")
                    for domain, addr in records.items():
                        print(f"  {domain} -> {addr}")
            except Exception as e:
                print(f"Failed to list DNS records: {e}")
            return
        
        if args[0] == "register" and len(args) >= 3 and self.manager.is_genesis:
            domain = args[1]
            addr = " ".join(args[2:])
            try:
                if hasattr(self.manager.network.dns_overlay, 'register'):
                    await self.manager.network.dns_overlay.register(domain, addr)
                    print(f"Registered: {domain} -> {addr}")
                else:
                    print("DNS registration not available")
            except Exception as e:
                print(f"Failed to register: {e}")
            return
        
        # Regular DNS resolution
        name = args[0]
        try:
            if hasattr(self.manager.network.dns_overlay, 'resolve'):
                result = await self.manager.network.dns_overlay.resolve(name)
                print(f"{name} -> {result}")
            else:
                print("DNS resolution not available")
        except Exception as e:
            print(f"Failed to resolve {name}: {e}")
    
    async def cmd_send(self, args: List[str]):
        """Send message to peer."""
        if len(args) < 2:
            print("Usage: send <peer_id> <message>")
            return
        
        peer_id = args[0]
        message = " ".join(args[1:])
        
        try:
            if hasattr(self.manager.network, 'send_message'):
                # Use low-level send_message interface
                success = await self.manager.network.send_message(peer_id, {"content": message, "type": "chat"})
                if success:
                    print(f"Message sent to {peer_id}")
                else:
                    print(f"Failed to send message to {peer_id}")
            else:
                print("Message sending not implemented")
        except Exception as e:
            print(f"Failed to send message: {e}")
    
    async def cmd_stats(self, args: List[str]):
        """Show node statistics."""
        metrics = await self.manager.collect_metrics()
        
        if RICH_AVAILABLE:
            table = Table(title="Node Statistics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            for key, value in metrics.items():
                if key == "uptime":
                    value = f"{value:.0f}s"
                elif key in ["bandwidth_in", "bandwidth_out"]:
                    value = f"{value / 1024 / 1024:.2f} MB"
                table.add_row(key.replace("_", " ").title(), str(value))
            
            console.print(table)
        else:
            print("\nNode Statistics:")
            for key, value in metrics.items():
                print(f"  {key}: {value}")
    
    async def cmd_loglevel(self, args: List[str]):
        """Set logging level."""
        if not args:
            print("Usage: loglevel <debug|info|warning|error>")
            return
        
        level = args[0].upper()
        try:
            logging.getLogger().setLevel(getattr(logging, level))
            print(f"Log level set to {level}")
        except AttributeError:
            print(f"Invalid log level: {level}")
    
    async def cmd_quit(self, args: List[str]):
        """Exit the shell."""
        print("Exiting shell...")
        self.manager.shutdown_event.set()


class StatusServer:
    """HTTP/WebSocket status endpoint server."""
    
    def __init__(self, manager: NodeManager, port: int):
        self.manager = manager
        self.port = port
        self.app = web.Application()
        self.setup_routes()
    
    def setup_routes(self):
        """Setup HTTP routes."""
        # Dashboard 
        self.app.router.add_get('/', self.handle_root)
        
        # API endpoints
        self.app.router.add_get('/api/info', self.handle_api_info)
        self.app.router.add_get('/api/status', self.handle_api_status)
        self.app.router.add_get('/api/peers', self.handle_api_peers)
        self.app.router.add_get('/api/dns', self.handle_api_dns)
        self.app.router.add_post('/api/connect', self.handle_api_connect)
        
        # Legacy endpoints
        self.app.router.add_get('/metrics', self.handle_metrics)
        self.app.router.add_get('/info', self.handle_info)
        self.app.router.add_get('/health', self.handle_health)
    
    async def handle_root(self, request: web.Request) -> web.Response:
        """Serve the dashboard HTML."""
        dashboard_path = Path(__file__).parent / 'dashboard' / 'index.html'
        if dashboard_path.exists():
            return web.FileResponse(dashboard_path)
        else:
            # Fallback to simple status page
            html = """
            <!DOCTYPE html>
            <html>
            <head><title>Enhanced CSP Node</title></head>
            <body>
                <h1>Enhanced CSP Node Status</h1>
                <p>Node is running. Visit /api/status for JSON data.</p>
                <ul>
                    <li><a href="/api/info">Node Info</a></li>
                    <li><a href="/api/status">Status</a></li>
                    <li><a href="/api/peers">Peers</a></li>
                    <li><a href="/api/dns">DNS Records</a></li>
                </ul>
            </body>
            </html>
            """
            return web.Response(text=html, content_type='text/html')

    async def handle_api_info(self, request: web.Request) -> web.Response:
        """API endpoint for node information."""
        try:
            info = {
                "node_id": str(self.manager.network.node_id),
                "version": "1.0.0",
                "is_genesis": self.manager.is_genesis,
                "network_id": getattr(self.manager.config, 'network_id', 'unknown'),
                "listen_address": f"{getattr(self.manager.config, 'listen_address', '0.0.0.0')}:{getattr(self.manager.config, 'listen_port', 30300)}",
                "capabilities": getattr(self.manager.config, 'node_capabilities', [])
            }
            return web.json_response(info)
        except Exception as e:
            self.manager.logger.error(f"Error in handle_api_info: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_api_status(self, request: web.Request) -> web.Response:
        """API endpoint for node metrics."""
        try:
            metrics = await self.manager.collect_metrics()
            return web.json_response(metrics)
        except Exception as e:
            self.manager.logger.error(f"Error in handle_api_status: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_api_peers(self, request: web.Request) -> web.Response:
        """API endpoint for peer list."""
        try:
            if hasattr(self.manager.network, 'get_peers'):
                peers = self.manager.network.get_peers()
            else:
                peers = []
            
            peer_list = []
            for peer in peers:
                peer_data = {
                    "id": str(getattr(peer, 'id', 'unknown')),
                    "address": getattr(peer, 'address', 'unknown'),
                    "port": getattr(peer, 'port', 0),
                    "latency": getattr(peer, 'latency', 0),
                    "reputation": getattr(peer, 'reputation', 0),
                    "last_seen": getattr(peer, 'last_seen', None)
                }
                if peer_data["last_seen"] and hasattr(peer_data["last_seen"], 'isoformat'):
                    peer_data["last_seen"] = peer_data["last_seen"].isoformat()
                peer_list.append(peer_data)
            
            return web.json_response(peer_list)
        except Exception as e:
            self.manager.logger.error(f"Error in handle_api_peers: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_api_dns(self, request: web.Request) -> web.Response:
        """API endpoint for DNS records."""
        try:
            if hasattr(self.manager.network, 'dns_overlay') and self.manager.network.dns_overlay:
                if hasattr(self.manager.network.dns_overlay, 'list_records'):
                    records = await self.manager.network.dns_overlay.list_records()
                else:
                    records = getattr(self.manager.network.dns_overlay, 'records', {})
            else:
                records = {}
            return web.json_response(records)
        except Exception as e:
            self.manager.logger.error(f"Error in handle_api_dns: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_api_connect(self, request: web.Request) -> web.Response:
        """API endpoint to connect to a peer."""
        try:
            data = await request.json()
            address = data.get('address')
            if address:
                if hasattr(self.manager.network, 'connect'):
                    await self.manager.network.connect(address)
                    return web.json_response({"status": "connecting", "address": address})
                else:
                    return web.json_response({"error": "Connect method not implemented"}, status=501)
            else:
                return web.json_response({"error": "No address provided"}, status=400)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
    
    async def handle_metrics(self, request: web.Request) -> web.Response:
        """Prometheus-compatible metrics endpoint."""
        metrics = await self.manager.collect_metrics()

        # Format as Prometheus metrics
        lines = [
            "# HELP enhanced_csp_peers Number of connected peers",
            "# TYPE enhanced_csp_peers gauge",
            f"enhanced_csp_peers {metrics.get('peers_connected', 0)}",
            "",
            "# HELP enhanced_csp_messages_sent Total messages sent",
            "# TYPE enhanced_csp_messages_sent counter",
            f"enhanced_csp_messages_sent {metrics.get('messages_sent', 0)}",
            "",
            "# HELP enhanced_csp_messages_received Total messages received",
            "# TYPE enhanced_csp_messages_received counter",
            f"enhanced_csp_messages_received {metrics.get('messages_received', 0)}",
            "",
            "# HELP enhanced_csp_bandwidth_in_bytes Bandwidth in (bytes)",
            "# TYPE enhanced_csp_bandwidth_in_bytes counter",
            f"enhanced_csp_bandwidth_in_bytes {metrics.get('bandwidth_in', 0)}",
            "",
            "# HELP enhanced_csp_bandwidth_out_bytes Bandwidth out (bytes)",
            "# TYPE enhanced_csp_bandwidth_out_bytes counter",
            f"enhanced_csp_bandwidth_out_bytes {metrics.get('bandwidth_out', 0)}",
            "",
            "# HELP enhanced_csp_uptime_seconds Node uptime in seconds",
            "# TYPE enhanced_csp_uptime_seconds gauge",
            f"enhanced_csp_uptime_seconds {metrics.get('uptime', 0)}",
        ]
        
        return web.Response(text="\n".join(lines), content_type="text/plain")
    
    async def handle_info(self, request: web.Request) -> web.Response:
        """Node information endpoint."""
        try:
            info = {
                "node_id": str(self.manager.network.node_id),
                "version": "1.0.0",
                "network_id": getattr(self.manager.config, 'network_id', 'unknown'),
                "is_genesis": self.manager.is_genesis,
                "capabilities": getattr(self.manager.config, 'node_capabilities', []),
                "security": {
                    "tls": self.manager.config.security.enable_tls,
                    "mtls": self.manager.config.security.enable_mtls,
                    "pq_crypto": self.manager.config.security.enable_pq_crypto,
                    "zero_trust": self.manager.config.security.enable_zero_trust,
                }
            }
            return web.json_response(info)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
    
    async def handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        if self.manager.network and getattr(self.manager.network, 'is_running', False):
            return web.json_response({"status": "healthy"})
        else:
            return web.json_response({"status": "unhealthy"}, status=503)
    
    async def start(self):
        """Start the status server."""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', self.port)
        await site.start()
        logging.getLogger("enhanced_csp.main").info(
            f"Status server started on http://0.0.0.0:{self.port}"
        )
        return runner


async def run_main(args: argparse.Namespace):
    """Main execution function."""
    manager = NodeManager(args)
    shell = InteractiveShell(manager) if not args.no_shell else None
    status_server = None
    
    # Setup signal handlers
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(
            sig, 
            lambda s=sig: asyncio.create_task(handle_signal(s, manager))
        )
    
    try:
        # Initialize node
        await manager.initialize()
        
        # Start status server if enabled
        if not args.no_status and AIOHTTP_AVAILABLE:
            server = StatusServer(manager, args.status_port)
            status_server = await server.start()
        elif not AIOHTTP_AVAILABLE and not args.no_status:
            manager.logger.warning("aiohttp not available - status server disabled")
        
        # Run interactive shell or wait for shutdown
        if shell and not args.no_shell:
            await shell.run()
        else:
            await manager.shutdown_event.wait()
            
    except Exception as e:
        manager.logger.error(f"Fatal error: {e}", exc_info=True)
        raise
    finally:
        # Cleanup
        if status_server:
            await status_server.cleanup()
        await manager.shutdown()


async def handle_signal(sig: signal.Signals, manager: NodeManager):
    """Handle system signals."""
    manager.logger.info(f"Received signal {sig.value}")
    manager.shutdown_event.set()


async def main():
    """Entry point."""
    args = parse_args()
    
    # Show banner
    if RICH_AVAILABLE and not args.no_shell:
        console.print("[bold cyan]Enhanced CSP Network Node[/bold cyan]")
        console.print("[dim]Version 1.0.0[/dim]")
        if args.genesis:
            console.print("[bold yellow]🌟 GENESIS NODE - First in the network[/bold yellow]")
        console.print()
    
    await run_main(args)


if __name__ == "__main__":
    asyncio.run(main())