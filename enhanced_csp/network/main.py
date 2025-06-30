#!/usr/bin/env python3
"""
Enhanced CSP Network - Production Entry Script
Boots, configures, and supervises a single Enhanced CSP peer-node with all features.
"""

import asyncio
import signal
import sys
import os
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import argparse
from datetime import datetime, timedelta
import ipaddress
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

# Enhanced CSP imports - adjusted for actual file structure
try:
    from enhanced_csp.network.core.node import NetworkConfig, EnhancedCSPNetwork
    from enhanced_csp.network.core.types import NodeID, NodeCapabilities
except ImportError:
    # Fallback to creating minimal stubs
    from dataclasses import dataclass, field
    from typing import Dict, Any
    
    @dataclass
    class SecurityConfig:
        enable_tls: bool = True
        enable_mtls: bool = False
        enable_pq_crypto: bool = True
        enable_zero_trust: bool = False
        tls_cert_path: Optional[str] = None
        tls_key_path: Optional[str] = None
        ca_cert_path: Optional[str] = None
        allowed_cipher_suites: List[str] = field(default_factory=lambda: ["TLS_AES_256_GCM_SHA384"])
        min_tls_version: str = "1.3"
        audit_log_path: Optional[Path] = None
        rate_limit_requests: int = 100
        rate_limit_window: int = 60
        enable_threat_detection: bool = True
        threat_detection_threshold: float = 0.8
        enable_intrusion_prevention: bool = True
        max_connection_rate: int = 50
        enable_compliance_mode: bool = False
        compliance_standards: List[str] = field(default_factory=list)
        data_retention_days: int = 90
        enable_data_encryption: bool = True
        encryption_algorithm: str = "AES-256-GCM"
        key_rotation_interval: int = 86400
        tls_rotation_interval: int = 2592000
        enable_ca_mode: bool = False
        trust_anchors: List[str] = field(default_factory=list)
    
    @dataclass
    class NetworkConfig:
        bootstrap_nodes: List[str] = field(default_factory=list)
        listen_address: str = "0.0.0.0"
        listen_port: int = 30300
        stun_servers: List[str] = field(default_factory=list)
        turn_servers: List[str] = field(default_factory=list)
        is_super_peer: bool = False
        enable_relay: bool = True
        enable_nat_traversal: bool = True
        enable_upnp: bool = True
        max_peers: int = 100
        peer_discovery_interval: int = 30
        peer_cleanup_interval: int = 300
        message_ttl: int = 64
        max_message_size: int = 1048576
        enable_compression: bool = True
        enable_encryption: bool = True
        node_capabilities: Dict[str, Any] = field(default_factory=dict)
        network_id: str = "enhanced-csp"
        protocol_version: str = "1.0.0"
        enable_metrics: bool = True
        metrics_interval: int = 60
        enable_dht: bool = True
        dht_bootstrap_nodes: List[str] = field(default_factory=list)
        routing_algorithm: str = "batman-adv"
        enable_qos: bool = True
        bandwidth_limit: int = 0
        enable_ipv6: bool = True
        dns_seeds: List[str] = field(default_factory=list)
        gossip_interval: int = 5
        gossip_fanout: int = 6
        enable_mdns: bool = True
        security: SecurityConfig = field(default_factory=SecurityConfig)
    
    class EnhancedCSPNetwork:
        def __init__(self, config: NetworkConfig):
            self.config = config
            self.node_id = f"node_{os.urandom(8).hex()}"
            self.is_running = False
            self.start_time = datetime.utcnow()
            self.stats = {}
            
        async def start(self):
            self.is_running = True
            
        async def stop(self):
            self.is_running = False
            
        def get_peers(self):
            return []
            
        @property
        def dns_overlay(self):
            return self
            
        async def resolve(self, name):
            return f"resolved_{name}"
            
        async def register(self, name, addr):
            pass
            
        async def list_records(self):
            return {}
            
        async def send_message(self, peer_id, message):
            pass

try:
    from enhanced_csp.security_hardening import SecurityConfig, SecurityOrchestrator
except ImportError:
    class SecurityOrchestrator:
        def __init__(self, config):
            self.config = config
            
        async def initialize(self):
            pass
            
        async def shutdown(self):
            pass
            
        async def monitor_threats(self):
            while True:
                await asyncio.sleep(60)
                
        async def rotate_tls_certificates(self):
            pass

try:
    from enhanced_csp.quantum_csp_engine import QuantumCSPEngine
except ImportError:
    class QuantumCSPEngine:
        def __init__(self, network):
            self.network = network
            
        async def initialize(self):
            pass
            
        async def shutdown(self):
            pass

try:
    from enhanced_csp.blockchain_csp_network import BlockchainCSPNetwork
except ImportError:
    class BlockchainCSPNetwork:
        def __init__(self, network):
            self.network = network
            
        async def initialize(self):
            pass
            
        async def shutdown(self):
            pass

# These imports are likely not needed for the main script
# from enhanced_csp.network.p2p.node import P2PNode
# from enhanced_csp.network.mesh.topology import MeshTopology
# from enhanced_csp.network.routing import BATMANRouter
# from enhanced_csp.network.dns.overlay import DNSOverlay

# Constants
DEFAULT_BOOTSTRAP_NODES = []  # Empty for genesis node
DEFAULT_STUN_SERVERS = ["stun:stun.l.google.com:19302", "stun:stun.cloudflare.com:3478"]
DEFAULT_TURN_SERVERS = []
DEFAULT_LISTEN_ADDRESS = "0.0.0.0"
DEFAULT_LISTEN_PORT = 30300
DEFAULT_STATUS_PORT = 8080
LOG_ROTATION_DAYS = 7
TLS_ROTATION_DAYS = 30
GENESIS_DNS_RECORDS = {
    # Initial DNS seeds for the network
    "seed1.web4ai": None,  # Will be set to this node's multiaddr
    "seed2.web4ai": None,  # Will be set to this node's multiaddr
    "bootstrap.web4ai": None,  # Will be set to this node's multiaddr
    "genesis.web4ai": None,  # Will be set to this node's multiaddr
}

# Global console for rich output
console = Console() if RICH_AVAILABLE else None


class NodeManager:
    """Manages the lifecycle of an Enhanced CSP node."""
    
    def __init__(self, config: NetworkConfig, args: argparse.Namespace):
        self.config = config
        self.args = args
        self.network: Optional[EnhancedCSPNetwork] = None
        self.security_orchestrator: Optional[SecurityOrchestrator] = None
        self.quantum_engine: Optional[QuantumCSPEngine] = None
        self.blockchain_network: Optional[BlockchainCSPNetwork] = None
        self.status_app: Optional[web.Application] = None
        self.shutdown_event = asyncio.Event()
        self.tasks: List[asyncio.Task] = []
        self.is_genesis = args.genesis
        
    async def initialize(self):
        """Initialize all node components."""
        self.logger.info("Initializing Enhanced CSP Node...")
        
        if self.is_genesis:
            self.logger.info("Starting as GENESIS node - First bootstrap and DNS seed")
            await self._initialize_genesis_node()
        
        # Create network instance
        self.network = EnhancedCSPNetwork(self.config)
        
        # Attach security orchestrator
        self.security_orchestrator = SecurityOrchestrator(self.config.security)
        await self.security_orchestrator.initialize()
        
        # Attach quantum engine if enabled
        if self.args.enable_quantum:
            self.logger.info("Initializing Quantum CSP Engine...")
            self.quantum_engine = QuantumCSPEngine(self.network)
            await self.quantum_engine.initialize()
        
        # Attach blockchain network if available
        if self.args.enable_blockchain:
            try:
                self.blockchain_network = BlockchainCSPNetwork(self.network)
                await self.blockchain_network.initialize()
                self.logger.info("Blockchain integration enabled")
            except Exception as e:
                self.logger.warning(f"Failed to initialize blockchain: {e}")
        
        # Start the network
        await self.network.start()
        self.logger.info(f"Node started with ID: {self.network.node_id}")
        
        # If genesis node, set up initial DNS records
        if self.is_genesis:
            await self._setup_genesis_dns()
        
        # Start background tasks
        self._start_background_tasks()
    
    async def _initialize_genesis_node(self):
        """Initialize special settings for the genesis node."""
        self.logger.info("Configuring genesis node settings...")
        
        # Ensure we're a super peer
        self.config.is_super_peer = True
        self.config.max_peers = 1000  # Higher limit for genesis
        
        # Disable bootstrap attempts since we ARE the bootstrap
        self.config.bootstrap_nodes = []
        
        # Enable all capabilities for genesis node
        self.config.node_capabilities.update({
            "relay": True,
            "storage": True,
            "compute": True,
            "dns": True,
            "bootstrap": True,
        })
        
        # Set up genesis-specific security
        self.config.security.enable_ca_mode = True  # Act as Certificate Authority
        self.config.security.trust_anchors = ["self"]  # Self-signed root
        
    async def _setup_genesis_dns(self):
        """Set up initial DNS records for the network."""
        self.logger.info("Setting up genesis DNS records...")
        
        # Get our public address
        public_ip = await self._get_public_ip()
        node_multiaddr = f"/ip4/{public_ip}/tcp/{self.config.listen_port}/p2p/{self.network.node_id}"
        
        # Register all genesis DNS names
        for domain, _ in GENESIS_DNS_RECORDS.items():
            try:
                await self.network.dns_overlay.register(domain, node_multiaddr)
                self.logger.info(f"Registered DNS: {domain} -> {node_multiaddr}")
            except Exception as e:
                self.logger.error(f"Failed to register {domain}: {e}")
        
        # Also register our node ID as a DNS name
        short_id = str(self.network.node_id)[:16]
        await self.network.dns_overlay.register(f"{short_id}.web4ai", node_multiaddr)
        
    async def _get_public_ip(self) -> str:
        """Get public IP address of this node."""
        # Try STUN first
        if self.config.stun_servers:
            try:
                import aiostun
                stun_client = aiostun.Client(self.config.stun_servers[0])
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
            if self.config.listen_address != "0.0.0.0":
                return self.config.listen_address
            return "127.0.0.1"  # Localhost fallback
        
    def _start_background_tasks(self):
        """Start background maintenance tasks."""
        # TLS key rotation
        if self.config.security.tls_rotation_interval:
            self.tasks.append(asyncio.create_task(self._tls_rotation_task()))
        
        # Metrics collection
        self.tasks.append(asyncio.create_task(self._metrics_collection_task()))
        
        # Security monitoring
        if self.security_orchestrator:
            self.tasks.append(asyncio.create_task(
                self.security_orchestrator.monitor_threats()
            ))
        
        # Genesis node maintenance
        if self.is_genesis:
            self.tasks.append(asyncio.create_task(self._genesis_maintenance_task()))
    
    async def _genesis_maintenance_task(self):
        """Maintenance tasks specific to genesis node."""
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                # Update DNS records if our IP changed
                current_ip = await self._get_public_ip()
                node_multiaddr = f"/ip4/{current_ip}/tcp/{self.config.listen_port}/p2p/{self.network.node_id}"
                
                for domain in GENESIS_DNS_RECORDS.keys():
                    existing = await self.network.dns_overlay.resolve(domain)
                    if existing != node_multiaddr:
                        await self.network.dns_overlay.register(domain, node_multiaddr)
                        self.logger.info(f"Updated DNS record: {domain}")
                
                # Log network statistics
                stats = await self.collect_metrics()
                self.logger.info(f"Genesis node stats: {stats['peers']} peers connected")
                
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
                await asyncio.sleep(60)  # Collect every minute
                # Collect and log metrics
                metrics = await self.collect_metrics()
                self.logger.debug(f"Metrics: {json.dumps(metrics)}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Metrics collection failed: {e}")
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect node metrics."""
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "node_id": str(self.network.node_id),
            "peers": len(self.network.get_peers()),
            "uptime": (datetime.utcnow() - self.network.start_time).total_seconds(),
            "messages_sent": self.network.stats.get("messages_sent", 0),
            "messages_received": self.network.stats.get("messages_received", 0),
            "bandwidth_in": self.network.stats.get("bandwidth_in", 0),
            "bandwidth_out": self.network.stats.get("bandwidth_out", 0),
            "is_genesis": self.is_genesis,
            "is_super_peer": self.config.is_super_peer,
        }
        
        # Add genesis-specific metrics
        if self.is_genesis:
            metrics.update({
                "dns_records": len(await self.network.dns_overlay.list_records()),
                "bootstrap_requests": self.network.stats.get("bootstrap_requests", 0),
                "network_age": (datetime.utcnow() - self.network.start_time).total_seconds(),
            })
        
        return metrics
    
    async def shutdown(self):
        """Gracefully shutdown the node."""
        self.logger.info("Shutting down Enhanced CSP Node...")
        self.shutdown_event.set()
        
        # Cancel background tasks
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Shutdown components
        if self.blockchain_network:
            await self.blockchain_network.shutdown()
        if self.quantum_engine:
            await self.quantum_engine.shutdown()
        if self.security_orchestrator:
            await self.security_orchestrator.shutdown()
        if self.network:
            await self.network.stop()
        
        self.logger.info("Node shutdown complete")
    
    @property
    def logger(self):
        return logging.getLogger("enhanced_csp.node_manager")


class InteractiveShell:
    """Interactive management shell for the node."""
    
    def __init__(self, manager: NodeManager):
        self.manager = manager
        self.commands = {
            "help": self.cmd_help,
            "peers": self.cmd_peers,
            "dns": self.cmd_dns,
            "send": self.cmd_send,
            "stats": self.cmd_stats,
            "loglevel": self.cmd_loglevel,
            "quit": self.cmd_quit,
        }
    
    async def run(self):
        """Run the interactive shell."""
        print("\nEnhanced CSP Node Interactive Shell")
        print("Type 'help' for available commands\n")
        
        while not self.manager.shutdown_event.is_set():
            try:
                if RICH_AVAILABLE:
                    cmd_line = await asyncio.to_thread(
                        Prompt.ask, "[cyan]csp>[/cyan]"
                    )
                else:
                    cmd_line = await asyncio.to_thread(input, "csp> ")
                
                if not cmd_line.strip():
                    continue
                
                parts = cmd_line.strip().split()
                cmd = parts[0].lower()
                args = parts[1:]
                
                if cmd in self.commands:
                    await self.commands[cmd](args)
                else:
                    print(f"Unknown command: {cmd}")
            except (EOFError, KeyboardInterrupt):
                break
            except Exception as e:
                print(f"Error: {e}")
    
    async def cmd_help(self, args: List[str]):
        """Show available commands."""
        print("\nAvailable commands:")
        print("  help              - Show this help message")
        print("  peers             - List connected peers")
        print("  dns <name>        - Resolve .web4ai domain")
        print("  send <peer> <msg> - Send message to peer")
        print("  stats             - Show node statistics")
        print("  loglevel <level>  - Set logging level")
        print("  quit              - Exit the shell\n")
    
    async def cmd_peers(self, args: List[str]):
        """List connected peers."""
        peers = self.manager.network.get_peers()
        if not peers:
            print("No connected peers")
            return
        
        if RICH_AVAILABLE:
            table = Table(title="Connected Peers")
            table.add_column("Peer ID", style="cyan")
            table.add_column("Address", style="green")
            table.add_column("Latency", style="yellow")
            
            for peer in peers:
                table.add_row(
                    str(peer.id)[:16] + "...",
                    peer.address,
                    f"{peer.latency:.2f}ms" if peer.latency else "N/A"
                )
            console.print(table)
        else:
            print(f"\nConnected peers ({len(peers)}):")
            for peer in peers:
                print(f"  {peer.id}: {peer.address}")
    
    async def cmd_dns(self, args: List[str]):
        """Resolve DNS name."""
        if not args:
            print("Usage: dns <name>")
            print("       dns list              - List all DNS records (genesis only)")
            print("       dns register <name> <addr> - Register DNS name (genesis only)")
            return
        
        if args[0] == "list" and self.manager.is_genesis:
            records = await self.manager.network.dns_overlay.list_records()
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
            return
        
        if args[0] == "register" and len(args) >= 3 and self.manager.is_genesis:
            domain = args[1]
            addr = " ".join(args[2:])
            try:
                await self.manager.network.dns_overlay.register(domain, addr)
                print(f"Registered: {domain} -> {addr}")
            except Exception as e:
                print(f"Failed to register: {e}")
            return
        
        # Regular DNS resolution
        name = args[0]
        try:
            result = await self.manager.network.dns_overlay.resolve(name)
            print(f"{name} -> {result}")
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
            await self.manager.network.send_message(peer_id, message)
            print(f"Message sent to {peer_id}")
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
        self.app.router.add_get('/metrics', self.handle_metrics)
        self.app.router.add_get('/info', self.handle_info)
        self.app.router.add_get('/health', self.handle_health)
    
    async def handle_metrics(self, request: web.Request) -> web.Response:
        """Prometheus-compatible metrics endpoint."""
        metrics = await self.manager.collect_metrics()
        
        # Format as Prometheus metrics
        output = []
        output.append(f"# HELP csp_peers_total Number of connected peers")
        output.append(f"# TYPE csp_peers_total gauge")
        output.append(f"csp_peers_total {metrics['peers']}")
        
        output.append(f"# HELP csp_uptime_seconds Node uptime in seconds")
        output.append(f"# TYPE csp_uptime_seconds counter")
        output.append(f"csp_uptime_seconds {metrics['uptime']}")
        
        output.append(f"# HELP csp_messages_sent_total Total messages sent")
        output.append(f"# TYPE csp_messages_sent_total counter")
        output.append(f"csp_messages_sent_total {metrics['messages_sent']}")
        
        output.append(f"# HELP csp_messages_received_total Total messages received")
        output.append(f"# TYPE csp_messages_received_total counter")
        output.append(f"csp_messages_received_total {metrics['messages_received']}")
        
        return web.Response(text="\n".join(output), content_type="text/plain")
    
    async def handle_info(self, request: web.Request) -> web.Response:
        """JSON info endpoint."""
        info = {
            "node_id": str(self.manager.network.node_id),
            "version": "1.0.0",
            "network": "enhanced-csp",
            "features": {
                "quantum": self.manager.quantum_engine is not None,
                "blockchain": self.manager.blockchain_network is not None,
                "super_peer": self.manager.args.super_peer,
                "genesis": self.manager.is_genesis,
            },
            "uptime": (datetime.utcnow() - self.manager.network.start_time).total_seconds(),
        }
        
        # Add multiaddr for easy connection
        if self.manager.is_genesis:
            public_ip = await self.manager._get_public_ip()
            info["multiaddr"] = f"/ip4/{public_ip}/tcp/{self.manager.config.listen_port}/p2p/{self.manager.network.node_id}"
            info["dns_names"] = list(GENESIS_DNS_RECORDS.keys())
        
        return web.json_response(info)
    
    async def handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        if self.manager.network and self.manager.network.is_running:
            return web.Response(text="OK", status=200)
        return web.Response(text="UNHEALTHY", status=503)
    
    async def start(self):
        """Start the status server."""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', self.port)
        await site.start()
        logging.getLogger("enhanced_csp.status").info(
            f"Status server started on http://0.0.0.0:{self.port}"
        )


def setup_logging(args: argparse.Namespace):
    """Setup logging configuration."""
    handlers = []
    
    # Console handler
    if RICH_AVAILABLE and not args.no_color:
        handlers.append(RichHandler(
            console=console,
            rich_tracebacks=True,
            tracebacks_show_locals=args.log_level == "DEBUG"
        ))
    else:
        handlers.append(logging.StreamHandler())
    
    # File handler
    if args.log_file:
        log_dir = Path(args.log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Use rotating file handler
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            args.log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        handlers=handlers,
        format='%(message)s' if RICH_AVAILABLE else '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Enhanced CSP Network Node",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Network configuration
    parser.add_argument(
        "--bootstrap",
        type=str,
        default=os.environ.get("CSP_BOOTSTRAP", ",".join(DEFAULT_BOOTSTRAP_NODES)),
        help="Comma-separated list of bootstrap node multiaddrs"
    )
    parser.add_argument(
        "--listen-address",
        type=str,
        default=os.environ.get("CSP_LISTEN_ADDRESS", DEFAULT_LISTEN_ADDRESS),
        help="Listen address for P2P connections"
    )
    parser.add_argument(
        "--listen-port",
        type=int,
        default=int(os.environ.get("CSP_LISTEN_PORT", DEFAULT_LISTEN_PORT)),
        help="Listen port for P2P connections"
    )
    
    # STUN/TURN servers
    parser.add_argument(
        "--stun",
        type=str,
        default=os.environ.get("CSP_STUN", ",".join(DEFAULT_STUN_SERVERS)),
        help="Comma-separated list of STUN servers"
    )
    parser.add_argument(
        "--turn",
        type=str,
        default=os.environ.get("CSP_TURN", ",".join(DEFAULT_TURN_SERVERS)),
        help="Comma-separated list of TURN servers"
    )
    
    # Node type
    parser.add_argument(
        "--super-peer",
        action="store_true",
        default=os.environ.get("CSP_SUPER_PEER", "false").lower() == "true",
        help="Run as a super peer"
    )
    parser.add_argument(
        "--genesis",
        action="store_true",
        default=os.environ.get("CSP_GENESIS", "false").lower() == "true",
        help="Run as the genesis (first) node in the network"
    )
    
    # Features
    parser.add_argument(
        "--enable-quantum",
        action="store_true",
        default=os.environ.get("CSP_ENABLE_QUANTUM", "false").lower() == "true",
        help="Enable quantum integration"
    )
    parser.add_argument(
        "--enable-blockchain",
        action="store_true",
        default=os.environ.get("CSP_ENABLE_BLOCKCHAIN", "false").lower() == "true",
        help="Enable blockchain integration"
    )
    
    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=os.environ.get("CSP_LOG_LEVEL", "INFO"),
        help="Logging level"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=os.environ.get("CSP_LOG_FILE"),
        help="Log file path"
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output"
    )
    
    # Status server
    parser.add_argument(
        "--status-port",
        type=int,
        default=int(os.environ.get("CSP_STATUS_PORT", 0)),
        help="HTTP status server port (0 to disable)"
    )
    
    # Security
    parser.add_argument(
        "--tls-rotation-days",
        type=int,
        default=int(os.environ.get("CSP_TLS_ROTATION_DAYS", TLS_ROTATION_DAYS)),
        help="TLS certificate rotation interval in days"
    )
    
    # Interactive shell
    parser.add_argument(
        "--no-shell",
        action="store_true",
        help="Disable interactive shell"
    )
    
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> NetworkConfig:
    """Build network configuration from arguments."""
    # Parse bootstrap nodes (empty if genesis)
    bootstrap_nodes = []
    if args.bootstrap and not args.genesis:
        bootstrap_nodes = [addr.strip() for addr in args.bootstrap.split(",") if addr.strip()]
    elif not args.genesis and not args.bootstrap:
        # Try to use default genesis DNS names
        bootstrap_nodes = [
            "bootstrap.web4ai",
            "genesis.web4ai",
            "seed1.web4ai",
            "seed2.web4ai"
        ]
    
    # Parse STUN/TURN servers
    stun_servers = []
    if args.stun:
        stun_servers = [s.strip() for s in args.stun.split(",") if s.strip()]
    
    turn_servers = []
    if args.turn:
        turn_servers = [s.strip() for s in args.turn.split(",") if s.strip()]
    
    # Build security config
    security_config = SecurityConfig(
        enable_tls=True,
        enable_mtls=True,
        enable_pq_crypto=True,
        enable_zero_trust=True,
        tls_cert_path=None,  # Auto-generate
        tls_key_path=None,   # Auto-generate
        ca_cert_path=None,   # Auto-generate
        allowed_cipher_suites=[
            "TLS_AES_256_GCM_SHA384",
            "TLS_CHACHA20_POLY1305_SHA256"
        ],
        min_tls_version="1.3",
        audit_log_path=Path(args.log_file).parent / "audit.log" if args.log_file else None,
        rate_limit_requests=100,
        rate_limit_window=60,
        enable_threat_detection=True,
        threat_detection_threshold=0.8,
        enable_intrusion_prevention=True,
        max_connection_rate=50,
        enable_compliance_mode=True,
        compliance_standards=["SOC2", "ISO27001"],
        data_retention_days=90,
        enable_data_encryption=True,
        encryption_algorithm="AES-256-GCM",
        key_rotation_interval=86400,  # 24 hours
        tls_rotation_interval=args.tls_rotation_days * 86400,
    )
    
    # Build network config
    config = NetworkConfig(
        bootstrap_nodes=bootstrap_nodes,
        listen_address=args.listen_address,
        listen_port=args.listen_port,
        stun_servers=stun_servers,
        turn_servers=turn_servers,
        is_super_peer=args.super_peer or args.genesis,  # Genesis nodes are always super peers
        enable_relay=True,
        enable_nat_traversal=True,
        enable_upnp=True,
        max_peers=100 if not (args.super_peer or args.genesis) else 500,
        peer_discovery_interval=30,
        peer_cleanup_interval=300,
        message_ttl=64,
        max_message_size=1024 * 1024,  # 1MB
        enable_compression=True,
        enable_encryption=True,
        node_capabilities={
            "relay": True,
            "storage": args.genesis,  # Genesis node stores network state
            "compute": False,
            "quantum": args.enable_quantum,
            "blockchain": args.enable_blockchain,
            "dns": args.genesis,  # Genesis node provides DNS
            "bootstrap": args.genesis,  # Genesis node is bootstrap
        },
        network_id="enhanced-csp-mainnet",
        protocol_version="1.0.0",
        enable_metrics=True,
        metrics_interval=60,
        enable_dht=True,
        dht_bootstrap_nodes=bootstrap_nodes if not args.genesis else [],
        routing_algorithm="batman-adv",
        enable_qos=True,
        bandwidth_limit=0,  # No limit
        enable_ipv6=True,
        dns_seeds=["seed1.web4ai", "seed2.web4ai"] if not args.genesis else [],
        gossip_interval=5,
        gossip_fanout=6,
        enable_mdns=True,
        security=security_config,
    )
    
    return config


async def run_main(args: Optional[argparse.Namespace] = None) -> NodeManager:
    """Main entry point for testing and programmatic usage."""
    if args is None:
        args = parse_args()
    
    # Setup logging
    setup_logging(args)
    logger = logging.getLogger("enhanced_csp.main")
    
    # Build configuration
    config = build_config(args)
    
    # Create node manager
    manager = NodeManager(config, args)
    
    # Setup signal handlers
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}")
        asyncio.create_task(manager.shutdown())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Initialize node
        await manager.initialize()
        
        # Start status server if enabled
        status_server = None
        if args.status_port > 0 and AIOHTTP_AVAILABLE:
            status_server = StatusServer(manager, args.status_port)
            await status_server.start()
        
        # Start interactive shell if enabled
        shell_task = None
        if not args.no_shell:
            shell = InteractiveShell(manager)
            shell_task = asyncio.create_task(shell.run())
        
        # Wait for shutdown
        await manager.shutdown_event.wait()
        
        # Cleanup
        if shell_task:
            shell_task.cancel()
            try:
                await shell_task
            except asyncio.CancelledError:
                pass
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise
    finally:
        await manager.shutdown()
    
    return manager


async def main():
    """Main entry point."""
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