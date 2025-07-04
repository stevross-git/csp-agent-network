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
    from aiohttp import web
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
    from enhanced_csp.network.core.node import EnhancedCSPNetwork
    from enhanced_csp.security_hardening import SecurityOrchestrator
    from enhanced_csp.quantum_csp_engine import QuantumCSPEngine
    from enhanced_csp.blockchain_csp_network import BlockchainCSPNetwork
except ImportError:
    # Fall back to relative imports (for direct execution)
    sys.path.insert(0, str(Path(__file__).parent))
    from core.config import (
        NetworkConfig, SecurityConfig, P2PConfig, MeshConfig, 
        DNSConfig, RoutingConfig
    )
    from core.node import EnhancedCSPNetwork
    sys.path.insert(0, str(Path(__file__).parent.parent))
    try:
        from security_hardening import SecurityOrchestrator
        from quantum_csp_engine import QuantumCSPEngine
        from blockchain_csp_network import BlockchainCSPNetwork
    except ImportError:
        # These modules may not exist, create placeholder classes
        class SecurityOrchestrator:
            def __init__(self, *args, **kwargs):
                pass
        class QuantumCSPEngine:
            def __init__(self, *args, **kwargs):
                pass
        class BlockchainCSPNetwork:
            def __init__(self, *args, **kwargs):
                pass

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
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.config = self._build_config()
        self.network: Optional[EnhancedCSPNetwork] = None
        self.security_orchestrator: Optional[SecurityOrchestrator] = None
        self.quantum_engine: Optional[QuantumCSPEngine] = None
        self.blockchain: Optional[BlockchainCSPNetwork] = None
        self.shutdown_event = asyncio.Event()
        self.tasks: List[asyncio.Task] = []
        self.logger = self._setup_logging()
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
                await asyncio.sleep(60)  # Every minute
                metrics = await self.collect_metrics()
                
                # Log key metrics
                self.logger.debug(f"Metrics: peers={metrics.get('peers', 0)}, "
                                f"messages_sent={metrics.get('messages_sent', 0)}, "
                                f"uptime={metrics.get('uptime', 0):.0f}s")
                
                # Store metrics for monitoring
                # TODO: Send to monitoring system
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Metrics collection failed: {e}")
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect current node metrics."""
        if self.network:
            return await self.network.metrics()
        return {
            "peers": 0,
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
        # Build security config
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
        
        # Build P2P config
        p2p = P2PConfig(
            listen_address=self.args.listen_address,
            listen_port=self.args.listen_port,
            bootstrap_nodes=self.args.bootstrap if not self.args.genesis else [],
            stun_servers=self.args.stun_servers or DEFAULT_STUN_SERVERS,
            turn_servers=self.args.turn_servers or [],
            max_peers=self.args.max_peers,
            enable_mdns=not self.args.no_mdns,
        )
        
        # Build mesh config
        mesh = MeshConfig(
            max_peers=self.args.max_peers,
        )
        
        # Build DNS config
        dns = DNSConfig(
            root_domain=".web4ai",
        )
        
        # Build routing config
        routing = RoutingConfig(
            enable_qos=self.args.qos,
        )
        
        # Build main network config
        config = NetworkConfig(
            security=security,
            p2p=p2p,
            mesh=mesh,
            dns=dns,
            routing=routing,
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
        peers = self.manager.network.get_peers()
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
                    str(peer.id)[:16] + "...",
                    f"{peer.address}:{peer.port}",
                    f"{peer.latency:.2f}ms" if peer.latency else "N/A",
                    f"{peer.reputation:.2f}"
                )
            console.print(table)
        else:
            print(f"\nConnected peers ({len(peers)}):")
            for peer in peers:
                print(f"  {peer.id}: {peer.address}:{peer.port}")
    
    async def cmd_dns(self, args: List[str]):
        """DNS operations."""
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
        info = {
            "node_id": str(self.manager.network.node_id),
            "version": "1.0.0",
            "is_genesis": self.manager.is_genesis,
            "network_id": self.manager.config.network_id,
            "listen_address": f"{self.manager.config.listen_address}:{self.manager.config.listen_port}",
            "capabilities": self.manager.config.node_capabilities
        }
        return web.json_response(info)

    async def handle_api_status(self, request: web.Request) -> web.Response:
        """API endpoint for node metrics."""
        metrics = await self.manager.collect_metrics()
        return web.json_response(metrics)

    async def handle_api_peers(self, request: web.Request) -> web.Response:
        """API endpoint for peer list."""
        peers = self.manager.network.get_peers()
        peer_list = [
            {
                "id": str(peer.id),
                "address": peer.address,
                "port": peer.port,
                "latency": peer.latency,
                "reputation": peer.reputation,
                "last_seen": peer.last_seen.isoformat() if peer.last_seen else None
            }
            for peer in peers
        ]
        return web.json_response(peer_list)

    async def handle_api_dns(self, request: web.Request) -> web.Response:
        """API endpoint for DNS records."""
        if hasattr(self.manager.network.dns_overlay, 'list_records'):
            records = await self.manager.network.dns_overlay.list_records()
            return web.json_response(records)
        else:
            # Fallback for stub implementation
            records = getattr(self.manager.network.dns_overlay, 'records', {})
            return web.json_response(records)

    async def handle_api_connect(self, request: web.Request) -> web.Response:
        """API endpoint to connect to a peer."""
        try:
            data = await request.json()
            address = data.get('address')
            if address:
                await self.manager.network.connect(address)
                return web.json_response({"status": "connecting", "address": address})
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
            f"enhanced_csp_peers {metrics.get('peers', 0)}",
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
        info = {
            "node_id": str(self.manager.network.node_id),
            "version": "1.0.0",
            "network_id": self.manager.config.network_id,
            "is_genesis": self.manager.is_genesis,
            "capabilities": self.manager.config.node_capabilities,
            "security": {
                "tls": self.manager.config.security.enable_tls,
                "mtls": self.manager.config.security.enable_mtls,
                "pq_crypto": self.manager.config.security.enable_pq_crypto,
                "zero_trust": self.manager.config.security.enable_zero_trust,
            }
        }
        return web.json_response(info)
    
    async def handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        if self.manager.network and self.manager.network.is_running:
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
    status_server = None  # Initialize status_server at the top
    
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