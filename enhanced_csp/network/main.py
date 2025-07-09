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
import time

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
                self.logger = logging.getLogger("security_orchestrator")
                self.logger.info("Security orchestrator placeholder initialized")
                
            async def start(self):
                return True
                
            async def stop(self):
                pass
                
        class QuantumCSPEngine:
            def __init__(self, *args, **kwargs):
                self.logger = logging.getLogger("quantum_engine")
                self.logger.info("Quantum engine placeholder initialized")
                
            async def start(self):
                return True
                
            async def stop(self):
                pass
                
        class BlockchainCSPNetwork:
            def __init__(self, *args, **kwargs):
                self.logger = logging.getLogger("blockchain_network")
                self.logger.info("Blockchain network placeholder initialized")
                
            async def start(self):
                return True
                
            async def stop(self):
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
    "seed1.web4ai": None,  # Will be set to this node's multiaddr
    "seed2.web4ai": None,  # Will be set to this node's multiaddr
    "bootstrap.web4ai": None,  # Will be set to this node's multiaddr
    "genesis.web4ai": None,  # Will be set to this node's multiaddr
    
    # New peoplesainetwork.com domains
    "genesis.peoplesainetwork.com": None,  # Will be set to this node's multiaddr
    "bootstrap.peoplesainetwork.com": None,  # Will be set to this node's multiaddr
    "seed1.peoplesainetwork.com": None,  # Will be set to this node's multiaddr
    "seed2.peoplesainetwork.com": None,  # Will be set to this node's multiaddr
    
    # Additional bootstrap points
    "boot1.peoplesainetwork.com": None,
    "boot2.peoplesainetwork.com": None,
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
        self.start_time = time.time()
        
    async def initialize(self):
        """Initialize the node and all subsystems."""
        self.logger.info("Initializing Enhanced CSP Node...")
        
        try:
            # Initialize security first
            self.logger.info("Initializing security orchestrator")
            self.security_orchestrator = SecurityOrchestrator()
            await self.security_orchestrator.start()
            
            # Initialize network
            self.network = EnhancedCSPNetwork(self.config)
            if not await self.network.start():
                raise RuntimeError("Failed to start network node")
            
            # Initialize optional components
            if self.args.enable_quantum:
                self.logger.info("Initializing quantum engine")
                self.quantum_engine = QuantumCSPEngine()
                await self.quantum_engine.start()
            
            if self.args.enable_blockchain:
                self.logger.info("Initializing blockchain network")
                self.blockchain = BlockchainCSPNetwork()
                await self.blockchain.start()
            
            self.logger.info(f"Node started with ID: {self.network.node_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize node: {e}")
            await self.shutdown()
            raise
    
    async def shutdown(self):
        """Shutdown all components gracefully."""
        self.logger.info("Shutting down Enhanced CSP Node...")
        
        # Cancel background tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Shutdown components
        if self.blockchain:
            await self.blockchain.stop()
        
        if self.quantum_engine:
            await self.quantum_engine.stop()
        
        if self.network:
            await self.network.stop()
        
        if self.security_orchestrator:
            await self.security_orchestrator.stop()
        
        self.logger.info("Node shutdown complete")
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect metrics from all components."""
        metrics = {
            "uptime": time.time() - self.start_time,
            "peers": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "bandwidth_in": 0,
            "bandwidth_out": 0,
            "memory_usage": 0,
            "cpu_usage": 0,
        }
        
        try:
            if self.network and hasattr(self.network, 'get_metrics'):
                network_metrics = await self.network.get_metrics()
                if network_metrics:
                    metrics.update(network_metrics)
            elif self.network:
                # Basic metrics if get_metrics doesn't exist
                if hasattr(self.network, 'get_peers'):
                    metrics["peers"] = len(self.network.get_peers())
                    
            # Add component-specific metrics
            if self.quantum_engine and hasattr(self.quantum_engine, 'get_metrics'):
                quantum_metrics = await self.quantum_engine.get_metrics()
                if quantum_metrics:
                    metrics["quantum"] = quantum_metrics
                    
            if self.blockchain and hasattr(self.blockchain, 'get_metrics'):
                blockchain_metrics = await self.blockchain.get_metrics()
                if blockchain_metrics:
                    metrics["blockchain"] = blockchain_metrics
                    
        except Exception as e:
            self.logger.warning(f"Failed to collect some metrics: {e}")
        
        return metrics
    
    def _build_config(self) -> NetworkConfig:
        """Build network configuration from command line arguments."""
        return NetworkConfig(
            network_id=self.args.network_id,
            listen_address=self.args.listen_address,
            listen_port=self.args.listen_port,
            bootstrap_nodes=self._parse_bootstrap_nodes(),
            node_capabilities={
                "relay": True,
                "dht": True,
                "mesh": True,
                "dns": self.args.genesis,
                "super_peer": self.args.super_peer,
                "quantum": self.args.enable_quantum,
                "blockchain": self.args.enable_blockchain,
            },
            security=SecurityConfig(
                enable_tls=True,
                enable_mtls=self.args.enable_mtls,
                enable_pq_crypto=self.args.enable_quantum,
                enable_zero_trust=True,
            ),
            p2p=P2PConfig(
                max_peers=self.args.max_peers,
                stun_servers=self._parse_stun_servers(),
                turn_servers=self._parse_turn_servers(),
                enable_upnp=not self.args.disable_upnp,
                enable_nat_pmp=not self.args.disable_nat_pmp,
                connection_timeout=30,
                keep_alive_interval=60,
            ),
            mesh=MeshConfig(
                topology="dynamic_partial",
                optimization_interval=30,
                redundancy_factor=3,
                enable_load_balancing=True,
            ),
            dns=DNSConfig(
                enable_overlay=self.args.genesis,
                records=GENESIS_DNS_RECORDS if self.args.genesis else {},
                cache_ttl=300,
            ),
            routing=RoutingConfig(
                algorithm="batman",
                metric="latency",
                enable_multipath=True,
                convergence_timeout=10,
            )
        )
    
    def _parse_bootstrap_nodes(self) -> List[str]:
        """Parse bootstrap nodes from command line."""
        if not self.args.bootstrap:
            return []
        
        nodes = []
        for addr in self.args.bootstrap:
            if addr.endswith('.peoplesainetwork.com') or addr.endswith('.web4ai'):
                # DNS-based bootstrap
                nodes.append(f"/dnsaddr/{addr}")
            else:
                # Direct multiaddr
                nodes.append(addr)
        
        return nodes
    
    def _parse_stun_servers(self) -> List[str]:
        """Parse STUN servers from command line."""
        if hasattr(self.args, 'stun_servers') and self.args.stun_servers:
            return self.args.stun_servers
        return DEFAULT_STUN_SERVERS
    
    def _parse_turn_servers(self) -> List[str]:
        """Parse TURN servers from command line."""
        if hasattr(self.args, 'turn_servers') and self.args.turn_servers:
            return self.args.turn_servers
        return DEFAULT_TURN_SERVERS
    
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


class InteractiveShell:
    """Interactive command shell for node management."""
    
    def __init__(self, manager: NodeManager):
        self.manager = manager
        self.commands = {
            'help': self.cmd_help,
            'status': self.cmd_status,
            'peers': self.cmd_peers,
            'connect': self.cmd_connect,
            'disconnect': self.cmd_disconnect,
            'send': self.cmd_send,
            'stats': self.cmd_stats,
            'loglevel': self.cmd_loglevel,
            'quit': self.cmd_quit,
            'exit': self.cmd_quit,
        }
    
    async def run(self):
        """Run the interactive shell."""
        if RICH_AVAILABLE:
            console.print("\n[bold green]Enhanced CSP Node Interactive Shell[/bold green]")
            console.print("Type 'help' for available commands\n")
        else:
            print("\nEnhanced CSP Node Interactive Shell")
            print("Type 'help' for available commands\n")
        
        while not self.manager.shutdown_event.is_set():
            try:
                if RICH_AVAILABLE:
                    command_line = Prompt.ask("csp>", console=console)
                else:
                    command_line = input("csp>: ")
                
                if not command_line.strip():
                    continue
                
                parts = command_line.strip().split()
                cmd = parts[0].lower()
                args = parts[1:] if len(parts) > 1 else []
                
                if cmd in self.commands:
                    await self.commands[cmd](args)
                else:
                    print(f"Unknown command: {cmd}. Type 'help' for available commands.")
                    
            except (EOFError, KeyboardInterrupt):
                print("\nExiting shell...")
                self.manager.shutdown_event.set()
                break
            except Exception as e:
                print(f"Command error: {e}")
    
    async def cmd_help(self, args: List[str]):
        """Show available commands."""
        help_text = """
Available commands:
  help                 - Show this help message
  status               - Show node status
  peers                - List connected peers
  connect <address>    - Connect to a peer
  disconnect <peer_id> - Disconnect from a peer
  send <peer_id> <msg> - Send message to peer
  stats                - Show detailed statistics
  loglevel <level>     - Set log level (debug, info, warning, error)
  quit/exit            - Exit the shell
"""
        print(help_text)
    
    async def cmd_status(self, args: List[str]):
        """Show node status."""
        if self.manager.network:
            print(f"Node ID: {self.manager.network.node_id}")
            print(f"Status: {'Running' if getattr(self.manager.network, 'is_running', False) else 'Stopped'}")
            print(f"Listen Address: {self.manager.config.listen_address}:{self.manager.config.listen_port}")
            print(f"Network ID: {self.manager.config.network_id}")
            if hasattr(self.manager.network, 'get_peers'):
                peers = self.manager.network.get_peers()
                print(f"Connected Peers: {len(peers)}")
        else:
            print("Network not initialized")
    
    async def cmd_peers(self, args: List[str]):
        """List connected peers."""
        if not self.manager.network or not hasattr(self.manager.network, 'get_peers'):
            print("No network or peer information available")
            return
        
        peers = self.manager.network.get_peers()
        if not peers:
            print("No connected peers")
            return
        
        if RICH_AVAILABLE:
            table = Table(title="Connected Peers")
            table.add_column("Peer ID", style="cyan")
            table.add_column("Address", style="magenta")
            table.add_column("Latency", style="green")
            table.add_column("Status", style="yellow")
            
            for peer in peers:
                table.add_row(
                    str(getattr(peer, 'id', 'unknown'))[:16] + '...',
                    getattr(peer, 'address', 'unknown'),
                    f"{getattr(peer, 'latency', 0):.1f}ms",
                    getattr(peer, 'status', 'connected')
                )
            
            console.print(table)
        else:
            print("\nConnected Peers:")
            for i, peer in enumerate(peers):
                print(f"  {i+1}. {getattr(peer, 'id', 'unknown')} - {getattr(peer, 'address', 'unknown')}")
    
    async def cmd_connect(self, args: List[str]):
        """Connect to a peer."""
        if not args:
            print("Usage: connect <address>")
            return
        
        address = args[0]
        
        if not self.manager.network or not hasattr(self.manager.network, 'connect'):
            print("Network not available")
            return
        
        try:
            await self.manager.network.connect(address)
            print(f"Connecting to {address}...")
        except Exception as e:
            print(f"Failed to connect: {e}")
    
    async def cmd_disconnect(self, args: List[str]):
        """Disconnect from a peer."""
        if not args:
            print("Usage: disconnect <peer_id>")
            return
        
        peer_id = args[0]
        
        if not self.manager.network or not hasattr(self.manager.network, 'disconnect'):
            print("Network not available")
            return
        
        try:
            await self.manager.network.disconnect(peer_id)
            print(f"Disconnected from {peer_id}")
        except Exception as e:
            print(f"Failed to disconnect: {e}")
    
    async def cmd_send(self, args: List[str]):
        """Send a message to a peer."""
        if len(args) < 2:
            print("Usage: send <peer_id> <message>")
            return
        
        peer_id = args[0]
        message = " ".join(args[1:])
        
        if not self.manager.network or not hasattr(self.manager.network, 'send_message'):
            print("Network not available")
            return
        
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
        
        # Static files for dashboard
        dashboard_static = Path(__file__).parent / 'dashboard' / 'static'
        if dashboard_static.exists():
            self.app.router.add_static('/static/', dashboard_static)
    
    async def handle_root(self, request: web.Request) -> web.Response:
        """Serve the dashboard HTML."""
        dashboard_path = Path(__file__).parent / 'dashboard' / 'index.html'
        if dashboard_path.exists():
            return web.FileResponse(dashboard_path)
        else:
            # Enhanced fallback with working status page
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Enhanced CSP Node - {self.manager.network.node_id if self.manager.network else 'Unknown'}</title>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
                    .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                    .status {{ padding: 10px; border-radius: 4px; margin: 10px 0; }}
                    .online {{ background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }}
                    .offline {{ background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }}
                    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
                    .metric {{ background: #f8f9fa; padding: 15px; border-radius: 4px; text-align: center; }}
                    .metric h3 {{ margin: 0 0 10px 0; color: #495057; }}
                    .metric .value {{ font-size: 24px; font-weight: bold; color: #007bff; }}
                    a {{ color: #007bff; text-decoration: none; margin-right: 15px; }}
                    a:hover {{ text-decoration: underline; }}
                    .refresh {{ background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }}
                    .refresh:hover {{ background: #0056b3; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Enhanced CSP Node Status</h1>
                    <div id="status" class="status online">● Node Online</div>
                    
                    <div class="metrics" id="metrics">
                        <div class="metric">
                            <h3>Node ID</h3>
                            <div class="value" id="node-id">Loading...</div>
                        </div>
                        <div class="metric">
                            <h3>Peers</h3>
                            <div class="value" id="peer-count">0</div>
                        </div>
                        <div class="metric">
                            <h3>Uptime</h3>
                            <div class="value" id="uptime">0s</div>
                        </div>
                        <div class="metric">
                            <h3>Messages</h3>
                            <div class="value" id="messages">0</div>
                        </div>
                    </div>
                    
                    <button class="refresh" onclick="refreshData()">Refresh Data</button>
                    
                    <h2>API Endpoints</h2>
                    <p>
                        <a href="/api/info">Node Info</a>
                        <a href="/api/status">Status & Metrics</a>
                        <a href="/api/peers">Peer List</a>
                        <a href="/api/dns">DNS Records</a>
                        <a href="/health">Health Check</a>
                    </p>
                </div>
                
                <script>
                    async function refreshData() {{
                        try {{
                            // Fetch node info
                            const infoRes = await fetch('/api/info');
                            if (infoRes.ok) {{
                                const info = await infoRes.json();
                                document.getElementById('node-id').textContent = info.node_id.substring(0, 12) + '...';
                                document.getElementById('status').className = 'status online';
                                document.getElementById('status').textContent = '● Node Online - ' + info.network_id;
                            }}
                            
                            // Fetch metrics
                            const statusRes = await fetch('/api/status');
                            if (statusRes.ok) {{
                                const metrics = await statusRes.json();
                                document.getElementById('peer-count').textContent = metrics.peers || 0;
                                document.getElementById('uptime').textContent = Math.floor(metrics.uptime || 0) + 's';
                                document.getElementById('messages').textContent = (metrics.messages_sent || 0) + '/' + (metrics.messages_received || 0);
                            }}
                            
                        }} catch (error) {{
                            console.error('Failed to refresh:', error);
                            document.getElementById('status').className = 'status offline';
                            document.getElementById('status').textContent = '● Connection Error';
                        }}
                    }}
                    
                    // Auto-refresh every 5 seconds
                    setInterval(refreshData, 5000);
                    
                    // Initial load
                    refreshData();
                </script>
            </body>
            </html>
            """
            return web.Response(text=html, content_type='text/html')

    async def handle_api_info(self, request: web.Request) -> web.Response:
        """API endpoint for node information."""
        try:
            info = {
                "node_id": str(self.manager.network.node_id) if self.manager.network else "unknown",
                "version": "1.0.0",
                "is_genesis": self.manager.is_genesis,
                "network_id": getattr(self.manager.config, 'network_id', 'enhanced-csp'),
                "listen_address": f"{self.manager.config.listen_address}:{self.manager.config.listen_port}",
                "capabilities": getattr(self.manager.config, 'node_capabilities', {}),
                "external_address": getattr(self.manager.network, 'external_address', None) if self.manager.network else None,
                "nat_type": getattr(self.manager.network, 'nat_type', None) if self.manager.network else None,
                "status": "running" if self.manager.network and getattr(self.manager.network, 'is_running', False) else "stopped"
            }
            return web.json_response(info)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_api_status(self, request: web.Request) -> web.Response:
        """API endpoint for node metrics."""
        try:
            # Collect metrics from the manager
            metrics = await self.manager.collect_metrics()
            
            # Add network-specific metrics if available
            if self.manager.network:
                network_metrics = {
                    "is_running": getattr(self.manager.network, 'is_running', False),
                    "node_id": str(self.manager.network.node_id),
                }
                if hasattr(self.manager.network, 'get_peers'):
                    network_metrics["peers"] = len(self.manager.network.get_peers())
                
                metrics.update(network_metrics)
            
            return web.json_response(metrics)
        except Exception as e:
            # Return basic metrics even if collection fails
            basic_metrics = {
                "uptime": time.time() - self.manager.start_time,
                "peers": 0,
                "messages_sent": 0,
                "messages_received": 0,
                "error": str(e)
            }
            return web.json_response(basic_metrics)

    async def handle_api_peers(self, request: web.Request) -> web.Response:
        """API endpoint for peer list."""
        try:
            if not self.manager.network or not hasattr(self.manager.network, 'get_peers'):
                return web.json_response([])
            
            peers = self.manager.network.get_peers()
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
            return web.json_response({"error": str(e)}, status=500)

    async def handle_api_dns(self, request: web.Request) -> web.Response:
        """API endpoint for DNS records."""
        try:
            if not self.manager.network or not hasattr(self.manager.network, 'dns_overlay'):
                return web.json_response({})
            
            dns_overlay = self.manager.network.dns_overlay
            
            if hasattr(dns_overlay, 'list_records'):
                records = await dns_overlay.list_records()
            else:
                # Fallback for stub implementation
                records = getattr(dns_overlay, 'records', {})
            
            return web.json_response(records)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_api_connect(self, request: web.Request) -> web.Response:
        """API endpoint to connect to a peer."""
        try:
            data = await request.json()
            address = data.get('address')
            
            if not address:
                return web.json_response({"error": "No address provided"}, status=400)
            
            if not self.manager.network or not hasattr(self.manager.network, 'connect'):
                return web.json_response({"error": "Network not available"}, status=503)
            
            await self.manager.network.connect(address)
            return web.json_response({"status": "connecting", "address": address})
            
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_metrics(self, request: web.Request) -> web.Response:
        """Prometheus-compatible metrics endpoint."""
        try:
            metrics = await self.manager.collect_metrics()
            
            # Convert to Prometheus format
            prometheus_metrics = []
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    prometheus_metrics.append(f"enhanced_csp_{key} {value}")
            
            return web.Response(
                text="\n".join(prometheus_metrics),
                content_type="text/plain"
            )
        except Exception as e:
            return web.Response(text=f"# Error collecting metrics: {e}", status=500)
    
    async def handle_info(self, request: web.Request) -> web.Response:
        """Legacy info endpoint."""
        return await self.handle_api_info(request)
    
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
  %(prog)s --bootstrap genesis.peoplesainetwork.com
  
  # Enable all features
  %(prog)s --enable-quantum --enable-blockchain --super-peer
"""
    )
    
    # Network options
    network = parser.add_argument_group('network')
    network.add_argument('--genesis', action='store_true',
                        help='Start as genesis node (first bootstrap)')
    network.add_argument('--bootstrap', nargs='*', default=[],
                        help='Bootstrap nodes (multiaddr or .peoplesainetwork.com domain)')
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
                    help='STUN servers for NAT traversal')
    nat.add_argument('--turn-servers', nargs='*',
                    help='TURN servers for NAT traversal')
    nat.add_argument('--disable-upnp', action='store_true',
                    help='Disable UPnP port mapping')
    nat.add_argument('--disable-nat-pmp', action='store_true',
                    help='Disable NAT-PMP port mapping')
    
    # Security options
    security = parser.add_argument_group('security')
    security.add_argument('--enable-mtls', action='store_true',
                         help='Enable mutual TLS authentication')
    security.add_argument('--enable-quantum', action='store_true',
                         help='Enable quantum-resistant cryptography')
    
    # Advanced features
    advanced = parser.add_argument_group('advanced features')
    advanced.add_argument('--enable-blockchain', action='store_true',
                         help='Enable blockchain integration')
    
    # Status server
    status = parser.add_argument_group('status server')
    status.add_argument('--no-status', action='store_true',
                       help='Disable HTTP status server')
    status.add_argument('--status-port', type=int, default=DEFAULT_STATUS_PORT,
                       help='Status server port (default: %(default)s)')
    
    # Interface options
    interface = parser.add_argument_group('interface')
    interface.add_argument('--no-shell', action='store_true',
                          help='Disable interactive shell')
    interface.add_argument('--no-color', action='store_true',
                          help='Disable colored output')
    
    # Logging options
    logging_group = parser.add_argument_group('logging')
    logging_group.add_argument('--log-level', default='INFO',
                              choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                              help='Set logging level (default: %(default)s)')
    logging_group.add_argument('--log-file', type=str,
                              help='Log to file instead of stdout')
    logging_group.add_argument('--debug', action='store_true',
                              help='Enable debug mode (equivalent to --log-level DEBUG)')
    
    args = parser.parse_args()
    
    # Post-process arguments
    if args.debug:
        args.log_level = 'DEBUG'
    
    return args


async def main():
    """Entry point."""
    try:
        args = parse_args()
        await run_main(args)
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Ensure we're using the right event loop policy on Windows
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main())