#!/usr/bin/env python3
"""
Quick fix script for Enhanced CSP Network issues
Run this script to patch the immediate problems
"""

import os
import sys
import logging
from pathlib import Path

def fix_core_node_py():
    """Fix the core node.py file"""
    
    # Add to the NetworkNode class __init__ method
    metrics_init_code = '''
    # Add metrics attribute to prevent AttributeError
    self.metrics = {
        'messages_sent': 0,
        'messages_received': 0,
        'peers_connected': 0,
        'bandwidth_in': 0,
        'bandwidth_out': 0,
        'routing_table_size': 0,
        'last_updated': time.time()
    }
    '''
    
    # Fix the _initialize_components method to handle BatmanRouting properly
    fixed_init_components = '''
    async def _initialize_components(self):
        """Initialize network components with better error handling"""
        try:
            # Import components here to avoid circular imports
            from ..p2p.transport import P2PTransport
            from ..p2p.discovery import HybridDiscovery
            from ..p2p.dht import KademliaDHT
            from ..p2p.nat import NATTraversal
            from ..mesh.topology import MeshTopologyManager
            from ..dns.overlay import DNSOverlay
            from ..routing.adaptive import AdaptiveRoutingEngine
            
            # Initialize transport
            if hasattr(self.config, 'p2p'):
                self.transport = P2PTransport(self.config.p2p)
            
            # Initialize discovery
            if hasattr(self.config, 'p2p'):
                self.discovery = HybridDiscovery(self.node_id, self.config.p2p)
            
            # Initialize DHT
            if getattr(self.config, 'enable_dht', True):
                self.dht = KademliaDHT(self.node_id, self.config.p2p)
            
            # Initialize NAT traversal
            if hasattr(self.config, 'p2p'):
                self.nat = NATTraversal(self.config.p2p)
            
            # Initialize mesh topology
            if getattr(self.config, 'enable_mesh', True):
                self.topology = MeshTopologyManager(self.node_id, self.config.mesh)
            
            # Initialize routing - handle BatmanRouting import error
            if getattr(self.config, 'enable_routing', True) and self.topology:
                try:
                    from ..mesh.routing import BatmanRouting
                    self.routing = BatmanRouting(self, self.topology)
                except ImportError as e:
                    logging.warning(f"BatmanRouting not available: {e}")
                    # Create a simple routing stub
                    self.routing = SimpleRoutingStub()
            
            # Initialize DNS overlay
            if getattr(self.config, 'enable_dns', True) and self.dht:
                self.dns = DNSOverlay(self.node_id, self.dht, self.config.dns)
            
            # Initialize adaptive routing
            if getattr(self.config, 'enable_adaptive_routing', True) and self.routing:
                self.adaptive_routing = AdaptiveRoutingEngine(
                    self, 
                    self.config.routing, 
                    self.routing
                )
            
            logging.info("Network components initialized successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize components: {e}")
            return False
    '''
    
    return metrics_init_code, fixed_init_components

def create_routing_stub():
    """Create a simple routing stub to prevent import errors"""
    
    routing_stub = '''
class SimpleRoutingStub:
    """Simple routing stub to prevent import errors"""
    
    def __init__(self):
        self.routing_table = {}
        self.is_running = False
    
    async def start(self):
        self.is_running = True
        logging.info("Simple routing stub started")
    
    async def stop(self):
        self.is_running = False
        logging.info("Simple routing stub stopped")
    
    def get_route(self, destination):
        return self.routing_table.get(destination)
    
    def add_route(self, destination, route):
        self.routing_table[destination] = route
    
    def get_all_routes(self, destination):
        return []
    '''
    
    return routing_stub

def fix_enhanced_csp_network():
    """Fix the EnhancedCSPNetwork class"""
    
    enhanced_network_fix = '''
class EnhancedCSPNetwork:
    """Enhanced CSP Network with proper metrics support"""
    
    def __init__(self, config=None):
        self.config = config or NetworkConfig()
        self.nodes = {}
        # Add metrics attribute to prevent AttributeError
        self.metrics = {
            'nodes_active': 0,
            'total_messages': 0,
            'network_health': 100.0,
            'last_updated': time.time()
        }
        self.is_running = False
    
    async def create_node(self, name="default"):
        """Create and start a network node with proper error handling"""
        try:
            node = NetworkNode(self.config)
            if await node.start():
                self.nodes[name] = node
                self.metrics['nodes_active'] = len(self.nodes)
                return node
            else:
                logging.error("Failed to start network node")
                return None
        except Exception as e:
            logging.error(f"Failed to create node: {e}")
            return None
    
    def get_metrics(self):
        """Get network metrics safely"""
        return self.metrics.copy()
    '''
    
    return enhanced_network_fix

def fix_main_py_metrics():
    """Fix the main.py metrics collection"""
    
    main_fix = '''
# In main.py, replace the metrics collection section with:

async def collect_metrics(network):
    """Collect metrics safely"""
    try:
        if hasattr(network, 'metrics') and network.metrics:
            metrics_data = network.metrics
            if isinstance(metrics_data, dict):
                logger.info(f"Network metrics collected: {len(metrics_data)} metrics")
            else:
                logger.info("Network metrics available")
        else:
            logger.warning("Network metrics not available")
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")

# Replace the problematic metrics collection loop:
async def safe_metrics_loop(network):
    """Safe metrics collection loop"""
    while True:
        try:
            await collect_metrics(network)
            await asyncio.sleep(60)  # Collect every minute
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in metrics loop: {e}")
            await asyncio.sleep(60)  # Continue despite errors
    '''
    
    return main_fix

def create_patch_files():
    """Create patch files for the fixes"""
    
    # Create patches directory
    patches_dir = Path("patches")
    patches_dir.mkdir(exist_ok=True)
    
    # Create node.py patch
    node_metrics, node_init = fix_core_node_py()
    with open(patches_dir / "node_fixes.py", "w") as f:
        f.write("# Fixes for enhanced_csp/network/core/node.py\n")
        f.write("# Add this to NetworkNode.__init__ method:\n")
        f.write(node_metrics)
        f.write("\n# Replace _initialize_components method with:\n")
        f.write(node_init)
        f.write("\n# Add this class to the file:\n")
        f.write(create_routing_stub())
    
    # Create enhanced network patch
    with open(patches_dir / "enhanced_network_fixes.py", "w") as f:
        f.write("# Fixes for enhanced_csp/network/core/node.py EnhancedCSPNetwork class\n")
        f.write(fix_enhanced_csp_network())
    
    # Create main.py patch
    with open(patches_dir / "main_fixes.py", "w") as f:
        f.write("# Fixes for enhanced_csp/network/main.py\n")
        f.write(fix_main_py_metrics())
    
    print("✅ Patch files created in 'patches' directory")

def print_manual_fixes():
    """Print manual fixes to apply"""
    
    print("\n🔧 MANUAL FIXES TO APPLY:")
    print("="*50)
    
    print("\n1. Fix NetworkNode.__init__ method:")
    print("   Add this line after self.capabilities = ...")
    print("   self.metrics = {'messages_sent': 0, 'messages_received': 0, 'peers_connected': 0}")
    
    print("\n2. Fix EnhancedCSPNetwork.__init__ method:")
    print("   Add this line after self.nodes = {}:")
    print("   self.metrics = {'nodes_active': 0, 'total_messages': 0, 'network_health': 100.0}")
    
    print("\n3. Fix main.py metrics collection:")
    print("   Replace the metrics collection line with:")
    print("   if hasattr(network, 'metrics') and network.metrics:")
    print("       logger.info(f'Metrics: {network.metrics}')")
    
    print("\n4. Fix BatmanRouting import:")
    print("   In _initialize_components method, wrap BatmanRouting import in try-except:")
    print("   try:")
    print("       from ..mesh.routing import BatmanRouting")
    print("       self.routing = BatmanRouting(self, self.topology)")
    print("   except ImportError:")
    print("       logging.warning('BatmanRouting not available, using stub')")
    print("       self.routing = SimpleRoutingStub()")

def main():
    """Main fix application"""
    
    print("🚀 Enhanced CSP Network Quick Fix Script")
    print("="*50)
    
    print("\n📋 Issues to fix:")
    print("1. 'EnhancedCSPNetwork' object has no attribute 'metrics'")
    print("2. name 'BatmanRouting' is not defined")
    print("3. Failed to initialize components")
    print("4. Metrics collection errors")
    
    # Create patch files
    create_patch_files()
    
    # Print manual instructions
    print_manual_fixes()
    
    print("\n🎯 IMMEDIATE ACTIONS:")
    print("1. Stop the current network process (Ctrl+C)")
    print("2. Apply the manual fixes above")
    print("3. Or copy code from the patch files in 'patches' directory")
    print("4. Restart the network:")
    print("   python -m enhanced_csp.network.main --bootstrap genesis.web4ai")
    
    print("\n📁 Files to modify:")
    print("- enhanced_csp/network/core/node.py")
    print("- enhanced_csp/network/main.py")
    
    print("\n✅ After applying fixes, the network should start without errors!")

if __name__ == "__main__":
    main()
