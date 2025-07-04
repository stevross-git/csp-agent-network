# Fixes for enhanced_csp/network/core/node.py
# Add this to NetworkNode.__init__ method:

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
    
# Replace _initialize_components method with:

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
    
# Add this class to the file:

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
    