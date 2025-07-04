# Fixes for enhanced_csp/network/core/node.py EnhancedCSPNetwork class

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
    