# Add these routes to the StatusServer class in main.py

async def handle_root(self, request: web.Request) -> web.Response:
    """Serve the dashboard HTML."""
    dashboard_path = Path(__file__).parent / 'dashboard' / 'index.html'
    if dashboard_path.exists():
        return web.FileResponse(dashboard_path)
    else:
        return web.Response(text="Dashboard not found", status=404)

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
    metrics = await self.manager.network.metrics()
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

# Update the setup_routes method:
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
