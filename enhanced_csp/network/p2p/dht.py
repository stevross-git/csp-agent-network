# enhanced_csp/network/p2p/dht.py
"""
Kademlia DHT wrapper for py-libp2p
Provides distributed hash table functionality for peer discovery and data storage
"""

import asyncio
import logging
import hashlib
import json
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta

try:
    from libp2p import new_node
    from libp2p.kademlia import KademliaServer
    from libp2p.peer.id import ID as PeerID
    from libp2p.peer.peerinfo import PeerInfo as LibP2PPeerInfo
    from libp2p.network.stream.net_stream_interface import INetStream
    LIBP2P_AVAILABLE = True
except ImportError:
    LIBP2P_AVAILABLE = False
    logger.warning("py-libp2p not available, using mock DHT")

from ..core.types import NodeID, PeerInfo, DHT as DHTProtocol
from ..core.node import NetworkNode


logger = logging.getLogger(__name__)


class KademliaDHT(DHTProtocol):
    """Kademlia DHT implementation with py-libp2p wrapper"""
    
    def __init__(self, node: NetworkNode):
        self.node = node
        self.kademlia: Optional[KademliaServer] = None
        self.libp2p_node = None
        self.bootstrap_nodes: List[Tuple[str, int]] = []
        
        # Cache for DHT operations
        self.cache: Dict[bytes, Tuple[bytes, datetime]] = {}
        self.cache_ttl = timedelta(minutes=10)
        
        # Replication settings
        self.replication_factor = 3
        self.republish_interval = 3600  # 1 hour
        
        self._tasks: List[asyncio.Task] = []
    
    async def start(self, listen_addr: str = "0.0.0.0", port: int = 0):
        """Start the Kademlia DHT"""
        if LIBP2P_AVAILABLE:
            await self._start_libp2p(listen_addr, port)
        else:
            await self._start_mock()
        
        # Start maintenance tasks
        self._tasks.extend([
            asyncio.create_task(self._republish_loop()),
            asyncio.create_task(self._cache_cleanup_loop())
        ])
        
        logger.info("Kademlia DHT started")
    
    async def stop(self):
        """Stop the DHT"""
        # Cancel tasks
        for task in self._tasks:
            task.cancel()
        
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        
        if self.kademlia:
            self.kademlia.stop()
        
        if self.libp2p_node:
            await self.libp2p_node.close()
        
        logger.info("Kademlia DHT stopped")
    
    async def _start_libp2p(self, listen_addr: str, port: int):
        """Start with actual libp2p implementation"""
        try:
            # Create libp2p node
            self.libp2p_node = await new_node(
                key_pair=self.node.private_key,  # Use node's Ed25519 key
                swarm_opt=[f"/ip4/{listen_addr}/tcp/{port}"]
            )
            
            # Create Kademlia server
            self.kademlia = KademliaServer(
                node_id=self.node.node_id.raw_id,
                storage=self._create_storage()
            )
            
            # Attach Kademlia protocol to libp2p node
            self.libp2p_node.get_mux().add_handler(
                "/kademlia/1.0.0", 
                self._handle_kademlia_stream
            )
            
            # Bootstrap if nodes are configured
            if self.bootstrap_nodes:
                await self.bootstrap(self.bootstrap_nodes)
            
        except Exception as e:
            logger.error(f"Failed to start libp2p DHT: {e}")
            raise
    
    async def _start_mock(self):
        """Start with mock implementation for testing"""
        logger.warning("Starting mock DHT (py-libp2p not available)")
        self.kademlia = MockKademliaServer(self.node.node_id.raw_id)
    
    def _create_storage(self):
        """Create storage backend for Kademlia"""
        # Could use LevelDB or other persistent storage
        return {}
    
    async def _handle_kademlia_stream(self, stream: INetStream):
        """Handle incoming Kademlia protocol stream"""
        # This would handle the actual Kademlia protocol messages
        # Implementation depends on libp2p stream handling
        pass
    
    async def bootstrap(self, nodes: List[Tuple[str, int]]):
        """Bootstrap the DHT with known nodes"""
        logger.info(f"Bootstrapping DHT with {len(nodes)} nodes")
        
        successful = 0
        for host, port in nodes:
            try:
                if self.kademlia:
                    await self.kademlia.bootstrap_node((host, port))
                    successful += 1
            except Exception as e:
                logger.error(f"Failed to bootstrap from {host}:{port}: {e}")
        
        logger.info(f"Successfully bootstrapped from {successful}/{len(nodes)} nodes")
        return successful > 0
    
    async def get(self, key: bytes) -> Optional[bytes]:
        """Get value from DHT"""
        # Check cache first
        if key in self.cache:
            value, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.cache_ttl:
                return value
        
        try:
            if self.kademlia:
                value = await self.kademlia.get(key)
                
                if value:
                    # Update cache
                    self.cache[key] = (value, datetime.now())
                    return value
            
            return None
            
        except Exception as e:
            logger.error(f"DHT get failed for key {key.hex()[:16]}: {e}")
            return None
    
    async def put(self, key: bytes, value: bytes) -> bool:
        """Store value in DHT with replication"""
        try:
            if self.kademlia:
                # Store in DHT
                await self.kademlia.set(key, value)
                
                # Update local cache
                self.cache[key] = (value, datetime.now())
                
                # Track for republishing
                await self._track_for_republish(key, value)
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"DHT put failed for key {key.hex()[:16]}: {e}")
            return False
    
    async def delete(self, key: bytes) -> bool:
        """Delete value from DHT (best effort)"""
        try:
            # Remove from local cache
            self.cache.pop(key, None)
            
            # DHT doesn't support direct deletion, but we can overwrite with empty
            return await self.put(key, b'')
            
        except Exception as e:
            logger.error(f"DHT delete failed for key {key.hex()[:16]}: {e}")
            return False
    
    async def find_peer(self, node_id: NodeID) -> Optional[PeerInfo]:
        """Find peer information by node ID"""
        try:
            # Use node ID as key
            key = node_id.raw_id
            
            # Try to find peer info in DHT
            data = await self.get(key)
            if data:
                peer_data = json.loads(data.decode())
                return self._deserialize_peer_info(peer_data)
            
            # If not found in DHT, try Kademlia's internal routing table
            if self.kademlia:
                closest = await self.kademlia.find_neighbors(node_id.raw_id, k=1)
                if closest and closest[0]['node_id'] == node_id.raw_id:
                    return self._kademlia_node_to_peer_info(closest[0])
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to find peer {node_id.to_base58()[:16]}: {e}")
            return None
    
    async def announce(self, key: bytes, port: int) -> bool:
        """Announce that we provide a resource"""
        try:
            # Create announcement data
            announcement = {
                'node_id': self.node.node_id.to_base58(),
                'addresses': self._get_announce_addresses(port),
                'timestamp': datetime.now().isoformat(),
                'ttl': 3600  # 1 hour
            }
            
            # Store announcement in DHT
            return await self.put(
                self._get_provider_key(key),
                json.dumps(announcement).encode()
            )
            
        except Exception as e:
            logger.error(f"Failed to announce for key {key.hex()[:16]}: {e}")
            return False
    
    async def find_providers(self, key: bytes, count: int = 10) -> List[Dict[str, Any]]:
        """Find nodes that provide a resource"""
        providers = []
        
        try:
            # Look for provider announcements
            provider_key = self._get_provider_key(key)
            
            # In a real implementation, this would query multiple nodes
            data = await self.get(provider_key)
            if data:
                announcement = json.loads(data.decode())
                if self._is_announcement_valid(announcement):
                    providers.append(announcement)
            
            # Could also check multiple provider keys for redundancy
            for i in range(min(count, 5)):
                alt_key = self._get_provider_key(key, suffix=i)
                data = await self.get(alt_key)
                if data:
                    announcement = json.loads(data.decode())
                    if self._is_announcement_valid(announcement):
                        providers.append(announcement)
            
            return providers[:count]
            
        except Exception as e:
            logger.error(f"Failed to find providers for {key.hex()[:16]}: {e}")
            return []
    
    async def find_closest_peers(self, target: NodeID, k: int = 20) -> List[Dict[str, Any]]:
        """Find k closest peers to target ID"""
        try:
            if self.kademlia:
                nodes = await self.kademlia.find_neighbors(target.raw_id, k=k)
                return [self._kademlia_node_to_dict(node) for node in nodes]
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to find closest peers: {e}")
            return []
    
    def _get_provider_key(self, key: bytes, suffix: int = 0) -> bytes:
        """Generate provider key for announcements"""
        base = b'provider:' + key
        if suffix > 0:
            base += f':{suffix}'.encode()
        return hashlib.sha256(base).digest()
    
    def _get_announce_addresses(self, port: int) -> List[str]:
        """Get addresses to announce"""
        addresses = []
        node_id = self.node.node_id.to_base58()
        
        # Get local IPs
        import socket
        hostname = socket.gethostname()
        try:
            # Try to get external IP
            import urllib.request
            external_ip = urllib.request.urlopen('https://api.ipify.org').read().decode('utf8')
            addresses.append(f"/ip4/{external_ip}/tcp/{port}/p2p/{node_id}")
            addresses.append(f"/ip4/{external_ip}/udp/{port}/quic/p2p/{node_id}")
        except:
            pass
        
        # Add local addresses
        for addr_info in socket.getaddrinfo(hostname, port):
            ip = addr_info[4][0]
            if ':' in ip:  # IPv6
                addresses.append(f"/ip6/{ip}/tcp/{port}/p2p/{node_id}")
            else:
                addresses.append(f"/ip4/{ip}/tcp/{port}/p2p/{node_id}")
        
        return addresses
    
    def _is_announcement_valid(self, announcement: Dict[str, Any]) -> bool:
        """Check if provider announcement is still valid"""
        try:
            timestamp = datetime.fromisoformat(announcement['timestamp'])
            ttl = announcement.get('ttl', 3600)
            age = (datetime.now() - timestamp).total_seconds()
            return age < ttl
        except:
            return False
    
    def _deserialize_peer_info(self, data: Dict[str, Any]) -> PeerInfo:
        """Convert stored data to PeerInfo"""
        # This would properly deserialize the peer info
        # For now, return a mock
        return None
    
    def _kademlia_node_to_peer_info(self, node: Dict[str, Any]) -> PeerInfo:
        """Convert Kademlia node data to PeerInfo"""
        # This would convert Kademlia's internal format
        return None
    
    def _kademlia_node_to_dict(self, node: Any) -> Dict[str, Any]:
        """Convert Kademlia node to dictionary"""
        return {
            'node_id': node.id.hex() if hasattr(node, 'id') else '',
            'address': str(node.address) if hasattr(node, 'address') else '',
            'distance': node.distance if hasattr(node, 'distance') else 0
        }
    
    async def _track_for_republish(self, key: bytes, value: bytes):
        """Track data for periodic republishing"""
        # In production, this would use persistent storage
        pass
    
    async def _republish_loop(self):
        """Periodically republish stored data"""
        while True:
            try:
                await asyncio.sleep(self.republish_interval)
                
                # Republish announcements and important data
                # This ensures data remains available despite node churn
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in republish loop: {e}")
    
    async def _cache_cleanup_loop(self):
        """Clean up expired cache entries"""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                now = datetime.now()
                expired = [
                    key for key, (_, timestamp) in self.cache.items()
                    if now - timestamp > self.cache_ttl
                ]
                
                for key in expired:
                    del self.cache[key]
                
                if expired:
                    logger.debug(f"Cleaned up {len(expired)} expired cache entries")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")


class MockKademliaServer:
    """Mock Kademlia server for testing without py-libp2p"""
    
    def __init__(self, node_id: bytes):
        self.node_id = node_id
        self.storage = {}
        self.routing_table = {}
    
    async def bootstrap_node(self, node: Tuple[str, int]):
        """Mock bootstrap"""
        logger.info(f"Mock bootstrap to {node}")
        return True
    
    async def get(self, key: bytes) -> Optional[bytes]:
        """Mock get"""
        return self.storage.get(key)
    
    async def set(self, key: bytes, value: bytes):
        """Mock set"""
        self.storage[key] = value
    
    async def find_neighbors(self, target: bytes, k: int = 20) -> List[Dict]:
        """Mock find neighbors"""
        return []
    
    def stop(self):
        """Mock stop"""
        pass