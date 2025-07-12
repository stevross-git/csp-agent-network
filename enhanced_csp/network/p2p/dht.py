# network/p2p/dht.py
"""
Kademlia Distributed Hash Table implementation for Enhanced CSP.
Provides decentralized key-value storage and peer discovery.
"""

import asyncio
import hashlib
import time
import logging
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json

from ..core.types import NodeID, PeerInfo, MessageType
from ..p2p.transport import P2PTransport
from ..utils import TaskManager

logger = logging.getLogger(__name__)

# Kademlia constants
K_BUCKET_SIZE = 20  # Maximum nodes per k-bucket
ALPHA = 3  # Concurrent RPCs for lookups
KEY_SIZE = 256  # bits
REPUBLISH_INTERVAL = 3600  # 1 hour
EXPIRATION_TIME = 86400  # 24 hours
RPC_TIMEOUT = 5  # seconds


@dataclass
class KBucketEntry:
    """Entry in a Kademlia k-bucket."""
    node_id: NodeID
    peer_info: PeerInfo
    last_seen: datetime = field(default_factory=datetime.utcnow)
    rtt: float = 0.0  # Round-trip time in ms
    
    def touch(self):
        """Update last seen time."""
        self.last_seen = datetime.utcnow()


class KBucket:
    """A single k-bucket in the routing table."""
    
    def __init__(self, range_min: int, range_max: int, k: int = K_BUCKET_SIZE):
        self.range_min = range_min
        self.range_max = range_max
        self.k = k
        self.entries: List[KBucketEntry] = []
        self.replacement_cache: List[KBucketEntry] = []
        
    def add_node(self, entry: KBucketEntry) -> bool:
        """Add a node to the bucket."""
        # Check if node already exists
        for i, existing in enumerate(self.entries):
            if existing.node_id == entry.node_id:
                # Move to end (most recently seen)
                self.entries.pop(i)
                self.entries.append(entry)
                entry.touch()
                return True
        
        # Add new node if space available
        if len(self.entries) < self.k:
            self.entries.append(entry)
            return True
            
        # Bucket full - add to replacement cache
        self.replacement_cache.append(entry)
        if len(self.replacement_cache) > self.k:
            self.replacement_cache.pop(0)
            
        return False
    
    def remove_node(self, node_id: NodeID):
        """Remove a node from the bucket."""
        self.entries = [e for e in self.entries if e.node_id != node_id]
        
        # Promote from replacement cache if available
        if self.replacement_cache and len(self.entries) < self.k:
            self.entries.append(self.replacement_cache.pop(0))
    
    def get_nodes(self, count: int = None) -> List[KBucketEntry]:
        """Get nodes from bucket, most recently seen first."""
        if count is None:
            return self.entries[:]
        return self.entries[-count:]


class RoutingTable:
    """Kademlia routing table."""
    
    def __init__(self, node_id: NodeID, k: int = K_BUCKET_SIZE):
        self.node_id = node_id
        self.k = k
        self.buckets: List[KBucket] = []
        
        # Initialize buckets
        for i in range(KEY_SIZE):
            range_min = 2**i if i > 0 else 0
            range_max = 2**(i+1) - 1
            self.buckets.append(KBucket(range_min, range_max, k))
    
    def distance(self, id1: NodeID, id2: NodeID) -> int:
        """Calculate XOR distance between two node IDs."""
        # Convert to integers for XOR
        int1 = int.from_bytes(id1.value.encode()[:32], 'big')
        int2 = int.from_bytes(id2.value.encode()[:32], 'big')
        return int1 ^ int2
    
    def bucket_index(self, node_id: NodeID) -> int:
        """Get bucket index for a node ID."""
        distance = self.distance(self.node_id, node_id)
        if distance == 0:
            return -1
        return distance.bit_length() - 1
    
    def add_node(self, peer_info: PeerInfo) -> bool:
        """Add a node to the routing table."""
        if peer_info.id == self.node_id:
            return False
            
        bucket_idx = self.bucket_index(peer_info.id)
        if bucket_idx < 0:
            return False
            
        entry = KBucketEntry(peer_info.id, peer_info)
        return self.buckets[bucket_idx].add_node(entry)
    
    def remove_node(self, node_id: NodeID):
        """Remove a node from the routing table."""
        bucket_idx = self.bucket_index(node_id)
        if bucket_idx >= 0:
            self.buckets[bucket_idx].remove_node(node_id)
    
    def find_closest_nodes(self, target: NodeID, count: int = K_BUCKET_SIZE) -> List[PeerInfo]:
        """Find the k closest nodes to a target ID."""
        # Get all nodes with their distances
        nodes_with_distance = []
        
        for bucket in self.buckets:
            for entry in bucket.entries:
                distance = self.distance(entry.node_id, target)
                nodes_with_distance.append((distance, entry.peer_info))
        
        # Sort by distance and return closest
        nodes_with_distance.sort(key=lambda x: x[0])
        return [peer_info for _, peer_info in nodes_with_distance[:count]]


@dataclass
class StoredValue:
    """Value stored in the DHT."""
    key: str
    value: Any
    publisher: NodeID
    timestamp: datetime
    ttl: int  # seconds
    
    def is_expired(self) -> bool:
        """Check if the value has expired."""
        age = (datetime.utcnow() - self.timestamp).total_seconds()
        return age > self.ttl


class KademliaDHT:
    """
    Kademlia Distributed Hash Table implementation.
    Provides distributed storage and peer discovery.
    """
    
    def __init__(self, node_id: NodeID, transport: P2PTransport):
        """Initialize Kademlia DHT."""
        self.node_id = node_id
        self.transport = transport
        self.routing_table = RoutingTable(node_id)
        
        # Local storage
        self.storage: Dict[str, StoredValue] = {}
        
        # Pending RPCs
        self.pending_rpcs: Dict[str, asyncio.Future] = {}
        
        self.task_manager = TaskManager()
        
        self.is_running = False
        
    async def start(self):
        """Start the DHT."""
        if self.is_running:
            return
            
        logger.info(f"Starting Kademlia DHT for node {self.node_id}")
        self.is_running = True
        
        # Register message handlers
        self.transport.register_handler(MessageType.DHT_QUERY, self._handle_dht_query)
        self.transport.register_handler(MessageType.DHT_RESPONSE, self._handle_dht_response)
        
        self.task_manager.create_task(self._maintenance_loop())
        self.task_manager.create_task(self._republish_loop())
        
    async def stop(self):
        """Stop the DHT."""
        if not self.is_running:
            return
            
        logger.info("Stopping Kademlia DHT")
        self.is_running = False
        
        await self.task_manager.cancel_all()
        
        # Clear pending RPCs
        for future in self.pending_rpcs.values():
            future.cancel()
        self.pending_rpcs.clear()
    
    # Public API
    
    async def store(self, key: str, value: Any, ttl: int = EXPIRATION_TIME) -> bool:
        """
        Store a key-value pair in the DHT.
        
        Args:
            key: The key to store
            value: The value to store (must be JSON serializable)
            ttl: Time-to-live in seconds
            
        Returns:
            True if stored successfully
        """
        # Store locally
        stored_value = StoredValue(
            key=key,
            value=value,
            publisher=self.node_id,
            timestamp=datetime.utcnow(),
            ttl=ttl
        )
        self.storage[key] = stored_value
        
        # Find k closest nodes to the key
        key_hash = self._hash_key(key)
        closest_nodes = self.routing_table.find_closest_nodes(
            NodeID(key_hash), 
            self.routing_table.k
        )
        
        # Store on closest nodes
        store_tasks = []
        for peer in closest_nodes:
            if peer.id != self.node_id:
                store_tasks.append(self._store_on_node(peer, key, value, ttl))
        
        if store_tasks:
            results = await asyncio.gather(*store_tasks, return_exceptions=True)
            success_count = sum(1 for r in results if r is True)
            logger.debug(f"Stored {key} on {success_count}/{len(store_tasks)} nodes")
            
        return True
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a value from the DHT.
        
        Args:
            key: The key to retrieve
            
        Returns:
            The value if found, None otherwise
        """
        # Check local storage first
        if key in self.storage:
            stored = self.storage[key]
            if not stored.is_expired():
                return stored.value
            else:
                del self.storage[key]
        
        # Find value on network
        key_hash = self._hash_key(key)
        result = await self._iterative_find_value(key, NodeID(key_hash))
        
        if result:
            # Cache the value locally
            self.storage[key] = StoredValue(
                key=key,
                value=result['value'],
                publisher=NodeID(result['publisher']),
                timestamp=datetime.utcnow(),
                ttl=result.get('ttl', EXPIRATION_TIME)
            )
            return result['value']
            
        return None
    
    async def find_node(self, node_id: NodeID) -> Optional[PeerInfo]:
        """
        Find a specific node in the network.
        
        Args:
            node_id: The node ID to find
            
        Returns:
            PeerInfo if found, None otherwise
        """
        # Check routing table first
        bucket_idx = self.routing_table.bucket_index(node_id)
        if bucket_idx >= 0:
            bucket = self.routing_table.buckets[bucket_idx]
            for entry in bucket.entries:
                if entry.node_id == node_id:
                    return entry.peer_info
        
        # Perform iterative node lookup
        closest_nodes = await self._iterative_find_node(node_id)
        
        for peer in closest_nodes:
            if peer.id == node_id:
                return peer
                
        return None
    
    def add_peer(self, peer_info: PeerInfo):
        """Add a peer to the routing table."""
        self.routing_table.add_node(peer_info)
    
    # RPC handlers
    
    async def _handle_dht_query(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming DHT query."""
        query_type = message.get('type')
        
        if query_type == 'ping':
            return {'type': 'pong', 'node_id': self.node_id.value}
            
        elif query_type == 'find_node':
            target_id = NodeID(message['target'])
            closest = self.routing_table.find_closest_nodes(target_id, K_BUCKET_SIZE)
            return {
                'type': 'nodes',
                'nodes': [self._peer_to_dict(p) for p in closest]
            }
            
        elif query_type == 'find_value':
            key = message['key']
            
            # Check if we have the value
            if key in self.storage:
                stored = self.storage[key]
                if not stored.is_expired():
                    return {
                        'type': 'value',
                        'value': stored.value,
                        'publisher': stored.publisher.value,
                        'ttl': stored.ttl
                    }
            
            # Return closest nodes instead
            key_hash = self._hash_key(key)
            closest = self.routing_table.find_closest_nodes(NodeID(key_hash), K_BUCKET_SIZE)
            return {
                'type': 'nodes',
                'nodes': [self._peer_to_dict(p) for p in closest]
            }
            
        elif query_type == 'store':
            key = message['key']
            value = message['value']
            ttl = message.get('ttl', EXPIRATION_TIME)
            publisher = NodeID(message['publisher'])
            
            self.storage[key] = StoredValue(
                key=key,
                value=value,
                publisher=publisher,
                timestamp=datetime.utcnow(),
                ttl=ttl
            )
            
            return {'type': 'store_ack', 'stored': True}
            
        else:
            return {'type': 'error', 'message': f'Unknown query type: {query_type}'}
    
    async def _handle_dht_response(self, message: Dict[str, Any]):
        """Handle DHT response."""
        request_id = message.get('request_id')
        if request_id in self.pending_rpcs:
            future = self.pending_rpcs.pop(request_id)
            if not future.done():
                future.set_result(message)
    
    # Internal methods
    
    async def _iterative_find_node(self, target: NodeID) -> List[PeerInfo]:
        """Perform iterative node lookup."""
        # Get initial closest nodes from routing table
        closest_nodes = self.routing_table.find_closest_nodes(target, ALPHA)
        queried_nodes: Set[NodeID] = {self.node_id}
        
        while True:
            # Query unqueried nodes
            unqueried = [n for n in closest_nodes if n.id not in queried_nodes]
            if not unqueried:
                break
                
            # Query up to ALPHA nodes in parallel
            query_tasks = []
            for peer in unqueried[:ALPHA]:
                queried_nodes.add(peer.id)
                query_tasks.append(self._query_find_node(peer, target))
            
            results = await asyncio.gather(*query_tasks, return_exceptions=True)
            
            # Process results
            new_nodes_found = False
            for result in results:
                if isinstance(result, list):
                    for peer in result:
                        if peer.id not in [n.id for n in closest_nodes]:
                            closest_nodes.append(peer)
                            new_nodes_found = True
            
            if not new_nodes_found:
                break
            
            # Keep only k closest
            closest_nodes.sort(key=lambda p: self.routing_table.distance(p.id, target))
            closest_nodes = closest_nodes[:K_BUCKET_SIZE]
        
        return closest_nodes
    
    async def _iterative_find_value(self, key: str, key_hash: NodeID) -> Optional[Dict[str, Any]]:
        """Perform iterative value lookup."""
        closest_nodes = self.routing_table.find_closest_nodes(key_hash, ALPHA)
        queried_nodes: Set[NodeID] = {self.node_id}
        
        while closest_nodes:
            # Query unqueried nodes
            unqueried = [n for n in closest_nodes if n.id not in queried_nodes]
            if not unqueried:
                break
            
            # Query up to ALPHA nodes
            query_tasks = []
            for peer in unqueried[:ALPHA]:
                queried_nodes.add(peer.id)
                query_tasks.append(self._query_find_value(peer, key))
            
            results = await asyncio.gather(*query_tasks, return_exceptions=True)
            
            # Check for value
            for result in results:
                if isinstance(result, dict) and result.get('type') == 'value':
                    return result
                elif isinstance(result, dict) and result.get('nodes'):
                    # Add new nodes to search
                    for node_dict in result['nodes']:
                        peer = self._dict_to_peer(node_dict)
                        if peer and peer.id not in [n.id for n in closest_nodes]:
                            closest_nodes.append(peer)
            
            # Keep searching with closest nodes
            closest_nodes.sort(key=lambda p: self.routing_table.distance(p.id, key_hash))
            closest_nodes = closest_nodes[:K_BUCKET_SIZE]
        
        return None
    
    async def _query_find_node(self, peer: PeerInfo, target: NodeID) -> List[PeerInfo]:
        """Query a peer for nodes close to target."""
        try:
            response = await self._send_rpc(peer, {
                'type': 'find_node',
                'target': target.value
            })
            
            if response and response.get('nodes'):
                return [self._dict_to_peer(n) for n in response['nodes'] if n]
                
        except Exception as e:
            logger.debug(f"find_node query to {peer.id} failed: {e}")
            
        return []
    
    async def _query_find_value(self, peer: PeerInfo, key: str) -> Optional[Dict[str, Any]]:
        """Query a peer for a value."""
        try:
            response = await self._send_rpc(peer, {
                'type': 'find_value',
                'key': key
            })
            return response
            
        except Exception as e:
            logger.debug(f"find_value query to {peer.id} failed: {e}")
            return None
    
    async def _store_on_node(self, peer: PeerInfo, key: str, value: Any, ttl: int) -> bool:
        """Store a value on a specific node."""
        try:
            response = await self._send_rpc(peer, {
                'type': 'store',
                'key': key,
                'value': value,
                'ttl': ttl,
                'publisher': self.node_id.value
            })
            
            return response and response.get('stored', False)
            
        except Exception as e:
            logger.debug(f"Store on {peer.id} failed: {e}")
            return False
    
    async def _send_rpc(self, peer: PeerInfo, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send an RPC query to a peer."""
        import uuid
        request_id = str(uuid.uuid4())
        
        # Create future for response
        future = asyncio.Future()
        self.pending_rpcs[request_id] = future
        
        # Send query
        message = {
            'request_id': request_id,
            **query
        }
        
        success = await self.transport.send(peer.address, {
            'type': MessageType.DHT_QUERY,
            'data': message
        })
        
        if not success:
            self.pending_rpcs.pop(request_id, None)
            return None
        
        try:
            # Wait for response
            response = await asyncio.wait_for(future, timeout=RPC_TIMEOUT)
            return response
        except asyncio.TimeoutError:
            logger.debug(f"RPC timeout to {peer.id}")
            return None
        finally:
            self.pending_rpcs.pop(request_id, None)
    
    def _hash_key(self, key: str) -> str:
        """Hash a key to node ID format."""
        hash_bytes = hashlib.sha256(key.encode()).digest()
        return f"Qm{hash_bytes.hex()[:44]}"
    
    def _peer_to_dict(self, peer: PeerInfo) -> Dict[str, Any]:
        """Convert PeerInfo to dictionary."""
        return {
            'node_id': peer.id.value,
            'address': peer.address,
            'port': peer.port,
            'capabilities': peer.capabilities.__dict__
        }
    
    def _dict_to_peer(self, data: Dict[str, Any]) -> Optional[PeerInfo]:
        """Convert dictionary to PeerInfo."""
        try:
            from ..core.types import NodeCapabilities
            return PeerInfo(
                id=NodeID(data['node_id']),
                address=data['address'],
                port=data['port'],
                capabilities=NodeCapabilities(**data.get('capabilities', {})),
                last_seen=datetime.utcnow()
            )
        except Exception as e:
            logger.error(f"Failed to parse peer info: {e}")
            return None
    
    # Background tasks
    
    async def _maintenance_loop(self):
        """Periodic maintenance tasks."""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                # Clean expired values
                expired_keys = []
                for key, stored in self.storage.items():
                    if stored.is_expired():
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del self.storage[key]
                
                if expired_keys:
                    logger.debug(f"Cleaned {len(expired_keys)} expired values")
                
                # Refresh buckets
                await self._refresh_buckets()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in DHT maintenance: {e}")
    
    async def _republish_loop(self):
        """Periodically republish stored values."""
        while self.is_running:
            try:
                await asyncio.sleep(REPUBLISH_INTERVAL)
                
                # Republish values we're responsible for
                for key, stored in list(self.storage.items()):
                    if stored.publisher == self.node_id:
                        await self.store(key, stored.value, stored.ttl)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in republish loop: {e}")
    
    async def _refresh_buckets(self):
        """Refresh k-buckets that haven't been accessed recently."""
        for i, bucket in enumerate(self.routing_table.buckets):
            if not bucket.entries:
                continue
                
            # Check if bucket needs refresh (no activity in last hour)
            most_recent = max(e.last_seen for e in bucket.entries)
            if (datetime.utcnow() - most_recent).total_seconds() > 3600:
                # Generate random ID in bucket range
                from ..utils.secure_random import secure_randint
                random_id = secure_randint(bucket.range_min, bucket.range_max)
                target = NodeID(f"Qm{random_id:064x}"[:48])
                
                # Perform lookup to refresh bucket
                await self._iterative_find_node(target)