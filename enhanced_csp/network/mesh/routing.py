# enhanced_csp/network/mesh/routing.py
"""
B.A.T.M.A.N.-inspired routing protocol implementation
Better Approach To Mobile Ad-hoc Networking - adapted for Enhanced CSP
"""

import asyncio
import logging
import datetime
import time
import random
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import struct
from typing import TYPE_CHECKING

from ..core.types import NodeID, MessageType


if TYPE_CHECKING:
    from .node import NetworkNode
from .topology import MeshTopologyManager


logger = logging.getLogger(__name__)

@dataclass
class RoutingEntry:
    """Entry in the routing table."""
    destination: NodeID
    next_hop: NodeID
    metric: float  # Lower is better
    sequence_number: int
    last_updated: datetime.datetime
    path: List[NodeID] = field(default_factory=list)
    
    def is_stale(self) -> bool:
        """Check if routing entry is stale."""
        age = (datetime.datetime.utcnow() - self.last_updated).total_seconds()
        return age > 300  # 5 minutes
    
@dataclass
class OriginatorMessage:
    """B.A.T.M.A.N. originator message"""
    originator_id: NodeID
    sequence_number: int
    ttl: int
    flags: int = 0
    gateway_flags: int = 0
    tq: int = 255  # Transmission Quality (0-255)
    timestamp: float = field(default_factory=time.time)
    
    def to_bytes(self) -> bytes:
        """Serialize to bytes for network transmission"""
        # Pack message: [originator_id(32)][seq(4)][ttl(1)][flags(1)][gw_flags(1)][tq(1)]
        return (
            self.originator_id.raw_id[:32] +  # 32 bytes
            struct.pack("!IBBBB", 
                       self.sequence_number,
                       self.ttl,
                       self.flags,
                       self.gateway_flags,
                       self.tq)
        )
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'OriginatorMessage':
        """Deserialize from bytes"""
        if len(data) < 40:
            raise ValueError("Invalid originator message size")
        
        originator_id = NodeID(raw_id=data[:32], public_key=None)
        seq, ttl, flags, gw_flags, tq = struct.unpack("!IBBBB", data[32:40])
        
        return cls(
            originator_id=originator_id,
            sequence_number=seq,
            ttl=ttl,
            flags=flags,
            gateway_flags=gw_flags,
            tq=tq
        )


@dataclass
class RoutingTableEntry:
    """Entry in B.A.T.M.A.N. routing table"""
    destination: NodeID
    next_hop: NodeID
    last_seqno: int
    tq: int  # Transmission quality (0-255)
    hop_count: int
    last_seen: float
    window_size: int = 64  # Sliding window for packet loss
    packet_window: int = 0  # Bit field for received packets
    
    def update_packet_window(self, seqno: int) -> bool:
        """Update sliding window with new sequence number"""
        if seqno <= self.last_seqno:
            # Old or duplicate packet
            bit_pos = self.last_seqno - seqno
            if bit_pos < self.window_size:
                # Check if already received
                if self.packet_window & (1 << bit_pos):
                    return False  # Duplicate
                self.packet_window |= (1 << bit_pos)
            return True
        
        # New packet - shift window
        shift = seqno - self.last_seqno
        if shift >= self.window_size:
            # Complete window reset
            self.packet_window = 1
        else:
            # Shift and set bit
            self.packet_window = (self.packet_window << shift) | 1
        
        self.last_seqno = seqno
        return True
    
    def calculate_packet_loss(self) -> float:
        """Calculate packet loss from sliding window"""
        if self.packet_window == 0:
            return 1.0
        
        # Count set bits (received packets)
        received = bin(self.packet_window).count('1')
        return 1.0 - (received / self.window_size)


class BatmanRouting:
    """B.A.T.M.A.N.-inspired routing protocol"""
    
    # Protocol constants
    OGM_INTERVAL = 1.0  # Originator message interval (seconds)
    PURGE_TIMEOUT = 200.0  # Route timeout (seconds)
    TQ_LOCAL_WINDOW_SIZE = 64
    TQ_GLOBAL_WINDOW_SIZE = 5
    TQ_MAX = 255
    TTL_DEFAULT = 50
    
    def __init__(self, node: 'NetworkNode', topology_manager: MeshTopologyManager):
        self.node = node
        self.topology = topology_manager
        
        # Routing table: destination -> best route
        self.routing_table: Dict[NodeID, RoutingTableEntry] = {}
        
        # Alternative routes: destination -> list of routes
        self.alternative_routes: Dict[NodeID, List[RoutingTableEntry]] = defaultdict(list)
        
        # Sequence number for our OGMs
        self.sequence_number = 0
        
        # Neighbor link quality tracking
        self.neighbor_tq: Dict[NodeID, int] = {}
        
        # Last OGM received from each neighbor
        self.last_ogm: Dict[NodeID, OriginatorMessage] = {}
        
        # Pending route requests
        self.pending_routes: Dict[NodeID, asyncio.Event] = {}
        
        # Tasks
        self._tasks: List[asyncio.Task] = []
    
    async def start(self):
        """Start routing protocol"""
        logger.info("Starting B.A.T.M.A.N. routing protocol")
        
        # Start protocol tasks
        self._tasks.extend([
            asyncio.create_task(self._ogm_sender_loop()),
            asyncio.create_task(self._route_maintenance_loop()),
            asyncio.create_task(self._tq_calculation_loop())
        ])
        
        # Register message handlers
        self.node.on_event('originator_message', self.handle_ogm)
    
    async def stop(self):
        """Stop routing protocol"""
        for task in self._tasks:
            task.cancel()
        
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
    
    async def _ogm_sender_loop(self):
        """Periodically send originator messages"""
        while True:
            try:
                await self.send_ogm()
                await asyncio.sleep(self.OGM_INTERVAL)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in OGM sender: {e}")
    
    # Replace the send_ogm method in BatmanRouting class with this fixed version:

    async def send_ogm(self):
        """Send originator message to all neighbors"""
        # Increment sequence number
        self.sequence_number += 1
        
        # Create OGM
        ogm = OriginatorMessage(
            originator_id=self.node.node_id,
            sequence_number=self.sequence_number,
            ttl=self.TTL_DEFAULT,
            tq=self.TQ_MAX  # Perfect quality for our own messages
        )
        
        # Determine if we're a gateway (super-peer)
        if hasattr(self.topology, 'super_peers') and self.node.node_id in self.topology.super_peers:
            ogm.gateway_flags = 1
        
        # Get neighbors - handle both old and new topology interfaces
        neighbors = set()
        
        # Try new interface first
        if hasattr(self.topology, 'peer_connections'):
            neighbors = self.topology.peer_connections.get(self.node.node_id, set())
        # Fallback to get_mesh_neighbors
        elif hasattr(self.topology, 'get_mesh_neighbors'):
            neighbors = set(self.topology.get_mesh_neighbors())
        
        for neighbor_id in neighbors:
            # Adjust TQ based on link quality to neighbor
            neighbor_tq = self.neighbor_tq.get(neighbor_id, self.TQ_MAX)
            ogm_copy = OriginatorMessage(
                originator_id=ogm.originator_id,
                sequence_number=ogm.sequence_number,
                ttl=ogm.ttl,
                flags=ogm.flags,
                gateway_flags=ogm.gateway_flags,
                tq=ogm.tq
            )
            
            await self.send_ogm_to_neighbor(neighbor_id, ogm_copy)
    
    async def send_ogm_to_neighbor(self, neighbor_id: NodeID, ogm: OriginatorMessage):
        """Send OGM to specific neighbor"""
        message = {
            'type': 'originator_message',
            'data': ogm.to_bytes().hex()
        }
        
        await self.node.send_message(neighbor_id, message)
    
    async def handle_ogm(self, data: Dict):
        """Handle received originator message"""
        try:
            # Parse OGM
            ogm_bytes = bytes.fromhex(data['data'])
            ogm = OriginatorMessage.from_bytes(ogm_bytes)
            sender_id = data.get('sender_id')  # Direct sender
            
            if not sender_id:
                return
            
            # Ignore our own OGMs
            if ogm.originator_id == self.node.node_id:
                return
            
            # Update last seen
            self.last_ogm[sender_id] = ogm
            
            # Check TTL
            if ogm.ttl <= 0:
                return
            
            # Process OGM
            await self._process_ogm(ogm, sender_id)
            
            # Forward OGM if TTL > 1
            if ogm.ttl > 1:
                await self._forward_ogm(ogm, sender_id)
                
        except Exception as e:
            logger.error(f"Error handling OGM: {e}")
    
    async def _process_ogm(self, ogm: OriginatorMessage, sender_id: NodeID):
        """Process received OGM and update routing table"""
        originator = ogm.originator_id
        
        # Calculate effective TQ (transmission quality)
        # TQ = OGM_TQ * link_quality_to_sender
        link_tq = self.neighbor_tq.get(sender_id, self.TQ_MAX)
        effective_tq = (ogm.tq * link_tq) // self.TQ_MAX
        
        # Get or create routing entry
        if originator in self.routing_table:
            entry = self.routing_table[originator]
            
            # Update sliding window
            is_new = entry.update_packet_window(ogm.sequence_number)
            
            if not is_new:
                # Duplicate packet
                return
            
            # Check if this is a better route
            if effective_tq > entry.tq or sender_id == entry.next_hop:
                # Update route
                entry.next_hop = sender_id
                entry.tq = effective_tq
                entry.hop_count = self.TTL_DEFAULT - ogm.ttl + 1
                entry.last_seen = time.time()
        else:
            # New route
            entry = RoutingTableEntry(
                destination=originator,
                next_hop=sender_id,
                last_seqno=ogm.sequence_number,
                tq=effective_tq,
                hop_count=self.TTL_DEFAULT - ogm.ttl + 1,
                last_seen=time.time()
            )
            self.routing_table[originator] = entry
            
            # Signal any waiting route requests
            if originator in self.pending_routes:
                self.pending_routes[originator].set()
        
        # Store as alternative route
        alt_entry = RoutingTableEntry(
            destination=originator,
            next_hop=sender_id,
            last_seqno=ogm.sequence_number,
            tq=effective_tq,
            hop_count=self.TTL_DEFAULT - ogm.ttl + 1,
            last_seen=time.time()
        )
        
        # Keep top 3 alternative routes
        alt_routes = self.alternative_routes[originator]
        alt_routes.append(alt_entry)
        alt_routes.sort(key=lambda r: r.tq, reverse=True)
        self.alternative_routes[originator] = alt_routes[:3]
    
    # Replace the _forward_ogm method in BatmanRouting class with this fixed version:

    async def _forward_ogm(self, ogm: OriginatorMessage, received_from: NodeID):
        """Forward OGM to other neighbors"""
        # Decrease TTL
        ogm.ttl -= 1
        
        # Apply forwarding penalty to TQ
        ogm.tq = (ogm.tq * 230) // 255  # ~10% penalty
        
        # Get our neighbors - handle both old and new topology interfaces
        neighbors = set()
        
        # Try new interface first
        if hasattr(self.topology, 'peer_connections'):
            neighbors = self.topology.peer_connections.get(self.node.node_id, set())
        # Fallback to get_mesh_neighbors
        elif hasattr(self.topology, 'get_mesh_neighbors'):
            neighbors = set(self.topology.get_mesh_neighbors())
        
        # Forward to all neighbors except sender
        for neighbor_id in neighbors:
            if neighbor_id != received_from:
                await self.send_ogm_to_neighbor(neighbor_id, ogm)
    
    async def _route_maintenance_loop(self):
        """Maintain routing table and remove stale routes"""
        while True:
            try:
                await asyncio.sleep(10)  # Every 10 seconds
                
                current_time = time.time()
                stale_routes = []
                
                # Check for stale routes
                for dest, entry in self.routing_table.items():
                    if current_time - entry.last_seen > self.PURGE_TIMEOUT:
                        stale_routes.append(dest)
                
                # Remove stale routes
                for dest in stale_routes:
                    del self.routing_table[dest]
                    self.alternative_routes.pop(dest, None)
                    logger.info(f"Purged stale route to {dest.to_base58()[:16]}")
                
                # Clean up alternative routes
                for dest, alt_routes in list(self.alternative_routes.items()):
                    self.alternative_routes[dest] = [
                        r for r in alt_routes
                        if current_time - r.last_seen <= self.PURGE_TIMEOUT
                    ]
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in route maintenance: {e}")
    
    async def _tq_calculation_loop(self):
        """Calculate link quality to neighbors"""
        while True:
            try:
                await asyncio.sleep(5)  # Every 5 seconds
                
                # Update TQ for each neighbor based on packet loss
                for neighbor_id in self.topology.peer_connections.get(self.node.node_id, set()):
                    # Get neighbor's route entry (to us)
                    if neighbor_id in self.routing_table:
                        entry = self.routing_table[neighbor_id]
                        packet_loss = entry.calculate_packet_loss()
                        
                        # Calculate new TQ
                        new_tq = int(self.TQ_MAX * (1.0 - packet_loss))
                        
                        # Smooth with exponential moving average
                        old_tq = self.neighbor_tq.get(neighbor_id, self.TQ_MAX)
                        self.neighbor_tq[neighbor_id] = (old_tq * 7 + new_tq) // 8
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in TQ calculation: {e}")
    
    def get_route(self, destination: NodeID) -> Optional[RoutingEntry]:
        """Get best route to destination"""
        if destination == self.node.node_id:
            # Route to self
            return RoutingEntry(
                destination=destination,
                next_hop=destination,
                metric=0.0,
                sequence_number=0,
                last_updated=datetime.datetime.utcnow(),
                path=[destination]
            )
        
        if destination in self.routing_table:
            entry = self.routing_table[destination]
            
            # Convert to standard RoutingEntry
            return RoutingEntry(
                destination=destination,
                next_hop=entry.next_hop,
                metric=float(self.TQ_MAX - entry.tq),  # Invert TQ for cost
                sequence_number=entry.last_seqno,
                last_updated=datetime.datetime.fromtimestamp(entry.last_seen),
                path=self._reconstruct_path(destination)
            )
        
        return None
    
    def get_all_routes(self, destination: NodeID) -> List[RoutingEntry]:
        """Get all known routes to destination"""
        routes = []
        
        # Add primary route
        primary = self.get_route(destination)
        if primary:
            routes.append(primary)
        
        # Add alternative routes
        if destination in self.alternative_routes:
            for alt_entry in self.alternative_routes[destination]:
                route = RoutingEntry(
                    destination=destination,
                    next_hop=alt_entry.next_hop,
                    metric=float(self.TQ_MAX - alt_entry.tq),
                    sequence_number=alt_entry.last_seqno,
                    last_updated=datetime.datetime.fromtimestamp(alt_entry.last_seen),
                    path=self._reconstruct_path(destination, alt_entry.next_hop)
                )
                routes.append(route)
        
        return routes
    
    def _reconstruct_path(self, destination: NodeID, 
                         next_hop: Optional[NodeID] = None) -> List[NodeID]:
        """Reconstruct path to destination"""
        # Simple path reconstruction
        # In full implementation, would track actual paths
        path = [self.node.node_id]
        
        if next_hop:
            path.append(next_hop)
        elif destination in self.routing_table:
            path.append(self.routing_table[destination].next_hop)
        
        if destination not in path:
            path.append(destination)
        
        return path
    
    async def find_route(self, destination: NodeID, timeout: float = 5.0) -> Optional[RoutingEntry]:
        """Find route to destination, waiting if necessary"""
        # Check if route exists
        route = self.get_route(destination)
        if route:
            return route
        
        # Create event for this destination
        if destination not in self.pending_routes:
            self.pending_routes[destination] = asyncio.Event()
        
        event = self.pending_routes[destination]
        
        try:
            # Wait for route discovery
            await asyncio.wait_for(event.wait(), timeout)
            
            # Route should now exist
            return self.get_route(destination)
            
        except asyncio.TimeoutError:
            logger.warning(f"Route discovery timeout for {destination.to_base58()[:16]}")
            return None
        finally:
            # Clean up
            self.pending_routes.pop(destination, None)
    
    def get_routing_metrics(self) -> Dict[str, any]:
        """Get routing protocol metrics"""
        total_routes = len(self.routing_table)
        active_neighbors = len(self.neighbor_tq)
        
        # Calculate average TQ
        avg_tq = 0.0
        if self.routing_table:
            avg_tq = sum(e.tq for e in self.routing_table.values()) / total_routes
        
        # Find best and worst routes
        best_route = None
        worst_route = None
        
        for dest, entry in self.routing_table.items():
            if not best_route or entry.tq > best_route[1].tq:
                best_route = (dest, entry)
            if not worst_route or entry.tq < worst_route[1].tq:
                worst_route = (dest, entry)
        
        return {
            'total_routes': total_routes,
            'active_neighbors': active_neighbors,
            'average_tq': avg_tq,
            'best_route': {
                'destination': best_route[0].to_base58() if best_route else None,
                'tq': best_route[1].tq if best_route else 0,
                'hops': best_route[1].hop_count if best_route else 0
            } if best_route else None,
            'worst_route': {
                'destination': worst_route[0].to_base58() if worst_route else None,
                'tq': worst_route[1].tq if worst_route else 0,
                'hops': worst_route[1].hop_count if worst_route else 0
            } if worst_route else None,
            'sequence_number': self.sequence_number
        }