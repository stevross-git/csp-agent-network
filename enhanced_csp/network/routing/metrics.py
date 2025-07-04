# enhanced_csp/network/routing/metrics.py
"""
Network metrics collection for adaptive routing
Measures RTT, bandwidth, packet loss, and jitter
"""
from typing import TYPE_CHECKING
import asyncio
import logging
import time
import statistics
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import struct
import random

from ..core.types import NodeID
if TYPE_CHECKING:
    from ..core.node import NetworkNode


logger = logging.getLogger(__name__)


@dataclass
class ProbePacket:
    """Network probe packet for measurements"""
    probe_id: int
    sequence: int
    timestamp: float
    probe_type: str  # 'rtt', 'bandwidth', 'jitter'
    payload_size: int
    path: List[NodeID]
    
    def to_bytes(self) -> bytes:
        """Serialize probe packet"""
        # Header: [probe_id(4)][seq(4)][timestamp(8)][type(1)][size(2)][path_len(1)]
        header = struct.pack(
            "!IIdBHB",
            self.probe_id,
            self.sequence,
            self.timestamp,
            ord(self.probe_type[0]),  # First char as type
            self.payload_size,
            len(self.path)
        )
        
        # Add path (node IDs)
        path_data = b''.join(node.raw_id[:8] for node in self.path)  # 8 bytes per node
        
        # Add payload
        payload = b'\x00' * max(0, self.payload_size - len(header) - len(path_data))
        
        return header + path_data + payload
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'ProbePacket':
        """Deserialize probe packet"""
        if len(data) < 20:
            raise ValueError("Invalid probe packet size")
        
        # Parse header
        probe_id, seq, timestamp, ptype, size, path_len = struct.unpack(
            "!IIdBHB", data[:20]
        )
        
        # Parse path
        path = []
        offset = 20
        for i in range(path_len):
            if offset + 8 <= len(data):
                node_id = NodeID(raw_id=data[offset:offset+8], public_key=None)
                path.append(node_id)
                offset += 8
        
        # Determine probe type
        type_map = {ord('r'): 'rtt', ord('b'): 'bandwidth', ord('j'): 'jitter'}
        probe_type = type_map.get(ptype, 'rtt')
        
        return cls(
            probe_id=probe_id,
            sequence=seq,
            timestamp=timestamp,
            probe_type=probe_type,
            payload_size=size,
            path=path
        )


@dataclass
class MeasurementSession:
    """Active measurement session"""
    session_id: int
    target: NodeID
    next_hop: NodeID
    probe_count: int = 10
    probes_sent: int = 0
    probes_received: int = 0
    rtt_samples: List[float] = field(default_factory=list)
    bandwidth_samples: List[float] = field(default_factory=list)
    send_times: Dict[int, float] = field(default_factory=dict)
    created: float = field(default_factory=time.time)
    completed: bool = False


class MetricsCollector:
    """Collects network metrics through active probing"""
    
    # Probe parameters
    RTT_PROBE_SIZE = 64  # bytes
    BANDWIDTH_PROBE_SIZE = 1400  # bytes
    PROBE_INTERVAL = 0.1  # seconds
    PROBE_TIMEOUT = 5.0  # seconds
    
    def __init__(self, node: 'NetworkNode'):
        self.node = node
        self.session_counter = 0
        self.active_sessions: Dict[int, MeasurementSession] = {}
        self.probe_handlers: Dict[int, asyncio.Future] = {}
        
        # Background tasks
        self._tasks: List[asyncio.Task] = []
    
    async def start(self):
        """Start metrics collector"""
        logger.info("Starting metrics collector")
        
        # Register probe handlers
        self.node.on_event('probe_packet', self.handle_probe)
        self.node.on_event('probe_response', self.handle_probe_response)
        
        # Start cleanup task
        self._tasks.append(
            asyncio.create_task(self._session_cleanup_loop())
        )
    
    async def stop(self):
        """Stop metrics collector"""
        for task in self._tasks:
            task.cancel()
        
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
    
    async def measure_path(self, path: List[NodeID], 
                          next_hop: NodeID) -> Optional[Dict[str, float]]:
        """Measure metrics for a path"""
        if len(path) < 2:
            return None
        
        target = path[-1]  # Destination
        
        # Create measurement session
        session = self._create_session(target, next_hop)
        
        try:
            # Send RTT probes
            await self._send_rtt_probes(session, path)
            
            # Send bandwidth probes
            await self._send_bandwidth_probes(session, path)
            
            # Wait for responses
            await asyncio.sleep(self.PROBE_TIMEOUT)
            
            # Calculate metrics
            metrics = self._calculate_metrics(session)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Path measurement failed: {e}")
            return None
        finally:
            # Clean up session
            self.active_sessions.pop(session.session_id, None)
    
    def _create_session(self, target: NodeID, next_hop: NodeID) -> MeasurementSession:
        """Create new measurement session"""
        self.session_counter += 1
        session = MeasurementSession(
            session_id=self.session_counter,
            target=target,
            next_hop=next_hop
        )
        self.active_sessions[session.session_id] = session
        return session
    
    async def _send_rtt_probes(self, session: MeasurementSession, 
                              path: List[NodeID]):
        """Send RTT measurement probes"""
        for i in range(session.probe_count):
            probe = ProbePacket(
                probe_id=session.session_id,
                sequence=i,
                timestamp=time.time(),
                probe_type='rtt',
                payload_size=self.RTT_PROBE_SIZE,
                path=path
            )
            
            # Record send time
            session.send_times[i] = probe.timestamp
            session.probes_sent += 1
            
            # Send probe
            await self._send_probe(probe, session.next_hop)
            
            # Small delay between probes
            await asyncio.sleep(self.PROBE_INTERVAL)
    
    async def _send_bandwidth_probes(self, session: MeasurementSession,
                                   path: List[NodeID]):
        """Send bandwidth measurement probes"""
        # Send burst of large probes
        burst_size = 5
        start_time = time.time()
        
        for i in range(burst_size):
            probe = ProbePacket(
                probe_id=session.session_id,
                sequence=session.probe_count + i,
                timestamp=time.time(),
                probe_type='bandwidth',
                payload_size=self.BANDWIDTH_PROBE_SIZE,
                path=path
            )
            
            session.send_times[probe.sequence] = probe.timestamp
            session.probes_sent += 1
            
            # Send without delay for bandwidth test
            await self._send_probe(probe, session.next_hop)
        
        # Record burst duration
        session.bandwidth_burst_duration = time.time() - start_time
    
    async def _send_probe(self, probe: ProbePacket, next_hop: NodeID):
        """Send probe packet"""
        message = {
            'type': 'probe_packet',
            'data': probe.to_bytes().hex()
        }
        
        await self.node.send_message(next_hop, message)
    
    async def handle_probe(self, data: Dict):
        """Handle incoming probe packet"""
        try:
            probe_bytes = bytes.fromhex(data['data'])
            probe = ProbePacket.from_bytes(probe_bytes)
            sender = data.get('sender_id')
            
            # Check if probe is for us
            if probe.path[-1] == self.node.node_id:
                # We're the target - send response
                await self._send_probe_response(probe, sender)
            else:
                # Forward probe along path
                await self._forward_probe(probe, sender)
                
        except Exception as e:
            logger.error(f"Error handling probe: {e}")
    
    async def _send_probe_response(self, probe: ProbePacket, sender: NodeID):
        """Send probe response back to origin"""
        response = {
            'type': 'probe_response',
            'probe_id': probe.probe_id,
            'sequence': probe.sequence,
            'probe_type': probe.probe_type,
            'sent_timestamp': probe.timestamp,
            'received_timestamp': time.time()
        }
        
        await self.node.send_message(sender, response)
    
    async def _forward_probe(self, probe: ProbePacket, sender: NodeID):
        """Forward probe to next hop in path"""
        # Find our position in path
        try:
            our_index = probe.path.index(self.node.node_id)
            if our_index < len(probe.path) - 1:
                # Forward to next hop
                next_hop = probe.path[our_index + 1]
                await self._send_probe(probe, next_hop)
        except ValueError:
            logger.warning("Received probe not meant for us")
    
    async def handle_probe_response(self, data: Dict):
        """Handle probe response"""
        try:
            probe_id = data['probe_id']
            sequence = data['sequence']
            sent_time = data['sent_timestamp']
            received_time = data['received_timestamp']
            probe_type = data['probe_type']
            
            # Find session
            if probe_id not in self.active_sessions:
                return
            
            session = self.active_sessions[probe_id]
            current_time = time.time()
            
            # Calculate RTT
            rtt = current_time - sent_time
            
            # Store samples based on probe type
            if probe_type == 'rtt':
                session.rtt_samples.append(rtt * 1000)  # Convert to ms
            elif probe_type == 'bandwidth':
                # For bandwidth, we care about the burst completion
                session.bandwidth_samples.append(received_time)
            
            session.probes_received += 1
            
            # Check if future is waiting
            if probe_id in self.probe_handlers:
                self.probe_handlers[probe_id].set_result(data)
                
        except Exception as e:
            logger.error(f"Error handling probe response: {e}")
    
    def _calculate_metrics(self, session: MeasurementSession) -> Dict[str, float]:
        """Calculate metrics from session data"""
        metrics = {
            'rtt': 0.0,
            'bandwidth': 0.0,
            'loss': 0.0,
            'jitter': 0.0
        }
        
        # Calculate packet loss
        if session.probes_sent > 0:
            metrics['loss'] = 1.0 - (session.probes_received / session.probes_sent)
        
        # Calculate RTT statistics
        if session.rtt_samples:
            metrics['rtt'] = statistics.median(session.rtt_samples)
            
            # Calculate jitter (variation in RTT)
            if len(session.rtt_samples) > 1:
                metrics['jitter'] = statistics.stdev(session.rtt_samples)
        
        # Estimate bandwidth from burst test
        if len(session.bandwidth_samples) >= 2:
            # Sort receive times
            session.bandwidth_samples.sort()
            
            # Calculate receive rate
            total_bytes = self.BANDWIDTH_PROBE_SIZE * len(session.bandwidth_samples)
            duration = session.bandwidth_samples[-1] - session.bandwidth_samples[0]
            
            if duration > 0:
                # Convert to Mbps
                metrics['bandwidth'] = (total_bytes * 8) / (duration * 1_000_000)
        
        return metrics
    
    async def _session_cleanup_loop(self):
        """Clean up old measurement sessions"""
        while True:
            try:
                await asyncio.sleep(60)  # Every minute
                
                current_time = time.time()
                expired = []
                
                for session_id, session in self.active_sessions.items():
                    if current_time - session.created > 300:  # 5 minutes
                        expired.append(session_id)
                
                for session_id in expired:
                    self.active_sessions.pop(session_id, None)
                    self.probe_handlers.pop(session_id, None)
                
                if expired:
                    logger.debug(f"Cleaned up {len(expired)} measurement sessions")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in session cleanup: {e}")
    
    async def continuous_monitoring(self, target: NodeID, next_hop: NodeID,
                                  interval: float = 30.0) -> asyncio.Task:
        """Start continuous monitoring of a path"""
        async def monitor():
            while True:
                try:
                    metrics = await self.measure_path([self.node.node_id, target], next_hop)
                    if metrics:
                        await self.node.emit_event('metrics_updated', {
                            'target': target,
                            'next_hop': next_hop,
                            'metrics': metrics
                        })
                    
                    await asyncio.sleep(interval)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    await asyncio.sleep(interval)
        
        return asyncio.create_task(monitor())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collector statistics"""
        total_probes = sum(s.probes_sent for s in self.active_sessions.values())
        total_responses = sum(s.probes_received for s in self.active_sessions.values())
        
        return {
            'active_sessions': len(self.active_sessions),
            'total_probes_sent': total_probes,
            'total_responses': total_responses,
            'response_rate': total_responses / total_probes if total_probes > 0 else 0.0
        }