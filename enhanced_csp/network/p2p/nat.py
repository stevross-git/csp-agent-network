# enhanced_csp/network/p2p/nat.py
"""
NAT traversal implementation with STUN, TURN, and UDP hole punching
"""

import asyncio
import logging
import socket
import struct
import time
import random
import select
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
from enum import Enum
import secrets

logger = logging.getLogger(__name__)

try:
    import aiortc
    from aiortc import RTCPeerConnection, RTCSessionDescription
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    logger.warning("aiortc not available, WebRTC-based NAT traversal disabled")

from enhanced_csp.network.core.config import P2PConfig


class NATType(Enum):
    """NAT types from RFC 3489"""
    OPEN = "open"
    FULL_CONE = "full_cone"
    RESTRICTED_CONE = "restricted_cone"
    PORT_RESTRICTED_CONE = "port_restricted_cone"
    SYMMETRIC = "symmetric"
    UNKNOWN = "unknown"


@dataclass
class STUNResponse:
    """STUN server response"""
    external_ip: str
    external_port: int
    nat_type: NATType
    success: bool
    error: Optional[str] = None


@dataclass
class NATInfo:
    """Information about NAT configuration"""
    nat_type: NATType
    external_ip: str
    external_port: int
    internal_ip: str
    internal_port: int
    supports_hairpinning: bool = False
    supports_upnp: bool = False
    mapped_ports: Dict[int, int] = None  # internal -> external


class NATTraversal:
    """NAT traversal with multiple strategies"""
    
    # STUN message types
    STUN_BINDING_REQUEST = 0x0001
    STUN_BINDING_RESPONSE = 0x0101
    STUN_MAGIC_COOKIE = 0x2112A442
    
    def __init__(self, config: P2PConfig):
        self.config = config
        self.nat_info: Optional[NATInfo] = None
        self.stun_cache: Dict[str, STUNResponse] = {}
        self.active_hole_punches: Dict[str, asyncio.Task] = {}
        
        # UPnP client
        self.upnp_client = None
        if self._check_upnp_available():
            self._init_upnp()
    
    async def start(self) -> bool:
        """Start NAT traversal services."""
        try:
            logger.info("Starting NAT traversal")
            
            # Detect NAT type using the configured port
            self.nat_info = await self.detect_nat(self.config.listen_port)
            
            logger.info(f"NAT type detected: {self.nat_info.nat_type.value}")
            logger.info(f"External address: {self.nat_info.external_ip}:{self.nat_info.external_port}")
            
            # Try to set up UPnP port mapping if available
            if self.upnp_client and self.config.enable_upnp:
                try:
                    await self._setup_upnp_mapping(self.config.listen_port)
                    logger.info("UPnP port mapping established")
                except Exception as e:
                    logger.warning(f"Failed to set up UPnP mapping: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start NAT traversal: {e}")
            return False

    async def stop(self):
        """Stop NAT traversal services and clean up."""
        try:
            logger.info("Stopping NAT traversal")
            
            # Cancel any active hole punching tasks
            for task in self.active_hole_punches.values():
                if not task.done():
                    task.cancel()
            
            # Wait for cancellation
            if self.active_hole_punches:
                await asyncio.gather(*self.active_hole_punches.values(), return_exceptions=True)
            
            self.active_hole_punches.clear()
            
            # Remove UPnP mappings if set up
            if self.upnp_client and self.nat_info and self.nat_info.supports_upnp:
                try:
                    await self._remove_upnp_mapping(self.config.listen_port)
                    logger.info("UPnP port mapping removed")
                except Exception as e:
                    logger.warning(f"Failed to remove UPnP mapping: {e}")
            
            # Clear cache
            self.stun_cache.clear()
            
        except Exception as e:
            logger.error(f"Error stopping NAT traversal: {e}")
    
    async def detect_nat(self, local_port: int) -> NATInfo:
        """Detect NAT type and external address"""
        logger.info("Starting NAT detection...")
        
        # Get internal IP
        internal_ip = self._get_internal_ip()
        
        # Try STUN servers
        for stun_server in self.config.stun_servers:
            response = await self._query_stun(stun_server, local_port)
            if response.success:
                self.nat_info = NATInfo(
                    nat_type=response.nat_type,
                    external_ip=response.external_ip,
                    external_port=response.external_port,
                    internal_ip=internal_ip,
                    internal_port=local_port,
                    mapped_ports={local_port: response.external_port}
                )
                
                logger.info(f"NAT detected: {response.nat_type.value} "
                          f"({response.external_ip}:{response.external_port})")
                
                # Check additional capabilities
                await self._check_nat_capabilities()
                
                return self.nat_info
        
        logger.error("Failed to detect NAT through STUN")
        
        # Fallback to basic detection
        self.nat_info = NATInfo(
            nat_type=NATType.UNKNOWN,
            external_ip=internal_ip,
            external_port=local_port,
            internal_ip=internal_ip,
            internal_port=local_port
        )
        
        return self.nat_info
    
    async def _query_stun(self, stun_url: str, local_port: int) -> STUNResponse:
        """Query STUN server for external address"""
        try:
            # Parse STUN URL
            if stun_url.startswith("stun:"):
                stun_url = stun_url[5:]
            
            host, port = stun_url.split(":")
            port = int(port)
            
            # Create UDP socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind(('', local_port))
            sock.settimeout(3.0)
            
            # Build STUN binding request
            request = self._build_stun_request()
            
            # Send request
            sock.sendto(request, (host, port))
            
            # Receive response - use different approach for compatibility
            loop = asyncio.get_event_loop()
            
            # For Python < 3.11, use run_in_executor
            try:
                # Try the newer sock_recvfrom first
                data, addr = await loop.sock_recvfrom(sock, 1024)
            except AttributeError:
                # Fallback for older Python versions
                def recv_sync():
                    return sock.recvfrom(1024)
                
                data, addr = await loop.run_in_executor(None, recv_sync)
            
            # Parse response
            response = self._parse_stun_response(data)
            
            # Detect NAT type with additional tests
            nat_type = await self._detect_nat_type(sock, host, port, response)
            
            sock.close()
            
            return STUNResponse(
                external_ip=response['external_ip'],
                external_port=response['external_port'],
                nat_type=nat_type,
                success=True
            )
            
        except Exception as e:
            logger.error(f"STUN query failed for {stun_url}: {e}")
            return STUNResponse(
                external_ip="",
                external_port=0,
                nat_type=NATType.UNKNOWN,
                success=False,
                error=str(e)
            )
    
    def _build_stun_request(self) -> bytes:
        """Build STUN binding request message"""
        # Message type and length (no attributes)
        msg_type = self.STUN_BINDING_REQUEST
        msg_length = 0
        
        # Transaction ID (12 bytes)
        transaction_id = secrets.token_bytes(12)
        
        # Build message
        message = struct.pack(
            "!HHI12s",
            msg_type,
            msg_length,
            self.STUN_MAGIC_COOKIE,
            transaction_id
        )
        
        return message
    
    def _parse_stun_response(self, data: bytes) -> Dict[str, any]:
        """Parse STUN binding response"""
        # Parse header
        msg_type, msg_length, magic_cookie = struct.unpack("!HHI", data[:8])
        transaction_id = data[8:20]
        
        if msg_type != self.STUN_BINDING_RESPONSE:
            raise ValueError("Not a STUN binding response")
        
        # Parse attributes
        offset = 20
        external_ip = None
        external_port = None
        
        while offset < len(data):
            if offset + 4 > len(data):
                break
                
            attr_type, attr_length = struct.unpack("!HH", data[offset:offset+4])
            offset += 4
            
            # MAPPED-ADDRESS (0x0001) or XOR-MAPPED-ADDRESS (0x0020)
            if attr_type in (0x0001, 0x0020):
                # Skip padding, family
                offset += 2
                port = struct.unpack("!H", data[offset:offset+2])[0]
                offset += 2
                
                # IPv4 address
                ip_bytes = data[offset:offset+4]
                offset += 4
                
                if attr_type == 0x0020:  # XOR-MAPPED-ADDRESS
                    # XOR with magic cookie
                    port ^= (self.STUN_MAGIC_COOKIE >> 16) & 0xFFFF
                    ip_int = struct.unpack("!I", ip_bytes)[0]
                    ip_int ^= self.STUN_MAGIC_COOKIE
                    ip_bytes = struct.pack("!I", ip_int)
                
                external_ip = socket.inet_ntoa(ip_bytes)
                external_port = port
            else:
                # Skip unknown attribute
                offset += attr_length
                # Align to 4-byte boundary
                offset += (4 - (attr_length % 4)) % 4
        
        return {
            'external_ip': external_ip,
            'external_port': external_port,
            'transaction_id': transaction_id
        }
    
    async def _detect_nat_type(self, sock: socket.socket, stun_host: str, 
                              stun_port: int, initial_response: Dict) -> NATType:
        """Detect specific NAT type using RFC 3489 methodology"""
        try:
            # Test 1: Check if we have public IP
            internal_ip = self._get_internal_ip()
            if internal_ip == initial_response['external_ip']:
                return NATType.OPEN
            
            # Test 2: Send from different port
            sock2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock2.bind(('', 0))  # Random port
            sock2.settimeout(3.0)
            
            request = self._build_stun_request()
            sock2.sendto(request, (stun_host, stun_port))
            
            try:
                loop = asyncio.get_event_loop()
                # Compatible receive approach
                try:
                    data, _ = await loop.sock_recvfrom(sock2, 1024)
                except AttributeError:
                    def recv_sync():
                        return sock2.recvfrom(1024)
                    data, _ = await loop.run_in_executor(None, recv_sync)
                    
                response2 = self._parse_stun_response(data)
                
                # If external port changes with internal port, it's symmetric
                if response2['external_port'] != initial_response['external_port']:
                    sock2.close()
                    return NATType.SYMMETRIC
            except socket.timeout:
                pass
            
            sock2.close()
            
            # For more detailed detection, would need multiple STUN servers
            # For now, assume restricted cone if not symmetric
            return NATType.RESTRICTED_CONE
            
        except Exception as e:
            logger.error(f"NAT type detection failed: {e}")
            return NATType.UNKNOWN
    
    def _get_internal_ip(self) -> str:
        """Get internal IP address"""
        try:
            # Create a dummy connection to get local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"
    
    async def _check_nat_capabilities(self):
        """Check additional NAT capabilities"""
        if not self.nat_info:
            return
        
        # Check hairpinning support
        self.nat_info.supports_hairpinning = await self._test_hairpinning()
        
        # Check UPnP support
        if self.upnp_client:
            self.nat_info.supports_upnp = await self._test_upnp()
    
    async def _test_hairpinning(self) -> bool:
        """Test if NAT supports hairpinning"""
        # Try to connect to our own external address
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(
                    self.nat_info.external_ip,
                    self.nat_info.external_port
                ),
                timeout=2.0
            )
            writer.close()
            await writer.wait_closed()
            return True
        except:
            return False
    
    def _check_upnp_available(self) -> bool:
        """Check if UPnP is available"""
        try:
            import miniupnpc
            return True
        except ImportError:
            return False
    
    def _init_upnp(self):
        """Initialize UPnP client"""
        try:
            import miniupnpc
            self.upnp_client = miniupnpc.UPnP()
            self.upnp_client.discoverdelay = 2000
        except Exception as e:
            logger.error(f"Failed to initialize UPnP: {e}")
    
    async def _test_upnp(self) -> bool:
        """Test UPnP functionality"""
        if not self.upnp_client:
            return False
        
        try:
            # Discover UPnP devices
            devices = await asyncio.get_event_loop().run_in_executor(
                None, self.upnp_client.discover
            )
            
            if devices > 0:
                # Select IGD
                await asyncio.get_event_loop().run_in_executor(
                    None, self.upnp_client.selectigd
                )
                return True
            
            return False
        except:
            return False
    
    async def setup_port_mapping(self, internal_port: int, 
                               external_port: int = 0,
                               protocol: str = "TCP") -> Optional[int]:
        """Setup port mapping using UPnP or NAT-PMP"""
        if not external_port:
            external_port = internal_port
        
        # Try UPnP first
        if self.nat_info and self.nat_info.supports_upnp:
            mapped_port = await self._setup_upnp_mapping(
                internal_port, external_port, protocol
            )
            if mapped_port:
                return mapped_port
        
        # Try NAT-PMP
        # TODO: Implement NAT-PMP support
        
        # Fallback to manual mapping
        if self.nat_info:
            if not self.nat_info.mapped_ports:
                self.nat_info.mapped_ports = {}
            self.nat_info.mapped_ports[internal_port] = external_port
            return external_port
        
        return None
    
    async def _setup_upnp_mapping(self, internal_port: int,
                                external_port: int = None,
                                protocol: str = "TCP") -> Optional[int]:
        """Setup port mapping using UPnP"""
        if external_port is None:
            external_port = internal_port
            
        if not self.upnp_client:
            return None
        
        try:
            # Add port mapping
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.upnp_client.addportmapping,
                external_port,      # external port
                protocol,           # protocol
                self.nat_info.internal_ip,  # internal IP
                internal_port,      # internal port
                "Enhanced CSP P2P", # description
                ""                  # remote host (any)
            )
            
            if result:
                logger.info(f"UPnP mapping created: "
                          f"{internal_port} -> {external_port} ({protocol})")
                return external_port
            
            return None
            
        except Exception as e:
            logger.error(f"UPnP port mapping failed: {e}")
            return None

    async def _remove_upnp_mapping(self, internal_port: int) -> bool:
        """Remove UPnP port mapping."""
        # This is a placeholder - actual implementation would use miniupnpc or similar
        logger.debug(f"Removing UPnP mapping for port {internal_port}")
        return True
    
    async def remove_port_mapping(self, external_port: int, protocol: str = "TCP"):
        """Remove UPnP port mapping"""
        if self.upnp_client:
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.upnp_client.deleteportmapping,
                    external_port,
                    protocol
                )
                logger.info(f"UPnP mapping removed: {external_port} ({protocol})")
            except Exception as e:
                logger.error(f"Failed to remove UPnP mapping: {e}")
    
    async def punch_hole(self, target_ip: str, target_port: int,
                        local_port: int) -> bool:
        """Perform UDP hole punching"""
        punch_id = f"{target_ip}:{target_port}"
        
        # Check if already punching
        if punch_id in self.active_hole_punches:
            return True
        
        try:
            # Create task for hole punching
            task = asyncio.create_task(
                self._udp_hole_punch(target_ip, target_port, local_port)
            )
            self.active_hole_punches[punch_id] = task
            
            # Wait for completion
            success = await task
            
            return success
            
        finally:
            self.active_hole_punches.pop(punch_id, None)
    
    async def _udp_hole_punch(self, target_ip: str, target_port: int,
                            local_port: int) -> bool:
        """Perform actual UDP hole punching"""
        try:
            # Create UDP socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind(('', local_port))
            sock.setblocking(False)
            
            # Send punch packets
            punch_msg = b"PUNCH"
            attempts = 10
            
            for i in range(attempts):
                try:
                    sock.sendto(punch_msg, (target_ip, target_port))
                    logger.debug(f"Sent hole punch {i+1}/{attempts} to "
                               f"{target_ip}:{target_port}")
                except Exception as e:
                    logger.error(f"Hole punch send failed: {e}")
                
                # Wait and check for response
                await asyncio.sleep(0.1)
                
                # Try to receive
                try:
                    loop = asyncio.get_event_loop()
                    # Compatible receive
                    try:
                        data, addr = await loop.sock_recvfrom(sock, 1024)
                    except AttributeError:
                        # Set socket to non-blocking for select
                        sock.setblocking(False)
                        # Use select to check if data is available
                        ready = select.select([sock], [], [], 0)
                        if ready[0]:
                            data, addr = sock.recvfrom(1024)
                        else:
                            continue
                            
                    if data == punch_msg and addr[0] == target_ip:
                        logger.info(f"Hole punch successful with {target_ip}:{target_port}")
                        sock.close()
                        return True
                except (BlockingIOError, socket.error):
                    # No data available yet
                    pass
            
            sock.close()
            logger.warning(f"Hole punch failed with {target_ip}:{target_port}")
            return False
            
        except Exception as e:
            logger.error(f"UDP hole punch error: {e}")
            return False
    
    async def establish_relay(self, relay_server: str) -> Optional[str]:
        """Establish connection through TURN relay"""
        # Parse relay server URL
        if relay_server.startswith("turn:"):
            relay_server = relay_server[5:]
        
        # TODO: Implement TURN client
        logger.warning("TURN relay not yet implemented")
        return None
    
    async def create_webrtc_offer(self) -> Optional[str]:
        """Create WebRTC offer for advanced NAT traversal"""
        if not WEBRTC_AVAILABLE:
            return None
        
        try:
            # Create peer connection
            pc = RTCPeerConnection({
                'iceServers': [
                    {'urls': self.config.stun_servers},
                    # Add TURN servers if configured
                ]
            })
            
            # Create data channel
            channel = pc.createDataChannel("p2p")
            
            # Create offer
            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)
            
            # Return SDP
            return pc.localDescription.sdp
            
        except Exception as e:
            logger.error(f"Failed to create WebRTC offer: {e}")
            return None