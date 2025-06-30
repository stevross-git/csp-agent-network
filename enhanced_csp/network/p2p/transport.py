# enhanced_csp/network/p2p/transport.py
"""
QUIC transport with TCP fallback implementation
Provides secure, multiplexed connections with 0-RTT support
"""

import asyncio
import logging
import ssl
import time
import datetime
import ipaddress
import tempfile
import random
import os
from pathlib import Path

logger = logging.getLogger(__name__)
from typing import Optional, Dict, List, Callable, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import struct
import json

try:
    from aioquic.asyncio import connect, serve
    from aioquic.asyncio.protocol import QuicConnectionProtocol
    from aioquic.quic.configuration import QuicConfiguration
    from aioquic.quic.events import QuicEvent, StreamDataReceived, ConnectionTerminated
    QUIC_AVAILABLE = True
except ImportError:
    QUIC_AVAILABLE = False
    logger.warning("aioquic not available, QUIC transport disabled")

from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.backends import default_backend

from ..core.types import (
    Transport, Connection, NodeID, PeerInfo, NetworkProtocol
)
from ..core.node import NetworkNode


class TransportState(Enum):
    """Transport connection state"""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    CLOSING = "closing"
    CLOSED = "closed"


@dataclass
class TransportStats:
    """Transport layer statistics"""
    bytes_sent: int = 0
    bytes_received: int = 0
    packets_sent: int = 0
    packets_received: int = 0
    rtt_ms: float = 0.0
    packet_loss: float = 0.0
    connections_established: int = 0
    connections_failed: int = 0


class MultiProtocolTransport(Transport):
    """Transport supporting QUIC with TCP fallback"""
    
    def __init__(self, node: NetworkNode):
        self.node = node
        self.config = node.config
        
        # Active connections
        self.connections: Dict[str, Connection] = {}
        
        # Listeners
        self.quic_server = None
        self.tcp_server = None
        
        # Statistics
        self.stats = TransportStats()
        
        # Connection handlers
        self.on_connection: Optional[Callable] = None
        
        # TLS configuration
        self._tls_cert = None
        self._tls_key = None
        self._tls_context = None
        self._cert_tempfile = None
        self._key_tempfile = None
        
        # Initialize TLS
        self._init_tls()
    
    def _init_tls(self):
        """Initialize TLS certificates and context"""
        # Generate self-signed certificate for node
        self._tls_key = self.node.private_key
        
        # Create certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, self.node.node_id.to_base58()),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Enhanced CSP Network"),
        ])
        
        cert_builder = x509.CertificateBuilder()
        cert_builder = cert_builder.subject_name(subject)
        cert_builder = cert_builder.issuer_name(issuer)
        cert_builder = cert_builder.public_key(self.node.public_key)
        cert_builder = cert_builder.serial_number(x509.random_serial_number())
        cert_builder = cert_builder.not_valid_before(datetime.datetime.utcnow())
        cert_builder = cert_builder.not_valid_after(
            datetime.datetime.utcnow() + datetime.timedelta(days=365)
        )
        
        # Add extensions
        cert_builder = cert_builder.add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
            ]),
            critical=False,
        )
        
        # Sign certificate
        self._tls_cert = cert_builder.sign(
            self._tls_key, hashes.SHA256(), default_backend()
        )
        
        # Create SSL context
        self._tls_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        self._tls_context.check_hostname = False
        self._tls_context.verify_mode = ssl.CERT_NONE  # We verify node IDs instead
        
        # Load cert and key
        cert_pem = self._tls_cert.public_bytes(serialization.Encoding.PEM)
        key_pem = self._tls_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )

        cert_tmp = tempfile.NamedTemporaryFile(delete=False)
        key_tmp = tempfile.NamedTemporaryFile(delete=False)
        cert_tmp.write(cert_pem)
        key_tmp.write(key_pem)
        cert_tmp.close()
        key_tmp.close()
        self._cert_tempfile = cert_tmp.name
        self._key_tempfile = key_tmp.name

        self._tls_context.load_cert_chain(self._cert_tempfile, self._key_tempfile)
    
    async def start(self):
        """Start transport listeners"""
        logger.info("Starting multi-protocol transport...")
        
        # Start QUIC listener
        if QUIC_AVAILABLE and self.config.p2p.enable_quic:
            await self._start_quic_server()
        
        # Start TCP listener
        await self._start_tcp_server()
        
        logger.info("Transport started")
    
    async def stop(self):
        """Stop transport"""
        logger.info("Stopping transport...")
        
        # Close all connections
        for conn in list(self.connections.values()):
            await conn.close()
        
        # Stop servers
        if self.quic_server:
            self.quic_server.close()
        
        if self.tcp_server:
            self.tcp_server.close()
            await self.tcp_server.wait_closed()

        if self._cert_tempfile:
            try:
                os.unlink(self._cert_tempfile)
            except OSError:
                pass
            self._cert_tempfile = None

        if self._key_tempfile:
            try:
                os.unlink(self._key_tempfile)
            except OSError:
                pass
            self._key_tempfile = None

        logger.info("Transport stopped")
    
    async def _start_quic_server(self):
        """Start QUIC server"""
        try:
            # Configure QUIC
            configuration = QuicConfiguration(
                is_client=False,
                max_datagram_size=65536,
            )
            
            # Load certificate
            configuration.load_cert_chain(
                certfile=self._cert_tempfile,
                keyfile=self._key_tempfile
            )
            
            # Configure ALPN
            configuration.alpn_protocols = ["enhanced-csp/1.0"]
            
            # Start server
            self.quic_server = await serve(
                self.config.p2p.listen_address,
                self.config.p2p.listen_port,
                configuration=configuration,
                create_protocol=self._create_quic_protocol,
            )
            
            logger.info(f"QUIC server listening on "
                       f"{self.config.p2p.listen_address}:{self.config.p2p.listen_port}")
            
        except Exception as e:
            logger.error(f"Failed to start QUIC server: {e}")
    
    async def _start_tcp_server(self):
        """Start TCP server with TLS"""
        try:
            self.tcp_server = await asyncio.start_server(
                self._handle_tcp_connection,
                self.config.p2p.listen_address,
                self.config.p2p.listen_port + 1,  # TCP on next port
                ssl=self._tls_context if self.config.enable_tls else None
            )
            
            logger.info(f"TCP server listening on "
                       f"{self.config.p2p.listen_address}:{self.config.p2p.listen_port + 1}")
            
        except Exception as e:
            logger.error(f"Failed to start TCP server: {e}")
    
    def _create_quic_protocol(self) -> QuicConnectionProtocol:
        """Create QUIC protocol handler"""
        return EnhancedCSPQuicProtocol(self)
    
    async def _handle_tcp_connection(self, reader: asyncio.StreamReader,
                                   writer: asyncio.StreamWriter):
        """Handle incoming TCP connection"""
        peer_addr = writer.get_extra_info('peername')
        logger.info(f"New TCP connection from {peer_addr}")
        
        try:
            # Create connection wrapper
            conn = TCPConnection(reader, writer, self)
            
            # Perform handshake
            peer_info = await conn.handshake()
            if not peer_info:
                await conn.close()
                return
            
            # Store connection
            conn_id = f"tcp:{peer_info.node_id.to_base58()}"
            self.connections[conn_id] = conn
            
            # Notify handler
            if self.on_connection:
                await self.on_connection(conn)
            
            # Start receiving
            await conn._receive_loop()
            
        except Exception as e:
            logger.error(f"TCP connection error: {e}")
        finally:
            writer.close()
            await writer.wait_closed()
    
    async def connect(self, address: str) -> Optional[Connection]:
        """Connect to a peer"""
        try:
            # Parse multiaddr
            protocol, host, port, peer_id = self._parse_multiaddr(address)
            
            # Try QUIC first
            if QUIC_AVAILABLE and protocol in ("quic", "any"):
                conn = await self._connect_quic(host, port, peer_id)
                if conn:
                    return conn
            
            # Fallback to TCP
            if protocol in ("tcp", "any"):
                conn = await self._connect_tcp(host, port, peer_id)
                if conn:
                    return conn
            
            logger.error(f"Failed to connect to {address}")
            return None
            
        except Exception as e:
            logger.error(f"Connection error: {e}")
            self.stats.connections_failed += 1
            return None
    
    async def _connect_quic(self, host: str, port: int, 
                          peer_id: str) -> Optional[Connection]:
        """Connect using QUIC"""
        try:
            configuration = QuicConfiguration(is_client=True)
            configuration.alpn_protocols = ["enhanced-csp/1.0"]
            configuration.verify_mode = ssl.CERT_NONE  # We verify node IDs
            
            async with connect(
                host, port,
                configuration=configuration,
                create_protocol=lambda: EnhancedCSPQuicProtocol(self, is_client=True)
            ) as protocol:
                # Create connection wrapper
                conn = QUICConnection(protocol, self)
                
                # Perform handshake
                peer_info = await conn.handshake(peer_id)
                if not peer_info:
                    return None
                
                # Store connection
                conn_id = f"quic:{peer_info.node_id.to_base58()}"
                self.connections[conn_id] = conn
                
                self.stats.connections_established += 1
                logger.info(f"QUIC connection established to {peer_id[:16]}...")
                
                return conn
                
        except Exception as e:
            logger.error(f"QUIC connection failed: {e}")
            return None
    
    async def _connect_tcp(self, host: str, port: int,
                         peer_id: str) -> Optional[Connection]:
        """Connect using TCP"""
        try:
            # Connect with timeout
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(
                    host, port,
                    ssl=self._tls_context if self.config.enable_tls else None
                ),
                timeout=self.config.p2p.connection_timeout
            )
            
            # Create connection wrapper
            conn = TCPConnection(reader, writer, self)
            
            # Perform handshake
            peer_info = await conn.handshake(peer_id)
            if not peer_info:
                await conn.close()
                return None
            
            # Store connection
            conn_id = f"tcp:{peer_info.node_id.to_base58()}"
            self.connections[conn_id] = conn
            
            self.stats.connections_established += 1
            logger.info(f"TCP connection established to {peer_id[:16]}...")
            
            # Start receive loop
            asyncio.create_task(conn._receive_loop())
            
            return conn
            
        except Exception as e:
            logger.error(f"TCP connection failed: {e}")
            return None
    
    def _parse_multiaddr(self, address: str) -> Tuple[str, str, int, str]:
        """Parse multiaddr format"""
        # Format: /ip4/1.2.3.4/tcp/4001/p2p/QmNodeID
        # or: /ip4/1.2.3.4/udp/4001/quic/p2p/QmNodeID
        
        parts = address.strip('/').split('/')
        
        if len(parts) < 6:
            raise ValueError(f"Invalid multiaddr: {address}")
        
        ip_type = parts[0]  # ip4 or ip6
        host = parts[1]
        transport = parts[2]  # tcp or udp
        port = int(parts[3])
        
        # Determine protocol
        if "quic" in parts:
            protocol = "quic"
            peer_id = parts[6]  # After quic/p2p/
        else:
            protocol = "tcp"
            peer_id = parts[5]  # After p2p/
        
        return protocol, host, port, peer_id
    
    async def listen(self, address: str) -> None:
        """Listen on additional address"""
        # Already handled by start()
        pass
    
    def get_stats(self) -> TransportStats:
        """Get transport statistics"""
        return self.stats


class EnhancedCSPQuicProtocol(QuicConnectionProtocol):
    """QUIC protocol handler for Enhanced CSP"""
    
    def __init__(self, transport: MultiProtocolTransport, is_client: bool = False):
        super().__init__()
        self.transport = transport
        self.is_client = is_client
        self.streams: Dict[int, QUICStream] = {}
    
    def quic_event_received(self, event: QuicEvent) -> None:
        """Handle QUIC events"""
        if isinstance(event, StreamDataReceived):
            stream_id = event.stream_id
            
            if stream_id not in self.streams:
                self.streams[stream_id] = QUICStream(stream_id, self)
            
            self.streams[stream_id].data_received(event.data, event.end_stream)
            
        elif isinstance(event, ConnectionTerminated):
            logger.info("QUIC connection terminated")
            # Clean up streams
            for stream in self.streams.values():
                stream.close()


class QUICStream:
    """Individual QUIC stream"""
    
    def __init__(self, stream_id: int, protocol: EnhancedCSPQuicProtocol):
        self.stream_id = stream_id
        self.protocol = protocol
        self.buffer = bytearray()
        self.message_handler = None
    
    def data_received(self, data: bytes, end_stream: bool):
        """Handle received data"""
        self.buffer.extend(data)
        
        # Try to parse messages
        while len(self.buffer) >= 4:
            # Read message length
            msg_len = struct.unpack("!I", self.buffer[:4])[0]
            
            if len(self.buffer) >= 4 + msg_len:
                # Extract message
                msg_data = self.buffer[4:4+msg_len]
                self.buffer = self.buffer[4+msg_len:]
                
                # Handle message
                if self.message_handler:
                    asyncio.create_task(self.message_handler(msg_data))
    
    def send(self, data: bytes):
        """Send data on stream"""
        # Prepend length
        msg = struct.pack("!I", len(data)) + data
        self.protocol._quic.send_stream_data(self.stream_id, msg)
    
    def close(self):
        """Close stream"""
        self.protocol._quic.send_stream_data(self.stream_id, b'', end_stream=True)


class BaseConnection(Connection):
    """Base connection implementation"""
    
    def __init__(self, transport: MultiProtocolTransport):
        self.transport = transport
        self.state = TransportState.CONNECTING
        self.remote_peer: Optional[PeerInfo] = None
        self.stats = {
            'bytes_sent': 0,
            'bytes_received': 0,
            'messages_sent': 0,
            'messages_received': 0,
            'last_activity': time.time()
        }
        self.message_handler: Optional[Callable] = None
    
    async def handshake(self, expected_peer_id: Optional[str] = None) -> Optional[PeerInfo]:
        """Perform connection handshake"""
        try:
            await asyncio.sleep(random.uniform(0, 0.2))

            handshake_msg = {
                'type': 'handshake',
                'node_id': self.transport.node.node_id.to_base58(),
                'version': '1.0',
                'capabilities': ['mesh', 'dns', 'routing', 'quantum'],
                'timestamp': time.time()
            }

            await self.send(json.dumps(handshake_msg).encode())

            data = await asyncio.wait_for(self.receive(), timeout=5.0)
            peer_msg = json.loads(data.decode())
            
            if peer_msg.get('type') != 'handshake':
                logger.error("Invalid handshake response")
                return None
            
            peer_id = peer_msg.get('node_id')
            
            # Verify peer ID if expected
            if expected_peer_id and peer_id != expected_peer_id:
                logger.error(f"Peer ID mismatch: expected {expected_peer_id}, "
                           f"got {peer_id}")
                return None
            
            # Create peer info
            # TODO: Create proper PeerInfo from handshake
            self.remote_peer = peer_msg  # Placeholder
            
            self.state = TransportState.CONNECTED
            return self.remote_peer
            
        except Exception as e:
            logger.error(f"Handshake failed: {e}")
            return None
    
    def update_stats(self, sent: int = 0, received: int = 0):
        """Update connection statistics"""
        self.stats['bytes_sent'] += sent
        self.stats['bytes_received'] += received
        self.stats['last_activity'] = time.time()
        
        # Update transport stats
        self.transport.stats.bytes_sent += sent
        self.transport.stats.bytes_received += received


class QUICConnection(BaseConnection):
    """QUIC connection implementation"""
    
    def __init__(self, protocol: EnhancedCSPQuicProtocol,
                 transport: MultiProtocolTransport):
        super().__init__(transport)
        self.protocol = protocol
        self.stream: Optional[QUICStream] = None
        self._recv_queue: asyncio.Queue[bytes] = asyncio.Queue()
        
        # Create primary stream
        self._create_stream()
    
    def _create_stream(self):
        """Create a new QUIC stream"""
        stream_id = self.protocol._quic.get_next_available_stream_id()
        self.stream = QUICStream(stream_id, self.protocol)
        self.protocol.streams[stream_id] = self.stream
        
        # Set message handler
        self.stream.message_handler = self._handle_message
    
    async def send(self, data: bytes) -> None:
        """Send data over QUIC"""
        if self.state != TransportState.CONNECTED:
            raise ConnectionError("Not connected")
        
        self.stream.send(data)
        self.update_stats(sent=len(data))
        self.stats['messages_sent'] += 1
    
    async def receive(self) -> bytes:
        """Receive data from QUIC."""
        return await self._recv_queue.get()
    
    async def _handle_message(self, data: bytes):
        """Handle received message"""
        self.update_stats(received=len(data))
        self.stats['messages_received'] += 1

        await self._recv_queue.put(data)
        if self.message_handler:
            await self.message_handler(data)
    
    async def close(self) -> None:
        """Close QUIC connection"""
        self.state = TransportState.CLOSING
        
        if self.stream:
            self.stream.close()
        
        self.state = TransportState.CLOSED


class TCPConnection(BaseConnection):
    """TCP connection implementation"""
    
    def __init__(self, reader: asyncio.StreamReader,
                 writer: asyncio.StreamWriter,
                 transport: MultiProtocolTransport):
        super().__init__(transport)
        self.reader = reader
        self.writer = writer
    
    async def send(self, data: bytes) -> None:
        """Send data over TCP"""
        if self.state != TransportState.CONNECTED:
            raise ConnectionError("Not connected")
        
        # Frame: [4 bytes length][data]
        frame = struct.pack("!I", len(data)) + data
        
        self.writer.write(frame)
        await self.writer.drain()
        
        self.update_stats(sent=len(frame))
        self.stats['messages_sent'] += 1
    
    async def receive(self) -> bytes:
        """Receive data from TCP"""
        # Read length header
        header = await self.reader.readexactly(4)
        msg_len = struct.unpack("!I", header)[0]
        
        # Read message
        data = await self.reader.readexactly(msg_len)
        
        self.update_stats(received=len(data))
        self.stats['messages_received'] += 1
        
        return data
    
    async def _receive_loop(self):
        """Continuous receive loop for TCP"""
        try:
            while self.state == TransportState.CONNECTED:
                data = await self.receive()
                
                if self.message_handler:
                    await self.message_handler(data)
                    
        except asyncio.IncompleteReadError:
            logger.info("TCP connection closed by peer")
        except Exception as e:
            logger.error(f"TCP receive error: {e}")
        finally:
            await self.close()
    
    async def close(self) -> None:
        """Close TCP connection"""
        self.state = TransportState.CLOSING
        
        self.writer.close()
        await self.writer.wait_closed()
        
        self.state = TransportState.CLOSED
