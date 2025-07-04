# enhanced_csp/network/p2p/transport.py
"""
Multi-protocol transport layer for Enhanced CSP.
Supports TCP, QUIC, and WebSocket transports.
"""

import asyncio
import logging
from typing import Dict, Optional, Any, Callable, Tuple, List
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json

from ..core.types import NetworkMessage, MessageType
from ..core.config import P2PConfig
from ..protocol_optimizer import BinaryProtocol, MessageType as BinaryMessageType

logger = logging.getLogger(__name__)


@dataclass
class Connection:
    """Represents a network connection."""
    address: str
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    protocol: str = "tcp"
    
    async def close(self):
        """Close the connection."""
        self.writer.close()
        await self.writer.wait_closed()


class P2PTransport(ABC):
    """
    Abstract base class for P2P transport implementations.
    Provides the interface for network communication.
    """
    
    def __init__(self, config: P2PConfig):
        """Initialize transport with configuration."""
        self.config = config
        self.connections: Dict[str, Connection] = {}
        self.server: Optional[asyncio.Server] = None
        self.is_running = False
        
    @abstractmethod
    async def start(self) -> bool:
        """Start the transport service."""
        pass
    
    @abstractmethod
    async def stop(self):
        """Stop the transport service."""
        pass
    
    @abstractmethod
    async def send(self, address: str, message: Any) -> bool:
        """Send a message to a peer."""
        pass
    
    @abstractmethod
    async def connect(self, address: str) -> bool:
        """Connect to a peer."""
        pass
    
    @abstractmethod
    def register_handler(self, message_type: str, handler: Callable):
        """Register a message handler."""
        pass


class MultiProtocolTransport(P2PTransport):
    """
    Enhanced transport supporting multiple protocols.
    Primary implementation uses TCP with optional QUIC support.
    """
    
    def __init__(self, config: P2PConfig):
        super().__init__(config)
        self.protocol = BinaryProtocol()
        self.message_handlers: Dict[str, List[Callable]] = {}
        
    async def start(self) -> bool:
        """Start transport services."""
        if self.is_running:
            return True
            
        logger.info(f"Starting P2P transport on {self.config.listen_address}:{self.config.listen_port}")
        
        try:
            # Start TCP server
            self.server = await asyncio.start_server(
                self._handle_connection,
                self.config.listen_address,
                self.config.listen_port
            )
            
            self.is_running = True
            
            # Start additional protocol servers if configured
            if self.config.enable_quic:
                logger.info("QUIC support enabled but not implemented in this version")
            
            logger.info("P2P transport started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start P2P transport: {e}")
            return False
    
    async def stop(self):
        """Stop transport services."""
        if not self.is_running:
            return
            
        logger.info("Stopping P2P transport")
        self.is_running = False
        
        # Close server
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        # Close all connections
        for conn in self.connections.values():
            await conn.close()
        self.connections.clear()
        
        logger.info("P2P transport stopped")
    
    async def connect(self, address: str) -> bool:
        """Connect to a peer."""
        if address in self.connections:
            return True
            
        try:
            # Parse address
            if ':' in address:
                host, port = address.split(':')
                port = int(port)
            else:
                host = address
                port = self.config.listen_port
            
            # Create connection
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=self.config.connection_timeout
            )
            
            conn = Connection(address, reader, writer)
            self.connections[address] = conn
            
            logger.info(f"Connected to peer {address}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to {address}: {e}")
            return False
    
    async def send(self, address: str, message: Any) -> bool:
        """
        Send a message to a peer.
        
        Args:
            address: Peer address (host:port or multiaddr)
            message: Message to send
            
        Returns:
            True if sent successfully
        """
        try:
            # Get or create connection
            if address not in self.connections:
                if not await self.connect(address):
                    return False
                    
            conn = self.connections.get(address)
            if not conn:
                return False
            
            # Determine message type
            msg_type = BinaryMessageType.DATA
            if isinstance(message, dict):
                if message.get('type') == 'ping':
                    msg_type = BinaryMessageType.PING
                elif message.get('type') == 'pong':
                    msg_type = BinaryMessageType.PONG
                elif message.get('type') in ['control', 'peer_exchange']:
                    msg_type = BinaryMessageType.CONTROL
            
            # Encode message
            encoded = self.protocol.encode_message(message, msg_type)
            
            # Send over connection
            conn.writer.write(encoded)
            await conn.writer.drain()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message to {address}: {e}")
            # Remove failed connection
            if address in self.connections:
                await self.connections[address].close()
                del self.connections[address]
            return False
    
    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle incoming connection."""
        addr = writer.get_extra_info('peername')
        address = f"{addr[0]}:{addr[1]}" if addr else "unknown"
        
        logger.info(f"New connection from {address}")
        
        # Store connection
        conn = Connection(address, reader, writer)
        self.connections[address] = conn
        
        try:
            # Read messages
            while True:
                # Read header first
                header_data = await reader.read(self.protocol.HEADER_SIZE)
                if not header_data:
                    break
                    
                # Decode header to get message length
                try:
                    _, _, _, length = self.protocol.decode_header_only(header_data)
                except Exception as e:
                    logger.error(f"Invalid header from {address}: {e}")
                    break
                
                # Read full message
                payload_data = await reader.read(length)
                if len(payload_data) < length:
                    logger.error(f"Incomplete message from {address}")
                    break
                
                # Decode message
                try:
                    message, msg_type, flags = self.protocol.decode_message(
                        header_data + payload_data
                    )
                    
                    # Dispatch to handlers
                    await self._dispatch_message(address, message, msg_type)
                    
                except Exception as e:
                    logger.error(f"Failed to decode message from {address}: {e}")
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Connection error with {address}: {e}")
        finally:
            # Clean up connection
            await conn.close()
            if address in self.connections:
                del self.connections[address]
            logger.info(f"Connection closed: {address}")
    
    def register_handler(self, message_type: str, handler: Callable):
        """Register a message handler."""
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        self.message_handlers[message_type].append(handler)
    
    async def _dispatch_message(self, sender: str, message: Any, msg_type: BinaryMessageType):
        """Dispatch message to registered handlers."""
        # Map binary message type to handler key
        handler_key = msg_type.name
        
        if handler_key in self.message_handlers:
            for handler in self.message_handlers[handler_key]:
                try:
                    await handler(sender, message)
                except Exception as e:
                    logger.error(f"Handler error for {handler_key}: {e}")