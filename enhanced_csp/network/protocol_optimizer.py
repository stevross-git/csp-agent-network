# enhanced_csp/network/protocol_optimizer.py
import struct
import time
import msgpack
from typing import Dict, Any, Tuple, Optional
from enum import IntEnum
import logging

logger = logging.getLogger(__name__)

class MessageType(IntEnum):
    """Optimized message type encoding"""
    PING = 0
    PONG = 1
    DATA = 2
    BATCH = 3
    COMPRESSED = 4
    CONTROL = 5
    STREAM_START = 6
    STREAM_DATA = 7
    STREAM_END = 8

class MessageFlags(IntEnum):
    """Message flags for fast-path processing"""
    COMPRESSED = 0x01
    BATCHED = 0x02
    PRIORITY = 0x04
    ENCRYPTED = 0x08
    REQUIRES_ACK = 0x10

class BinaryProtocol:
    """High-performance binary protocol for network communication"""
    
    # Fixed: B=version(1) B=type(1) H=flags(2) I=length(4) Q=timestamp(8) = 16 bytes
    HEADER_FORMAT = "!BBHIQ"
    HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
    MAX_MESSAGE_SIZE = 16 * 1024 * 1024  # 16MB limit
    
    def __init__(self, version: int = 1):
        self.version = version
        
    def encode_message(self, 
                      message: Dict[str, Any], 
                      msg_type: MessageType,
                      compressed: bool = False,
                      batched: bool = False,
                      priority: bool = False) -> bytes:
        """Encode message to optimized binary format"""
        # Serialize payload
        payload = msgpack.packb(message, use_bin_type=True)
        
        # Validate size
        if len(payload) > self.MAX_MESSAGE_SIZE:
            raise ValueError(f"Message too large: {len(payload)} bytes")
        
        # Calculate flags
        flags = 0
        if compressed:
            flags |= MessageFlags.COMPRESSED
        if batched:
            flags |= MessageFlags.BATCHED
        if priority:
            flags |= MessageFlags.PRIORITY
        
        # Microsecond timestamp
        timestamp = int(time.time() * 1000000)
        
        # Pack header
        header = struct.pack(
            self.HEADER_FORMAT,
            self.version,
            msg_type,
            flags,
            len(payload),
            timestamp
        )
        
        return header + payload
    
    def decode_message(self, data: bytes) -> Tuple[Dict[str, Any], MessageType, int]:
        """Decode binary message with validation"""
        if len(data) < self.HEADER_SIZE:
            raise ValueError(f"Invalid message: too short ({len(data)} bytes)")
        
        # Parse header
        try:
            version, msg_type, flags, length, timestamp = struct.unpack(
                self.HEADER_FORMAT, data[:self.HEADER_SIZE]
            )
        except struct.error as e:
            raise ValueError(f"Invalid message header: {e}")
        
        # Validate version
        if version != self.version:
            raise ValueError(f"Unsupported protocol version: {version}")
        
        # Validate length
        if length > self.MAX_MESSAGE_SIZE:
            raise ValueError(f"Message length exceeds limit: {length}")
            
        if len(data) < self.HEADER_SIZE + length:
            raise ValueError(f"Message truncated: expected {length} bytes, got {len(data) - self.HEADER_SIZE}")
        
        # Extract payload
        payload_data = data[self.HEADER_SIZE:self.HEADER_SIZE + length]
        
        try:
            message = msgpack.unpackb(payload_data, raw=False)
        except Exception as e:
            raise ValueError(f"Invalid message payload: {e}")
        
        # Add metadata
        message["_timestamp"] = timestamp / 1000000.0
        message["_flags"] = flags
        message["_compressed"] = bool(flags & MessageFlags.COMPRESSED)
        message["_batched"] = bool(flags & MessageFlags.BATCHED)
        message["_priority"] = bool(flags & MessageFlags.PRIORITY)
        
        return message, MessageType(msg_type), flags
    
    def decode_header_only(self, data: bytes) -> Tuple[int, MessageType, int, int]:
        """Fast header-only decode for routing decisions"""
        if len(data) < self.HEADER_SIZE:
            raise ValueError("Insufficient data for header")
            
        version, msg_type, flags, length, _ = struct.unpack(
            self.HEADER_FORMAT, data[:self.HEADER_SIZE]
        )
        
        return version, MessageType(msg_type), flags, length