# enhanced_csp/network/zero_copy.py
import mmap
import os
import tempfile
import platform
from typing import Optional, Dict, Any, Tuple
from threading import Lock
import struct
import logging

logger = logging.getLogger(__name__)

class ZeroCopyRingBuffer:
    """Zero-copy ring buffer with platform compatibility and metrics"""
    
    def __init__(self, size: int = 10 * 1024 * 1024, name: str = None):
        self.size = size
        self.name = name or f"csp_buffer_{os.getpid()}"
        self.platform = platform.system()
        
        # Platform-specific setup
        if self.platform == "Linux" and os.path.exists("/dev/shm"):
            self.filepath = f"/dev/shm/{self.name}"
        else:
            # Windows and macOS fallback
            self.temp_dir = tempfile.gettempdir()
            self.filepath = os.path.join(self.temp_dir, self.name)
        
        # Create file
        self._create_buffer_file()
        
        # Memory map with platform-specific access
        self.file = open(self.filepath, "r+b")
        
        if self.platform == "Windows":
            # Windows needs ACCESS_WRITE for read/write
            self.mmap = mmap.mmap(self.file.fileno(), self.size, access=mmap.ACCESS_WRITE)
        else:
            # Unix platforms
            self.mmap = mmap.mmap(self.file.fileno(), self.size)
        
        # Ring buffer state
        self._state_lock = Lock()
        self.write_pos = 0
        self.read_pos = 0
        self.wrap_count = 0
        self.dropped_messages = 0
        
        # Header format: size(4) + checksum(4) + next_offset(4) = 12 bytes
        self.HEADER_SIZE = 12
        
    def write_message(self, data: bytes) -> Optional[int]:
        """Write message to ring buffer with wrap-around"""
        msg_size = len(data)
        total_size = self.HEADER_SIZE + msg_size
        
        with self._state_lock:
            # Check if we need to wrap
            if self.write_pos + total_size > self.size:
                # Check if we have room at the beginning
                if total_size > self.read_pos:
                    # Buffer is full
                    self.dropped_messages += 1
                    logger.warning(f"Ring buffer full, dropped message. Total dropped: {self.dropped_messages}")
                    return None
                
                # Write wrap marker at current position
                if self.write_pos + 4 <= self.size:
                    self.mmap[self.write_pos:self.write_pos + 4] = struct.pack("<I", 0xFFFFFFFF)
                
                # Wrap to beginning
                self.write_pos = 0
                self.wrap_count += 1
            
            # Calculate simple checksum
            checksum = sum(data) & 0xFFFFFFFF
            
            # Calculate next offset (for reader navigation)
            next_offset = (self.write_pos + total_size) % self.size
            
            # Write header
            header = struct.pack("<III", msg_size, checksum, next_offset)
            self.mmap[self.write_pos:self.write_pos + self.HEADER_SIZE] = header
            
            # Write data
            data_start = self.write_pos + self.HEADER_SIZE
            self.mmap[data_start:data_start + msg_size] = data
            
            # Update position
            message_offset = self.write_pos
            self.write_pos = next_offset
            
            return message_offset
    
    def read_message(self, offset: int) -> Tuple[Optional[bytes], Optional[int]]:
        """Read message from ring buffer, returns (data, next_offset)"""
        with self._state_lock:
            # Check for wrap marker
            if offset + 4 <= self.size:
                marker = struct.unpack("<I", self.mmap[offset:offset + 4])[0]
                if marker == 0xFFFFFFFF:
                    # Wrap marker, jump to beginning
                    return None, 0
            
            # Validate offset
            if offset < 0 or offset + self.HEADER_SIZE > self.size:
                logger.error(f"Invalid read offset: {offset}")
                return None, None
            
            # Read header
            header_data = self.mmap[offset:offset + self.HEADER_SIZE]
            msg_size, expected_checksum, next_offset = struct.unpack("<III", header_data)
            
            # Validate size
            if offset + self.HEADER_SIZE + msg_size > self.size:
                logger.error(f"Invalid message size: {msg_size}")
                return None, None
            
            # Read data
            data_start = offset + self.HEADER_SIZE
            data = bytes(self.mmap[data_start:data_start + msg_size])
            
            # Verify checksum
            actual_checksum = sum(data) & 0xFFFFFFFF
            if actual_checksum != expected_checksum:
                logger.error("Checksum mismatch in ring buffer")
                return None, None
            
            # Update read position
            self.read_pos = next_offset
            
            return data, next_offset
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics including fullness metrics"""
        with self._state_lock:
            # Calculate used space (accounting for wrap)
            if self.write_pos >= self.read_pos:
                used = self.write_pos - self.read_pos
            else:
                used = self.size - self.read_pos + self.write_pos
            
            free = self.size - used
            
            return {
                "size": self.size,
                "used": used,
                "free": free,
                "utilization": (used / self.size) * 100,
                "wrap_count": self.wrap_count,
                "dropped_messages": self.dropped_messages,
                "platform": self.platform,
                "filepath": self.filepath,
                "near_full": (used / self.size) > 0.9  # Alert threshold
            }