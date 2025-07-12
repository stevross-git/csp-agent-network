# enhanced_csp/network/compression.py
import time
import zlib
import lz4.frame
import brotli
import snappy
import zstandard
import msgpack
import struct
import threading
from enum import Enum
from typing import Union, Dict, Any, Optional, Set, List
from dataclasses import dataclass, field
from .utils.structured_logging import get_logger
from contextvars import ContextVar
from .utils import ThreadSafeStats

logger = get_logger("compression")

# Context variables for thread-safe compression contexts
_zstd_context: ContextVar[Optional['ZstdContext']] = ContextVar('zstd_context', default=None)


class CompressionAlgorithm(Enum):
    """Supported compression algorithms"""

    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"
    BROTLI = "brotli"
    SNAPPY = "snappy"
    ZSTD = "zstd"


@dataclass
class CompressionConfig:
    """Configuration for message compression."""

    min_compress_bytes: int = 64
    zstd_dict_size: int = 8192
    incompressible_mimes: Set[str] = field(default_factory=lambda: {
        "image/jpeg",
        "image/png",
    })
    max_decompress_bytes: int = 50 * 1024 * 1024

@dataclass
class ZstdContext:
    """Thread-safe container for Zstandard compression contexts"""
    dict_data: Optional[zstandard.ZstdCompressionDict] = None
    compressor: Optional[zstandard.ZstdCompressor] = None
    decompressor: Optional[zstandard.ZstdDecompressor] = None
    lock: threading.RLock = field(default_factory=threading.RLock)

class MessageCompressor:
    """High-performance message compression with thread safety and metrics export"""
    
    def __init__(self, 
                 default_algorithm: CompressionAlgorithm = CompressionAlgorithm.LZ4,
                 config: CompressionConfig = None):
        self.default_algorithm = default_algorithm
        self.config = config or CompressionConfig()
        
        defaults = {
            "total_bytes_original": 0,
            "total_bytes_compressed": 0,
            "compression_time_ms": 0,
            "decompression_time_ms": 0,
            "skipped_count": 0,
            "compression_ratio_sum": 0.0,
            "compression_count": 0,
        }
        self.compression_stats = ThreadSafeStats(defaults)
        
    def train_dictionary(self, samples: List[bytes]) -> None:
        """Train Zstandard dictionary for better compression of similar messages"""
        if not samples:
            return
            
        try:
            dict_data = zstandard.train_dictionary(
                self.config.zstd_dict_size,
                samples
            )
            
            # Create new context
            ctx = ZstdContext(
                dict_data=zstandard.ZstdCompressionDict(dict_data),
                compressor=zstandard.ZstdCompressor(dict_data=dict_data, level=3),
                decompressor=zstandard.ZstdDecompressor(dict_data=dict_data)
            )
            
            # Set in context var (thread-safe)
            _zstd_context.set(ctx)
            
            with logger.context(operation="train_dict"):
                logger.info(
                    f"Trained zstd dictionary with {len(samples)} samples"
                )
        except Exception as e:
            with logger.context(operation="train_dict"):
                logger.warning(f"Failed to train zstd dictionary: {e}")
    
    def _looks_already_compressed(self, data: bytes, sample_size: int = 512) -> bool:
        """Enhanced check if data appears to be already compressed"""
        if len(data) < sample_size:
            sample = data
        else:
            sample = data[:sample_size]
        
        # Check for common compressed file headers
        compressed_headers = [
            b'\x1f\x8b',  # gzip
            b'PK',        # zip
            b'\x42\x5a',  # bzip2
            b'\x04\x22\x4d\x18',  # lz4
            b'\x28\xb5\x2f\xfd',  # zstd
            b'\xff\xd8\xff',  # JPEG
            b'\x89\x50\x4e\x47',  # PNG
        ]
        
        for header in compressed_headers:
            if sample.startswith(header):
                return True
        
        # Enhanced entropy check with secondary validation
        byte_counts = [0] * 256
        for byte in sample:
            byte_counts[byte] += 1
        
        # Calculate simple entropy metric
        unique_bytes = sum(1 for count in byte_counts if count > 0)
        entropy = unique_bytes / 256
        
        # High entropy alone isn't enough - also check compressibility
        if entropy > 0.95:
            # Try quick compression test
            try:
                test_compressed = zlib.compress(sample, level=1)
                compression_ratio = len(test_compressed) / len(sample)
                # If compression doesn't help much, data is likely already compressed
                return compression_ratio > 0.97
            except Exception:
                return True
                
        return False
        
    def compress(self, 
                 data: Union[bytes, str, Dict[str, Any]], 
                 algorithm: CompressionAlgorithm = None,
                 content_type: Optional[str] = None) -> tuple[bytes, str]:
        """Compress data with specified algorithm and safety checks"""
        algorithm = algorithm or self.default_algorithm
        
        # Serialize if needed
        if isinstance(data, dict):
            data = msgpack.packb(
                data, 
                use_bin_type=True,
                use_single_float=True,  # Preserve float32
                strict_types=True       # Ensure bytes stay bytes
            )
        elif isinstance(data, str):
            data = data.encode('utf-8')
        
        # Skip compression for small or incompressible data
        if (len(data) < self.config.min_compress_bytes or
            (content_type and content_type in self.config.incompressible_mimes) or
            self._looks_already_compressed(data)):
            self.compression_stats.increment("skipped_count")
            return data, CompressionAlgorithm.NONE.value
        
        start_time = time.perf_counter()
        try:
            compressed_data = self._compress_with_algorithm(data, algorithm)
            
            # Secondary check: don't use compression if it made data larger
            compression_ratio = len(compressed_data) / len(data)
            if compression_ratio >= 0.97:
                self.compression_stats.increment("skipped_count")
                return data, CompressionAlgorithm.NONE.value
                
        except Exception as e:
            logger.warning(f"Compression failed with {algorithm}: {e}")
            return data, CompressionAlgorithm.NONE.value
            
        compression_time = (time.perf_counter() - start_time) * 1000
        
        # Update stats thread-safely
        self.compression_stats.increment("total_bytes_original", len(data))
        self.compression_stats.increment("total_bytes_compressed", len(compressed_data))
        self.compression_stats.increment("compression_time_ms", compression_time)
        self.compression_stats.increment("compression_ratio_sum", compression_ratio)
        self.compression_stats.increment("compression_count")
        
        return compressed_data, algorithm.value
    
    def _compress_with_algorithm(self, data: bytes, algorithm: CompressionAlgorithm) -> bytes:
        """Apply specific compression algorithm with thread safety"""
        if algorithm == CompressionAlgorithm.NONE:
            return data
        elif algorithm == CompressionAlgorithm.GZIP:
            return zlib.compress(data, level=6)
        elif algorithm == CompressionAlgorithm.LZ4:
            return lz4.frame.compress(data, compression_level=0)
        elif algorithm == CompressionAlgorithm.BROTLI:
            return brotli.compress(data, quality=4)
        elif algorithm == CompressionAlgorithm.SNAPPY:
            return snappy.compress(data)
        elif algorithm == CompressionAlgorithm.ZSTD:
            ctx = _zstd_context.get()
            if ctx and ctx.compressor:
                with ctx.lock:
                    return ctx.compressor.compress(data)
            else:
                # Fallback to default compressor
                cctx = zstandard.ZstdCompressor(level=3)
                return cctx.compress(data)
    
    def export_stats(self) -> Dict[str, Any]:
        """Export and reset statistics for metrics collection"""
        stats = self.compression_stats.snapshot()
        count = stats.get("compression_count", 0)
        ratio = stats.get("compression_ratio_sum", 0.0)
        avg_ratio = ratio / count if count > 0 else 1.0
        stats["average_compression_ratio"] = avg_ratio
        stats["space_saved_bytes"] = (
            stats.get("total_bytes_original", 0) - stats.get("total_bytes_compressed", 0)
        )
        self.compression_stats.reset({k: 0 for k in stats.keys() if k != "average_compression_ratio" and k != "space_saved_bytes"})
        return stats
