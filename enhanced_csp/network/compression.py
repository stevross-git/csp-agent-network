# enhanced_csp/network/compression.py
"""
Adaptive Compression Pipeline for Enhanced CSP Network
Provides 50-80% bandwidth reduction through intelligent algorithm selection.
"""

import asyncio
import time
<<<<<<< HEAD
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib
import msgpack

# Compression libraries with graceful fallbacks
try:
    import lz4.frame
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False

try:
    import brotli
    BROTLI_AVAILABLE = True
except ImportError:
    BROTLI_AVAILABLE = False

import zlib  # Always available in Python standard library
import gzip

from .core.config import P2PConfig
from .utils import get_logger

logger = get_logger(__name__)
=======
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
>>>>>>> 1871c497b6c6ccafca331c9065069c220ca63f43



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
class CompressionConfig:
    """Configuration for adaptive compression."""
    min_compress_bytes: int = 128
    default_algorithm: str = 'lz4'
    enable_adaptive_selection: bool = True
    dictionary_training: bool = True
    max_dictionary_size: int = 64 * 1024  # 64KB
    compression_level: int = 3
    enable_content_analysis: bool = True


@dataclass
class CompressionStats:
    """Statistics for compression algorithm performance."""
    total_bytes_input: int = 0
    total_bytes_output: int = 0
    total_time_seconds: float = 0.0
    operation_count: int = 0
    last_ratio: float = 1.0
    last_speed: float = 0.0  # bytes per second
    
    @property
    def avg_ratio(self) -> float:
        """Average compression ratio."""
        if self.total_bytes_input == 0:
            return 1.0
        return self.total_bytes_output / self.total_bytes_input
    
    @property
    def avg_speed(self) -> float:
        """Average compression speed in bytes/second."""
        if self.total_time_seconds == 0:
            return 0.0
        return self.total_bytes_input / self.total_time_seconds


@dataclass
class ContentProfile:
    """Profile of content for optimal compression selection."""
    json_like_score: float = 0.0
    binary_score: float = 0.0
    repetition_score: float = 0.0
    entropy: float = 0.0
    size_category: str = 'small'  # small, medium, large


class CompressionDictionary:
    """Compression dictionary for improved ratios on similar content."""
    
    def __init__(self, max_size: int = 64 * 1024):
        self.max_size = max_size
        self.samples: List[bytes] = []
        self.dictionaries: Dict[str, Any] = {}
        self.last_trained = 0.0
        
<<<<<<< HEAD
    def add_sample(self, data: bytes):
        """Add sample data for dictionary training."""
        if len(data) < 100:  # Skip very small samples
            return
            
        self.samples.append(data)
=======
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
>>>>>>> 1871c497b6c6ccafca331c9065069c220ca63f43
        
        # Limit sample count to prevent memory bloat
        if len(self.samples) > 1000:
            self.samples = self.samples[-500:]  # Keep most recent 500
    
    def train_dictionaries(self):
        """Train compression dictionaries from samples."""
        if len(self.samples) < 10:  # Need enough samples
            return
            
        try:
            # Combine samples for training
            training_data = b''.join(self.samples[-100:])  # Use last 100 samples
            
            # Train Zstandard dictionary if available
            if ZSTD_AVAILABLE:
                try:
                    dict_data = zstd.train_dictionary(self.max_size, [training_data])
                    self.dictionaries['zstd'] = zstd.ZstdCompressionDict(dict_data)
                    logger.debug("Trained Zstandard compression dictionary")
                except Exception as e:
                    logger.debug(f"Failed to train Zstandard dictionary: {e}")
            
            self.last_trained = time.time()
            
<<<<<<< HEAD
        except Exception as e:
            logger.error(f"Dictionary training failed: {e}")


class AdaptiveCompressionPipeline:
    """
    Multi-stage compression with intelligent algorithm selection.
    Achieves 50-80% bandwidth reduction through adaptive optimization.
    """
=======
            with logger.context(operation="train_dict"):
                logger.info(
                    f"Trained zstd dictionary with {len(samples)} samples"
                )
        except Exception as e:
            with logger.context(operation="train_dict"):
                logger.warning(f"Failed to train zstd dictionary: {e}")
>>>>>>> 1871c497b6c6ccafca331c9065069c220ca63f43
    
    def __init__(self, config: CompressionConfig):
        self.config = config
        
        # Available compression algorithms
        self.algorithms: Dict[str, Callable] = {}
        self.decompression_algorithms: Dict[str, Callable] = {}
        self._setup_algorithms()
        
        # Performance tracking
        self.algorithm_stats: Dict[str, CompressionStats] = defaultdict(CompressionStats)
        
        # Dictionary for improved compression
        self.dictionary = CompressionDictionary(config.max_dictionary_size)
        
        # Content analysis cache
        self.content_profiles: Dict[str, ContentProfile] = {}
        
        # Performance optimization
        self.last_algorithm_selection = {}
        self.selection_cache_ttl = 60.0  # Cache selections for 1 minute
        
    def _setup_algorithms(self):
        """Setup available compression algorithms."""
        # Always available: zlib and gzip
        self.algorithms['zlib'] = self._compress_zlib
        self.algorithms['gzip'] = self._compress_gzip
        self.decompression_algorithms['zlib'] = self._decompress_zlib
        self.decompression_algorithms['gzip'] = self._decompress_gzip
        
        # Optional algorithms
        if LZ4_AVAILABLE:
            self.algorithms['lz4'] = self._compress_lz4
            self.decompression_algorithms['lz4'] = self._decompress_lz4
            
        if ZSTD_AVAILABLE:
            self.algorithms['zstd'] = self._compress_zstd
            self.decompression_algorithms['zstd'] = self._decompress_zstd
            
        if BROTLI_AVAILABLE:
            self.algorithms['brotli'] = self._compress_brotli
            self.decompression_algorithms['brotli'] = self._decompress_brotli
        
<<<<<<< HEAD
        logger.info(f"Initialized compression with algorithms: {list(self.algorithms.keys())}")
    
    async def compress_batch(self, messages: List[Dict[str, Any]]) -> Tuple[bytes, str, Dict[str, Any]]:
        """Compress entire batch with optimal algorithm."""
=======
        # Skip compression for small or incompressible data
        if (len(data) < self.config.min_compress_bytes or
            (content_type and content_type in self.config.incompressible_mimes) or
            self._looks_already_compressed(data)):
            self.compression_stats.increment("skipped_count")
            return data, CompressionAlgorithm.NONE.value
        
        start_time = time.perf_counter()
>>>>>>> 1871c497b6c6ccafca331c9065069c220ca63f43
        try:
            # Serialize batch
            batch_data = msgpack.packb({
                'messages': messages,
                'count': len(messages),
                'timestamp': time.time(),
                'batch_id': hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
            })
            
            # Skip compression if too small
            if len(batch_data) < self.config.min_compress_bytes:
                return batch_data, 'none', {'ratio': 1.0, 'algorithm': 'none'}
            
            # Select optimal algorithm
            algorithm = await self._select_optimal_algorithm(batch_data)
            
            # Compress with selected algorithm
            start_time = time.perf_counter()
            compressed = await self._compress_async(batch_data, algorithm)
            compression_time = time.perf_counter() - start_time
            
            # Update algorithm performance stats
            ratio = len(compressed) / len(batch_data)
            speed = len(batch_data) / max(compression_time, 0.0001)
            self._update_algorithm_stats(algorithm, len(batch_data), len(compressed), compression_time)
            
            # Add to dictionary training samples
            if self.config.dictionary_training:
                self.dictionary.add_sample(batch_data)
            
            metadata = {
                'algorithm': algorithm,
                'ratio': ratio,
                'original_size': len(batch_data),
                'compressed_size': len(compressed),
                'compression_time_ms': compression_time * 1000,
                'speed_mbps': (len(batch_data) / (1024 * 1024)) / max(compression_time, 0.0001)
            }
            
            return compressed, algorithm, metadata
            
<<<<<<< HEAD
        except Exception as e:
            logger.error(f"Batch compression failed: {e}")
            # Return uncompressed data on failure
            return msgpack.packb(messages), 'none', {'ratio': 1.0, 'algorithm': 'none', 'error': str(e)}
    
    async def decompress_batch(self, compressed_data: bytes, algorithm: str) -> List[Dict[str, Any]]:
        """Decompress batch data."""
        try:
            if algorithm == 'none':
                # Data was not compressed
                batch_data = msgpack.unpackb(compressed_data, raw=False)
                if isinstance(batch_data, dict) and 'messages' in batch_data:
                    return batch_data['messages']
                return batch_data if isinstance(batch_data, list) else [batch_data]
            
            # Decompress data
            start_time = time.perf_counter()
            decompressed = await self._decompress_async(compressed_data, algorithm)
            decompression_time = time.perf_counter() - start_time
            
            # Deserialize batch
            batch_data = msgpack.unpackb(decompressed, raw=False)
            
            logger.debug(f"Decompressed {len(compressed_data)} -> {len(decompressed)} bytes "
                        f"in {decompression_time*1000:.2f}ms using {algorithm}")
            
            if isinstance(batch_data, dict) and 'messages' in batch_data:
                return batch_data['messages']
            return batch_data if isinstance(batch_data, list) else [batch_data]
=======
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
>>>>>>> 1871c497b6c6ccafca331c9065069c220ca63f43
