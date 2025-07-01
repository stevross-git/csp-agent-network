# tests/network/test_compression.py
import pytest
import msgpack
from enhanced_csp.network.compression import (
    MessageCompressor, CompressionAlgorithm, CompressionConfig
)

class TestCompression:
    """Test compression functionality"""
    
    @pytest.fixture
    def compressor(self):
        return MessageCompressor()
    
    @pytest.mark.parametrize("algorithm", list(CompressionAlgorithm))
    def test_round_trip(self, compressor, algorithm):
        """Test compression round-trip for all algorithms"""
        data = b"Hello, World!" * 100
        
        compressed, algo_used = compressor.compress(data, algorithm)
        decompressed = compressor.decompress(compressed, algo_used)
        
        assert decompressed == data
    
    def test_small_data_skip(self, compressor):
        """Test that small data skips compression"""
        data = b"tiny"
        compressed, algo = compressor.compress(data)
        
        assert compressed == data
        assert algo == CompressionAlgorithm.NONE.value
    
    def test_incompressible_skip(self, compressor):
        """Test that incompressible data is detected"""
        # JPEG header
        data = b'\xff\xd8\xff' + b'x' * 1000
        compressed, algo = compressor.compress(data)
        
        assert algo == CompressionAlgorithm.NONE.value
    
    def test_decompression_bomb_protection(self, compressor):
        """Test protection against decompression bombs"""
        # Create a highly compressible payload
        data = b'A' * (200 * 1024 * 1024)  # 200MB of 'A's
        compressed, algo = compressor.compress(data, CompressionAlgorithm.ZSTD)
        
        # Should fail with safety limit
        compressor.config.max_decompress_bytes = 100 * 1024 * 1024  # 100MB limit
        with pytest.raises(ValueError):
            compressor.decompress(compressed, algo)

# tests/network/test_batching.py
import pytest
import asyncio
from enhanced_csp.network.batching import MessageBatcher, BatchConfig

class TestBatching:
    """Test message batching functionality"""
    
    @pytest.fixture
    async def batcher(self):
        config = BatchConfig(
            max_batch_size=5,
            max_wait_time_ms=50
        )
        
        sent_batches = []
        
        async def capture_batch(batch):
            sent_batches.append(batch)
        
        batcher = MessageBatcher(config, send_callback=capture_batch)
        await batcher.start()
        
        yield batcher, sent_batches
        
        await batcher.stop()
    
    @pytest.mark.asyncio
    async def test_size_trigger(self, batcher):
        """Test batch flush on size limit"""
        batcher_obj, sent_batches = batcher
        
        # Add messages up to limit
        for i in range(5):
            await batcher_obj.add_message({"id": i})
        
        # Should trigger immediate flush
        await asyncio.sleep(0.01)
        
        assert len(sent_batches) == 1
        assert sent_batches[0]["count"] == 5
    
    @pytest.mark.asyncio
    async def test_time_trigger(self, batcher):
        """Test batch flush on time limit"""
        batcher_obj, sent_batches = batcher
        
        # Add fewer messages than size limit
        await batcher_obj.add_message({"id": 1})
        await batcher_obj.add_message({"id": 2})
        
        # Wait for time trigger
        await asyncio.sleep(0.1)
        
        assert len(sent_batches) == 1
        assert sent_batches[0]["count"] == 2
    
    @pytest.mark.asyncio
    async def test_priority_bypass(self, batcher):
        """Test high priority message bypass"""
        batcher_obj, sent_batches = batcher
        
        # High priority message should bypass batching
        await batcher_obj.add_message({"id": 1, "urgent": True}, priority=9)
        
        await asyncio.sleep(0.01)
        
        assert len(sent_batches) == 1
        assert sent_batches[0]["type"] == "immediate"