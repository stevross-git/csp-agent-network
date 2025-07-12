import asyncio
import pytest

from enhanced_csp.network.utils import MessageBatcher, BatchConfig

@pytest.mark.asyncio
async def test_batcher_size_trigger():
    batches = []

    async def sender(batch):
        batches.append(batch)
        return True

    batcher = MessageBatcher(sender, BatchConfig(max_messages=3, flush_interval=0.5))
    await batcher.start()
    await batcher.add_message({"id": 1})
    await batcher.add_message({"id": 2})
    assert batches == []
    await batcher.add_message({"id": 3})
    await asyncio.sleep(0.05)
    assert len(batches) == 1
    assert batches[0]["count"] == 3
    await batcher.stop()

@pytest.mark.asyncio
async def test_batcher_time_trigger():
    batches = []

    async def sender(batch):
        batches.append(batch)
        return True

    batcher = MessageBatcher(sender, BatchConfig(max_messages=5, flush_interval=0.05))
    await batcher.start()
    await batcher.add_message({"id": 1})
    await asyncio.sleep(0.1)
    assert len(batches) == 1
    assert batches[0]["count"] == 1
    await batcher.stop()

@pytest.mark.asyncio
async def test_batcher_priority_bypass():
    batches = []

    async def sender(batch):
        batches.append(batch)
        return True

    batcher = MessageBatcher(sender, BatchConfig(max_messages=5, urgent_priority=5))
    await batcher.start()
    await batcher.add_message({"id": 1}, priority=10)
    await asyncio.sleep(0.05)
    assert len(batches) == 1
    assert batches[0]["count"] == 1
    await batcher.stop()
