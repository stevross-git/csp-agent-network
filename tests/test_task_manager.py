import asyncio
import pytest

from enhanced_csp.network.utils import TaskManager
from enhanced_csp.network.core.node import NetworkNode
from enhanced_csp.network.core.types import NetworkConfig


@pytest.mark.asyncio
async def test_task_manager_cancel():
    tm = TaskManager()

    async def sleeper():
        await asyncio.sleep(1)

    task = tm.create_task(sleeper())
    await asyncio.sleep(0.05)
    await tm.cancel_all()
    assert task.cancelled()


@pytest.mark.asyncio
async def test_node_tasks_cleaned(monkeypatch):
    async def dummy_init(self):
        pass

    monkeypatch.setattr(NetworkNode, "_init_components", dummy_init)
    node = NetworkNode(NetworkConfig())
    await node.start()
    assert len(node.task_manager.tasks) >= 2
    await node.stop()
    assert len(node.task_manager.tasks) == 0
