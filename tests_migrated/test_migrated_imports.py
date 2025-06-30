import pytest

from enhanced_csp.ai_comm import AdvancedAICommChannel, AdvancedCommPattern
from enhanced_csp.agents import BaseAgent, DataCleanerAgent, PlannerAgent
from enhanced_csp.api import CSPLogStore
from enhanced_csp.memory import ChromaVectorStore
from enhanced_csp.protocols import create_csp_message


@pytest.mark.asyncio
async def test_comm_channel():
    channel = AdvancedAICommChannel('test', AdvancedCommPattern.NEURAL_MESH)
    assert channel.channel_id == 'test'
    await channel.establish_neural_mesh([])


def test_agents_and_utils():
    agent = BaseAgent('a')
    cleaner = DataCleanerAgent('c')
    msg = cleaner.receive_csp_message({'sender': 'a', 'msg_id': '1', 'task': {'intent': 'deduplicate_records'}})
    assert msg['sender'] == 'c'
    planner = PlannerAgent('p')
    task_msg = planner.create_task_message('c', 'cleanup', 'ref', {})
    assert task_msg['sender'] == 'p'
    log = CSPLogStore()
    log.log('x')
    assert log.get_logs()[0]['msg'] == 'x'
    store = ChromaVectorStore()
    store.store('k', 'v')
    assert store.search('v')['ids'][0] == 'k'
    m = create_csp_message('x', 'y', 'z')
    assert m['sender'] == 'x'
