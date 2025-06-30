import asyncio
from enhanced_csp.ai_comm import AdvancedAICommChannel, AdvancedCommPattern
from enhanced_csp.agents import PlannerAgent, DataCleanerAgent
from enhanced_csp.network.core.types import NetworkConfig
from enhanced_csp.network.core.node import NetworkNode


async def main() -> None:
    node_a = NetworkNode(NetworkConfig())
    node_b = NetworkNode(NetworkConfig())

    channel = AdvancedAICommChannel("demo", AdvancedCommPattern.NEURAL_MESH)
    node_a.register_channel("demo", channel)
    node_b.register_channel("demo", channel)

    planner = PlannerAgent("planner")
    cleaner = DataCleanerAgent("cleaner")
    node_a.register_agent(planner.agent_id, planner)
    node_b.register_agent(cleaner.agent_id, cleaner)

    msg = planner.create_task_message("cleaner", "deduplicate_records", "ref", {})
    await node_b.handle_raw_message(node_b._serialize_message(msg))
    print("Demo executed")


if __name__ == "__main__":
    asyncio.run(main())
