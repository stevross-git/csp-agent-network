"""
Enhanced CSP Memory Architecture
================================

Four-layer memory system for advanced AI-agent collaboration
-----------------------------------------------------------
1. **Working Memory** – Fast, temporary storage for active processes  
2. **Shared Memory**  – Inter-agent communication and synchronization  
3. **Crystallized Memory** – Persistent pattern storage and knowledge formation  
4. **Collective Memory** – Emergent intelligence and network-wide insights
"""

from collections import defaultdict
from typing import Dict

from .working_memory import WorkingMemory, WorkingCache
from .shared_memory import SharedMemory, SyncManager, ConsistencyLevel, SyncProtocol
from .crystallized_memory import CrystallizedStore, CrystalDB, MemoryCrystal, CrystalType
from .collective_memory import (
    CollectiveInsightEngine,
    PatternMiner,
    CollectiveInsight,
    InsightType,
)

__version__ = "1.0.0"

__all__ = [
    # Working Memory
    "WorkingMemory",
    "WorkingCache",
    # Shared Memory
    "SharedMemory",
    "SyncManager",
    "ConsistencyLevel",
    "SyncProtocol",
    # Crystallized Memory
    "CrystallizedStore",
    "CrystalDB",
    "MemoryCrystal",
    "CrystalType",
    # Collective Memory
    "CollectiveInsightEngine",
    "PatternMiner",
    "CollectiveInsight",
    "InsightType",
    # Coordinator
    "MemoryCoordinator",
]


# ======================================================================
# Memory Coordinator
# ======================================================================


class MemoryCoordinator:
    """
    Coordinates all four memory layers for seamless operation.
    Manages data flow and interactions between layers.
    """

    # ------------------------------------------------------------------
    # construction / lifecycle
    # ------------------------------------------------------------------
    def __init__(self, namespace: str = "default"):
        self.namespace = namespace
        self._last_key: Dict[str, str] = {}  # agent_id → most-recent WM key

        # initialise layers
        self.working_memory_agents: Dict[str, WorkingMemory] = {}
        self.shared_memory = SharedMemory(namespace)
        self.crystallized_store = CrystallizedStore(namespace)
        self.collective_engine = CollectiveInsightEngine()

        # queues / flags
        self._interaction_queue = []
        self._coordination_running = False

    async def start(self):
        """Start background services for all layers."""
        await self.shared_memory.start()
        await self.crystallized_store.start()
        await self.collective_engine.start()
        self._coordination_running = True

    async def stop(self):
        """Gracefully stop all layers."""
        self._coordination_running = False
        for wm in self.working_memory_agents.values():
            await wm.stop()
        await self.shared_memory.stop()
        await self.crystallized_store.stop()
        await self.collective_engine.stop()

    # ------------------------------------------------------------------
    # agent registration
    # ------------------------------------------------------------------
    def register_agent(self, agent_id: str, working_memory_mb: float = 256) -> WorkingMemory:
        """Create per-agent working memory and register with Shared Memory."""
        wm = WorkingMemory(agent_id, working_memory_mb)
        self.working_memory_agents[agent_id] = wm
        self.shared_memory.sync_manager.register_agent(agent_id)
        return wm

    # ------------------------------------------------------------------
    # core interaction flow
    # ------------------------------------------------------------------
    async def process_interaction(self, interaction_data: dict):
        """
        Ingest a single interaction event and propagate it through all layers.
        Returns a summary dict indicating what happened.
        """
        participants = interaction_data.get("participants", [])

        # 1  Working Memory
        for agent_id in participants:
            wm = self.working_memory_agents.get(agent_id)
            if wm:
                key = f"interaction_{interaction_data.get('id', '')}"
                wm.store(key, interaction_data)
                self._last_key[agent_id] = key

        # 2  Shared Memory
        is_shared = len(participants) > 1
        if is_shared:
            obj_id = f"shared_{interaction_data.get('id', '')}"
            self.shared_memory.create_object(
                obj_id, "interaction", interaction_data, set(participants)
            )

        # 3  Crystallized Memory
        crystal_id = await self.crystallized_store.process_interaction(interaction_data)

        # 4  Collective queue
        self._interaction_queue.append(interaction_data)
        if len(self._interaction_queue) >= 10:
            await self._run_collective_analysis()

        return {
            "crystal_formed": crystal_id is not None,
            "crystal_id": crystal_id,
            "shared": is_shared,
        }

    # ------------------------------------------------------------------
    # collective analysis
    # ------------------------------------------------------------------
    async def _run_collective_analysis(self):
        if not self._interaction_queue:
            return

        memory_data = {
            "working_memory_interactions": self._interaction_queue[-100:],
            "crystallized_memories": await self._get_recent_crystals(),
            "shared_memory_states": self._get_shared_memory_states(),
        }

        await self.collective_engine.process_memory_data(memory_data)
        self._interaction_queue = self._interaction_queue[-50:]  # keep tail only

    async def _get_recent_crystals(self):
        crystals = []
        for cid in list(self.crystallized_store.crystals)[-50:]:
            c = self.crystallized_store.crystals[cid]
            crystals.append(
                {
                    "id": c.id,
                    "type": c.type.value,
                    "participants": c.participants,
                    "strength": c.strength,
                    "content": c.content,
                }
            )
        return crystals

    def _get_shared_memory_states(self):
        states = []
        for oid in self.shared_memory.list_objects():
            info = self.shared_memory.get_object_info(oid)
            if info:
                states.append(
                    {
                        "id": oid,
                        "participants": info["participants"],
                        "version": info["version"],
                        "status": "synced" if not info["locked_by"] else "locked",
                    }
                )
        return states

    # ------------------------------------------------------------------
    # querying across layers
    # ------------------------------------------------------------------
    async def query_memory(self, agent_id: str, query: dict):
        """
        Return a dict with data pulled from Working, Shared, Crystallized
        and Collective layers relevant to *agent_id*.
        """
        results = {"working": None, "shared": [], "crystallized": [], "insights": []}

        # working memory
        wm = self.working_memory_agents.get(agent_id)
        if wm:
            key = query.get("key")
            if key:
                results["working"] = wm.retrieve(key)
            else:
                last_key = self._last_key.get(agent_id)
                if last_key:
                    results["working"] = wm.retrieve(last_key)

        # shared memory
        for oid in self.shared_memory.list_objects()[:10]:
            info = self.shared_memory.get_object_info(oid)
            if info and agent_id in info["participants"]:
                data = await self.shared_memory.read(oid, agent_id)
                if data:
                    results["shared"].append({"id": oid, "data": data})

        # crystallized memory
        crit = {"participant": agent_id, "min_strength": query.get("min_strength", 0.5)}
        crystals = await self.crystallized_store.search(crit)
        results["crystallized"] = [
            {
                "id": c.id,
                "type": c.type.value,
                "strength": c.strength,
                "content": c.content,
            }
            for c in crystals[:10]
        ]

        # collective insights
        filters = {"contributor": agent_id, "min_confidence": query.get("min_confidence", 0.6)}
        insights = await self.collective_engine.get_insights(filters)
        results["insights"] = [
            {"id": i.id, "type": i.type.value, "description": i.description, "impact": i.impact_score}
            for i in insights[:10]
        ]

        return results

    # ------------------------------------------------------------------
    # statistics
    # ------------------------------------------------------------------
    def get_memory_stats(self):
        return {
            "working_memory": {aid: wm.get_usage() for aid, wm in self.working_memory_agents.items()},
            "shared_memory": self.shared_memory.get_stats(),
            "crystallized_memory": self.crystallized_store.get_stats(),
            "collective_memory": self.collective_engine.get_network_stats(),
        }


# ======================================================================
# -- END OF FILE --
# ======================================================================
