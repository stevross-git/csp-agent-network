"""
Enhanced CSP Memory Architecture
================================

Four-layer memory system for advanced AI agent collaboration:
1. Working Memory - Fast, temporary storage for active processes
2. Shared Memory - Inter-agent communication and synchronization
3. Crystallized Memory - Persistent pattern storage and knowledge formation
4. Collective Memory - Emergent intelligence and network-wide insights
"""

from .working_memory import WorkingMemory, WorkingCache
from .shared_memory import SharedMemory, SyncManager, ConsistencyLevel, SyncProtocol
from .crystallized_memory import CrystallizedStore, CrystalDB, MemoryCrystal, CrystalType
from .collective_memory import CollectiveInsightEngine, PatternMiner, CollectiveInsight, InsightType

__version__ = "1.0.0"

__all__ = [
    # Working Memory Layer
    "WorkingMemory",
    "WorkingCache",
    
    # Shared Memory Layer
    "SharedMemory",
    "SyncManager",
    "ConsistencyLevel",
    "SyncProtocol",
    
    # Crystallized Memory Layer
    "CrystallizedStore",
    "CrystalDB",
    "MemoryCrystal",
    "CrystalType",
    
    # Collective Memory Layer
    "CollectiveInsightEngine",
    "PatternMiner",
    "CollectiveInsight",
    "InsightType",
    
    # Coordinator
    "MemoryCoordinator"
]


class MemoryCoordinator:
    """
    Coordinates all four memory layers for seamless operation.
    Manages data flow and interactions between layers.
    """
    
    def __init__(self, namespace: str = "default"):
        self.namespace = namespace
        
        
        
        # Initialize all memory layers
        self.working_memory_agents = {}  # agent_id -> WorkingMemory
        self.shared_memory = SharedMemory(namespace)
        self.crystallized_store = CrystallizedStore(namespace)
        self.collective_engine = CollectiveInsightEngine()
        
        # Coordination queues
        self._interaction_queue = []
        self._coordination_running = False
        
    async def start(self):
        """Start all memory layers"""
        await self.shared_memory.start()
        await self.crystallized_store.start()
        await self.collective_engine.start()
        self._coordination_running = True
        
    async def stop(self):
        """Stop all memory layers"""
        self._coordination_running = False
        
        # Stop all working memories
        for wm in self.working_memory_agents.values():
            await wm.stop()
        
        await self.shared_memory.stop()
        await self.crystallized_store.stop()
        await self.collective_engine.stop()
    
    def register_agent(self, agent_id: str, working_memory_mb: float = 256):
        """Register a new agent with the memory system"""
        # Create working memory for agent
        self.working_memory_agents[agent_id] = WorkingMemory(agent_id, working_memory_mb)
        
        # Register with shared memory sync
        self.shared_memory.sync_manager.register_agent(agent_id)
        
        return self.working_memory_agents[agent_id]
    
    async def process_interaction(self, interaction_data: dict):
        """Process an interaction through all memory layers"""
        participants = interaction_data.get('participants', [])
        
        # 1. Store in working memory of participants
        for agent_id in participants:
            if agent_id in self.working_memory_agents:
                wm = self.working_memory_agents[agent_id]
                wm.store(f"interaction_{interaction_data.get('id', '')}", interaction_data)
        
        # 2. Check if interaction should be shared
        if len(participants) > 1:
            # Create shared memory object
            object_id = f"shared_{interaction_data.get('id', '')}"
            self.shared_memory.create_object(
                object_id,
                "interaction",
                interaction_data,
                set(participants)
            )
        
        # 3. Process for crystallization
        crystal_id = await self.crystallized_store.process_interaction(interaction_data)
        
        # 4. Add to collective analysis queue
        self._interaction_queue.append(interaction_data)
        
        # 5. Trigger collective analysis if queue is large enough
        if len(self._interaction_queue) >= 10:
            await self._run_collective_analysis()
        
        return {
            'crystal_formed': crystal_id is not None,
            'crystal_id': crystal_id,
            'shared': len(participants) > 1
        }
    
    async def _run_collective_analysis(self):
        """Run collective analysis on queued interactions"""
        if not self._interaction_queue:
            return
        
        # Prepare memory data for collective engine
        memory_data = {
            'working_memory_interactions': self._interaction_queue[-100:],
            'crystallized_memories': await self._get_recent_crystals(),
            'shared_memory_states': self._get_shared_memory_states()
        }
        
        # Generate insights
        insight_ids = await self.collective_engine.process_memory_data(memory_data)
        
        # Clear processed interactions
        self._interaction_queue = self._interaction_queue[-50:]  # Keep last 50
        
        return insight_ids
    
    async def _get_recent_crystals(self):
        """Get recent crystals for collective analysis"""
        crystals = []
        for crystal_id in list(self.crystallized_store.crystals.keys())[-50:]:
            crystal = self.crystallized_store.crystals[crystal_id]
            crystals.append({
                'id': crystal.id,
                'type': crystal.type.value,
                'participants': crystal.participants,
                'strength': crystal.strength,
                'content': crystal.content
            })
        return crystals
    
    def _get_shared_memory_states(self):
        """Get shared memory states for collective analysis"""
        states = []
        for object_id in self.shared_memory.list_objects():
            info = self.shared_memory.get_object_info(object_id)
            if info:
                states.append({
                    'id': object_id,
                    'participants': info['participants'],
                    'version': info['version'],
                    'status': 'synced' if not info['locked_by'] else 'locked'
                })
        return states
    
    async def query_memory(self, agent_id: str, query: dict):
        """Query across all memory layers"""
        results = {
            'working': None,
            'shared': [],
            'crystallized': [],
            'insights': []
        }
        
        # Query working memory
        if agent_id in self.working_memory_agents:
            wm = self.working_memory_agents[agent_id]
            key = query.get('key')
            if key:
                results['working'] = wm.retrieve(key)
        
        shared_objects = self.shared_memory.list_objects()
        for obj_id in shared_objects[:10]:  # Limit results
            info = self.shared_memory.get_object_info(obj_id)
            if info and agent_id in info['participants']:
                data = await self.shared_memory.read(obj_id, agent_id)
                if data:
                    results['shared'].append({
                        'id': obj_id,
                        'data': data
                    })

        
        # Query crystallized memory
        crystal_query = {
            'participant': agent_id,
            'min_strength': query.get('min_strength', 0.5)
        }
        crystals = await self.crystallized_store.search(crystal_query)
        results['crystallized'] = [
            {
                'id': c.id,
                'type': c.type.value,
                'strength': c.strength,
                'content': c.content
            }
            for c in crystals[:10]  # Limit results
        ]
        
        # Query collective insights
        insight_filters = {
            'contributor': agent_id,
            'min_confidence': query.get('min_confidence', 0.6)
        }
        insights = await self.collective_engine.get_insights(insight_filters)
        results['insights'] = [
            {
                'id': i.id,
                'type': i.type.value,
                'description': i.description,
                'impact': i.impact_score
            }
            for i in insights[:10]  # Limit results
        ]
        
        return results
    
    def get_memory_stats(self):
        """Get statistics from all memory layers"""
        stats = {
            'working_memory': {},
            'shared_memory': {},
            'crystallized_memory': {},
            'collective_memory': {}
        }
        
        # Working memory stats
        for agent_id, wm in self.working_memory_agents.items():
            stats['working_memory'][agent_id] = wm.get_usage()
        
        # Shared memory stats
        stats['shared_memory'] = self.shared_memory.get_stats()
        
        # Crystallized memory stats
        stats['crystallized_memory'] = self.crystallized_store.get_stats()
        
        # Collective memory stats
        stats['collective_memory'] = self.collective_engine.get_network_stats()
        
        return stats