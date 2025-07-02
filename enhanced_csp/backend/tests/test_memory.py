"""
Test suite for the Enhanced CSP Memory Architecture
"""

import asyncio
import pytest
from datetime import timedelta
import numpy as np

# Assuming the memory module is in the backend/memory directory
from backend.memory import (
    WorkingMemory, WorkingCache,
    SharedMemory, SyncManager, ConsistencyLevel,
    CrystallizedStore, CrystalDB, CrystalType,
    CollectiveInsightEngine, PatternMiner, InsightType,
    MemoryCoordinator
)


class TestWorkingMemory:
    """Test Working Memory implementation"""
    
    @pytest.mark.asyncio
    async def test_basic_storage_retrieval(self):
        """Test basic store and retrieve operations"""
        wm = WorkingMemory("agent_1", capacity_mb=10)
        await wm.start()
        
        # Store data
        assert wm.store("key1", "value1")
        assert wm.store("key2", {"data": "complex"})
        
        # Retrieve data
        assert wm.retrieve("key1") == "value1"
        assert wm.retrieve("key2") == {"data": "complex"}
        assert wm.retrieve("nonexistent") is None
        
        await wm.stop()
    
    @pytest.mark.asyncio
    async def test_ttl_expiration(self):
        """Test TTL-based expiration"""
        wm = WorkingMemory("agent_2", capacity_mb=10)
        await wm.start()
        
        # Store with short TTL
        wm.store("temp_key", "temp_value", ttl=timedelta(seconds=1))
        
        # Should exist immediately
        assert wm.retrieve("temp_key") == "temp_value"
        
        # Wait for expiration
        await asyncio.sleep(1.5)
        
        # Should be expired
        assert wm.retrieve("temp_key") is None
        
        await wm.stop()
    
    @pytest.mark.asyncio
    async def test_capacity_limit(self):
        """Test memory capacity limits"""
        wm = WorkingMemory("agent_3", capacity_mb=0.001)  # 1KB
        await wm.start()
        
        # Store small items
        assert wm.store("small1", "x")
        assert wm.store("small2", "y")
        
        # Try to store large item that exceeds capacity
        large_data = "x" * 2000  # ~2KB
        assert not wm.store("large", large_data)
        
        await wm.stop()
    
    def test_working_cache(self):
        """Test WorkingCache functionality"""
        cache = WorkingCache(max_size=3)
        
        # Add items
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        
        # Access items
        assert cache.get("a") == 1
        assert cache.get("b") == 2
        
        # Add item that triggers eviction
        cache.put("d", 4)
        
        # Least recently used (c) should be evicted
        assert cache.get("c") is None
        assert cache.get("d") == 4


class TestSharedMemory:
    """Test Shared Memory implementation"""
    
    @pytest.mark.asyncio
    async def test_create_and_access_object(self):
        """Test creating and accessing shared objects"""
        sm = SharedMemory("test_namespace")
        await sm.start()
        
        # Create shared object
        participants = {"agent_1", "agent_2"}
        assert sm.create_object("obj1", "data", {"value": 42}, participants)
        
        # Read from authorized agents
        data = await sm.read("obj1", "agent_1")
        assert data == {"value": 42}
        
        data = await sm.read("obj1", "agent_2")
        assert data == {"value": 42}
        
        # Unauthorized access
        data = await sm.read("obj1", "agent_3")
        assert data is None
        
        await sm.stop()
    
    @pytest.mark.asyncio
    async def test_write_and_sync(self):
        """Test writing and synchronization"""
        sm = SharedMemory("test_namespace", consistency=ConsistencyLevel.STRONG)
        await sm.start()
        
        participants = {"agent_1", "agent_2"}
        sm.create_object("obj2", "counter", {"count": 0}, participants)
        
        # Write from agent_1
        await sm.write("obj2", "agent_1", {"count": 5})
        
        # Read from agent_2 should see update
        data = await sm.read("obj2", "agent_2")
        assert data["count"] == 5
        
        await sm.stop()
    
    @pytest.mark.asyncio
    async def test_locking(self):
        """Test object locking"""
        sm = SharedMemory("test_namespace")
        await sm.start()
        
        participants = {"agent_1", "agent_2"}
        sm.create_object("obj3", "resource", {"data": "shared"}, participants)
        
        # Acquire lock
        assert await sm.acquire_lock("obj3", "agent_1", timeout=1.0)
        
        # Second agent cannot acquire lock
        assert not await sm.acquire_lock("obj3", "agent_2", timeout=0.5)
        
        # Release lock
        sm.release_lock("obj3", "agent_1")
        
        # Now second agent can acquire
        assert await sm.acquire_lock("obj3", "agent_2", timeout=1.0)
        
        await sm.stop()


class TestCrystallizedMemory:
    """Test Crystallized Memory implementation"""
    
    @pytest.mark.asyncio
    async def test_crystal_formation(self):
        """Test crystal formation from interactions"""
        cs = CrystallizedStore("test_namespace")
        await cs.start()
        
        # Create similar interactions
        interactions = []
        for i in range(5):
            interaction = {
                "id": f"interaction_{i}",
                "participants": ["agent_1", "agent_2"],
                "action": "collaborate",
                "context": {"task": "problem_solving"},
                "outcomes": ["success"]
            }
            interactions.append(interaction)
        
        # Process interactions
        crystal_ids = []
        for interaction in interactions:
            crystal_id = await cs.process_interaction(interaction)
            if crystal_id:
                crystal_ids.append(crystal_id)
        
        # Should form at least one crystal from repeated pattern
        assert len(crystal_ids) > 0
        
        # Retrieve crystal
        crystal = await cs.retrieve(crystal_ids[0])
        assert crystal is not None
        assert crystal.type in [CrystalType.PATTERN, CrystalType.SOLUTION]
        assert "agent_1" in crystal.participants
        
        await cs.stop()
    
    @pytest.mark.asyncio
    async def test_crystal_reinforcement(self):
        """Test crystal reinforcement"""
        cs = CrystallizedStore("test_namespace")
        await cs.start()
        
        # Create initial interaction
        interaction = {
            "participants": ["agent_1", "agent_2"],
            "action": "learn",
            "context": {"topic": "optimization"}
        }
        
        # Process multiple times to form and reinforce
        crystal_id = None
        for _ in range(10):
            result = await cs.process_interaction(interaction)
            if result:
                crystal_id = result
        
        if crystal_id:
            crystal = await cs.retrieve(crystal_id)
            assert crystal.strength > 0.5  # Should be reinforced
            assert crystal.reinforcement_count > 0
        
        await cs.stop()
    
    @pytest.mark.asyncio
    async def test_crystal_search(self):
        """Test crystal search functionality"""
        cs = CrystallizedStore("test_namespace")
        await cs.start()
        
        # Create crystals with different participants
        interactions = [
            {
                "participants": ["agent_1", "agent_2"],
                "action": "solve",
                "context": {"problem": "A"}
            },
            {
                "participants": ["agent_2", "agent_3"],
                "action": "analyze",
                "context": {"data": "B"}
            }
        ]
        
        for interaction in interactions * 5:  # Repeat to trigger formation
            await cs.process_interaction(interaction)
        
        # Search for agent_2's crystals
        results = await cs.search({"participant": "agent_2"})
        assert len(results) > 0
        
        # All results should include agent_2
        for crystal in results:
            assert "agent_2" in crystal.participants
        
        await cs.stop()


class TestCollectiveMemory:
    """Test Collective Memory implementation"""
    
    @pytest.mark.asyncio
    async def test_insight_generation(self):
        """Test insight generation from patterns"""
        engine = CollectiveInsightEngine()
        await engine.start()
        
        # Create memory data with patterns
        memory_data = {
            "working_memory_interactions": [
                {
                    "participants": ["agent_1", "agent_2", "agent_3"],
                    "action": "collaborate",
                    "context": {"task": "optimization"}
                } for _ in range(10)
            ],
            "crystallized_memories": [],
            "shared_memory_states": []
        }
        
        # Process to generate insights
        insight_ids = await engine.process_memory_data(memory_data)
        
        # Should generate insights from patterns
        assert len(insight_ids) > 0
        
        # Get insights
        insights = await engine.get_insights()
        assert len(insights) > 0
        
        # Check insight properties
        insight = insights[0]
        assert insight.type in InsightType
        assert len(insight.contributors) >= 2
        assert insight.confidence > 0
        
        await engine.stop()
    
    @pytest.mark.asyncio
    async def test_pattern_mining(self):
        """Test pattern mining capabilities"""
        miner = PatternMiner()
        
        # Create data sources
        data_sources = {
            "interactions": [
                {
                    "participants": ["a", "b"],
                    "strength": 0.8,
                    "content": "data1"
                } for _ in range(20)
            ]
        }
        
        # Mine patterns
        patterns = await miner.mine_patterns(data_sources)
        
        # Should find patterns
        assert len(patterns) > 0
        
        # Check pattern types
        pattern_types = {p["type"] for p in patterns}
        assert len(pattern_types) > 0  # Should find at least one type
    
    @pytest.mark.asyncio
    async def test_insight_network(self):
        """Test insight network connections"""
        engine = CollectiveInsightEngine()
        await engine.start()
        
        # Generate multiple related insights
        for i in range(5):
            memory_data = {
                "working_memory_interactions": [
                    {
                        "participants": ["agent_1", "agent_2"],
                        "action": f"action_{i % 2}",  # Alternate actions
                        "context": {"phase": i}
                    } for _ in range(5)
                ],
                "crystallized_memories": [],
                "shared_memory_states": []
            }
            await engine.process_memory_data(memory_data)
        
        # Check network formation
        stats = engine.get_network_stats()
        assert stats["total_insights"] > 0
        
        await engine.stop()


class TestMemoryCoordinator:
    """Test Memory Coordinator integration"""
    
    @pytest.mark.asyncio
    async def test_full_memory_flow(self):
        """Test complete memory flow through all layers"""
        coordinator = MemoryCoordinator("test")
        await coordinator.start()
        
        # Register agents
        coordinator.register_agent("agent_1", working_memory_mb=10)
        coordinator.register_agent("agent_2", working_memory_mb=10)
        
        # Process interaction
        interaction = {
            "id": "test_interaction_1",
            "participants": ["agent_1", "agent_2"],
            "action": "collaborate",
            "context": {"task": "integration_test"},
            "timestamp": "2024-01-01T00:00:00"
        }
        
        result = await coordinator.process_interaction(interaction)
        
        # Check processing results
        assert "shared" in result
        assert result["shared"] is True  # Multi-agent interaction
        
        # Query memory
        query_result = await coordinator.query_memory("agent_1", {"key": "interaction_test_interaction_1"})
        
        # Should have data in working memory
        assert query_result["working"] is not None
        
        # Get stats
        stats = coordinator.get_memory_stats()
        assert "working_memory" in stats
        assert "agent_1" in stats["working_memory"]
        
        await coordinator.stop()
    
    @pytest.mark.asyncio
    async def test_cross_layer_query(self):
        """Test querying across all memory layers"""
        coordinator = MemoryCoordinator("test")
        await coordinator.start()
        
        # Register agent
        coordinator.register_agent("agent_1")
        
        # Process multiple interactions to populate all layers
        for i in range(15):
            interaction = {
                "id": f"interaction_{i}",
                "participants": ["agent_1"],
                "action": "process",
                "context": {"iteration": i}
            }
            await coordinator.process_interaction(interaction)
        
        # Force collective analysis
        await coordinator._run_collective_analysis()
        
        # Query across layers
        results = await coordinator.query_memory("agent_1", {
            "min_strength": 0.3,
            "min_confidence": 0.5
        })
        
        # Should have results from multiple layers
        assert results["working"] is not None or len(results["shared"]) > 0
        # May have crystallized memories and insights if patterns formed
        
        await coordinator.stop()


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])