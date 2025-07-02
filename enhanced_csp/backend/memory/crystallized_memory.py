"""
Crystallized Memory Layer Implementation
========================================

This module implements the Crystallized Memory layer for persistent,
structured memory formation from agent interactions.
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np
from collections import defaultdict
import pickle
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class CrystalType(Enum):
    """Types of memory crystals"""
    KNOWLEDGE = "knowledge"
    SKILL = "skill"
    PATTERN = "pattern"
    SOLUTION = "solution"
    RELATIONSHIP = "relationship"


@dataclass
class MemoryCrystal:
    """Represents a crystallized memory structure"""
    id: str
    type: CrystalType
    participants: List[str]
    strength: float  # 0.0 to 1.0
    content: Dict[str, Any]
    formation_time: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    reinforcement_count: int = 0
    decay_rate: float = 0.01
    connections: Set[str] = field(default_factory=set)  # Connected crystal IDs
    
    def reinforce(self, strength_delta: float = 0.1):
        """Reinforce this crystal, increasing its strength"""
        self.strength = min(1.0, self.strength + strength_delta)
        self.reinforcement_count += 1
        self.last_accessed = datetime.now()
        self.access_count += 1
    
    def decay(self):
        """Apply decay to crystal strength"""
        time_since_access = (datetime.now() - self.last_accessed).total_seconds() / 3600  # hours
        decay_amount = self.decay_rate * time_since_access
        self.strength = max(0.0, self.strength - decay_amount)
    
    def is_stable(self) -> bool:
        """Check if crystal is stable (high strength, frequently accessed)"""
        return self.strength > 0.7 and self.access_count > 5


class CrystalFormationEngine:
    """Engine for forming memory crystals from interactions"""
    
    def __init__(self):
        self.formation_threshold = 0.5
        self.pattern_buffer: List[Dict[str, Any]] = []
        self.pattern_window = 100  # Number of interactions to consider
        
    async def analyze_interaction(self, interaction_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze interaction for potential crystal formation"""
        self.pattern_buffer.append(interaction_data)
        
        # Keep buffer size limited
        if len(self.pattern_buffer) > self.pattern_window:
            self.pattern_buffer = self.pattern_buffer[-self.pattern_window:]
        
        # Check for crystallization patterns
        pattern_strength = await self._calculate_pattern_strength(interaction_data)
        
        if pattern_strength > self.formation_threshold:
            return await self._extract_crystal_content(interaction_data, pattern_strength)
        
        return None
    
    async def _calculate_pattern_strength(self, interaction_data: Dict[str, Any]) -> float:
        """Calculate the strength of pattern in recent interactions"""
        if len(self.pattern_buffer) < 3:
            return 0.0
        
        # Look for repeated patterns
        similar_count = 0
        for past_interaction in self.pattern_buffer[-10:]:
            similarity = self._calculate_similarity(interaction_data, past_interaction)
            if similarity > 0.7:
                similar_count += 1
        
        # Pattern strength based on repetition and recency
        pattern_strength = similar_count / 10.0
        
        # Boost for multi-agent patterns
        if len(interaction_data.get('participants', [])) > 2:
            pattern_strength *= 1.2
        
        return min(1.0, pattern_strength)
    
    def _calculate_similarity(self, data1: Dict[str, Any], data2: Dict[str, Any]) -> float:
        """Calculate similarity between two interactions"""
        # Simplified similarity calculation
        score = 0.0
        
        # Check participant overlap
        participants1 = set(data1.get('participants', []))
        participants2 = set(data2.get('participants', []))
        if participants1 and participants2:
            overlap = len(participants1.intersection(participants2)) / len(participants1.union(participants2))
            score += overlap * 0.3
        
        # Check action similarity
        if data1.get('action') == data2.get('action'):
            score += 0.3
        
        # Check context similarity
        context1 = data1.get('context', {})
        context2 = data2.get('context', {})
        common_keys = set(context1.keys()).intersection(set(context2.keys()))
        if common_keys:
            score += len(common_keys) / max(len(context1), len(context2)) * 0.4
        
        return score
    
    async def _extract_crystal_content(self, interaction_data: Dict[str, Any], 
                                     strength: float) -> Dict[str, Any]:
        """Extract content for crystal formation"""
        # Determine crystal type
        crystal_type = self._determine_crystal_type(interaction_data)
        
        # Extract relevant content
        content = {
            'core_pattern': interaction_data.get('action'),
            'context': interaction_data.get('context', {}),
            'outcomes': interaction_data.get('outcomes', []),
            'participants': interaction_data.get('participants', []),
            'formation_trigger': 'pattern_recognition',
            'metadata': {
                'interaction_count': len(self.pattern_buffer),
                'pattern_strength': strength
            }
        }
        
        return {
            'type': crystal_type,
            'content': content,
            'strength': strength
        }
    
    def _determine_crystal_type(self, interaction_data: Dict[str, Any]) -> CrystalType:
        """Determine the type of crystal based on interaction"""
        action = interaction_data.get('action', '').lower()
        
        if 'learn' in action or 'understand' in action:
            return CrystalType.KNOWLEDGE
        elif 'solve' in action or 'fix' in action:
            return CrystalType.SOLUTION
        elif 'perform' in action or 'execute' in action:
            return CrystalType.SKILL
        elif 'connect' in action or 'relate' in action:
            return CrystalType.RELATIONSHIP
        else:
            return CrystalType.PATTERN


class CrystalDB:
    """Database interface for crystallized memory storage"""
    
    def __init__(self, storage_path: str = "./crystallized_memory"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._index: Dict[str, Dict[str, Any]] = {}
        self._load_index()
    
    def _load_index(self):
        """Load crystal index from disk"""
        index_path = self.storage_path / "crystal_index.json"
        if index_path.exists():
            with open(index_path, 'r') as f:
                self._index = json.load(f)
    
    def _save_index(self):
        """Save crystal index to disk"""
        index_path = self.storage_path / "crystal_index.json"
        with open(index_path, 'w') as f:
            json.dump(self._index, f, indent=2)
    
    async def store_crystal(self, crystal: MemoryCrystal) -> bool:
        """Store a crystal to persistent storage"""
        try:
            # Store crystal data
            crystal_path = self.storage_path / f"{crystal.id}.pkl"
            with open(crystal_path, 'wb') as f:
                pickle.dump(crystal, f)
            
            # Update index
            self._index[crystal.id] = {
                'type': crystal.type.value,
                'participants': crystal.participants,
                'strength': crystal.strength,
                'formation_time': crystal.formation_time.isoformat(),
                'connections': list(crystal.connections)
            }
            self._save_index()
            
            return True
        except Exception as e:
            logger.error(f"Error storing crystal: {e}")
            return False
    
    async def retrieve_crystal(self, crystal_id: str) -> Optional[MemoryCrystal]:
        """Retrieve a crystal from storage"""
        try:
            crystal_path = self.storage_path / f"{crystal_id}.pkl"
            if crystal_path.exists():
                with open(crystal_path, 'rb') as f:
                    crystal = pickle.load(f)
                crystal.access_count += 1
                crystal.last_accessed = datetime.now()
                # Re-save with updated access info
                await self.store_crystal(crystal)
                return crystal
        except Exception as e:
            logger.error(f"Error retrieving crystal: {e}")
        return None
    
    async def search_crystals(self, criteria: Dict[str, Any]) -> List[str]:
        """Search for crystals matching criteria"""
        matching_ids = []
        
        for crystal_id, metadata in self._index.items():
            if self._matches_criteria(metadata, criteria):
                matching_ids.append(crystal_id)
        
        return matching_ids
    
    def _matches_criteria(self, metadata: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
        """Check if metadata matches search criteria"""
        for key, value in criteria.items():
            if key == 'type' and metadata.get('type') != value:
                return False
            elif key == 'participant' and value not in metadata.get('participants', []):
                return False
            elif key == 'min_strength' and metadata.get('strength', 0) < value:
                return False
        return True
    
    async def get_connected_crystals(self, crystal_id: str) -> List[str]:
        """Get IDs of crystals connected to given crystal"""
        if crystal_id in self._index:
            return self._index[crystal_id].get('connections', [])
        return []


class CrystallizedStore:
    """
    Main interface for the Crystallized Memory layer.
    Manages formation, storage, and retrieval of memory crystals.
    """
    
    def __init__(self, namespace: str = "default"):
        self.namespace = namespace
        self.crystals: Dict[str, MemoryCrystal] = {}
        self.formation_engine = CrystalFormationEngine()
        self.crystal_db = CrystalDB(f"./crystallized_memory/{namespace}")
        
        # Crystal network graph
        self.crystal_connections: Dict[str, Set[str]] = defaultdict(set)
        
        # Background tasks
        self._decay_task: Optional[asyncio.Task] = None
        self._consolidation_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.stats = {
            'total_crystals': 0,
            'crystals_formed': 0,
            'crystals_reinforced': 0,
            'crystals_decayed': 0,
            'average_strength': 0.0
        }
        self._pattern_counts: Dict[str, int] = defaultdict(int)
        self._pattern_to_crystal: Dict[str, str] = {}
        logger.info(f"CrystallizedStore initialized for namespace '{namespace}'")
    
    async def start(self):
        """Start crystallized memory services"""
        # Load existing crystals from storage
        await self._load_crystals()
        
        # Start background tasks
        self._decay_task = asyncio.create_task(self._decay_loop())
        self._consolidation_task = asyncio.create_task(self._consolidation_loop())
    
    async def stop(self):
        """Stop crystallized memory services"""
        # Cancel background tasks
        if self._decay_task:
            self._decay_task.cancel()
        if self._consolidation_task:
            self._consolidation_task.cancel()
        
        await asyncio.gather(
            self._decay_task,
            self._consolidation_task,
            return_exceptions=True
        )
    
    async def process_interaction(self, interaction_data: Dict[str, Any]) -> Optional[str]:
        """
        Form a new crystal after the same (action + participants) pattern
        is seen ≥ 3 times, or reinforce an existing crystal on every repeat.
        """
        participants = interaction_data.get("participants", [])
        action       = interaction_data.get("action", "")
        pattern_key  = f"{action}|{'|'.join(sorted(participants))}"

        # ── repeat counter ──────────────────────────────────────────────
        self._pattern_counts[pattern_key] += 1
        repeats = self._pattern_counts[pattern_key]

        # ── already crystallised? → reinforce ───────────────────────────
        if pattern_key in self._pattern_to_crystal:
            cid = self._pattern_to_crystal[pattern_key]
            crystal = await self.retrieve(cid)
            if crystal:
                crystal.reinforce(0.05)
                self.stats["crystals_reinforced"] += 1
                await self.crystal_db.store_crystal(crystal)
            return cid

        # ── form a new crystal after ≥ 3 repeats ────────────────────────
        if repeats < 3:
            return None

        crystal_id = f"crystal_{self.namespace}_{int(time.time()*1000)}"
        crystal_type = self.formation_engine._determine_crystal_type(interaction_data)
        crystal_content = {
            "core_pattern": action,
            "context": interaction_data.get("context", {}),
            "outcomes": interaction_data.get("outcomes", []),
            "participants": participants,
            "formation_trigger": "pattern_count",
            "metadata": {"interaction_count": repeats, "pattern_strength": 1.0},
        }

        crystal = MemoryCrystal(
            id=crystal_id,
            type=crystal_type,
            participants=participants,
            strength=1.0,
            content=crystal_content,
        )

        # link with existing crystals
        crystal.connections = await self._find_connections(crystal)

        # persist
        self.crystals[crystal_id] = crystal
        self._pattern_to_crystal[pattern_key] = crystal_id
        await self.crystal_db.store_crystal(crystal)

        self.stats["crystals_formed"]  += 1
        self.stats["total_crystals"]    = len(self.crystals)

        logger.info(f"Formed new crystal: {crystal_id} (type: {crystal.type.value})")
        return crystal_id

    
    async def retrieve(self, crystal_id: str) -> Optional[MemoryCrystal]:
        """Retrieve a crystal by ID"""
        if crystal_id in self.crystals:
            crystal = self.crystals[crystal_id]
            crystal.access_count += 1
            crystal.last_accessed = datetime.now()
            return crystal
        
        # Try loading from storage
        crystal = await self.crystal_db.retrieve_crystal(crystal_id)
        if crystal:
            self.crystals[crystal_id] = crystal
            return crystal
        
        return None
    
    async def search(self, query: Dict[str, Any]) -> List[MemoryCrystal]:
        """Search for crystals matching query"""
        crystal_ids = await self.crystal_db.search_crystals(query)
        crystals = []
        
        for crystal_id in crystal_ids:
            crystal = await self.retrieve(crystal_id)
            if crystal:
                crystals.append(crystal)
        
        return crystals
    
    async def get_network(self, crystal_id: str, depth: int = 1) -> Dict[str, Any]:
        """Get crystal network around a specific crystal"""
        if crystal_id not in self.crystals:
            return {}
        
        network = {'nodes': [], 'edges': []}
        visited = set()
        
        async def explore(cid: str, current_depth: int):
            if cid in visited or current_depth > depth:
                return
            
            visited.add(cid)
            crystal = await self.retrieve(cid)
            if not crystal:
                return
            
            # Add node
            network['nodes'].append({
                'id': cid,
                'type': crystal.type.value,
                'strength': crystal.strength,
                'participants': crystal.participants
            })
            
            # Add edges and explore connections
            for connected_id in crystal.connections:
                network['edges'].append({
                    'source': cid,
                    'target': connected_id
                })
                await explore(connected_id, current_depth + 1)
        
        await explore(crystal_id, 0)
        return network
    
    async def _load_crystals(self):
        """Load crystals from storage"""
        # Load most recently accessed crystals
        for crystal_id in list(self.crystal_db._index.keys())[:100]:
            crystal = await self.crystal_db.retrieve_crystal(crystal_id)
            if crystal:
                self.crystals[crystal_id] = crystal
    
    async def _find_connections(self, new_crystal: MemoryCrystal) -> Set[str]:
        """Find connections to existing crystals"""
        connections = set()
        
        for crystal_id, crystal in self.crystals.items():
            # Check participant overlap
            participant_overlap = set(new_crystal.participants).intersection(
                set(crystal.participants)
            )
            
            # Check content similarity
            if participant_overlap or self._content_similarity(
                new_crystal.content, crystal.content
            ) > 0.5:
                connections.add(crystal_id)
                crystal.connections.add(new_crystal.id)
        
        return connections
    
    def _content_similarity(self, content1: Dict[str, Any], 
                          content2: Dict[str, Any]) -> float:
        """Calculate content similarity between crystals"""
        # Simplified similarity - could use embeddings in production
        score = 0.0
        
        if content1.get('core_pattern') == content2.get('core_pattern'):
            score += 0.5
        
        # Context overlap
        ctx1 = set(content1.get('context', {}).keys())
        ctx2 = set(content2.get('context', {}).keys())
        if ctx1 and ctx2:
            score += len(ctx1.intersection(ctx2)) / len(ctx1.union(ctx2)) * 0.5
        
        return score
    
    async def _reinforce_related_crystals(self, interaction_data: Dict[str, Any]):
        """Reinforce crystals related to interaction"""
        participants = set(interaction_data.get('participants', []))
        
        for crystal in self.crystals.values():
            if set(crystal.participants).intersection(participants):
                crystal.reinforce(0.05)
                self.stats['crystals_reinforced'] += 1
    
    async def _decay_loop(self):
        """Background task to decay crystal strengths"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run hourly
                
                for crystal in list(self.crystals.values()):
                    crystal.decay()
                    
                    # Remove very weak crystals
                    if crystal.strength < 0.1:
                        del self.crystals[crystal.id]
                        self.stats['crystals_decayed'] += 1
                
                # Update average strength
                if self.crystals:
                    self.stats['average_strength'] = sum(
                        c.strength for c in self.crystals.values()
                    ) / len(self.crystals)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Decay loop error: {e}")
    
    async def _consolidation_loop(self):
        """Background task to consolidate related crystals"""
        while True:
            try:
                await asyncio.sleep(7200)  # Run every 2 hours
                
                # Find highly connected, similar crystals to merge
                consolidation_candidates = []
                
                for crystal in self.crystals.values():
                    if len(crystal.connections) > 5 and crystal.strength > 0.8:
                        consolidation_candidates.append(crystal)
                
                # Perform consolidation (simplified)
                # In production, would merge similar crystals into stronger ones
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Consolidation loop error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get crystallized memory statistics"""
        return {
            **self.stats,
            'crystal_types': defaultdict(int, {
                crystal.type.value: 1 
                for crystal in self.crystals.values()
            }),
            'stable_crystals': sum(
                1 for c in self.crystals.values() if c.is_stable()
            )
        }