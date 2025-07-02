"""
Shared Memory Layer Implementation
==================================

This module implements the Shared Memory layer for inter-agent communication
and collaboration in the Enhanced-CSP system.
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import logging
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)


class ConsistencyLevel(Enum):
    """Consistency levels for shared memory operations"""
    EVENTUAL = "eventual"
    STRONG = "strong"
    CAUSAL = "causal"


@dataclass
class SharedObject:
    """Represents a shared memory object"""
    id: str
    type: str
    data: Any
    participants: Set[str]
    version: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    locks: Set[str] = field(default_factory=set)  # Agent IDs holding locks
    
    def update_version(self):
        """Increment version and update timestamp"""
        self.version += 1
        self.updated_at = datetime.now()


class SyncProtocol(Enum):
    """Synchronization protocols"""
    GOSSIP = "gossip"
    BROADCAST = "broadcast"
    CONSENSUS = "consensus"
    CRDT = "crdt"  # Conflict-free Replicated Data Type


class SyncManager:
    """
    Manages synchronization between agents for shared memory consistency.
    Implements various consistency protocols.
    """
    
    def __init__(self, consistency_level: ConsistencyLevel = ConsistencyLevel.EVENTUAL):
        self.consistency_level = consistency_level
        self._sync_queues: Dict[str, asyncio.Queue] = {}
        self._sync_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self._version_vectors: Dict[str, Dict[str, int]] = defaultdict(dict)
        self._sync_tasks: List[asyncio.Task] = []
        self._running = False
        
        logger.info(f"SyncManager initialized with {consistency_level.value} consistency")
    
    async def start(self):
        """Start synchronization services"""
        self._running = True
        # Start sync protocol handlers
        self._sync_tasks.append(asyncio.create_task(self._gossip_protocol()))
        self._sync_tasks.append(asyncio.create_task(self._consensus_protocol()))
    
    async def stop(self):
        """Stop synchronization services"""
        self._running = False
        for task in self._sync_tasks:
            task.cancel()
        await asyncio.gather(*self._sync_tasks, return_exceptions=True)
    
    def register_agent(self, agent_id: str):
        """Register an agent for synchronization"""
        if agent_id not in self._sync_queues:
            self._sync_queues[agent_id] = asyncio.Queue()
            self._version_vectors[agent_id] = {}
    
    def unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        if agent_id in self._sync_queues:
            del self._sync_queues[agent_id]
            del self._version_vectors[agent_id]
    
    async def propagate_update(self, object_id: str, update_data: Dict[str, Any], 
                             source_agent: str, protocol: SyncProtocol = SyncProtocol.GOSSIP):
        """Propagate an update to other agents"""
        update_message = {
            'object_id': object_id,
            'data': update_data,
            'source': source_agent,
            'timestamp': datetime.now().isoformat(),
            'protocol': protocol.value
        }
        
        if protocol == SyncProtocol.BROADCAST:
            await self._broadcast_update(update_message)
        elif protocol == SyncProtocol.GOSSIP:
            await self._gossip_update(update_message)
        elif protocol == SyncProtocol.CONSENSUS:
            await self._consensus_update(update_message)
    
    async def _broadcast_update(self, update_message: Dict[str, Any]):
        """Broadcast update to all agents"""
        tasks = []
        for agent_id, queue in self._sync_queues.items():
            if agent_id != update_message['source']:
                tasks.append(queue.put(update_message))
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _gossip_update(self, update_message: Dict[str, Any]):
        """Gossip protocol - randomly propagate to subset of agents"""
        import random
        agents = list(self._sync_queues.keys())
        if update_message['source'] in agents:
            agents.remove(update_message['source'])
        
        # Select random subset (fanout = 3)
        selected = random.sample(agents, min(3, len(agents)))
        for agent_id in selected:
            await self._sync_queues[agent_id].put(update_message)
    
    async def _consensus_update(self, update_message: Dict[str, Any]):
        """Consensus protocol - ensure majority agreement"""
        # Simplified Raft-like consensus
        responses = []
        for agent_id, queue in self._sync_queues.items():
            if agent_id != update_message['source']:
                # Request vote
                response = await self._request_vote(agent_id, update_message)
                responses.append(response)
        
        # Check if majority agrees
        votes = sum(1 for r in responses if r)
        if votes >= len(self._sync_queues) // 2:
            await self._broadcast_update(update_message)
    
    async def _request_vote(self, agent_id: str, update_message: Dict[str, Any]) -> bool:
        """Request vote from agent for consensus"""
        # Simplified voting mechanism
        return True  # In real implementation, would check version vectors, etc.
    
    async def _gossip_protocol(self):
        """Background gossip protocol handler"""
        while self._running:
            try:
                await asyncio.sleep(1)  # Gossip interval
                # Implement anti-entropy gossip
                for agent_id, versions in self._version_vectors.items():
                    # Compare version vectors and sync differences
                    pass
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Gossip protocol error: {e}")
    
    async def _consensus_protocol(self):
        """Background consensus protocol handler"""
        while self._running:
            try:
                await asyncio.sleep(5)  # Consensus check interval
                # Implement consensus verification
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Consensus protocol error: {e}")
    
    def add_sync_callback(self, object_id: str, callback: Callable):
        """Add callback for object synchronization events"""
        self._sync_callbacks[object_id].append(callback)
    
    def get_sync_status(self) -> Dict[str, Any]:
        """Get synchronization status"""
        return {
            'consistency_level': self.consistency_level.value,
            'active_agents': len(self._sync_queues),
            'sync_queues': {aid: q.qsize() for aid, q in self._sync_queues.items()},
            'version_vectors': dict(self._version_vectors)
        }


class SharedMemory:
    """
    Shared Memory implementation for inter-agent communication.
    Provides distributed shared objects with configurable consistency.
    """
    
    def __init__(self, namespace: str = "default", 
                 consistency: ConsistencyLevel = ConsistencyLevel.EVENTUAL):
        self.namespace = namespace
        self.consistency = consistency
        
        # Shared object storage
        self._objects: Dict[str, SharedObject] = {}
        self._object_locks = defaultdict(threading.Lock)
        
        # Sync manager
        self.sync_manager = SyncManager(consistency)
        
        # Subscriptions
        self._subscriptions: Dict[str, Set[str]] = defaultdict(set)  # object_id -> agent_ids
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Statistics
        self.stats = {
            'total_objects': 0,
            'total_reads': 0,
            'total_writes': 0,
            'sync_operations': 0,
            'conflicts_resolved': 0
        }
        
        logger.info(f"SharedMemory initialized for namespace '{namespace}'")
    
    async def start(self):
        """Start shared memory services"""
        await self.sync_manager.start()
    
    async def stop(self):
        """Stop shared memory services"""
        await self.sync_manager.stop()
    
    def create_object(self, object_id: str, object_type: str, 
                     initial_data: Any, participants: Set[str]) -> bool:
        """Create a new shared object"""
        try:
            if object_id in self._objects:
                logger.warning(f"Object {object_id} already exists")
                return False
            
            shared_obj = SharedObject(
                id=object_id,
                type=object_type,
                data=initial_data,
                participants=participants
            )
            
            with self._object_locks[object_id]:
                self._objects[object_id] = shared_obj
                self.stats['total_objects'] += 1
            
            # Register participants with sync manager
            for agent_id in participants:
                self.sync_manager.register_agent(agent_id)
            
            # Notify participants
            asyncio.create_task(self._notify_participants(object_id, 'created', participants))
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating shared object: {e}")
            return False
    
    async def read(self, object_id: str, agent_id: str) -> Optional[Any]:
        """Read shared object data"""
        if object_id not in self._objects:
            return None
        
        obj = self._objects[object_id]
        if agent_id not in obj.participants:
            logger.warning(f"Agent {agent_id} not authorized for object {object_id}")
            return None
        
        self.stats['total_reads'] += 1
        
        # Return copy to prevent direct modification
        return self._deep_copy(obj.data)
    
    async def write(self, object_id: str, agent_id: str, 
                   data: Any, merge_strategy: str = "overwrite") -> bool:
        """Write to shared object"""
        if object_id not in self._objects:
            return False
        
        obj = self._objects[object_id]
        if agent_id not in obj.participants:
            logger.warning(f"Agent {agent_id} not authorized for object {object_id}")
            return False
        
        with self._object_locks[object_id]:
            old_data = obj.data
            
            if merge_strategy == "overwrite":
                obj.data = data
            elif merge_strategy == "merge":
                obj.data = self._merge_data(old_data, data)
            elif merge_strategy == "crdt":
                obj.data = self._crdt_merge(old_data, data)
            
            obj.update_version()
            self.stats['total_writes'] += 1
        
        # Propagate update
        await self.sync_manager.propagate_update(
            object_id,
            {'data': data, 'version': obj.version},
            agent_id,
            SyncProtocol.BROADCAST if self.consistency == ConsistencyLevel.STRONG else SyncProtocol.GOSSIP
        )
        
        # Notify subscribers
        await self._notify_subscribers(object_id, 'updated', agent_id)
        
        return True
    
    def subscribe(self, object_id: str, agent_id: str, callback: Optional[Callable] = None):
        """Subscribe to object updates"""
        if object_id in self._objects:
            obj = self._objects[object_id]
            if agent_id in obj.participants:
                self._subscriptions[object_id].add(agent_id)
                if callback:
                    self._callbacks[object_id].append(callback)
                return True
        return False
    
    def unsubscribe(self, object_id: str, agent_id: str):
        """Unsubscribe from object updates"""
        if object_id in self._subscriptions:
            self._subscriptions[object_id].discard(agent_id)
    
    async def acquire_lock(self, object_id: str, agent_id: str, 
                          timeout: float = 5.0) -> bool:
        """Acquire exclusive lock on object"""
        if object_id not in self._objects:
            return False
        
        obj = self._objects[object_id]
        if agent_id not in obj.participants:
            return False
        
        # Simple lock implementation - could be enhanced with distributed locking
        start_time = time.time()
        while obj.locks and time.time() - start_time < timeout:
            await asyncio.sleep(0.1)
        
        if not obj.locks:
            obj.locks.add(agent_id)
            return True
        return False
    
    def release_lock(self, object_id: str, agent_id: str):
        """Release lock on object"""
        if object_id in self._objects:
            self._objects[object_id].locks.discard(agent_id)
    
    def get_object_info(self, object_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a shared object"""
        if object_id not in self._objects:
            return None
        
        obj = self._objects[object_id]
        return {
            'id': obj.id,
            'type': obj.type,
            'participants': list(obj.participants),
            'version': obj.version,
            'created_at': obj.created_at.isoformat(),
            'updated_at': obj.updated_at.isoformat(),
            'locked_by': list(obj.locks),
            'data_size': len(str(obj.data).encode('utf-8'))
        }
    
    def list_objects(self, participant: Optional[str] = None) -> List[str]:
        """List shared objects, optionally filtered by participant"""
        if participant:
            return [oid for oid, obj in self._objects.items() 
                   if participant in obj.participants]
        return list(self._objects.keys())
    
    async def _notify_participants(self, object_id: str, event: str, participants: Set[str]):
        """Notify participants of object events"""
        # Implementation depends on messaging system
        pass
    
    async def _notify_subscribers(self, object_id: str, event: str, source_agent: str):
        """Notify subscribers of object updates"""
        if object_id in self._callbacks:
            for callback in self._callbacks[object_id]:
                try:
                    await callback(object_id, event, source_agent)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
    
    def _deep_copy(self, data: Any) -> Any:
        """Create deep copy of data"""
        import copy
        return copy.deepcopy(data)
    
    def _merge_data(self, old_data: Any, new_data: Any) -> Any:
        """Merge data using simple strategy"""
        if isinstance(old_data, dict) and isinstance(new_data, dict):
            merged = old_data.copy()
            merged.update(new_data)
            return merged
        return new_data
    
    def _crdt_merge(self, old_data: Any, new_data: Any) -> Any:
        """Merge using CRDT (Conflict-free Replicated Data Type) strategy"""
        # Simplified CRDT merge - in practice would use proper CRDT types
        return self._merge_data(old_data, new_data)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get shared memory statistics"""
        return {
            **self.stats,
            'sync_status': self.sync_manager.get_sync_status(),
            'active_subscriptions': sum(len(subs) for subs in self._subscriptions.values())
        }