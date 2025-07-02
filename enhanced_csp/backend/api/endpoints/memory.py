# backend/api/endpoints/memory.py
"""
Memory Management API Endpoints
===============================
FastAPI endpoints for the Enhanced CSP Memory System providing access to
all four memory layers: Working, Shared, Crystallized, and Collective Memory.
"""

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional, Any, Set
import asyncio
import logging
import json
from datetime import datetime
from pydantic import BaseModel, Field

# Import memory system components
from backend.memory import (
    MemoryCoordinator,
    WorkingMemory,
    ConsistencyLevel,
    CrystalType,
    InsightType
)
from backend.schemas.api_schemas import BaseResponse, ErrorResponse
from backend.auth.auth_system import get_current_user, UserInfo

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/memory", tags=["memory-management"])

# Global memory coordinator instance
memory_coordinator = MemoryCoordinator("production")

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.agent_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, agent_id: Optional[str] = None):
        await websocket.accept()
        self.active_connections.append(websocket)
        if agent_id:
            if agent_id not in self.agent_connections:
                self.agent_connections[agent_id] = set()
            self.agent_connections[agent_id].add(websocket)

    def disconnect(self, websocket: WebSocket, agent_id: Optional[str] = None):
        self.active_connections.remove(websocket)
        if agent_id and agent_id in self.agent_connections:
            self.agent_connections[agent_id].discard(websocket)

    async def broadcast(self, message: dict):
        """Broadcast to all connections"""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

    async def send_to_agent(self, agent_id: str, message: dict):
        """Send to specific agent connections"""
        if agent_id in self.agent_connections:
            for connection in self.agent_connections[agent_id]:
                try:
                    await connection.send_json(message)
                except:
                    pass

manager = ConnectionManager()

# ============================================================================
# Request/Response Models
# ============================================================================

class RegisterAgentRequest(BaseModel):
    agent_id: str = Field(..., description="Unique agent identifier")
    working_memory_mb: float = Field(256, description="Working memory capacity in MB")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional agent metadata")

class ProcessInteractionRequest(BaseModel):
    id: str = Field(..., description="Interaction ID")
    participants: List[str] = Field(..., description="List of participating agent IDs")
    action: str = Field(..., description="Action type")
    context: Dict[str, Any] = Field(..., description="Interaction context data")
    timestamp: Optional[str] = Field(None, description="ISO format timestamp")

class MemoryQueryRequest(BaseModel):
    key: Optional[str] = Field(None, description="Specific key to query")
    min_strength: float = Field(0.5, description="Minimum crystal strength")
    min_confidence: float = Field(0.6, description="Minimum insight confidence")
    layer_filter: Optional[List[str]] = Field(None, description="Filter by memory layers")
    time_range: Optional[Dict[str, str]] = Field(None, description="Time range filter")

class StoreMemoryRequest(BaseModel):
    agent_id: str = Field(..., description="Agent ID")
    key: str = Field(..., description="Memory key")
    value: Any = Field(..., description="Memory value")
    ttl_seconds: Optional[int] = Field(None, description="Time to live in seconds")
    share_with: Optional[List[str]] = Field(None, description="Agent IDs to share with")

class SharedMemoryRequest(BaseModel):
    object_id: str = Field(..., description="Shared object ID")
    object_type: str = Field(..., description="Object type")
    data: Dict[str, Any] = Field(..., description="Object data")
    participants: List[str] = Field(..., description="Authorized participants")
    consistency_level: str = Field("STRONG", description="Consistency level: STRONG, EVENTUAL, WEAK")

# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@router.on_event("startup")
async def startup_memory_system():
    """Initialize memory system on startup"""
    try:
        await memory_coordinator.start()
        logger.info("✅ Memory system started successfully")
    except Exception as e:
        logger.error(f"❌ Failed to start memory system: {e}")
        raise

@router.on_event("shutdown")
async def shutdown_memory_system():
    """Cleanup memory system on shutdown"""
    try:
        await memory_coordinator.stop()
        logger.info("✅ Memory system stopped successfully")
    except Exception as e:
        logger.error(f"❌ Error stopping memory system: {e}")

# ============================================================================
# Core Memory Management Endpoints
# ============================================================================

@router.get("/stats", response_model=Dict[str, Any])
async def get_memory_stats(
    layer: Optional[str] = Query(None, description="Specific layer to get stats for"),
    current_user: UserInfo = Depends(get_current_user)
):
    """
    Get comprehensive memory statistics across all layers.
    
    Returns statistics for:
    - Working Memory: capacity, usage, cache performance
    - Shared Memory: active objects, sync status
    - Crystallized Memory: crystal count, formation rate
    - Collective Memory: insights, patterns, network health
    """
    try:
        stats = memory_coordinator.get_memory_stats()
        
        # Filter by layer if specified
        if layer and layer in stats:
            return {
                "success": True,
                "layer": layer,
                "stats": stats[layer],
                "timestamp": datetime.now().isoformat()
            }
        
        # Add computed metrics
        total_memory_mb = sum(
            agent_stats.get('used_mb', 0) 
            for agent_stats in stats.get('working_memory', {}).values()
        )
        
        return {
            "success": True,
            "stats": stats,
            "summary": {
                "total_working_memory_mb": total_memory_mb,
                "active_agents": len(stats.get('working_memory', {})),
                "shared_objects": stats.get('shared_memory', {}).get('total_objects', 0),
                "crystals_formed": stats.get('crystallized_memory', {}).get('total_crystals', 0),
                "collective_insights": stats.get('collective_memory', {}).get('total_insights', 0)
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting memory stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/register-agent", response_model=Dict[str, Any])
async def register_agent(
    request: RegisterAgentRequest,
    current_user: UserInfo = Depends(get_current_user)
):
    """
    Register a new agent with the memory system.
    
    This creates a working memory allocation for the agent and
    registers it with the shared memory synchronization system.
    """
    try:
        # Check if agent already exists
        if request.agent_id in memory_coordinator.working_memory_agents:
            return {
                "success": False,
                "error": "Agent already registered",
                "agent_id": request.agent_id
            }
        
        # Register agent
        working_memory = memory_coordinator.register_agent(
            request.agent_id,
            request.working_memory_mb
        )
        
        # Broadcast registration event
        await manager.broadcast({
            "event": "agent_registered",
            "agent_id": request.agent_id,
            "working_memory_mb": request.working_memory_mb,
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "success": True,
            "agent_id": request.agent_id,
            "working_memory": {
                "capacity_mb": working_memory.capacity_bytes / (1024 * 1024),
                "used_mb": working_memory.used_bytes / (1024 * 1024),
                "cache_size": len(working_memory._cache._cache)
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error registering agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process-interaction", response_model=Dict[str, Any])
async def process_interaction(
    request: ProcessInteractionRequest,
    background_tasks: BackgroundTasks,
    current_user: UserInfo = Depends(get_current_user)
):
    """
    Process an interaction through all memory layers.
    
    This endpoint:
    1. Stores in working memory of participants
    2. Creates shared memory objects for multi-agent interactions
    3. Checks for crystallization patterns
    4. Triggers collective analysis when appropriate
    """
    try:
        # Add timestamp if not provided
        if not request.timestamp:
            request.timestamp = datetime.now().isoformat()
        
        interaction_data = request.dict()
        
        # Process through memory coordinator
        result = await memory_coordinator.process_interaction(interaction_data)
        
        # Broadcast interaction event to participants
        for agent_id in request.participants:
            await manager.send_to_agent(agent_id, {
                "event": "interaction_processed",
                "interaction_id": request.id,
                "result": result,
                "timestamp": datetime.now().isoformat()
            })
        
        # Schedule collective analysis in background if needed
        if len(memory_coordinator._interaction_queue) >= 8:
            background_tasks.add_task(
                memory_coordinator._run_collective_analysis
            )
        
        return {
            "success": True,
            "interaction_id": request.id,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error processing interaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query/{agent_id}", response_model=Dict[str, Any])
async def query_memory(
    agent_id: str,
    query: MemoryQueryRequest,
    current_user: UserInfo = Depends(get_current_user)
):
    """
    Query memory across all layers for a specific agent.
    
    Returns results from:
    - Working memory (if key specified)
    - Shared memory objects
    - Crystallized memories
    - Collective insights
    """
    try:
        # Verify agent exists
        if agent_id not in memory_coordinator.working_memory_agents:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        # Execute query
        results = await memory_coordinator.query_memory(agent_id, query.dict())
        
        # Filter by layers if specified
        if query.layer_filter:
            filtered_results = {}
            for layer in query.layer_filter:
                if layer in results:
                    filtered_results[layer] = results[layer]
            results = filtered_results
        
        return {
            "success": True,
            "agent_id": agent_id,
            "results": results,
            "query": query.dict(),
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error querying memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Working Memory Endpoints
# ============================================================================

@router.post("/working/store", response_model=Dict[str, Any])
async def store_working_memory(
    request: StoreMemoryRequest,
    current_user: UserInfo = Depends(get_current_user)
):
    """Store data in agent's working memory"""
    try:
        if request.agent_id not in memory_coordinator.working_memory_agents:
            raise HTTPException(status_code=404, detail=f"Agent {request.agent_id} not found")
        
        wm = memory_coordinator.working_memory_agents[request.agent_id]
        
        # Convert TTL to timedelta if provided
        ttl = None
        if request.ttl_seconds:
            from datetime import timedelta
            ttl = timedelta(seconds=request.ttl_seconds)
        
        # Store in working memory
        success = wm.store(request.key, request.value, ttl)
        
        if not success:
            return {
                "success": False,
                "error": "Failed to store - possibly out of memory",
                "agent_id": request.agent_id,
                "key": request.key
            }
        
        # Handle sharing if requested
        if request.share_with:
            object_id = f"shared_{request.key}_{datetime.now().timestamp()}"
            participants = set([request.agent_id] + request.share_with)
            memory_coordinator.shared_memory.create_object(
                object_id,
                "shared_data",
                {"key": request.key, "value": request.value},
                participants
            )
        
        return {
            "success": True,
            "agent_id": request.agent_id,
            "key": request.key,
            "stored_bytes": len(str(request.value).encode('utf-8')),
            "shared_with": request.share_with or [],
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error storing in working memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/working/{agent_id}/{key}", response_model=Dict[str, Any])
async def retrieve_working_memory(
    agent_id: str,
    key: str,
    current_user: UserInfo = Depends(get_current_user)
):
    """Retrieve data from agent's working memory"""
    try:
        if agent_id not in memory_coordinator.working_memory_agents:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        wm = memory_coordinator.working_memory_agents[agent_id]
        value = wm.retrieve(key)
        
        if value is None:
            return {
                "success": False,
                "error": "Key not found or expired",
                "agent_id": agent_id,
                "key": key
            }
        
        return {
            "success": True,
            "agent_id": agent_id,
            "key": key,
            "value": value,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving from working memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/working/{agent_id}/clear", response_model=Dict[str, Any])
async def clear_working_memory(
    agent_id: str,
    current_user: UserInfo = Depends(get_current_user)
):
    """Clear all working memory for an agent"""
    try:
        if agent_id not in memory_coordinator.working_memory_agents:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        wm = memory_coordinator.working_memory_agents[agent_id]
        previous_usage = wm.get_usage()
        
        wm.clear()
        
        return {
            "success": True,
            "agent_id": agent_id,
            "cleared": {
                "items": previous_usage['item_count'],
                "memory_mb": previous_usage['used_mb']
            },
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing working memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Shared Memory Endpoints
# ============================================================================

@router.post("/shared/create", response_model=Dict[str, Any])
async def create_shared_object(
    request: SharedMemoryRequest,
    current_user: UserInfo = Depends(get_current_user)
):
    """Create a new shared memory object"""
    try:
        # Parse consistency level
        consistency = ConsistencyLevel[request.consistency_level]
        
        # Create shared object
        success = memory_coordinator.shared_memory.create_object(
            request.object_id,
            request.object_type,
            request.data,
            set(request.participants)
        )
        
        if not success:
            return {
                "success": False,
                "error": "Object already exists",
                "object_id": request.object_id
            }
        
        # Notify participants
        for agent_id in request.participants:
            await manager.send_to_agent(agent_id, {
                "event": "shared_object_created",
                "object_id": request.object_id,
                "object_type": request.object_type,
                "timestamp": datetime.now().isoformat()
            })
        
        return {
            "success": True,
            "object_id": request.object_id,
            "participants": request.participants,
            "timestamp": datetime.now().isoformat()
        }
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Invalid consistency level: {request.consistency_level}")
    except Exception as e:
        logger.error(f"Error creating shared object: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/shared/list", response_model=Dict[str, Any])
async def list_shared_objects(
    agent_id: Optional[str] = Query(None, description="Filter by agent participation"),
    current_user: UserInfo = Depends(get_current_user)
):
    """List all shared memory objects"""
    try:
        all_objects = memory_coordinator.shared_memory.list_objects()
        objects_info = []
        
        for obj_id in all_objects:
            info = memory_coordinator.shared_memory.get_object_info(obj_id)
            if info:
                # Filter by agent if specified
                if agent_id and agent_id not in info['participants']:
                    continue
                
                objects_info.append({
                    'id': obj_id,
                    'type': info['type'],
                    'participants': list(info['participants']),
                    'version': info['version'],
                    'locked': info['locked_by'] is not None,
                    'created_at': info['created_at']
                })
        
        return {
            "success": True,
            "objects": objects_info,
            "total": len(objects_info),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error listing shared objects: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Crystallized Memory Endpoints
# ============================================================================

@router.get("/crystallized/list", response_model=Dict[str, Any])
async def list_crystals(
    crystal_type: Optional[str] = Query(None, description="Filter by crystal type"),
    min_strength: float = Query(0.0, description="Minimum strength threshold"),
    participant: Optional[str] = Query(None, description="Filter by participant"),
    current_user: UserInfo = Depends(get_current_user)
):
    """List crystallized memories with optional filters"""
    try:
        crystals = []
        
        for crystal_id, crystal in memory_coordinator.crystallized_store.crystals.items():
            # Apply filters
            if crystal_type and crystal.type.value != crystal_type:
                continue
            if crystal.strength < min_strength:
                continue
            if participant and participant not in crystal.participants:
                continue
            
            crystals.append({
                'id': crystal.id,
                'type': crystal.type.value,
                'participants': list(crystal.participants),
                'strength': crystal.strength,
                'formation_count': crystal.formation_count,
                'last_accessed': crystal.last_accessed.isoformat(),
                'content_preview': str(crystal.content)[:100] + "..." if len(str(crystal.content)) > 100 else str(crystal.content)
            })
        
        # Sort by strength
        crystals.sort(key=lambda x: x['strength'], reverse=True)
        
        return {
            "success": True,
            "crystals": crystals,
            "total": len(crystals),
            "filters": {
                "crystal_type": crystal_type,
                "min_strength": min_strength,
                "participant": participant
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error listing crystals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/crystallized/{crystal_id}", response_model=Dict[str, Any])
async def get_crystal_details(
    crystal_id: str,
    current_user: UserInfo = Depends(get_current_user)
):
    """Get detailed information about a specific crystal"""
    try:
        if crystal_id not in memory_coordinator.crystallized_store.crystals:
            raise HTTPException(status_code=404, detail=f"Crystal {crystal_id} not found")
        
        crystal = memory_coordinator.crystallized_store.crystals[crystal_id]
        
        # Get connected crystals
        connections = []
        if crystal_id in memory_coordinator.crystallized_store.crystal_connections:
            for connected_id in memory_coordinator.crystallized_store.crystal_connections[crystal_id]:
                if connected_id in memory_coordinator.crystallized_store.crystals:
                    connected = memory_coordinator.crystallized_store.crystals[connected_id]
                    connections.append({
                        'id': connected_id,
                        'type': connected.type.value,
                        'strength': connected.strength
                    })
        
        return {
            "success": True,
            "crystal": {
                'id': crystal.id,
                'type': crystal.type.value,
                'participants': list(crystal.participants),
                'strength': crystal.strength,
                'formation_count': crystal.formation_count,
                'last_accessed': crystal.last_accessed.isoformat(),
                'created_at': crystal.timestamp.isoformat(),
                'content': crystal.content,
                'metadata': crystal.metadata,
                'connections': connections
            },
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting crystal details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Collective Memory Endpoints
# ============================================================================

@router.get("/collective/insights", response_model=Dict[str, Any])
async def get_collective_insights(
    insight_type: Optional[str] = Query(None, description="Filter by insight type"),
    min_confidence: float = Query(0.6, description="Minimum confidence score"),
    contributor: Optional[str] = Query(None, description="Filter by contributor agent"),
    limit: int = Query(20, description="Maximum results to return"),
    current_user: UserInfo = Depends(get_current_user)
):
    """Get collective insights from the network"""
    try:
        filters = {
            'min_confidence': min_confidence
        }
        
        if contributor:
            filters['contributor'] = contributor
        
        # Get insights from collective engine
        all_insights = await memory_coordinator.collective_engine.get_insights(filters)
        
        # Apply additional filters
        insights = []
        for insight in all_insights:
            if insight_type and insight.type.value != insight_type:
                continue
            
            insights.append({
                'id': insight.id,
                'type': insight.type.value,
                'description': insight.description,
                'confidence': insight.confidence,
                'impact_score': insight.impact_score,
                'contributors': list(insight.contributors),
                'timestamp': insight.timestamp.isoformat(),
                'metadata': insight.metadata
            })
            
            if len(insights) >= limit:
                break
        
        # Sort by impact score
        insights.sort(key=lambda x: x['impact_score'], reverse=True)
        
        return {
            "success": True,
            "insights": insights,
            "total": len(insights),
            "network_stats": memory_coordinator.collective_engine.get_network_stats(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting collective insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/collective/analyze", response_model=Dict[str, Any])
async def trigger_collective_analysis(
    background_tasks: BackgroundTasks,
    force: bool = Query(False, description="Force analysis even with small queue"),
    current_user: UserInfo = Depends(get_current_user)
):
    """Manually trigger collective analysis"""
    try:
        queue_size = len(memory_coordinator._interaction_queue)
        
        if queue_size < 5 and not force:
            return {
                "success": False,
                "error": f"Insufficient interactions in queue ({queue_size}/5). Use force=true to override.",
                "queue_size": queue_size
            }
        
        # Run analysis in background
        background_tasks.add_task(
            memory_coordinator._run_collective_analysis
        )
        
        return {
            "success": True,
            "message": "Collective analysis scheduled",
            "queue_size": queue_size,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error triggering collective analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# WebSocket Endpoint for Real-time Updates
# ============================================================================

@router.websocket("/stream")
async def memory_stream(
    websocket: WebSocket,
    agent_id: Optional[str] = Query(None, description="Subscribe to specific agent updates")
):
    """
    WebSocket endpoint for real-time memory updates.
    
    Clients can subscribe to:
    - General memory events (all agents)
    - Agent-specific events
    - Memory statistics updates
    """
    await manager.connect(websocket, agent_id)
    
    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "event": "connected",
            "agent_id": agent_id,
            "timestamp": datetime.now().isoformat()
        })
        
        # Send initial stats
        stats = memory_coordinator.get_memory_stats()
        await websocket.send_json({
            "event": "stats_update",
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for any message from client
                data = await websocket.receive_json()
                
                # Handle different message types
                if data.get("type") == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    })
                elif data.get("type") == "get_stats":
                    stats = memory_coordinator.get_memory_stats()
                    await websocket.send_json({
                        "event": "stats_update",
                        "stats": stats,
                        "timestamp": datetime.now().isoformat()
                    })
                elif data.get("type") == "subscribe":
                    # Handle subscription changes
                    new_agent_id = data.get("agent_id")
                    if new_agent_id != agent_id:
                        manager.disconnect(websocket, agent_id)
                        agent_id = new_agent_id
                        await manager.connect(websocket, agent_id)
                        await websocket.send_json({
                            "event": "subscription_changed",
                            "agent_id": agent_id,
                            "timestamp": datetime.now().isoformat()
                        })
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await websocket.send_json({
                    "error": "Invalid JSON",
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await websocket.send_json({
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
                
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    finally:
        manager.disconnect(websocket, agent_id)

# ============================================================================
# Utility Endpoints
# ============================================================================

@router.post("/optimize", response_model=Dict[str, Any])
async def optimize_memory(
    layer: Optional[str] = Query(None, description="Specific layer to optimize"),
    background_tasks: BackgroundTasks,
    current_user: UserInfo = Depends(get_current_user)
):
    """
    Trigger memory optimization routines.
    
    This can include:
    - Working memory garbage collection
    - Crystal consolidation
    - Shared memory cleanup
    - Pattern re-analysis
    """
    try:
        optimization_tasks = []
        
        if not layer or layer == "working":
            # Optimize working memory for all agents
            for agent_id, wm in memory_coordinator.working_memory_agents.items():
                # Clean expired items
                expired_count = 0
                for key in list(wm._memory.keys()):
                    if wm._memory[key].is_expired():
                        wm.remove(key)
                        expired_count += 1
                
                optimization_tasks.append({
                    "layer": "working",
                    "agent_id": agent_id,
                    "expired_removed": expired_count
                })
        
        if not layer or layer == "crystallized":
            # Trigger crystal decay and consolidation
            background_tasks.add_task(
                memory_coordinator.crystallized_store._consolidation_loop
            )
            optimization_tasks.append({
                "layer": "crystallized",
                "action": "consolidation_scheduled"
            })
        
        if not layer or layer == "shared":
            # Clean up orphaned shared objects
            cleaned = 0
            for obj_id in list(memory_coordinator.shared_memory.objects.keys()):
                info = memory_coordinator.shared_memory.get_object_info(obj_id)
                if info and len(info['participants']) == 0:
                    del memory_coordinator.shared_memory.objects[obj_id]
                    cleaned += 1
            
            optimization_tasks.append({
                "layer": "shared",
                "orphaned_removed": cleaned
            })
        
        # Broadcast optimization event
        await manager.broadcast({
            "event": "memory_optimized",
            "tasks": optimization_tasks,
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "success": True,
            "optimization_tasks": optimization_tasks,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error optimizing memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health", response_model=Dict[str, Any])
async def memory_health_check():
    """Check memory system health status"""
    try:
        health_status = {
            "status": "healthy",
            "checks": {}
        }
        
        # Check working memory
        working_memory_ok = len(memory_coordinator.working_memory_agents) > 0
        health_status["checks"]["working_memory"] = {
            "status": "ok" if working_memory_ok else "warning",
            "agents": len(memory_coordinator.working_memory_agents)
        }
        
        # Check shared memory
        try:
            shared_stats = memory_coordinator.shared_memory.get_stats()
            health_status["checks"]["shared_memory"] = {
                "status": "ok",
                "objects": shared_stats.get('total_objects', 0)
            }
        except:
            health_status["checks"]["shared_memory"] = {"status": "error"}
            health_status["status"] = "degraded"
        
        # Check crystallized memory
        try:
            crystal_stats = memory_coordinator.crystallized_store.get_stats()
            health_status["checks"]["crystallized_memory"] = {
                "status": "ok",
                "crystals": crystal_stats.get('total_crystals', 0)
            }
        except:
            health_status["checks"]["crystallized_memory"] = {"status": "error"}
            health_status["status"] = "degraded"
        
        # Check collective memory
        try:
            collective_stats = memory_coordinator.collective_engine.get_network_stats()
            health_status["checks"]["collective_memory"] = {
                "status": "ok",
                "insights": collective_stats.get('total_insights', 0)
            }
        except:
            health_status["checks"]["collective_memory"] = {"status": "error"}
            health_status["status"] = "degraded"
        
        return {
            "success": True,
            "health": health_status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error checking memory health: {e}")
        return {
            "success": False,
            "health": {
                "status": "error",
                "error": str(e)
            },
            "timestamp": datetime.now().isoformat()
        }

# ============================================================================
# Admin Endpoints (require elevated permissions)
# ============================================================================

@router.delete("/admin/reset", response_model=Dict[str, Any])
async def reset_memory_system(
    confirm: bool = Query(False, description="Confirm system reset"),
    current_user: UserInfo = Depends(get_current_user)
):
    """
    Reset the entire memory system (CAUTION: This will clear all data)
    
    Requires admin privileges.
    """
    # Check admin privileges
    if not current_user.roles or "admin" not in current_user.roles:
        raise HTTPException(status_code=403, detail="Admin privileges required")
    
    if not confirm:
        return {
            "success": False,
            "error": "Reset not confirmed. Set confirm=true to proceed.",
            "warning": "This will delete all memory data!"
        }
    
    try:
        # Stop the system
        await memory_coordinator.stop()
        
        # Clear all data
        memory_coordinator.working_memory_agents.clear()
        memory_coordinator.shared_memory.objects.clear()
        memory_coordinator.crystallized_store.crystals.clear()
        memory_coordinator._interaction_queue.clear()
        
        # Restart the system
        await memory_coordinator.start()
        
        # Broadcast reset event
        await manager.broadcast({
            "event": "system_reset",
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "success": True,
            "message": "Memory system reset successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error resetting memory system: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Export router for inclusion in main app
__all__ = ["router", "memory_coordinator", "manager"]