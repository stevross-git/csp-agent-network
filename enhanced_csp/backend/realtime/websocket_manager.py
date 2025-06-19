# File: backend/realtime/websocket_manager.py
"""
Real-time WebSocket Manager for CSP Visual Designer
=================================================
Handles real-time collaboration, live updates, and event streaming
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Set, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from uuid import UUID, uuid4
from datetime import datetime, timedelta
from collections import defaultdict

from fastapi import WebSocket, WebSocketDisconnect
import redis.asyncio as redis
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ============================================================================
# EVENT TYPES AND SCHEMAS
# ============================================================================

class EventType(str, Enum):
    """WebSocket event types"""
    # Design events
    DESIGN_UPDATE = "design_update"
    NODE_CREATE = "node_create"
    NODE_UPDATE = "node_update"
    NODE_DELETE = "node_delete"
    CONNECTION_CREATE = "connection_create"
    CONNECTION_UPDATE = "connection_update"
    CONNECTION_DELETE = "connection_delete"
    
    # Collaboration events
    USER_JOIN = "user_join"
    USER_LEAVE = "user_leave"
    CURSOR_MOVE = "cursor_move"
    NODE_SELECT = "node_select"
    NODE_DRAG_START = "node_drag_start"
    NODE_DRAG_END = "node_drag_end"
    
    # Execution events
    EXECUTION_START = "execution_start"
    EXECUTION_UPDATE = "execution_update"
    EXECUTION_COMPLETE = "execution_complete"
    EXECUTION_ERROR = "execution_error"
    
    # System events
    SYSTEM_STATUS = "system_status"
    ERROR = "error"
    HEARTBEAT = "heartbeat"

@dataclass
class WebSocketEvent:
    """WebSocket event structure"""
    type: EventType
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    sender_id: Optional[str] = None
    design_id: Optional[UUID] = None
    session_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "sender_id": self.sender_id,
            "design_id": str(self.design_id) if self.design_id else None,
            "session_id": self.session_id
        }

@dataclass
class UserSession:
    """User session information"""
    user_id: str
    session_id: str
    websocket: WebSocket
    design_id: Optional[UUID] = None
    last_activity: datetime = field(default_factory=datetime.now)
    cursor_position: Optional[Dict[str, float]] = None
    selected_nodes: Set[str] = field(default_factory=set)
    is_active: bool = True

class ConnectionManager:
    """WebSocket connection manager with Redis pub/sub for scaling"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.active_connections: Dict[str, UserSession] = {}
        self.design_subscribers: Dict[UUID, Set[str]] = defaultdict(set)
        self.execution_subscribers: Dict[str, Set[str]] = defaultdict(set)
        self.cleanup_task: Optional[asyncio.Task] = None
        self.heartbeat_interval = 30  # seconds
        self.session_timeout = 300  # 5 minutes
        
        # Start background tasks
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        self.cleanup_task = asyncio.create_task(self._cleanup_inactive_sessions())
        
        if self.redis_client:
            asyncio.create_task(self._redis_message_handler())
    
    async def connect(self, websocket: WebSocket, user_id: str, design_id: Optional[UUID] = None) -> str:
        """Accept a new WebSocket connection"""
        await websocket.accept()
        
        session_id = str(uuid4())
        session = UserSession(
            user_id=user_id,
            session_id=session_id,
            websocket=websocket,
            design_id=design_id
        )
        
        self.active_connections[session_id] = session
        
        # Subscribe to design updates
        if design_id:
            self.design_subscribers[design_id].add(session_id)
        
        # Send welcome message
        await self.send_personal_message(session_id, WebSocketEvent(
            type=EventType.USER_JOIN,
            data={
                "session_id": session_id,
                "user_id": user_id,
                "design_id": str(design_id) if design_id else None,
                "message": "Connected successfully"
            }
        ))
        
        # Notify other users in the same design
        if design_id:
            await self.broadcast_to_design(design_id, WebSocketEvent(
                type=EventType.USER_JOIN,
                data={
                    "user_id": user_id,
                    "session_id": session_id,
                    "design_id": str(design_id)
                },
                sender_id=session_id
            ), exclude_sender=True)
        
        logger.info(f"User {user_id} connected with session {session_id}")
        return session_id
    
    async def disconnect(self, session_id: str):
        """Disconnect a WebSocket session"""
        if session_id not in self.active_connections:
            return
        
        session = self.active_connections[session_id]
        
        # Remove from design subscribers
        if session.design_id:
            self.design_subscribers[session.design_id].discard(session_id)
            
            # Notify other users
            await self.broadcast_to_design(session.design_id, WebSocketEvent(
                type=EventType.USER_LEAVE,
                data={
                    "user_id": session.user_id,
                    "session_id": session_id,
                    "design_id": str(session.design_id)
                },
                sender_id=session_id
            ), exclude_sender=True)
        
        # Remove from execution subscribers
        for execution_id in list(self.execution_subscribers.keys()):
            self.execution_subscribers[execution_id].discard(session_id)
        
        # Close WebSocket connection
        try:
            await session.websocket.close()
        except Exception:
            pass
        
        # Remove session
        del self.active_connections[session_id]
        
        logger.info(f"User {session.user_id} disconnected (session {session_id})")
    
    async def send_personal_message(self, session_id: str, event: WebSocketEvent):
        """Send message to a specific session"""
        if session_id not in self.active_connections:
            return False
        
        session = self.active_connections[session_id]
        session.last_activity = datetime.now()
        
        try:
            await session.websocket.send_text(json.dumps(event.to_dict()))
            return True
        except WebSocketDisconnect:
            await self.disconnect(session_id)
            return False
        except Exception as e:
            logger.error(f"Error sending message to session {session_id}: {e}")
            await self.disconnect(session_id)
            return False
    
    async def broadcast_to_design(self, design_id: UUID, event: WebSocketEvent, 
                                exclude_sender: bool = False):
        """Broadcast message to all users viewing a design"""
        if design_id not in self.design_subscribers:
            return
        
        subscribers = self.design_subscribers[design_id].copy()
        successful_sends = 0
        
        for session_id in subscribers:
            if exclude_sender and session_id == event.sender_id:
                continue
            
            if await self.send_personal_message(session_id, event):
                successful_sends += 1
        
        # Also publish to Redis for scaling across multiple instances
        if self.redis_client:
            await self._publish_to_redis(f"design:{design_id}", event)
        
        return successful_sends
    
    async def broadcast_to_execution(self, execution_id: str, event: WebSocketEvent):
        """Broadcast message to all users monitoring an execution"""
        if execution_id not in self.execution_subscribers:
            return
        
        subscribers = self.execution_subscribers[execution_id].copy()
        successful_sends = 0
        
        for session_id in subscribers:
            if await self.send_personal_message(session_id, event):
                successful_sends += 1
        
        # Publish to Redis
        if self.redis_client:
            await self._publish_to_redis(f"execution:{execution_id}", event)
        
        return successful_sends
    
    async def subscribe_to_execution(self, session_id: str, execution_id: str):
        """Subscribe session to execution updates"""
        if session_id in self.active_connections:
            self.execution_subscribers[execution_id].add(session_id)
            logger.info(f"Session {session_id} subscribed to execution {execution_id}")
    
    async def unsubscribe_from_execution(self, session_id: str, execution_id: str):
        """Unsubscribe session from execution updates"""
        self.execution_subscribers[execution_id].discard(session_id)
        logger.info(f"Session {session_id} unsubscribed from execution {execution_id}")
    
    async def update_user_cursor(self, session_id: str, x: float, y: float):
        """Update user cursor position"""
        if session_id not in self.active_connections:
            return
        
        session = self.active_connections[session_id]
        session.cursor_position = {"x": x, "y": y}
        session.last_activity = datetime.now()
        
        # Broadcast cursor position to other users in the same design
        if session.design_id:
            await self.broadcast_to_design(session.design_id, WebSocketEvent(
                type=EventType.CURSOR_MOVE,
                data={
                    "user_id": session.user_id,
                    "position": {"x": x, "y": y}
                },
                sender_id=session_id,
                design_id=session.design_id
            ), exclude_sender=True)
    
    async def update_node_selection(self, session_id: str, selected_nodes: List[str]):
        """Update user's selected nodes"""
        if session_id not in self.active_connections:
            return
        
        session = self.active_connections[session_id]
        session.selected_nodes = set(selected_nodes)
        session.last_activity = datetime.now()
        
        # Broadcast selection to other users
        if session.design_id:
            await self.broadcast_to_design(session.design_id, WebSocketEvent(
                type=EventType.NODE_SELECT,
                data={
                    "user_id": session.user_id,
                    "selected_nodes": selected_nodes
                },
                sender_id=session_id,
                design_id=session.design_id
            ), exclude_sender=True)
    
    async def handle_node_drag(self, session_id: str, node_id: str, 
                             position: Dict[str, float], is_start: bool = True):
        """Handle node drag events"""
        if session_id not in self.active_connections:
            return
        
        session = self.active_connections[session_id]
        session.last_activity = datetime.now()
        
        event_type = EventType.NODE_DRAG_START if is_start else EventType.NODE_DRAG_END
        
        if session.design_id:
            await self.broadcast_to_design(session.design_id, WebSocketEvent(
                type=event_type,
                data={
                    "user_id": session.user_id,
                    "node_id": node_id,
                    "position": position
                },
                sender_id=session_id,
                design_id=session.design_id
            ), exclude_sender=True)
    
    async def get_design_users(self, design_id: UUID) -> List[Dict[str, Any]]:
        """Get list of users currently viewing a design"""
        if design_id not in self.design_subscribers:
            return []
        
        users = []
        for session_id in self.design_subscribers[design_id]:
            if session_id in self.active_connections:
                session = self.active_connections[session_id]
                users.append({
                    "user_id": session.user_id,
                    "session_id": session_id,
                    "cursor_position": session.cursor_position,
                    "selected_nodes": list(session.selected_nodes),
                    "last_activity": session.last_activity.isoformat()
                })
        
        return users
    
    async def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        now = datetime.now()
        active_count = sum(1 for session in self.active_connections.values() 
                          if (now - session.last_activity).seconds < 60)
        
        return {
            "total_connections": len(self.active_connections),
            "active_connections": active_count,
            "designs_with_users": len([d for d in self.design_subscribers.values() if d]),
            "executions_with_subscribers": len([e for e in self.execution_subscribers.values() if e]),
            "uptime": "calculated_elsewhere"  # This would be calculated by the main app
        }
    
    async def _publish_to_redis(self, channel: str, event: WebSocketEvent):
        """Publish event to Redis for inter-instance communication"""
        if not self.redis_client:
            return
        
        try:
            await self.redis_client.publish(f"websocket:{channel}", json.dumps(event.to_dict()))
        except Exception as e:
            logger.error(f"Failed to publish to Redis: {e}")
    
    async def _redis_message_handler(self):
        """Handle incoming Redis pub/sub messages"""
        if not self.redis_client:
            return
        
        pubsub = self.redis_client.pubsub()
        await pubsub.subscribe("websocket:*")
        
        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    await self._handle_redis_message(message)
        except Exception as e:
            logger.error(f"Redis message handler error: {e}")
        finally:
            await pubsub.close()
    
    async def _handle_redis_message(self, message):
        """Handle a message received from Redis"""
        try:
            channel = message["channel"].decode("utf-8")
            data = json.loads(message["data"].decode("utf-8"))
            
            # Parse channel to determine routing
            if channel.startswith("websocket:design:"):
                design_id = UUID(channel.split(":")[-1])
                event = WebSocketEvent(**data)
                await self.broadcast_to_design(design_id, event, exclude_sender=True)
            
            elif channel.startswith("websocket:execution:"):
                execution_id = channel.split(":")[-1]
                event = WebSocketEvent(**data)
                await self.broadcast_to_execution(execution_id, event)
                
        except Exception as e:
            logger.error(f"Error handling Redis message: {e}")
    
    async def _cleanup_inactive_sessions(self):
        """Background task to clean up inactive sessions"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                now = datetime.now()
                inactive_sessions = []
                
                for session_id, session in self.active_connections.items():
                    if (now - session.last_activity).seconds > self.session_timeout:
                        inactive_sessions.append(session_id)
                
                for session_id in inactive_sessions:
                    logger.info(f"Cleaning up inactive session: {session_id}")
                    await self.disconnect(session_id)
                    
            except Exception as e:
                logger.error(f"Session cleanup error: {e}")
    
    async def send_heartbeat(self):
        """Send heartbeat to all active connections"""
        event = WebSocketEvent(
            type=EventType.HEARTBEAT,
            data={"timestamp": datetime.now().isoformat()}
        )
        
        for session_id in list(self.active_connections.keys()):
            await self.send_personal_message(session_id, event)
    
    async def shutdown(self):
        """Shutdown the connection manager"""
        # Cancel cleanup task
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Disconnect all sessions
        for session_id in list(self.active_connections.keys()):
            await self.disconnect(session_id)
        
        logger.info("WebSocket connection manager shut down")

# Global connection manager instance
connection_manager = ConnectionManager()

# WebSocket endpoint handlers
async def websocket_endpoint(websocket: WebSocket, user_id: str, design_id: Optional[str] = None):
    """Main WebSocket endpoint handler"""
    session_id = None
    
    try:
        # Convert design_id to UUID if provided
        design_uuid = UUID(design_id) if design_id else None
        
        # Connect user
        session_id = await connection_manager.connect(websocket, user_id, design_uuid)
        
        # Send current design users if applicable
        if design_uuid:
            current_users = await connection_manager.get_design_users(design_uuid)
            await connection_manager.send_personal_message(session_id, WebSocketEvent(
                type=EventType.SYSTEM_STATUS,
                data={
                    "current_users": current_users,
                    "design_id": str(design_uuid)
                }
            ))
        
        # Message handling loop
        while True:
            try:
                # Receive message
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Update last activity
                if session_id in connection_manager.active_connections:
                    connection_manager.active_connections[session_id].last_activity = datetime.now()
                
                # Handle different message types
                await handle_websocket_message(session_id, message)
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await connection_manager.send_personal_message(session_id, WebSocketEvent(
                    type=EventType.ERROR,
                    data={"message": "Invalid JSON format"}
                ))
            except Exception as e:
                logger.error(f"WebSocket message handling error: {e}")
                await connection_manager.send_personal_message(session_id, WebSocketEvent(
                    type=EventType.ERROR,
                    data={"message": f"Message handling error: {str(e)}"}
                ))
                
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    finally:
        if session_id:
            await connection_manager.disconnect(session_id)

async def handle_websocket_message(session_id: str, message: Dict[str, Any]):
    """Handle incoming WebSocket messages"""
    message_type = message.get("type")
    data = message.get("data", {})
    
    if message_type == "cursor_move":
        await connection_manager.update_user_cursor(
            session_id, 
            data.get("x", 0), 
            data.get("y", 0)
        )
    
    elif message_type == "node_select":
        await connection_manager.update_node_selection(
            session_id, 
            data.get("selected_nodes", [])
        )
    
    elif message_type == "node_drag_start":
        await connection_manager.handle_node_drag(
            session_id,
            data.get("node_id"),
            data.get("position", {}),
            is_start=True
        )
    
    elif message_type == "node_drag_end":
        await connection_manager.handle_node_drag(
            session_id,
            data.get("node_id"),
            data.get("position", {}),
            is_start=False
        )
    
    elif message_type == "subscribe_execution":
        execution_id = data.get("execution_id")
        if execution_id:
            await connection_manager.subscribe_to_execution(session_id, execution_id)
    
    elif message_type == "unsubscribe_execution":
        execution_id = data.get("execution_id")
        if execution_id:
            await connection_manager.unsubscribe_from_execution(session_id, execution_id)
    
    elif message_type == "ping":
        await connection_manager.send_personal_message(session_id, WebSocketEvent(
            type=EventType.HEARTBEAT,
            data={"pong": True, "timestamp": datetime.now().isoformat()}
        ))
    
    else:
        logger.warning(f"Unknown message type: {message_type}")

# Utility functions for broadcasting events
async def broadcast_design_event(design_id: UUID, event_type: EventType, data: Dict[str, Any], 
                                sender_id: Optional[str] = None):
    """Broadcast a design-related event to all subscribers"""
    event = WebSocketEvent(
        type=event_type,
        data=data,
        design_id=design_id,
        sender_id=sender_id
    )
    
    return await connection_manager.broadcast_to_design(design_id, event)

async def broadcast_execution_event(execution_id: str, event_type: EventType, data: Dict[str, Any]):
    """Broadcast an execution-related event to all subscribers"""
    event = WebSocketEvent(
        type=event_type,
        data=data,
        session_id=execution_id
    )
    
    return await connection_manager.broadcast_to_execution(execution_id, event)

# Initialize connection manager with Redis
async def init_websocket_manager(redis_client: redis.Redis):
    """Initialize the WebSocket manager with Redis client"""
    global connection_manager
    connection_manager.redis_client = redis_client
    logger.info("✅ WebSocket manager initialized with Redis support")

async def shutdown_websocket_manager():
    """Shutdown the WebSocket manager"""
    await connection_manager.shutdown()
