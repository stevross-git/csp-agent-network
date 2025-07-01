"""
WebSocket Network Integration
============================

Extensions to WebSocket manager for network-aware collaboration.
"""

from typing import Dict, Any, Optional, Set, List
from uuid import UUID
from datetime import datetime
import logging

from backend.realtime.websocket_manager import WebSocketEvent, EventType

logger = logging.getLogger(__name__)


class NetworkAwareWebSocketManager:
    """Extension for network-aware WebSocket management."""
    
    def __init__(self, base_manager, network_service):
        self.base_manager = base_manager
        self.network_service = network_service
        self.remote_sessions: Dict[str, Dict[str, Any]] = {}  # Track remote users
        
    async def handle_network_design_update(self, update: Dict[str, Any]):
        """Handle design update from network node."""
        design_id = UUID(update["design_id"])
        event_data = update["event"]
        sender_node = update.get("sender")
        
        # Create WebSocket event from network update
        event = WebSocketEvent(
            type=EventType(event_data["type"]),
            data=event_data["data"],
            user_id=f"network:{sender_node[:8]}" if sender_node else "network:unknown",
            timestamp=datetime.fromisoformat(event_data["timestamp"])
        )
        
        # Broadcast to local clients only (avoid echo)
        await self.base_manager.broadcast_to_design(
            design_id, 
            event,
            exclude_network=True  # Custom flag to prevent network re-broadcast
        )
        
    async def register_remote_user(self, design_id: str, node_id: str, user_info: Dict[str, Any]):
        """Register a remote user from another node."""
        session_key = f"{node_id}:{user_info['user_id']}"
        
        self.remote_sessions[session_key] = {
            "design_id": design_id,
            "node_id": node_id,
            "user_info": user_info,
            "last_seen": datetime.utcnow()
        }
        
        # Notify local users
        await self.base_manager.broadcast_to_design(
            UUID(design_id),
            WebSocketEvent(
                type=EventType.USER_JOIN,
                data={
                    "user_id": session_key,
                    "display_name": f"{user_info.get('name', 'Remote User')} (Network)",
                    "is_remote": True,
                    "node_id": node_id[:8]
                }
            )
        )
        
    async def get_all_design_users(self, design_id: UUID) -> List[Dict[str, Any]]:
        """Get all users including remote network users."""
        # Get local users
        local_users = await self.base_manager.get_design_users(design_id)
        
        # Add remote users
        remote_users = []
        design_id_str = str(design_id)
        
        for session_key, session in self.remote_sessions.items():
            if session["design_id"] == design_id_str:
                remote_users.append({
                    "user_id": session_key,
                    "display_name": session["user_info"].get("name", "Remote User"),
                    "is_remote": True,
                    "node_id": session["node_id"][:8],
                    "last_activity": session["last_seen"].isoformat()
                })
                
        return local_users + remote_users
