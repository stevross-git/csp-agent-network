# enhanced_csp/network/examples/chat_application.py
"""
Example: Decentralized Chat Application using Enhanced CSP Network Stack
"""

import asyncio
import sys
import json
from datetime import datetime
from typing import Dict, Set, Optional

from enhanced_csp.network import create_network, NetworkConfig, P2PConfig


class DecentralizedChat:
    """Simple decentralized chat application"""
    
    def __init__(self, username: str):
        self.username = username
        self.network = None
        self.chat_rooms: Dict[str, Set[str]] = {}  # room -> participants
        self.message_history: Dict[str, list] = {}  # room -> messages
        
    async def start(self, bootstrap_nodes: list = None):
        """Start the chat application"""
        print(f"🚀 Starting decentralized chat for {self.username}...")
        
        # Configure network
        config = NetworkConfig(
            p2p=P2PConfig(
                bootstrap_nodes=bootstrap_nodes or [],
                enable_mdns=True,  # Find peers on local network
                max_peers=30
            )
        )
        
        # Create and start network
        self.network = await create_network(config)
        
        # Register our username in DNS
        await self._register_username()
        
        # Set up message handlers
        self.network.node.on_event('chat_message', self._handle_chat_message)
        self.network.node.on_event('join_room', self._handle_join_room)
        self.network.node.on_event('leave_room', self._handle_leave_room)
        self.network.node.on_event('room_sync', self._handle_room_sync)
        
        # Get our info
        info = self.network.get_network_info()
        print(f"✅ Chat started!")
        print(f"📍 Node ID: {info['node_id'][:16]}...")
        print(f"🌐 DNS name: {self.username}.web4ai")
        print(f"🔗 External address: {info.get('external_address', 'Unknown')}")
        print(f"👥 Connected peers: {info.get('peers', 0)}")
        print("-" * 50)
    
    async def _register_username(self):
        """Register username in DNS overlay"""
        # Register A record with our IP
        await self.network.dns.update_record(
            f"{self.username}.web4ai",
            self.network.dns.DNSRecordType.A,
            self.network.nat.nat_info.external_ip if self.network.nat.nat_info else "127.0.0.1"
        )
        
        # Register TXT record with node ID
        await self.network.dns.update_record(
            f"{self.username}.web4ai",
            self.network.dns.DNSRecordType.TXT,
            f"node={self.network.node.node_id.to_base58()}"
        )
    
    async def join_room(self, room_name: str):
        """Join a chat room"""
        if room_name not in self.chat_rooms:
            self.chat_rooms[room_name] = {self.username}
            self.message_history[room_name] = []
        else:
            self.chat_rooms[room_name].add(self.username)
        
        # Announce join
        await self._broadcast_to_room(room_name, {
            'type': 'join_room',
            'room': room_name,
            'user': self.username,
            'timestamp': datetime.now().isoformat()
        })
        
        print(f"📢 Joined room: {room_name}")
        
        # Request room sync from peers
        await self._request_room_sync(room_name)
    
    async def leave_room(self, room_name: str):
        """Leave a chat room"""
        if room_name in self.chat_rooms:
            self.chat_rooms[room_name].discard(self.username)
            
            # Announce leave
            await self._broadcast_to_room(room_name, {
                'type': 'leave_room',
                'room': room_name,
                'user': self.username,
                'timestamp': datetime.now().isoformat()
            })
            
            print(f"👋 Left room: {room_name}")
    
    async def send_message(self, room_name: str, message: str):
        """Send message to a room"""
        if room_name not in self.chat_rooms:
            print(f"❌ Not in room: {room_name}")
            return
        
        chat_message = {
            'type': 'chat_message',
            'room': room_name,
            'user': self.username,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store locally
        self.message_history[room_name].append(chat_message)
        
        # Broadcast to room
        await self._broadcast_to_room(room_name, chat_message)
    
    async def send_private_message(self, recipient: str, message: str):
        """Send private message to a user"""
        # Resolve recipient's node
        response = await self.network.dns.resolve(
            f"{recipient}.web4ai",
            self.network.dns.DNSRecordType.TXT
        )
        
        if not response or not response.records:
            print(f"❌ User {recipient} not found")
            return
        
        # Extract node ID from TXT record
        txt_value = response.records[0].value
        if txt_value.startswith("node="):
            node_id = txt_value[5:]
            
            # Send private message
            await self.network.send_message(node_id, {
                'type': 'chat_message',
                'room': '@private',
                'user': self.username,
                'message': message,
                'timestamp': datetime.now().isoformat()
            })
            
            print(f"📨 Sent private message to {recipient}")
    
    async def _broadcast_to_room(self, room_name: str, message: Dict):
        """Broadcast message to all room participants"""
        # In a real implementation, we'd maintain room membership
        # For now, broadcast to all connected peers
        await self.network.node.broadcast_message(message)
    
    async def _request_room_sync(self, room_name: str):
        """Request room history from peers"""
        await self.network.node.broadcast_message({
            'type': 'room_sync_request',
            'room': room_name,
            'user': self.username
        })
    
    async def _handle_chat_message(self, data: Dict):
        """Handle incoming chat message"""
        room = data.get('room', '')
        user = data.get('user', '')
        message = data.get('message', '')
        timestamp = data.get('timestamp', '')
        
        # Store message
        if room not in self.message_history:
            self.message_history[room] = []
        self.message_history[room].append(data)
        
        # Display message
        if room == '@private':
            print(f"\n🔒 [Private] {user}: {message}")
        else:
            print(f"\n💬 [{room}] {user}: {message}")
        print("> ", end='', flush=True)
    
    async def _handle_join_room(self, data: Dict):
        """Handle room join notification"""
        room = data.get('room', '')
        user = data.get('user', '')
        
        if room not in self.chat_rooms:
            self.chat_rooms[room] = set()
        self.chat_rooms[room].add(user)
        
        if user != self.username:
            print(f"\n➕ {user} joined {room}")
            print("> ", end='', flush=True)
    
    async def _handle_leave_room(self, data: Dict):
        """Handle room leave notification"""
        room = data.get('room', '')
        user = data.get('user', '')
        
        if room in self.chat_rooms:
            self.chat_rooms[room].discard(user)
        
        if user != self.username:
            print(f"\n➖ {user} left {room}")
            print("> ", end='', flush=True)
    
    async def _handle_room_sync(self, data: Dict):
        """Handle room sync response"""
        # In a full implementation, this would sync room history
        pass
    
    def list_rooms(self):
        """List current rooms"""
        print("\n📋 Current rooms:")
        for room, users in self.chat_rooms.items():
            print(f"  • {room} ({len(users)} users): {', '.join(users)}")
    
    def show_history(self, room_name: str, limit: int = 10):
        """Show room message history"""
        if room_name not in self.message_history:
            print(f"❌ No history for room: {room_name}")
            return
        
        messages = self.message_history[room_name][-limit:]
        print(f"\n📜 History for {room_name} (last {limit} messages):")
        
        for msg in messages:
            timestamp = msg.get('timestamp', '')[:16]  # Trim microseconds
            user = msg.get('user', '')
            message = msg.get('message', '')
            print(f"  [{timestamp}] {user}: {message}")
    
    def show_network_stats(self):
        """Show network statistics"""
        info = self.network.get_network_info()
        stats = self.network.get_stats()
        
        print("\n📊 Network Statistics:")
        print(f"  • Node ID: {info['node_id'][:16]}...")
        print(f"  • NAT Type: {info.get('nat_type', 'Unknown')}")
        print(f"  • Connected Peers: {info.get('peers', 0)}")
        print(f"  • Active Routes: {info.get('routes', 0)}")
        print(f"  • Super Peers: {info.get('super_peers', 0)}")
        print(f"  • Active Flows: {info.get('active_flows', 0)}")
        
        if 'adaptive' in stats and 'average_metrics' in stats['adaptive']:
            metrics = stats['adaptive']['average_metrics']
            print(f"  • Avg Latency: {metrics.get('avg_rtt_ms', 0):.1f}ms")
            print(f"  • Avg Bandwidth: {metrics.get('avg_bandwidth_mbps', 0):.1f} Mbps")
            print(f"  • Avg Packet Loss: {metrics.get('avg_packet_loss', 0)*100:.1f}%")


async def main():
    """Main chat application"""
    # Get username
    if len(sys.argv) > 1:
        username = sys.argv[1]
    else:
        username = input("Enter your username: ")
    
    # Bootstrap nodes (optional)
    bootstrap_nodes = []
    if len(sys.argv) > 2:
        bootstrap_nodes = sys.argv[2].split(',')
    
    # Create chat instance
    chat = DecentralizedChat(username)
    
    try:
        # Start chat
        await chat.start(bootstrap_nodes)
        
        # Join default room
        await chat.join_room("general")
        
        # Command loop
        print("\nCommands:")
        print("  /join <room> - Join a room")
        print("  /leave <room> - Leave a room")
        print("  /msg <user> <message> - Send private message")
        print("  /rooms - List rooms")
        print("  /history <room> - Show room history")
        print("  /stats - Show network stats")
        print("  /quit - Exit")
        print("\nType messages to send to current room (general)")
        
        current_room = "general"
        
        # Input handling in background
        async def handle_input():
            nonlocal current_room
            
            while True:
                try:
                    # Get input (blocking, but in executor)
                    line = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: input("> ")
                    )
                    
                    if line.startswith('/'):
                        parts = line.split(' ', 2)
                        cmd = parts[0][1:]  # Remove /
                        
                        if cmd == 'quit':
                            break
                        elif cmd == 'join' and len(parts) > 1:
                            room = parts[1]
                            await chat.join_room(room)
                            current_room = room
                        elif cmd == 'leave' and len(parts) > 1:
                            await chat.leave_room(parts[1])
                        elif cmd == 'msg' and len(parts) > 2:
                            user = parts[1]
                            message = parts[2]
                            await chat.send_private_message(user, message)
                        elif cmd == 'rooms':
                            chat.list_rooms()
                        elif cmd == 'history' and len(parts) > 1:
                            chat.show_history(parts[1])
                        elif cmd == 'stats':
                            chat.show_network_stats()
                        else:
                            print(f"Unknown command: {cmd}")
                    else:
                        # Regular message to current room
                        if line.strip():
                            await chat.send_message(current_room, line)
                            
                except Exception as e:
                    print(f"Error: {e}")
        
        # Run input handler
        await handle_input()
        
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        if chat.network:
            await chat.network.stop()


if __name__ == "__main__":
    asyncio.run(main())