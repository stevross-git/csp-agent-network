# examples/optimized_chat_application.py
"""
Chat Application with Network Optimization
"""

import asyncio
import logging
from typing import Dict, Any

from enhanced_csp.agents import BaseAgent
from enhanced_csp.ai_comm import AdvancedCommPattern

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatAgent(BaseAgent):
    """Chat agent with optimized networking"""
    
    def __init__(self, agent_id: str, username: str):
        # Enable aggressive batching for chat messages
        config = {
            'network_optimization': {
                'enabled': True,
                'compression': {
                    'default_algorithm': 'lz4',
                    'min_size_bytes': 100  # Compress even small messages
                },
                'batching': {
                    'max_wait_ms': 100,  # Higher latency tolerance for chat
                    'max_size': 50
                },
                'connection_pool': {
                    'max': 20  # Limit connections for chat app
                },
                'protocol': {
                    'max_message_mb': 2  # 2MB limit for chat (prevents huge pastes)
                }
            }
        }
        super().__init__(agent_id, config)
        self.username = username
        self.chat_history = []
        
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process chat-related tasks"""
        task_type = task.get('type')
        
        if task_type == 'send_message':
            return await self.send_chat_message(task['message'], task.get('channel', 'general'))
        elif task_type == 'join_channel':
            return await self.join_channel(task['channel'])
        else:
            return {'error': f'Unknown task type: {task_type}'}
            
    async def send_chat_message(self, message: str, channel: str) -> Dict[str, Any]:
        """Send a chat message"""
        chat_msg = {
            'type': 'chat_message',
            'username': self.username,
            'message': message,
            'timestamp': asyncio.get_event_loop().time()
        }
        
        # Messages under 500 chars get normal priority
        # Longer messages (like pastes) get lower priority
        priority = 5 if len(message) < 500 else 3
        
        success = await self.send_message(
            channel_id=channel,
            recipient='*',  # Broadcast
            message=chat_msg,
            priority=priority
        )
        
        if success:
            self.chat_history.append(chat_msg)
            
        return {'success': success}
        
    async def join_channel(self, channel_name: str) -> Dict[str, Any]:
        """Join a chat channel"""
        # Create optimized broadcast channel for chat
        channel = await self.create_channel(
            channel_id=channel_name,
            pattern=AdvancedCommPattern.BROADCAST,
            endpoint=f"http://chat-server/channels/{channel_name}"
        )
        
        # Send join notification (high priority)
        await self.send_message(
            channel_id=channel_name,
            recipient='*',
            message={
                'type': 'user_joined',
                'username': self.username
            },
            priority=8
        )
        
        return {'channel': channel_name, 'status': 'joined'}

async def main():
    """Run optimized chat demonstration"""
    
    # Create chat agents
    alice = ChatAgent('alice_001', 'Alice')
    bob = ChatAgent('bob_001', 'Bob')
    
    # Start agents
    await alice.start()
    await bob.start()
    
    # Join channels
    await alice.join_channel('general')
    await bob.join_channel('general')
    
    # Simulate chat conversation
    messages = [
        ("Alice", "Hi everyone! 👋"),
        ("Bob", "Hey Alice! How's the new network optimization working?"),
        ("Alice", "It's amazing! Messages are batched automatically."),
        ("Bob", "And the compression is transparent. Love it!"),
        ("Alice", "Let me paste this config file...\n" + "config:\n  key: value\n" * 20),
        ("Bob", "Wow, that was fast despite being large!")
    ]
    
    for username, message in messages:
        agent = alice if username == "Alice" else bob
        await agent.send_chat_message(message, 'general')
        await asyncio.sleep(0.5)  # Simulate typing
        
    # Check metrics (note: get_metrics is synchronous)
    logger.info("=== Network Optimization Metrics ===")
    for agent in [alice, bob]:
        metrics = agent.channels['general'].get_metrics()  # No await needed
        logger.info(f"{agent.username}: {metrics}")
        
    # Cleanup
    await alice.stop()
    await bob.stop()

if __name__ == "__main__":
    asyncio.run(main())